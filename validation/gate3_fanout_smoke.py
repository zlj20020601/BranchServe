"""Small live fan-out comparison; feasibility smoke, not a formal benchmark."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import urllib.request

from gate2_paired_suite import Client, MODEL_ID, ROOT, Suite, delta, metric, prompt


def bounded_cache_salt(identity):
    return 'bs-' + hashlib.sha256(identity.encode('utf-8')).hexdigest()


def complete(client, ids, salt, max_tokens=32, ignore_eos=False):
    salt = bounded_cache_salt(salt)
    if len(salt.encode('utf-8')) > 128:
        raise ValueError('cache_salt exceeds connector limit')
    body = dict(model=MODEL_ID, prompt=ids, temperature=0, max_tokens=max_tokens,
                return_token_ids=True, cache_salt=salt)
    if ignore_eos:
        body.update(ignore_eos=True, min_tokens=max_tokens)
    req = urllib.request.Request(client.url + '/v1/completions',
        data=json.dumps(body).encode(),
        headers=dict(client.headers, **{'Content-Type': 'application/json'}))
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.load(response)
    choice = data['choices'][0]
    assert len(choice['token_ids']) == data['usage']['completion_tokens']
    assert data['usage']['prompt_tokens'] == len(ids)
    return dict(elapsed_s=time.perf_counter() - start, text=choice['text'],
                token_ids=choice['token_ids'], usage=data['usage'], finish_reason=choice['finish_reason'])


def execution_order(repeat, alternate):
    orders = [
        ['pack', 'spread_recompute', 'spread_retrieve'],
        ['spread_retrieve', 'spread_recompute', 'pack'],
        ['spread_recompute', 'pack', 'spread_retrieve'],
        ['spread_retrieve', 'pack', 'spread_recompute'],
        ['pack', 'spread_retrieve', 'spread_recompute'],
        ['spread_recompute', 'spread_retrieve', 'pack'],
    ]
    return orders[(repeat - 1) % 6] if alternate else orders[0]


def pending_workloads(rows, workloads, repeat, name):
    completed = {r.get('context_sha256') for r in rows
                 if r['measured'] and r['repeat'] == repeat and r['arm'] == name}
    return [w for w in workloads if (w['context_sha256'] if w else None) not in completed]


def arm(suite, name, child, output, repeat=1, warmup=0):
    salt = f'{suite.args.run_id}-{repeat}-{name}-warmup{warmup}'
    prefix = prompt(2609103000, 248320)[:8448]
    rng = random.Random(2609103001)
    suffixes = [[rng.randrange(1000, 247320) for _ in range(64)] for _ in range(5)]
    parent_ids = prefix + suffixes[0]
    branch_ids = [prefix + suffixes[i+1] for i in range(4)]
    max_tokens = 32
    expected_cached = [8448] * 4
    workload = getattr(suite.args, 'workload', None)
    if workload is not None:
        salt += '-' + workload['context_sha256']
        from prepare_qmsum import common_length
        parent_ids = workload['parent_ids']
        branch_ids = [b['prompt_ids'] for b in workload['branches']]
        max_tokens = 512
        expected_cached = [common_length([parent_ids, ids]) // 528 * 528 for ids in branch_ids]
    suite.check_parent()
    parent = complete(suite.parent, parent_ids, salt)
    suite.parent_count += 1
    suite.check_parent()
    clients = [suite.parent] * 4 if name == 'pack' else [suite.parent] * 2 + [child] * 2
    before0, before1 = suite.parent.metrics(), child.metrics()
    pressure = getattr(suite.args, 'pressure', 0)
    background = []
    pressure_info = dict(configured=pressure, kind='finite synthetic decode background',
                         background_prompt_tokens=16, background_output_tokens=2048)
    # Drain background before global counter validation, but exclude drain from foreground timing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, pressure)) as bg_pool:
        try:
            for slot in range(pressure):
                background.append(bg_pool.submit(complete, suite.parent, parent_ids[:16],
                    salt + f'-background-{slot}', 2048, True))
            if pressure:
                deadline = time.monotonic() + 30
                while True:
                    observed = suite.parent.metrics()
                    running = metric(observed, 'vllm:num_requests_running')
                    waiting = metric(observed, 'vllm:num_requests_waiting')
                    if running == pressure and waiting == 0 and not any(f.done() for f in background):
                        break
                    if time.monotonic() >= deadline or any(f.done() for f in background):
                        raise RuntimeError('Background pressure readiness failed')
                    time.sleep(.05)
                pressure_info.update(running_at_dispatch=running, waiting_at_dispatch=waiting)
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(complete, c, branch_ids[i], salt, max_tokens)
                           for i, c in enumerate(clients)]
                branches = [f.result() for f in futures]
            elapsed = time.perf_counter() - start
            pressure_info['unfinished_at_foreground_end'] = sum(not f.done() for f in background)
            if pressure:
                observed = suite.parent.metrics()
                pressure_info['running_at_foreground_end'] = metric(observed, 'vllm:num_requests_running')
                pressure_info['waiting_at_foreground_end'] = metric(observed, 'vllm:num_requests_waiting')
        finally:
            primary_error = sys.exc_info()[1]
            background_results, background_errors = [], []
            for future in background:
                try:
                    background_results.append(future.result())
                except Exception as error:
                    background_errors.append(error)
            if background_errors:
                suite.record('background_errors', errors=[repr(e) for e in background_errors])
                if primary_error is None:
                    raise background_errors[0]
    pressure_info['background_results'] = background_results
    after0, after1 = suite.parent.metrics(), child.metrics()
    suite.parent_count += (4 if name == 'pack' else 2) + pressure
    suite.check_parent()
    sources = [{s: delta(a, b, 'vllm:prompt_tokens_by_source_total', source=s)
                for s in ['local_compute', 'local_cache_hit', 'external_kv_transfer']}
               for a, b in [(before0, after0), (before1, after1)]]
    counts = [delta(a, b, 'vllm:request_success_total')
              for a, b in [(before0, after0), (before1, after1)]]
    assert counts == ([4 + pressure, 0] if name == 'pack' else [2 + pressure, 2]), counts
    raw_sources = [dict(s) for s in sources]
    if pressure:
        sources[0]['local_compute'] -= 16 * pressure
    checks = dict(request_accounting=True)
    if pressure:
        checks['background_full_overlap'] = (
            pressure_info['unfinished_at_foreground_end'] == pressure
            and pressure_info['running_at_foreground_end'] == pressure
            and pressure_info['waiting_at_foreground_end'] == 0)
        checks['background_token_accounting'] = all(
            b['usage']['prompt_tokens'] == 16 and b['usage']['completion_tokens'] == 2048
            for b in background_results)
    local_indices = range(4) if name == 'pack' else range(2)
    checks['local_path'] = sources[0] == dict(
        local_compute=float(sum(len(branch_ids[i]) - expected_cached[i] for i in local_indices)),
        local_cache_hit=float(sum(expected_cached[i] for i in local_indices)), external_kv_transfer=0.)
    if name == 'spread_recompute':
        checks['remote_path'] = (sources[1]['local_compute'] == sum(map(len, branch_ids[2:]))
            and sources[1]['external_kv_transfer'] in (None, 0)
            and sources[1]['local_cache_hit'] in (None, 0))
    elif name == 'spread_retrieve':
        checks['remote_path'] = sources[1] == dict(
            local_compute=float(sum(len(branch_ids[i]) - expected_cached[i] for i in [2, 3])),
            local_cache_hit=0., external_kv_transfer=float(sum(expected_cached[2:])))
    for index, branch in enumerate(branches):
        if workload is not None:
            branch.update(query_id=workload['branches'][index]['id'],
                          query=workload['branches'][index]['query'],
                          reference_answers=workload['branches'][index]['answers'])
    row = dict(arm=name, repeat=repeat, warmup=warmup, measured=warmup == 0,
               background_concurrency=pressure,
               makespan_s=elapsed, sources=sources, checks=checks,
               branches=branches, parent=parent, completion_tokens=sum(b['usage']['completion_tokens'] for b in branches))
    if pressure:
        row.update(pressure=pressure_info, raw_sources=raw_sources)
    if workload is not None:
        row['context_sha256'] = workload['context_sha256']
    output['arms'].append(row)
    (suite.output / 'fanout.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in row.items() if k not in ['branches', 'parent', 'pressure']}), flush=True)
    assert all(checks.values()), checks


def run(args):
    if args.manifest:
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw)
        args.workload = manifest['groups'][args.group_index]
        args.manifest_sha256 = hashlib.sha256(raw).hexdigest()
    workloads = manifest['groups'] if args.manifest and args.all_groups else [getattr(args, 'workload', None)]
    with (ROOT / 'artifacts/gate2_exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cmd = Path(f'/proc/{args.worker_pid}/cmdline').read_bytes().split(b'\0')
        assert cmd[cmd.index(b'--port')+1] == b'8001'
        assert b'--no-enable-prefix-caching' in cmd and b'--kv-transfer-config' not in cmd
        suite = Suite(args)
        public = Client(8001)
        assert all(metric(public.metrics(), n) == 0 for n in ['vllm:num_requests_running', 'vllm:num_requests_waiting'])
        output = dict(status='running', protocol=dict(fanout=4, prefix=8448, suffix=64,
            max_tokens=32, repeats=args.repeats, warmups_per_arm=args.warmups,
            kind='synthetic_repeated' if args.alternate_order else 'smoke',
            orders=[execution_order(r, args.alternate_order) for r in range(1, args.repeats + 1)],
            caveat='One synthetic input, no background pressure; not representative of real workloads.'), arms=[])
        if args.manifest:
            output['protocol'].update(kind='qmsum_smoke', prefix=None, suffix=None, max_tokens=512,
                manifest_sha256=args.manifest_sha256, context_sha256=args.workload['context_sha256'],
                shared_tokens=args.workload['shared_tokens'],
                aligned_shared_tokens=args.workload['aligned_shared_tokens'],
                enable_thinking=False, caveat='One genuine meeting with four original queries; not an agent trace or formal benchmark.')
            if args.all_groups:
                output['protocol'].update(kind='qmsum_baseline', context_sha256=None,
                    shared_tokens=None, aligned_shared_tokens=None, groups=len(workloads),
                    batching='strategy-major; warmups on first meeting per strategy block',
                    caveat='No background load; strategy-major batches retain time-order confounding; not a production trace.')
        if args.pressure:
            output['protocol'].update(pressure=args.pressure,
                background_kind='finite synthetic decode, unique salt, 16 input / 2048 output tokens',
                caveat='Controlled pressure smoke with real foreground queries, not production traffic; drain excluded.')
        if args.resume_from:
            prior_raw = args.resume_from.read_bytes()
            prior = json.loads(prior_raw)
            assert prior['protocol'] == output['protocol'], 'Resume protocol mismatch'
            assert all(all(r['checks'].values()) for r in prior['arms']), 'Invalid prior rows'
            keys = [(r['repeat'], r['arm'], r.get('context_sha256'))
                    for r in prior['arms'] if r['measured']]
            assert len(keys) == len(set(keys)), 'Duplicate prior measurements'
            output.update(arms=prior['arms'], resumed_from=str(args.resume_from),
                          prior_sha256=hashlib.sha256(prior_raw).hexdigest(),
                          resumed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        restore = False
        try:
            restore = True
            os.kill(args.worker_pid, signal.SIGTERM)
            suite.wait_gpu_free()
            for repeat in range(1, args.repeats + 1):
                for name in execution_order(repeat, args.alternate_order):
                    pending = pending_workloads(output['arms'], workloads, repeat, name)
                    if not pending:
                        continue
                    if name != 'pack' or suite.mode is None:
                        suite.start('retrieve' if name == 'spread_retrieve' else 'recompute')
                    for warmup in range(1, args.warmups + 1):
                        args.workload = workloads[0]
                        arm(suite, name, suite.child, output, repeat, warmup)
                    for workload in pending:
                        args.workload = workload
                        arm(suite, name, suite.child, output, repeat)
            output['status'] = 'complete'
        except Exception as error:
            output.update(status='failed', error=repr(error))
            raise
        finally:
            try:
                if restore:
                    suite.parent_count = None
                    suite.start('recompute', restore=True)
                    output['restored_worker_pid'] = suite.proc.pid
            except Exception as error:
                output.update(status='failed', restoration_error=repr(error))
                raise
            finally:
                (suite.output / 'fanout.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
                suite.log.close()
        if args.rouge_deps and output['status'] == 'complete':
            from score_qmsum import score_run
            score_run(suite.output / 'fanout.json', args.manifest, args.rouge_deps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--worker-pid', required=True, type=int)
    parser.add_argument('--lmcache-pid', required=True, type=int)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--alternate-order', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--group-index', type=int, default=0)
    parser.add_argument('--all-groups', action='store_true')
    parser.add_argument('--rouge-deps', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--pressure', type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error('repeats must be positive and warmups nonnegative')
    if args.all_groups and not args.manifest:
        parser.error('--all-groups requires --manifest')
    if not 0 <= args.pressure <= 8:
        parser.error('--pressure must be between 0 and 8')
    run(args)
