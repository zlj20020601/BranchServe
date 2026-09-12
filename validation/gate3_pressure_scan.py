"""Bounded three-meeting pressure scan, gated on successful salt-fix smoke."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from gate2_paired_suite import Client, ROOT, Suite, metric
from gate3_fanout_smoke import arm, execution_order


def select_workloads(groups):
    ordered = sorted(groups, key=lambda g: (g['shared_tokens'], g['context_sha256']))
    return [ordered[i] for i in sorted({0, len(ordered) // 2, len(ordered) - 1})]


def measurement_key(row):
    return (row['repeat'], row['arm'], row['background_concurrency'], row['context_sha256'])


def run(args):
    base_run_id = args.run_id
    resumed = json.loads(args.resume_from.read_text()) if args.resume_from else None
    deadline = time.monotonic() + 1200
    while resumed is None:
        prior = json.loads(args.after.read_text())
        if prior['status'] == 'failed':
            raise RuntimeError('Prerequisite smoke failed: ' + str(prior.get('error')))
        if prior['status'] == 'complete' and prior.get('restored_worker_pid'):
            if len(prior['arms']) != 3 or not all(all(r['checks'].values()) for r in prior['arms']):
                raise RuntimeError('Prerequisite coverage/checks failed')
            args.worker_pid = prior['restored_worker_pid']
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Prerequisite smoke timeout')
        time.sleep(5)
    if resumed is not None:
        args.worker_pid = resumed['restored_worker_pid']
    raw = args.manifest.read_bytes()
    workloads = select_workloads(json.loads(raw)['groups'])
    with (ROOT / 'artifacts/gate2_exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cmd = Path(f'/proc/{args.worker_pid}/cmdline').read_bytes().split(b'\0')
        assert cmd[cmd.index(b'--port') + 1] == b'8001'
        assert b'--no-enable-prefix-caching' in cmd and b'--kv-transfer-config' not in cmd
        public_metrics = Client(8001).metrics()
        assert all(metric(public_metrics, n) == 0 for n in ['vllm:num_requests_running', 'vllm:num_requests_waiting'])
        suite = Suite(args)
        output = dict(status='running', protocol=dict(kind='qmsum_pressure_pilot',
            manifest_sha256=hashlib.sha256(raw).hexdigest(), fanout=4, pressures=[0, 2, 4, 8],
            repeats=2, warmups_per_strategy_block=1, max_tokens=512,
            selected_contexts=[g['context_sha256'] for g in workloads],
            selected_shared_tokens=[g['shared_tokens'] for g in workloads],
            selection='min, median, max shared length; fixed before outcomes',
            caveat='Three-meeting pilot; strategy-major ordering, synthetic decode pressure; not production traffic.'), arms=[])
        path = suite.output / 'fanout.json'
        if resumed is not None:
            assert resumed['protocol'] == output['protocol'], 'Resume protocol mismatch'
            assert all(all(r['checks'].values()) for r in resumed['arms'])
            keys = [measurement_key(r) for r in resumed['arms'] if r['measured']]
            expected = {(rep, name, p, w['context_sha256']) for rep in [1, 2]
                        for name in execution_order(rep, True) for p in [0, 2, 4, 8]
                        for w in workloads}
            assert len(keys) == len(set(keys)) and set(keys) <= expected
            output.update(arms=resumed['arms'], resumed_from=str(args.resume_from),
                          resumed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        completed = {measurement_key(r) for r in output['arms'] if r['measured']}
        path.write_text(json.dumps(output, indent=2))
        try:
            suite.check_parent()
            os.kill(args.worker_pid, signal.SIGTERM)
            suite.wait_gpu_free()
            for repeat in [1, 2]:
                for name in execution_order(repeat, True):
                    if all((repeat, name, p, w['context_sha256']) in completed
                           for p in [0, 2, 4, 8] for w in workloads):
                        continue
                    if name != 'pack' or suite.mode is None:
                        suite.start('retrieve' if name == 'spread_retrieve' else 'recompute')
                    args.pressure = 0
                    args.workload = workloads[0]
                    arm(suite, name, suite.child, output, repeat, warmup=1)
                    levels = [0, 2, 4, 8] if repeat == 1 else [8, 4, 2, 0]
                    for pressure in levels:
                        args.pressure = pressure
                        # Keep cache namespaces distinct across pressure levels.
                        args.run_id = base_run_id + f'-p{pressure}'
                        for workload in workloads if repeat == 1 else list(reversed(workloads)):
                            key = (repeat, name, pressure, workload['context_sha256'])
                            if key in completed:
                                continue
                            args.workload = workload
                            arm(suite, name, suite.child, output, repeat)
                            completed.add(key)
                    args.run_id = base_run_id
            output['status'] = 'complete'
        except Exception as error:
            output.update(status='failed', error=repr(error))
            raise
        finally:
            try:
                suite.parent_count = None
                suite.start('recompute', restore=True)
                output['restored_worker_pid'] = suite.proc.pid
            except Exception as error:
                output.update(status='failed', restoration_error=repr(error))
                raise
            finally:
                path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
                suite.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--lmcache-pid', type=int, required=True)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    run(args)
