"""Inspect actual outputs for two historical mismatches, then restore GPU1."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import urllib.request

from gate2_paired_suite import Client, MODEL_ID, ROOT, Suite, delta, prompt


def request(client, seed, salt):
    before = client.metrics()
    body = dict(model=MODEL_ID, prompt=prompt(seed, 248320), temperature=0,
                max_tokens=32, return_token_ids=True, cache_salt=salt)
    req = urllib.request.Request(client.url + '/v1/completions',
        data=json.dumps(body).encode(),
        headers=dict(client.headers, **{'Content-Type': 'application/json'}))
    with urllib.request.urlopen(req, timeout=120) as response:
        answer = json.load(response)
    choice = answer['choices'][0]
    ids = choice['token_ids']
    assert len(ids) == answer['usage']['completion_tokens']
    after = client.metrics()
    assert delta(before, after, 'vllm:request_success_total') == 1
    return dict(text=choice['text'], token_ids=ids, usage=answer['usage'],
        finish_reason=choice['finish_reason'], sources={s: delta(before, after,
        'vllm:prompt_tokens_by_source_total', source=s) for s in
        ['local_compute', 'local_cache_hit', 'external_kv_transfer']})


def run(args):
    with (ROOT / 'artifacts/gate2_exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cmd = Path(f'/proc/{args.worker_pid}/cmdline').read_bytes().split(b'\0')
        assert cmd[cmd.index(b'--port') + 1] == b'8001'
        assert b'--no-enable-prefix-caching' in cmd and b'--kv-transfer-config' not in cmd
        suite = Suite(args)
        suite.check_parent()
        public = Client(8001)
        from gate2_paired_suite import metric
        assert all(metric(public.metrics(), name) == 0 for name in
                   ['vllm:num_requests_running', 'vllm:num_requests_waiting'])
        rows = []
        restore = False
        result = {'status': 'running', 'cases': rows}
        path = suite.output / 'actual_outputs.json'
        try:
            for seed in [2609101013, 2609101073]:
                salt = f'{args.run_id}-{seed}'
                parent = request(suite.parent, seed, salt)
                suite.parent_count += 1
                suite.check_parent()
                recompute = request(public, seed, salt)
                assert parent['sources']['local_compute'] == 8512
                assert recompute['sources']['local_compute'] == 8512
                rows.append(dict(seed=seed, salt=salt, parent=parent, recompute=recompute))
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            restore = True
            os.kill(args.worker_pid, signal.SIGTERM)
            suite.wait_gpu_free()
            suite.start('retrieve')
            for row in rows:
                suite.check_parent()
                value = request(suite.child, row['seed'], row['salt'])
                assert value['sources'] == dict(local_compute=64., local_cache_hit=0.,
                                                external_kv_transfer=8448.)
                suite.check_parent()
                row['retrieve'] = value
                row['comparison'] = {}
                for mode in ['recompute', 'retrieve']:
                    a, b = row['parent']['token_ids'], row[mode]['token_ids']
                    first = next((i for i in range(max(len(a), len(b)))
                                  if a[i:i+1] != b[i:i+1]), None)
                    row['comparison'][mode] = dict(equal=a == b, first_difference=first,
                        different_positions=sum(a[i:i+1] != b[i:i+1]
                                                for i in range(max(len(a), len(b)))))
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps(row, ensure_ascii=False), flush=True)
            result['status'] = 'complete'
        except Exception as error:
            result.update(status='failed', error=repr(error))
            raise
        finally:
            if restore:
                suite.parent_count = None
                suite.start('recompute', restore=True)
                result['restored_worker_pid'] = suite.proc.pid
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            suite.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--worker-pid', required=True, type=int)
    parser.add_argument('--lmcache-pid', required=True, type=int)
    run(parser.parse_args())
