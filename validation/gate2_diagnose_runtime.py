"""Reproduce LMCache store faults, validate a process-local workaround and outputs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

from gate2_paired_suite import Client, CONNECTOR, ENV, MODEL, MODEL_ID, ROOT, delta, metric, prompt


class Diagnosis:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / 'artifacts' / args.run_id
        self.out.mkdir(exist_ok=False)
        self.events = (self.out / 'events.jsonl').open('x', buffering=1)
        self.workers = {}
        self.server = None
        self.phase = None
        self.sequence = 0
        self.counts = {}

    def record(self, event, **data):
        row = dict(event=event, time=time.strftime('%Y-%m-%d %H:%M:%S'), phase=self.phase, **data)
        self.events.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ['parent', 'child', 'result']}, ensure_ascii=False), flush=True)
        return row

    def terminate(self, process):
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=90)

    def stop_workers(self):
        for proc in self.workers.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.workers.values():
            proc.wait(timeout=90)
        self.workers.clear()
        time.sleep(2)

    def launch_server(self, mode):
        self.stop_workers()
        self.terminate(self.server)
        self.phase = mode
        self.server_log = self.out / f'lmcache_{mode}.log'
        env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
        env.pop('CUDA_VISIBLE_DEVICES', None)
        args = [str(ENV / 'bin/python'), str(ROOT / 'lmcache_copy_diagnostic.py'), '--copy-mode', mode,
            'server', '--host', '127.0.0.1', '--port', '5555', '--chunk-size', '528',
            '--separate-object-groups', '--l1-size-gb', '100', '--eviction-policy', 'LRU']
        with self.server_log.open('x') as log:
            self.server = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        self.record('server_start', pid=self.server.pid, copy_mode=mode)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError('LMCache server exited')
            try:
                with socket.create_connection(('127.0.0.1', 5555), timeout=1):
                    return
            except OSError:
                time.sleep(2)
        raise RuntimeError('LMCache port readiness timeout')

    def launch_worker(self, gpu, mode, public=False):
        if gpu in self.workers:
            self.terminate(self.workers.pop(gpu))
            time.sleep(2)
        self.sequence += 1
        port = (8000 if public else 18000) + gpu
        env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'], CUDA_VISIBLE_DEVICES=str(gpu))
        for key in ['LMCACHE_CONFIG_FILE', 'VLLM_API_KEY', 'VLLM_KV_TRANSFER_CONFIG', 'VLLM_KV_CONNECTOR']:
            env.pop(key, None)
        args = [str(ENV / 'bin/vllm'), 'serve', MODEL, '--served-model-name', MODEL_ID,
            '--host', '0.0.0.0' if public else '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
            '--max-model-len', '16384', '--gpu-memory-utilization', '0.88', '--max-num-batched-tokens', '1024']
        args += ['--enable-prefix-caching', '--kv-transfer-config', json.dumps(CONNECTOR)] if mode == 'retrieve' else ['--no-enable-prefix-caching']
        path = self.out / f'worker_{self.sequence:02d}_gpu{gpu}_{mode}_{port}.log'
        with path.open('x') as log:
            self.workers[gpu] = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        self.record('worker_start', gpu=gpu, mode=mode, port=port, pid=self.workers[gpu].pid)
        return port

    def wait_worker(self, gpu, port, mode):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.workers[gpu].poll() is not None:
                raise RuntimeError(f'GPU{gpu} worker exited')
            try:
                client = Client(port)
                client.get('/health')
                values = client.metrics()
                expected = 'True' if mode == 'retrieve' else 'False'
                if metric(values, 'vllm:cache_config_info', enable_prefix_caching=expected) != 1:
                    raise RuntimeError('Runtime cache configuration mismatch')
                self.counts[port] = metric(values, 'vllm:request_success_total')
                self.record('worker_ready', gpu=gpu, port=port, mode=mode)
                return
            except (OSError, ValueError):
                time.sleep(3)
        raise RuntimeError('Worker startup timeout')

    def request(self, port, seed, cache_salt=None):
        c = Client(port)
        before = c.metrics()
        if metric(before, 'vllm:request_success_total') != self.counts[port]:
            raise RuntimeError(f'Unexpected requests on port {port}')
        ids = prompt(seed, 248320)
        body = {'model': MODEL_ID, 'prompt': ids, 'temperature': 0, 'max_tokens': 32,
            'return_token_ids': True, 'logprobs': 2, 'return_tokens_as_token_ids': True}
        if cache_salt is not None:
            body['cache_salt'] = cache_salt
        req = urllib.request.Request(c.url + '/v1/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            answer = json.load(response)
        choice = answer['choices'][0]
        if not choice.get('token_ids') or len(choice['token_ids']) != answer['usage']['completion_tokens']:
            raise RuntimeError('Missing or inconsistent generated token IDs')
        after = c.metrics()
        if delta(before, after, 'vllm:request_success_total') != 1:
            raise RuntimeError('Request count mismatch')
        self.counts[port] += 1
        result = {'seed': seed, 'usage': answer['usage'], 'text': choice['text'],
            'token_ids': choice['token_ids'], 'logprobs': choice.get('logprobs'),
            'output_sha256': hashlib.sha256(choice['text'].encode()).hexdigest(),
            'sources': {s: delta(before, after, 'vllm:prompt_tokens_by_source_total', source=s)
                for s in ['local_compute', 'local_cache_hit', 'external_kv_transfer']}}
        return result

    def transfer(self, index, seed, kind):
        offset = self.server_log.stat().st_size
        parent = self.request(18000, seed)
        deadline = time.monotonic() + 15
        tokens, errors = 0, 0
        while time.monotonic() < deadline:
            with self.server_log.open('rb') as handle:
                handle.seek(offset)
                text = handle.read().decode('utf-8', 'replace')
            tokens = sum(map(int, re.findall(r'Stored (\d+) tokens', text)))
            errors = text.count('Cannot store keys due to exception')
            if tokens == 8448 or errors:
                break
            time.sleep(0.2)
        if tokens != 8448 or errors:
            self.record('store_failure', index=index, seed=seed, kind=kind, stored_tokens=tokens,
                        store_errors=errors, parent=parent)
            return False
        child = self.request(18001, seed)
        sources = child['sources']
        source_ok = sources == {'local_compute': 64.0, 'local_cache_hit': 0.0, 'external_kv_transfer': 8448.0}
        self.record('transfer', index=index, seed=seed, kind=kind, stored_tokens=tokens,
            source_ok=source_ok, tokens_equal=parent['token_ids'] == child['token_ids'], parent=parent, child=child)
        if not source_ok:
            raise RuntimeError('Unexpected child KV source')
        return True

    def fixed(self, mode):
        self.launch_server(mode)
        for gpu in [0, 1]:
            self.launch_worker(gpu, 'retrieve')
        for gpu in [0, 1]:
            self.wait_worker(gpu, 18000 + gpu, 'retrieve')
        for index in range(1, 21):
            if not self.transfer(index, 2609091600 + index * 10, 'fixed_process'):
                return False
        return True

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [(self.args.parent_pid, b'8000'), (self.args.child_pid, b'8001'), (self.args.server_pid, b'5555')]
        for pid, port in specs:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'--port' not in args or args[args.index(b'--port') + 1] != port:
                raise RuntimeError(f'PID {pid} no longer matches expected port')
            self.record('original_process', pid=pid, argv=[s.decode() for s in args if s])
        for port in [8000, 8001]:
            v = Client(port).metrics()
            if metric(v, 'vllm:num_requests_running') != 0 or metric(v, 'vllm:num_requests_waiting') != 0:
                raise RuntimeError('Existing worker busy')
        status = 'failed'
        try:
            for pid in [self.args.parent_pid, self.args.child_pid]:
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while any(Path(f'/proc/{pid}').exists() for pid in [self.args.parent_pid, self.args.child_pid]):
                if time.monotonic() > deadline:
                    raise RuntimeError('Old worker did not exit')
                time.sleep(1)
            os.kill(self.args.server_pid, signal.SIGTERM)
            time.sleep(5)
            self.record('original_stack_stopped')
            baseline_ok = self.fixed('observe')
            self.record('baseline_result', completed_20=baseline_ok)
            fixed_ok = self.fixed('split')
            self.record('split_result', completed_20=fixed_ok)
            if not fixed_ok:
                raise RuntimeError('Boundary splitting did not resolve storage/transfer failure')
            for cycle in range(1, 4):
                self.launch_worker(1, 'retrieve')
                self.wait_worker(1, 18001, 'retrieve')
                if not self.transfer(cycle, 2609095000 + cycle, 'child_restart'):
                    raise RuntimeError('Store failed after child restart')
            self.record('restart_result', completed=3)
            self.stop_workers()
            for gpu in [0, 1]:
                self.launch_worker(gpu, 'recompute')
            for gpu in [0, 1]:
                self.wait_worker(gpu, 18000 + gpu, 'recompute')
            for seed in [2609091610, 2609091620, 2609091630]:
                for gpu in [0, 1]:
                    for repeat in [1, 2]:
                        result = self.request(18000 + gpu, seed)
                        if result['sources']['local_compute'] != 8512:
                            raise RuntimeError('Correctness control did not fully recompute')
                        self.record('recompute_control', seed=seed, gpu=gpu, repeat=repeat, result=result)
            status = 'complete'
        except Exception as error:
            self.record('failure', error=repr(error))
            raise
        finally:
            self.record('restore_start')
            self.stop_workers()
            if self.server is None or self.server.poll() is not None:
                raise RuntimeError('LMCache server unavailable during restoration')
            self.launch_worker(0, 'retrieve', public=True)
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(0, 8000, 'retrieve')
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, server_pid=self.server.pid,
                worker_pids={str(gpu): p.pid for gpu, p in self.workers.items()}, copy_mode=self.phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--server-pid', type=int, required=True)
    Diagnosis(parser.parse_args()).run()
