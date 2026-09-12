"""Repeat full recomputation of the exact inputs that diverged during retrieval."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ROOT, metric


class Controls(Diagnosis):
    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = [json.loads(s) for s in Path(self.args.source).read_text().splitlines()]
        seeds = [r['seed'] for r in rows if r['event'] == 'transfer' and r['phase'] == 'split'
                 and r['kind'] == 'fixed_process' and not r['tokens_equal']]
        last = rows[-1]
        if last['event'] != 'finished' or last['status'] != 'complete':
            raise RuntimeError('Primary diagnostic run has not finished')
        pids = {} if self.args.fresh_stack else last['worker_pids']
        if self.args.fresh_stack:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            if apps.strip():
                raise RuntimeError('Fresh-stack mode requires both GPUs to be free')
        for gpu in ([] if self.args.fresh_stack else [0, 1]):
            pid, port = pids[str(gpu)], 8000 + gpu
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'--port' not in args or args[args.index(b'--port') + 1] != str(port).encode():
                raise RuntimeError('Worker identity changed')
            values = Client(port).metrics()
            if metric(values, 'vllm:num_requests_running') != 0 or metric(values, 'vllm:num_requests_waiting') != 0:
                raise RuntimeError('Worker busy')
        self.record('protocol', seeds=seeds, repetitions=2, original_worker_pids=pids)
        status = 'failed'
        try:
            if self.args.fresh_stack:
                self.launch_server('split')
            for pid in pids.values():
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while any(Path(f'/proc/{pid}').exists() for pid in pids.values()):
                if time.monotonic() > deadline:
                    raise RuntimeError('Original worker did not stop')
                time.sleep(1)
            for gpu in [0, 1]:
                self.launch_worker(gpu, 'recompute')
            for gpu in [0, 1]:
                self.wait_worker(gpu, 18000 + gpu, 'recompute')
            for seed in seeds:
                for gpu in [0, 1]:
                    for repeat in [1, 2]:
                        result = self.request(18000 + gpu, seed)
                        if result['sources']['local_compute'] != 8512:
                            raise RuntimeError('Not full recomputation')
                        self.record('recompute_control', seed=seed, gpu=gpu, repeat=repeat, result=result)
            status = 'complete'
        finally:
            self.stop_workers()
            self.launch_worker(0, 'retrieve', public=True)
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(0, 8000, 'retrieve')
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, worker_pids={str(g): p.pid for g, p in self.workers.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--fresh-stack', action='store_true')
    Controls(parser.parse_args()).run()
