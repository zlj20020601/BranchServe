"""Isolate prefill chunk size without any external or local prefix cache."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ROOT, metric


class ChunkControl(Diagnosis):
    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def configured(args, **kwargs):
            if not public:
                args = list(args)
                args[args.index('--max-num-batched-tokens') + 1] = '528'
            return original(args, **kwargs)

        subprocess.Popen = configured
        try:
            return super().launch_worker(gpu, mode, public)
        finally:
            subprocess.Popen = original

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = self.args.child_pid
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'--port' not in args or args[args.index(b'--port') + 1] != b'8001' or b'--no-enable-prefix-caching' not in args:
            raise RuntimeError('Original worker is not the expected recompute worker')
        values = Client(8001).metrics()
        if any(metric(values, key) != 0 for key in ['vllm:num_requests_running', 'vllm:num_requests_waiting']):
            raise RuntimeError('Worker busy')
        self.counts[8001] = metric(values, 'vllm:request_success_total')
        for repeat in [1, 2]:
            result = self.request(8001, 2609091770)
            self.record('chunk_control', chunk=1024, repeat=repeat, result=result)
        status = 'failed'
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Worker shutdown timeout')
                time.sleep(1)
            self.launch_worker(1, 'recompute')
            self.wait_worker(1, 18001, 'recompute')
            for repeat in [1, 2]:
                result = self.request(18001, 2609091770)
                if result['sources'] != {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}:
                    raise RuntimeError('Control unexpectedly reused cache')
                self.record('chunk_control', chunk=528, repeat=repeat, result=result)
            status = 'complete'
        finally:
            self.stop_workers()
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, worker_pid=self.workers[1].pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    ChunkControl(parser.parse_args()).run()
