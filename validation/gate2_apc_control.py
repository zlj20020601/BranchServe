"""Isolate cold APC from LMCache, then compare aligned prefill traces."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ROOT, metric


class APCControl(Diagnosis):
    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def configured(args, **kwargs):
            if not public:
                args = list(args)
                if mode == 'apc':
                    args[args.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
                else:
                    args[args.index('--max-num-batched-tokens') + 1] = '528'
                args += ['--worker-extension-cls', 'gate2_apc_probe.ProbeWorkerExtension']
                kwargs['env']['PYTHONPATH'] = str(ROOT)
                kwargs['env']['STATE_PROBE_OUT'] = str(self.out / mode)
                kwargs['env']['VLLM_SERVER_DEV_MODE'] = '1'
                if '--kv-transfer-config' in args:
                    raise RuntimeError('APC control must not have a connector')
            return original(args, **kwargs)

        subprocess.Popen = configured
        try:
            return super().launch_worker(gpu, mode, public)
        finally:
            subprocess.Popen = original

    def reset(self):
        request = urllib.request.Request('http://127.0.0.1:18001/reset_prefix_cache', data=b'', method='POST')
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get('success') is not True:
            raise RuntimeError('Prefix reset failed')
        self.record('cache_reset', response=result)

    def control(self, port, mode, repeat):
        result = self.request(port, 2609091770)
        if result['sources'] != {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}:
            raise RuntimeError('Control did not fully recompute')
        self.record('apc_control', mode=mode, repeat=repeat, traced=port == 18001 and repeat == 1, result=result)

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = self.args.child_pid
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'--port' not in args or args[args.index(b'--port') + 1] != b'8001' or b'--no-enable-prefix-caching' not in args or b'--kv-transfer-config' in args:
            raise RuntimeError('Original worker identity changed')
        values = Client(8001).metrics()
        if any(metric(values, key) != 0 for key in ['vllm:num_requests_running', 'vllm:num_requests_waiting']):
            raise RuntimeError('Worker busy')
        self.counts[8001] = metric(values, 'vllm:request_success_total')
        self.control(8001, 'recompute1024', 1)
        status = 'failed'
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Worker shutdown timeout')
                time.sleep(1)
            for mode in ['apc', 'recompute528']:
                self.phase = mode
                self.launch_worker(1, mode)
                self.wait_worker(1, 18001, 'retrieve' if mode == 'apc' else 'recompute')
                for repeat in [1, 2]:
                    if mode == 'apc':
                        self.reset()
                    marker = self.out / mode / 'ACTIVE'
                    if repeat == 1:
                        marker.touch()
                    else:
                        marker.unlink()
                    self.control(18001, mode, repeat)
                self.stop_workers()
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
    APCControl(parser.parse_args()).run()
