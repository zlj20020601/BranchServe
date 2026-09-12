"""Capture a real cold APC input and restore GPU1 afterward."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error

from gate2_apc_control import APCControl
from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ROOT, metric


class KernelControl(APCControl):
    def request(self, *args, **kwargs):
        try:
            return super().request(*args, **kwargs)
        except urllib.error.HTTPError as error:
            self.record('http_failure', code=error.code, body=error.read().decode(errors='replace'))
            raise

    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def configured(args, **kwargs):
            if not public:
                args = list(args)
                args[args.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
                module = 'gate2_kernel_intervene' if self.args.intervene else 'gate2_kernel_probe'
                args += ['--worker-extension-cls', module + '.ProbeWorkerExtension']
                kwargs['env'].update(PYTHONPATH=str(ROOT), VLLM_SERVER_DEV_MODE='1',
                                     KERNEL_PROBE_OUT=str(self.out / 'kernel'))
            return original(args, **kwargs)

        subprocess.Popen = configured
        try:
            return Diagnosis.launch_worker(self, gpu, mode, public)
        finally:
            subprocess.Popen = original

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = self.args.child_pid
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if (b'--port' not in args or args[args.index(b'--port') + 1] != b'8001'
                or b'--no-enable-prefix-caching' not in args or b'--kv-transfer-config' in args):
            raise RuntimeError('Original worker identity changed')
        values = Client(8001).metrics()
        if any(metric(values, k) != 0 for k in ['vllm:num_requests_running', 'vllm:num_requests_waiting']):
            raise RuntimeError('Worker busy')
        status = 'failed'
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Shutdown timeout')
                time.sleep(1)
            self.launch_worker(1, 'apc')
            self.wait_worker(1, 18001, 'retrieve')
            results = []
            marker = self.out / 'kernel/ACTIVE'
            modes = ['native', 'all4096', 'first2048', 'all2048', 'all2048', 'first2048', 'all4096', 'native'] if self.args.intervene else ['capture', 'plain']
            if self.args.boundary:
                assert self.args.intervene
                modes = ['native', 'all4096', 'firstboundary2048', 'boundary2048', 'boundary2048', 'firstboundary2048', 'all4096', 'native']
            for repeat, mode in enumerate(modes, 1):
                self.reset()
                if self.args.intervene:
                    (self.out / 'kernel/MODE.json').write_text(json.dumps({'request': repeat, 'mode': mode}))
                elif repeat == 1:
                    marker.touch()
                else:
                    marker.unlink()
                result = self.request(18001, 2609091770)
                self.record('kernel_control', repeat=repeat, mode=mode, captured=not self.args.intervene and repeat == 1, result=result)
                assert result['sources'] == {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}
                results.append(result['token_ids'])
            if self.args.intervene:
                assert all(results[i] == results[7-i] for i in range(4)), 'Mode was not repeatable'
                assert results[0] == results[1], 'Direct launch control changed output'
                interventions = [json.loads(s) for s in (self.out / 'kernel/interventions.jsonl').read_text().splitlines()]
                for repeat, mode in enumerate(modes, 1):
                    calls = [v for v in interventions if v['request'] == repeat]
                    changed = sum(v['changed'] for v in calls)
                    assert len(calls) == 384, f'Unexpected kernel call count: {len(calls)}'
                    assert all(v['tokens'] == 528 for v in calls)
                    assert changed == {'native': 0, 'first2048': 1, 'all2048': 384, 'all4096': 384,
                                       'firstboundary2048': 1, 'boundary2048': 128}[mode]
            else:
                assert results[0] == results[1], 'Instrumentation changed output'
                assert (self.out / 'kernel/first_rmsnorm.pt').exists(), 'Kernel not intercepted'
            status = 'complete'
        finally:
            self.stop_workers()
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, worker_pid=self.workers[1].pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--child-pid', required=True, type=int)
    parser.add_argument('--intervene', action='store_true')
    parser.add_argument('--boundary', action='store_true')
    KernelControl(parser.parse_args()).run()
