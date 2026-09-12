"""Same-GPU paired operator traces with cold requests and service restoration."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from gate2_apc_control import APCControl
from gate2_connector_control import ConnectorControl
from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ROOT, metric


class OperatorControl(ConnectorControl):
    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def configured(args, **kwargs):
            if not public:
                args = list(args)
                if mode == 'apc':
                    args[args.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
                else:
                    index = args.index('--kv-transfer-config') + 1
                    config = json.loads(args[index])
                    config['kv_connector_extra_config']['lmcache.mp.port'] = 15555
                    args[index] = json.dumps(config)
                args += ['--worker-extension-cls', 'gate2_operator_probe.ProbeWorkerExtension']
                kwargs['env'].update(PYTHONPATH=str(ROOT), VLLM_SERVER_DEV_MODE='1',
                                     OPERATOR_PROBE_OUT=str(self.out / mode))
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
        assert b'--port' in args and args[args.index(b'--port') + 1] == b'8001'
        assert b'--no-enable-prefix-caching' in args and b'--kv-transfer-config' not in args
        values = Client(8001).metrics()
        assert all(metric(values, k) == 0 for k in ['vllm:num_requests_running', 'vllm:num_requests_waiting'])
        status = 'failed'
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Shutdown timeout')
                time.sleep(1)
            for mode in ['apc', 'retrieve']:
                if mode == 'retrieve':
                    self.private_server()
                self.launch_worker(1, mode)
                self.wait_worker(1, 18001, 'retrieve')
                results = []
                for repeat in [1, 2]:
                    APCControl.reset(self)
                    marker = self.out / mode / 'ACTIVE'
                    if repeat == 1:
                        marker.touch()
                    else:
                        marker.unlink()
                    offset = self.server_log.stat().st_size if mode == 'retrieve' else 0
                    result = self.request(18001, 2609091770, cache_salt=f'{self.args.run_id}-{mode}-{repeat}')
                    assert result['sources'] == {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}
                    extra = {}
                    if mode == 'retrieve':
                        deadline = time.monotonic() + 20
                        while time.monotonic() < deadline:
                            with self.server_log.open('rb') as handle:
                                handle.seek(offset)
                                log = handle.read().decode(errors='replace')
                            stored = sum(map(int, re.findall(r'Stored (\d+) tokens', log)))
                            errors = log.count('Cannot store keys due to exception')
                            if stored == 8448 or errors:
                                break
                            time.sleep(.2)
                        extra = {'stored_tokens': stored, 'store_errors': errors}
                        assert stored == 8448 and errors == 0
                    self.record('operator_control', mode=mode, repeat=repeat, traced=repeat == 1, result=result, **extra)
                    results.append(result['token_ids'])
                assert results[0] == results[1], 'Tracing altered output'
                assert (self.out / mode / 'DONE').exists(), 'Trace did not reach attention'
                self.stop_workers()
            status = 'complete'
        finally:
            self.stop_workers()
            self.terminate(self.server)
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, worker_pid=self.workers[1].pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    OperatorControl(parser.parse_args()).run()
