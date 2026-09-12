"""Cold connector controls on GPU1 with a private LMCache service."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import Client, ENV, ROOT, metric


class ConnectorControl(Diagnosis):
    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def configured(args, **kwargs):
            if not public:
                args = list(args)
                index = args.index('--kv-transfer-config') + 1
                config = json.loads(args[index])
                config['kv_connector_extra_config']['lmcache.mp.port'] = 15555
                args[index] = json.dumps(config)
                args += ['--worker-extension-cls', 'gate2_apc_probe.ProbeWorkerExtension']
                kwargs['env']['PYTHONPATH'] = str(ROOT)
                kwargs['env']['STATE_PROBE_OUT'] = str(self.out / 'connector')
            return original(args, **kwargs)

        subprocess.Popen = configured
        try:
            return super().launch_worker(gpu, mode, public)
        finally:
            subprocess.Popen = original

    def private_server(self):
        for port in [15555, 18080]:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
        self.server_log = self.out / 'lmcache.log'
        env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
        env.pop('CUDA_VISIBLE_DEVICES', None)
        args = [str(ENV / 'bin/python'), str(ROOT / 'lmcache_copy_diagnostic.py'),
                '--copy-mode', 'split', '--synchronize-copy', 'server', '--host', '127.0.0.1',
                '--port', '15555', '--http-port', '18080', '--chunk-size', '528', '--separate-object-groups',
                '--l1-size-gb', '100', '--eviction-policy', 'LRU']
        with self.server_log.open('x') as log:
            self.server = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        self.record('private_server_start', pid=self.server.pid, port=15555, synchronized=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError('Private LMCache exited')
            try:
                with socket.create_connection(('127.0.0.1', 15555), timeout=1):
                    return
            except OSError:
                time.sleep(2)
        raise RuntimeError('Private LMCache startup timeout')

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
        status = 'failed'
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while Path(f'/proc/{pid}').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Worker shutdown timeout')
                time.sleep(1)
            self.private_server()
            self.launch_worker(1, 'retrieve')
            self.wait_worker(1, 18001, 'retrieve')
            for repeat in [1, 2]:
                marker = self.out / 'connector' / 'ACTIVE'
                if repeat == 1:
                    marker.touch()
                else:
                    marker.unlink()
                offset = self.server_log.stat().st_size
                salt = f'{self.args.run_id}-{repeat}'
                result = self.request(18001, 2609091770, cache_salt=salt)
                if result['sources'] != {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}:
                    raise RuntimeError('Connector control was not cold')
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
                self.record('connector_control', repeat=repeat, traced=repeat == 1, cache_salt=salt,
                            stored_tokens=stored, store_errors=errors, result=result)
                if stored != 8448 or errors:
                    raise RuntimeError('Store did not complete cleanly')
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
    ConnectorControl(parser.parse_args()).run()
