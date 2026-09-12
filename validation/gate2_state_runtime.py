"""Capture a cold parent and external retrieval with identical runtime flags."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from gate2_diagnose_runtime import Diagnosis
from gate2_paired_suite import ROOT, Client, metric


class StateDiagnosis(Diagnosis):
    def launch_worker(self, gpu, mode, public=False):
        original = subprocess.Popen

        def instrument(args, **kwargs):
            if not public:
                args = list(args) + ['--worker-extension-cls', 'gate2_state_probe.ProbeWorkerExtension']
                if self.args.eager:
                    args += ['--enforce-eager']
                kwargs['env']['STATE_PROBE_OUT'] = str(self.out / f'gpu{gpu}')
                kwargs['env']['PYTHONPATH'] = str(ROOT)
                if gpu == 1 and self.args.repair_attention:
                    kwargs['env']['STATE_REPAIR_ATTN'] = '1'
            return original(args, **kwargs)

        subprocess.Popen = instrument
        try:
            return super().launch_worker(gpu, mode, public)
        finally:
            subprocess.Popen = original

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = [(self.args.parent_pid, 8000), (self.args.child_pid, 8001), (self.args.server_pid, 5555)]
        for pid, port in specs:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'--port' not in args or args[args.index(b'--port') + 1] != str(port).encode():
                raise RuntimeError('Original process identity changed')
        for port in [8000, 8001]:
            values = Client(port).metrics()
            if any(metric(values, key) != 0 for key in ['vllm:num_requests_running', 'vllm:num_requests_waiting']):
                raise RuntimeError('Worker busy')
        status = 'failed'
        try:
            for pid, port in specs[:2]:
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 90
            while any(Path(f'/proc/{pid}').exists() for pid, _ in specs[:2]):
                if time.monotonic() > deadline:
                    raise RuntimeError('Original worker shutdown timeout')
                time.sleep(1)
            os.kill(self.args.server_pid, signal.SIGTERM)
            time.sleep(5)
            self.launch_server('split')
            for gpu in [0, 1]:
                self.launch_worker(gpu, 'retrieve')
            for gpu in [0, 1]:
                self.wait_worker(gpu, 18000 + gpu, 'retrieve')
                (self.out / f'gpu{gpu}' / 'ACTIVE').touch()
            self.record('protocol', seed=self.args.seed, eager=self.args.eager, instrumentation=True,
                        repair_attention=self.args.repair_attention, synchronize_copy=self.args.synchronize_copy)
            if not self.transfer(1, self.args.seed, 'state_probe'):
                raise RuntimeError('Transfer failed')
            if self.args.extra_controls:
                for gpu in [0, 1]:
                    (self.out / f'gpu{gpu}' / 'ACTIVE').unlink()
                for index in [4, 5, 7, 13, 16, 17, 19, 20]:
                    seed = 2609091600 + index * 10
                    if seed != self.args.seed and not self.transfer(index, seed, 'untraced_control'):
                        raise RuntimeError('Additional transfer failed')
            status = 'complete'
        finally:
            self.record('restore_start')
            # Reset the server too: it can retain CUDA IPC allocations after worker exit.
            self.stop_workers()
            self.terminate(self.server)
            self.server = None
            self.phase = 'restore'
            self.launch_server('restore')
            self.launch_worker(0, 'retrieve', public=True)
            self.launch_worker(1, 'recompute', public=True)
            self.wait_worker(0, 8000, 'retrieve')
            self.wait_worker(1, 8001, 'recompute')
            self.record('finished', status=status, server_pid=self.server.pid,
                        worker_pids={str(g): p.pid for g, p in self.workers.items()})

    def launch_server(self, mode):
        # The parent uses the label as the copy-mode flag; restoration also needs split.
        if mode == 'restore':
            old = self.out
            self.out = old / 'restored'
            self.out.mkdir(exist_ok=True)
            try:
                return super().launch_server('split')
            finally:
                self.out = old
        original = subprocess.Popen

        def instrument(args, **kwargs):
            if self.args.synchronize_copy:
                args = list(args)
                args.insert(2, '--synchronize-copy')
            return original(args, **kwargs)

        subprocess.Popen = instrument
        try:
            return super().launch_server(mode)
        finally:
            subprocess.Popen = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2609091760)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--repair-attention', action='store_true')
    parser.add_argument('--synchronize-copy', action='store_true')
    parser.add_argument('--extra-controls', action='store_true')
    StateDiagnosis(parser.parse_args()).run()
