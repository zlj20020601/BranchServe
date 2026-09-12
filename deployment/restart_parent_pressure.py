"""Recover the stopped Parent, then run the corrected pressure smoke."""
import json
import os
import socket
import subprocess
import time

from gate2_paired_suite import Client, CONNECTOR, ENV, MODEL, MODEL_ID, ROOT


def main():
    os.chdir(ROOT)
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8000)) == 0:
            raise RuntimeError('Port 8000 already occupied; refusing duplicate Parent')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
    for key in ['LMCACHE_CONFIG_FILE', 'VLLM_KV_TRANSFER_CONFIG', 'VLLM_KV_CONNECTOR', 'VLLM_API_KEY']:
        env.pop(key, None)
    log = ROOT / 'artifacts/parent_saltfix_20260910.log'
    with log.open('x') as handle:
        proc = subprocess.Popen([str(ENV / 'bin/vllm'), 'serve', MODEL,
            '--served-model-name', MODEL_ID, '--host', '127.0.0.1', '--port', '8000',
            '--dtype', 'bfloat16', '--max-model-len', '16384',
            '--gpu-memory-utilization', '0.88', '--max-num-batched-tokens', '1024',
            '--enable-prefix-caching', '--kv-transfer-config', json.dumps(CONNECTOR)],
            env=env, stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
    print('Parent starting pid=' + str(proc.pid), flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('Parent exited; inspect ' + str(log))
        try:
            Client(8000).get('/health')
            break
        except OSError:
            time.sleep(3)
    else:
        raise RuntimeError('Parent readiness timeout')
    print('Parent healthy; starting pressure 4 three-arm smoke', flush=True)
    subprocess.run([str(ENV / 'bin/python'), '-u', 'gate3_fanout_smoke.py',
        '--run-id', 'gate3_qmsum_p4_saltfix_20260910', '--worker-pid', '22099',
        '--lmcache-pid', '1871', '--manifest', 'artifacts/qmsum_manifest_v2_20260910.json',
        '--pressure', '4', '--repeats', '1', '--warmups', '0',
        '--rouge-deps', 'artifacts/longbench_eval_deps'],
        env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)


if __name__ == '__main__':
    main()
