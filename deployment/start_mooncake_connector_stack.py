"""Start an N-GPU LMCache connector stack with list-based argv quoting.

Set ``GPU_IDS`` and ``WORKER_PORTS`` as comma-separated lists, for example
``GPU_IDS=0,1,2 WORKER_PORTS=8000,8001,8002 python ...``.  The first worker is
the default parent worker; the remaining workers are available for retrieval
or recomputation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ENV = "/root/autodl-tmp/conda_envs/branchserve"
MODEL = "/root/autodl-tmp/models/Qwen3.5-4B"
ROOT = Path("/root/autodl-tmp/branchserve")
CONNECTOR = {
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_connector_extra_config": {"lmcache.mp.host": "tcp://localhost", "lmcache.mp.port": 5555},
}


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def current_workers(ports):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            decoded = [part.decode(errors="ignore") for part in args if part]
        except OSError:
            continue
        if "vllm" not in " ".join(decoded) or "--port" not in decoded:
            continue
        port = decoded[decoded.index("--port") + 1]
        if port in {str(item) for item in ports}:
            found.append((int(entry.name), port))
    return found


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
            return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "0,1"),
                    help="comma-separated physical GPU ids")
    ap.add_argument("--worker-ports", default=os.environ.get("WORKER_PORTS", "8000,8001"),
                    help="comma-separated worker ports")
    args = ap.parse_args()
    gpus = parse_csv(args.gpu_ids, int)
    ports = parse_csv(args.worker_ports, int)
    if len(gpus) != len(ports) or not gpus:
        ap.error("--gpu-ids and --worker-ports must contain the same non-zero number of items")

    for pid, _ in current_workers(ports):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    while current_workers(ports) and time.monotonic() < deadline:
        time.sleep(1)
    (ROOT / "logs").mkdir(exist_ok=True)
    procs = []
    labels = ["parent"] + [f"worker{i}" for i in range(1, len(gpus))]
    for gpu, port, label in zip(gpus, ports, labels):
        argv = [
            f"{ENV}/bin/vllm", "serve", MODEL,
            "--served-model-name", "qwen3.5-4b", "--host", "0.0.0.0",
            "--port", str(port), "--dtype", "bfloat16", "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.88", "--max-num-batched-tokens", "1024",
            "--enable-prefix-caching", "--kv-transfer-config", json.dumps(CONNECTOR),
        ]
        log = (ROOT / "logs" / f"mooncake_{label}_20260912.log").open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PATH=f"{ENV}/bin:" + os.environ.get("PATH", ""))
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        procs.append(proc)
        print(f"started {label} pid={proc.pid} port={port}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if all(health(port) for port in ports):
            print("connector_stack_ready", flush=True)
            return
        if any(proc.poll() is not None for proc in procs):
            raise RuntimeError("connector worker exited; inspect logs")
        time.sleep(3)
    raise RuntimeError("connector stack readiness timeout")


if __name__ == "__main__":
    main()
