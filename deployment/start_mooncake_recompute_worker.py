"""Replace only GPU1 with the no-cache recompute worker for Mooncake replay."""
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ENV = "/root/autodl-tmp/conda_envs/branchserve"
MODEL = "/root/autodl-tmp/models/Qwen3.5-4B"
ROOT = Path("/root/autodl-tmp/branchserve")


def workers_on(port):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [x.decode(errors="ignore") for x in (entry / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if "vllm" in " ".join(args) and "--port" in args:
            i = args.index("--port")
            if i + 1 < len(args) and args[i + 1] == str(port):
                found.append(int(entry.name))
    return found


def main():
    for pid in workers_on(8001):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    while workers_on(8001) and time.monotonic() < deadline:
        time.sleep(1)
    (ROOT / "logs").mkdir(exist_ok=True)
    argv = [f"{ENV}/bin/vllm", "serve", MODEL, "--served-model-name", "qwen3.5-4b",
            "--host", "0.0.0.0", "--port", "8001", "--dtype", "bfloat16",
            "--max-model-len", "16384", "--gpu-memory-utilization", "0.88",
            "--max-num-batched-tokens", "1024", "--no-enable-prefix-caching"]
    log = (ROOT / "logs" / "mooncake_recompute_20260912.log").open("a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PATH=f"{ENV}/bin:" + os.environ.get("PATH", ""))
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"started recompute pid={proc.pid}", flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=3):
                print("recompute_ready", flush=True)
                return
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("recompute worker exited; inspect logs")
            time.sleep(3)
    raise RuntimeError("recompute readiness timeout")


if __name__ == "__main__":
    main()
