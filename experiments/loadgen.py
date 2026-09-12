#!/usr/bin/env python3
"""背景负载器:给指定 worker 打 N 路持续解码压力(独立于实验 harness)。

用法: python loadgen.py --port 8000 --count 6 --seconds 90
每路循环发送长生成请求(锚定固定前缀,min_tokens+ignore_eos 纯解码负载)。
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, ".")
from multiround_strategy import build_history, prefix_to, chat

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--label", default="loadgen")
    ap.add_argument("--output-tokens", type=int, default=2048)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3.5-4B",
                                        local_files_only=True)
    blocks = build_history(tok, 8192 + 512, args.label)
    base, n = prefix_to(blocks, tok, 8192)
    print(f"[loadgen] base={n} tokens -> :{args.port} count={args.count} "
          f"duration={args.seconds}s", flush=True)

    stop = time.time() + args.seconds
    stats = {"done": 0, "fail": 0}
    lock = threading.Lock()

    def slot(i):
        seq = 0
        while time.time() < stop:
            suffix = f"\nLoad slot {i} seq {seq}: continue the deterministic response."
            try:
                chat(args.port, base + suffix, args.output_tokens,
                     f"load-{i}-{seq}", min_tokens=args.output_tokens, ignore_eos=True)
                with lock:
                    stats["done"] += 1
            except Exception:
                with lock:
                    stats["fail"] += 1
            seq += 1

    threads = [threading.Thread(target=slot, args=(i,)) for i in range(args.count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[loadgen] finished", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
