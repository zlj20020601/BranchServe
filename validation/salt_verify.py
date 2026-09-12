#!/usr/bin/env python3
"""验证 cache_salt 能否让 child 绕过 LMCache 仓库(统一栈上的 Recompute 机制)。

判定(关键在 salted child 先发,:8001 冷):
  salted sources = local_compute≈全量 & 无 external_kv_transfer → salt 生效
  salted sources = external_kv_transfer>0                      → lmcache 键忽略 salt,机制不可用
"""
import json
import sys
import time

sys.path.insert(0, ".")
from multiround_strategy import build_history, prefix_to, chat, fetch_sources, diff

from transformers import AutoTokenizer

LABEL = sys.argv[1] if len(sys.argv) > 1 else "salttest-20260912"
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3.5-4B", local_files_only=True)
blocks = build_history(tok, 8192 + 512, LABEL)
history, n = prefix_to(blocks, tok, 8192)

out = {"history_tokens": n}

# 1. parent on :8000 (connector auto-stores KV)
p0 = fetch_sources(8000)
parent = chat(8000, history, 8, "salt-parent")
p1 = fetch_sources(8000)
out["parent"] = {"latency_ms": parent["latency_ms"], "prompt_tokens": parent["prompt_tokens"],
                 "sources": diff(p0, p1)}
print("[parent]", json.dumps(out["parent"]))
time.sleep(0.5)

# 2. SALTED child FIRST on cold :8001  ← 判定请求
b0 = fetch_sources(8001)
salted = chat(8001, history + "\nBranch 9: summarize with salt.", 8,
              "salt-child", cache_salt="recompute-arm-v1")
b1 = fetch_sources(8001)
out["salted_child"] = {"latency_ms": salted["latency_ms"],
                       "prompt_tokens": salted["prompt_tokens"],
                       "sources": diff(b0, b1)}
print("[salted ]", json.dumps(out["salted_child"]))

# 3. saltless child (对照:正常 transfer 路径)
a0 = fetch_sources(8001)
plain = chat(8001, history + "\nBranch 8: summarize without salt.", 8, "nosalt-child")
a1 = fetch_sources(8001)
out["nosalt_child"] = {"latency_ms": plain["latency_ms"],
                       "prompt_tokens": plain["prompt_tokens"],
                       "sources": diff(a0, a1)}
print("[nosalt ]", json.dumps(out["nosalt_child"]))

# verdict
def _num(v):
    return v if isinstance(v, (int, float)) else 0


s = out["salted_child"]["sources"]
if "error" in s:
    verdict = "METRICS_UNAVAILABLE"
elif _num(s.get("src:external_kv_transfer", 0)) > 0:
    verdict = "SALT_IGNORED_BY_STORE"
elif _num(s.get("src:local_compute", 0)) > 0.8 * n:
    verdict = "SALT_WORKS_FULL_RECOMPUTE"
elif _num(s.get("src:local_cache_hit", 0)) > 0:
    verdict = "SALT_PARTIAL(unexpected local hit)"
else:
    verdict = f"UNCLEAR: {s}"
out["verdict"] = verdict
print("[verdict]", verdict)

with open("artifacts/salt_verify_20260912.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("[done] artifacts/salt_verify_20260912.json")
