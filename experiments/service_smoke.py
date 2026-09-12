#!/usr/bin/env python3
"""BranchServe router service 烟测:真实流量经独立 :9000 服务路由。

场景 A(低压力):parent + 4 child 并发 → 预期全部 PACK(child 响应 branchserve.action)
场景 B(高压力,需先起 loadgen 打 gpu-0):parent + 4 child → 预期 RETRIEVE + gpu-1 transfer
用法: python service_smoke.py --router http://127.0.0.1:9000 --label svc-smoke-a
"""
import argparse
import json
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, ".")
from multiround_strategy import build_history, prefix_to, fetch_sources, diff

from transformers import AutoTokenizer

TRANSFER_RE = re.compile(
    r"^vllm:prompt_tokens_by_source_total\{[^}]*source=\"?(\w+)\"?[^}]*\}\s+([\d.eE+]+)\s*$")


def post_json(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://127.0.0.1:9000")
    ap.add_argument("--worker0", default="http://127.0.0.1:8000")
    ap.add_argument("--worker1", default="http://127.0.0.1:8001")
    ap.add_argument("--label", required=True)
    ap.add_argument("--fanout", type=int, default=4)
    ap.add_argument("--child-max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3.5-4B",
                                        local_files_only=True)
    blocks = build_history(tok, 8192 + 512, args.label)
    history, n = prefix_to(blocks, tok, 8192)
    out = {"label": args.label, "history_tokens": n}

    health = get_json(args.router + "/health")
    print("[router]", json.dumps(health))
    out["router_health"] = health

    # ---- parent
    session = f"{args.label}-parent"
    t0 = time.perf_counter()
    presp = post_json(args.router + "/v1/chat/completions", {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": history}],
        "max_tokens": 8, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "branchserve": {"session_id": session},
    })
    out["parent"] = {"latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                     "meta": presp.get("branchserve"),
                     "prompt_tokens": (presp.get("usage") or {}).get("prompt_tokens")}
    print("[parent]", json.dumps(out["parent"]))
    time.sleep(0.5)

    # ---- children(并发,走 router)
    b0 = {8000: fetch_sources(8000), 8001: fetch_sources(8001)}
    results = [None] * args.fanout

    def child(i):
        suffix = f"\nBranch {i}: summarize the branch-specific findings above."
        results[i] = post_json(args.router + "/v1/chat/completions", {
            "model": "qwen3.5-4b",
            "messages": [{"role": "user", "content": history + suffix}],
            "max_tokens": args.child_max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "branchserve": {"session_id": f"{args.label}-c{i}",
                            "parent_session_id": session,
                            "branch_id": f"b{i}"},
        })

    threads = [threading.Thread(target=child, args=(i,)) for i in range(args.fanout)]
    t1 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out["child_wall_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    a1 = {8000: fetch_sources(8000), 8001: fetch_sources(8001)}
    out["children"] = [{"latency_ms": r.get("branchserve", {}).get("latency_ms"),
                        "action": r.get("branchserve", {}).get("action"),
                        "worker": r.get("branchserve", {}).get("worker"),
                        "pressure": r.get("branchserve", {}).get("pressure"),
                        "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}
                       for r in results]
    out["sources_delta"] = {":8000": diff(b0[8000], a1[8000]), ":8001": diff(b0[8001], a1[8001])}
    for c in out["children"]:
        print("[child]", json.dumps(c, ensure_ascii=False))
    print("[gpu-delta]", json.dumps(out["sources_delta"], ensure_ascii=False))

    stats = get_json(args.router + "/stats")
    out["router_stats_recent"] = stats.get("decisions_recent", [])[-8:]
    out["router_store"] = stats.get("store")
    actions = [c["action"] for c in out["children"]]
    verdict = ("ALL_PACK" if all(a == "pack" for a in actions)
               else "ALL_RETRIEVE" if all(a == "retrieve" for a in actions)
               else f"MIXED:{actions}")
    out["verdict"] = verdict
    print("[verdict]", verdict)

    path = args.out or f"artifacts/service_smoke_{args.label}.json"
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done]", path)


if __name__ == "__main__":
    main()
