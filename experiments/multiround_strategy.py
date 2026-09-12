#!/usr/bin/env python3
"""Multi-round clean-state strategy runner: PACK / Retrieve / Recompute.

研究问题:双 GPU 长上下文 Agent fan-out,子请求组选哪条路径最快。
每策略独立进程调用、干净状态;每轮 = 增量 parent + fan-out children。

Round r: history = 固定长文本前缀(逐字节稳定追加), target = initial + r*append
  1. parent(history, max_tokens=8) -> parent worker :8000 (APC 使 r>0 只算增量)
  2. [retrieve] 轮询 lmcache server 日志新增 Stored 行,等 store 就绪(或超时,记录)
  3. [recompute] 每轮前 POST /reset_prefix_cache 清 child worker 本地 APC
  4. children(history + branch_suffix_i, 4 并发) -> pack:?8000;否则 :8001
     pack/retrieve 不清 child 本地缓存(r>=2 的摊销动态是测量对象)

每轮记录 sources 差分(local_compute/local_cache_hit/external_kv_transfer)
区分 GPU0 本地 / GPU1 本地 / 外部传输三处 cache 状态。
产物 json: header + 每轮一条 + summary。

用法(nmb1, branchserve env):
  pack:      python multiround_strategy.py --strategy pack
  recompute: python multiround_strategy.py --strategy recompute
  retrieve:  (connector 栈就绪后) python multiround_strategy.py --strategy retrieve \
               --server-log logs/lmcache_mooncake_20260912.log
"""
import argparse
import json
import math
import re
import statistics
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")
STORED_RE = re.compile(r"Stored (\d+) tokens in ([\d.]+) seconds")
LABEL_DEFAULT = "multiround-fixed"  # 固定 label:文本跨运行逐字节可复现;换 label=全新前缀


# ---------------------------------------------------------------- metrics
SRC_LINE_RE = re.compile(
    r"^vllm:prompt_tokens_by_source_total\{[^}]*source=\"?(\w+)\"?[^}]*\}\s+([\d.eE+]+)\s*$")
SUM_LINE_RE = re.compile(
    r"^vllm:(request_prefill_kv_computed_tokens_sum|prefix_cache_queries_total"
    r"|prefix_cache_hits_total)\s+([\d.eE+]+)\s*$")


def fetch_sources(port):
    """Return {src:<name>, <sum keys>} for cache-attribution counters, or None."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        m = SRC_LINE_RE.match(line)
        if m:
            out[f"src:{m.group(1)}"] = float(m.group(2))
            continue
        m = SUM_LINE_RE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out if out else None


def diff(before, after):
    if before is None or after is None:
        return {"error": "metrics unavailable"}
    d = {}
    for k in set(before) | set(after):
        b, a = before.get(k), after.get(k)
        if b is None or a is None:
            d[k] = {"before": b, "after": a}
        elif a != b:
            d[k] = round(a - b, 1)
    return d


# ---------------------------------------------------------------- http
def chat(port, content, max_tokens, tag, min_tokens=None, ignore_eos=False, cache_salt=None):
    body = {
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if min_tokens is not None:
        body["min_tokens"] = min_tokens
    if ignore_eos:
        body["ignore_eos"] = True
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode())
    ms = (time.perf_counter() - t0) * 1000
    usage = data.get("usage", {})
    choice = (data.get("choices") or [{}])[0]
    return {
        "tag": tag, "latency_ms": round(ms, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }


def reset_prefix_cache(port):
    for path in ("/reset_prefix_cache", "/v1/reset_prefix_cache"):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", data=b"{}",
                method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return path
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- history
def build_history(tokenizer, max_tokens_needed, label):
    """Deterministic block text; returns list of block strings (appended only)."""
    blocks = []
    index = 0
    total = 0
    text = ""
    while total < max_tokens_needed:
        block = (f"Repository fact block {index:05d} for {label}: BranchServe keeps "
                 f"workflow context block {index:05d} with tool schema key "
                 f"{index * 7919} for deterministic replay and branch reuse.\n")
        text += block
        total = len(tokenizer.encode(text, add_special_tokens=False))
        blocks.append(block)
        index += 1
    return blocks


def prefix_to(blocks, tokenizer, target_tokens):
    """Join blocks until tokenized length first reaches >= target; return (text, actual)."""
    text = ""
    for b in blocks:
        candidate = text + b
        n = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n >= target_tokens:
            return candidate, n
        text = candidate
    return text, len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- store watch
def scan_stored(path, offset):
    """Scan server log beyond byte offset; return (chunks, tokens, cpu_s, end_offset)."""
    if path is None or offset < 0:
        return -1, 0, 0.0, offset
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return -1, 0, 0.0, offset
    chunks, tokens, cpu = 0, 0, 0.0
    for m in STORED_RE.finditer(data.decode("utf-8", "replace")):
        chunks += 1
        tokens += int(m.group(1))
        cpu += float(m.group(2))
    return chunks, tokens, cpu, offset + len(data)


class StoreWatcher(threading.Thread):
    """Observe new 'Stored' lines until quiescence; never blocks the caller."""

    def __init__(self, path, offset, quiet_s=1.5, timeout_s=60.0, poll_s=0.25):
        super().__init__(daemon=True)
        self.path, self.offset = path, offset
        self.quiet_s, self.timeout_s, self.poll_s = quiet_s, timeout_s, poll_s
        self.first_ms = None
        self.last_ms = None
        self.complete_ms = None
        self.chunks = 0
        self.tokens = 0
        self.cpu_busy_ms = 0.0
        self.timeout = False

    def run(self):
        t0 = time.perf_counter()
        last_growth = t0
        while time.perf_counter() - t0 < self.timeout_s:
            chunks, tokens, cpu, _ = scan_stored(self.path, self.offset)
            now = time.perf_counter()
            if chunks > self.chunks:
                if self.first_ms is None:
                    self.first_ms = (now - t0) * 1000
                self.chunks, self.tokens = chunks, tokens
                self.cpu_busy_ms = cpu * 1000
                self.last_ms = (now - t0) * 1000
                last_growth = now
            elif now - last_growth >= self.quiet_s:
                self.complete_ms = (now - t0) * 1000
                break
            time.sleep(self.poll_s)
        else:
            self.timeout = True
            self.complete_ms = (time.perf_counter() - t0) * 1000

    def result(self):
        return {
            "first_line_ms": round(self.first_ms, 1) if self.first_ms is not None else None,
            "last_line_ms": round(self.last_ms, 1) if self.last_ms is not None else None,
            "complete_ms": round(self.complete_ms, 1) if self.complete_ms is not None else None,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "cpu_busy_ms": round(self.cpu_busy_ms, 1),
            "timeout": self.timeout,
        }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["pack", "retrieve", "recompute"], required=True)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--parent-port", type=int, default=8000)
    ap.add_argument("--child-port", type=int, default=8001)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--initial-tokens", type=int, default=8192)
    ap.add_argument("--append-tokens", type=int, default=2048)
    ap.add_argument("--fanout", type=int, default=4)
    ap.add_argument("--child-max-tokens", type=int, default=256)
    ap.add_argument("--parent-max-tokens", type=int, default=8)
    ap.add_argument("--server-log", default="logs/lmcache_mooncake_20260912.log")
    ap.add_argument("--settle-s", type=float, default=0.25)
    ap.add_argument("--store-timeout-s", type=float, default=60.0)
    ap.add_argument("--store-quiet-s", type=float, default=1.5,
                    help="静默判停窗口:server 日志这么多秒无新 Stored 行视为存完")
    ap.add_argument("--wait-full-store", action="store_true",
                    help="保守派发:等 store 静默判停后才发 child(默认乐观派发)")
    ap.add_argument("--label", default=LABEL_DEFAULT,
                    help="history 文本 label;换 label 即全新前缀(重跑免重起栈)")
    ap.add_argument("--reset-child-each-round", action="store_true",
                    help="recompute 臂改用 APC-on+每轮 reset 语义(默认 False:"
                         "与 431 基线一致,child worker 需以 --no-enable-prefix-caching 启动)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    suffixes = [f"\nBranch {i}: summarize the branch-specific findings above in one short paragraph."
                for i in range(args.fanout)]

    child_target_port = args.parent_port if args.strategy == "pack" else args.child_port

    out_path = args.out or (f"artifacts/multiround_{args.strategy}_"
                            + time.strftime("%Y%m%d_%H%M%S") + ".json")
    records = []
    out = open(out_path, "w", buffering=1)

    def rec(o):
        records.append(o)
        out.write(json.dumps(o, ensure_ascii=False) + "\n")

    # ---- history construction (fixed text, append-only)
    max_needed = args.initial_tokens + (args.rounds - 1) * args.append_tokens + 512
    t_build = time.time()
    blocks = build_history(tok, max_needed, args.label)
    rec({"type": "header", "strategy": args.strategy, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
         "label": args.label,
         "dispatch_policy": "wait_full_store" if args.wait_full_store else "optimistic_after_parent",
         "rounds": args.rounds, "initial_tokens": args.initial_tokens,
         "append_tokens": args.append_tokens, "fanout": args.fanout,
         "child_max_tokens": args.child_max_tokens, "parent_max_tokens": args.parent_max_tokens,
         "parent_port": args.parent_port, "child_port": args.child_port,
         "child_target_port": child_target_port, "blocks": len(blocks),
         "reset_child_each_round": bool(args.reset_child_each_round),
         "suffix_tokens": [len(tok.encode(s, add_special_tokens=False)) for s in suffixes],
         "build_s": round(time.time() - t_build, 2),
         "workers": {"gpu-0": f":{args.parent_port}", "gpu-1": f":{args.child_port}"}})

    arm_start = time.perf_counter()
    baseline = {args.parent_port: fetch_sources(args.parent_port),
                args.child_port: fetch_sources(args.child_port)}
    server_offset = 0
    if args.strategy == "retrieve":
        try:
            with open(args.server_log, "rb") as f:
                f.seek(0, 2)
                server_offset = f.tell()
        except OSError:
            server_offset = -1

    prev_history_tokens = 0
    for r in range(args.rounds):
        target = args.initial_tokens + r * args.append_tokens
        history, history_tokens = prefix_to(blocks, tok, target)
        round0 = time.perf_counter()

        # ---- 1. parent (incremental prefill via APC on parent worker)
        p_before = fetch_sources(args.parent_port)
        parent = chat(args.parent_port, history, args.parent_max_tokens, f"r{r}-parent")
        parent_after = fetch_sources(args.parent_port)
        parent_src = diff(p_before, parent_after)

        # ---- 2. store watch (retrieve only): 旁路观察,默认不阻塞派发
        watcher = None
        if args.strategy == "retrieve" and server_offset >= 0:
            watcher = StoreWatcher(args.server_log, server_offset,
                                   quiet_s=args.store_quiet_s,
                                   timeout_s=args.store_timeout_s)
            watcher.start()
            if args.wait_full_store:
                watcher.join(args.store_timeout_s + 5)
        time.sleep(args.settle_s)

        # ---- 3. recompute: optional APC eviction (baseline semantics = APC-off worker)
        reset_path = None
        if args.strategy == "recompute" and args.reset_child_each_round:
            reset_path = reset_prefix_cache(args.child_port)

        # ---- 4. children fan-out (concurrent)
        c_before = {args.parent_port: fetch_sources(args.parent_port),
                    args.child_port: fetch_sources(args.child_port)}
        children = [None] * args.fanout

        def run_child(i):
            children[i] = chat(child_target_port, history + suffixes[i],
                               args.child_max_tokens, f"r{r}-c{i}")

        threads = [threading.Thread(target=run_child, args=(i,)) for i in range(args.fanout)]
        t_children = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        children_done = time.perf_counter()

        c_after = {args.parent_port: fetch_sources(args.parent_port),
                   args.child_port: fetch_sources(args.child_port)}
        store_rec = None
        if watcher is not None:
            watcher.join(args.store_timeout_s)
            store_rec = watcher.result()
            _, _, _, server_offset = scan_stored(args.server_log, server_offset)
        lat = sorted(c["latency_ms"] for c in children if c)
        n = len(lat)
        p95_idx = max(0, int(math.ceil(0.95 * n)) - 1) if n else None
        rec({
            "type": "round", "round_id": r,
            "history_tokens": history_tokens,
            "append_actual": history_tokens - prev_history_tokens,
            "prev_history_tokens": prev_history_tokens,
            "parent_latency_ms": parent["latency_ms"],
            "parent_sources": parent_src,
            "store": store_rec,
            "reset_child_cache": reset_path,
            "children": children,
            "child_makespan_ms": round((children_done - t_children) * 1000, 3),
            "child_latency_p50_ms": round(statistics.median(lat), 3) if n else None,
            "child_latency_p95_ms": round(lat[p95_idx], 3) if n and p95_idx is not None else None,
            "child_latency_max_ms": round(lat[-1], 3) if n else None,
            "round_wall_ms": round((time.perf_counter() - round0) * 1000, 3),
            "cumulative_makespan_ms": round((time.perf_counter() - arm_start) * 1000, 3),
            "sources_delta": {
                "gpu-0": diff(c_before[args.parent_port], c_after[args.parent_port]),
                "gpu-1": diff(c_before[args.child_port], c_after[args.child_port]),
            },
        })
        print(f"[r{r}] hist={history_tokens} parent={parent['latency_ms']:.0f}ms "
              f"store={store_rec['complete_ms'] if store_rec else '-'}ms"
              f"(chunks={store_rec['chunks'] if store_rec else '-'}) "
              f"makespan={(children_done - t_children) * 1000:.0f}ms",
              flush=True)
        prev_history_tokens = history_tokens

    total = (time.perf_counter() - arm_start) * 1000
    rounds = [x for x in records if x.get("type") == "round"]
    rec({"type": "summary", "strategy": args.strategy,
         "rounds_done": len(rounds),
         "child_makespans_ms": [x["child_makespan_ms"] for x in rounds],
         "child_latency_p50_ms": [x["child_latency_p50_ms"] for x in rounds],
         "child_latency_p95_ms": [x["child_latency_p95_ms"] for x in rounds],
         "total_wall_ms": round(total, 3),
         "baseline_metrics": baseline})
    print(f"[done] {out_path} total={total / 1000:.1f}s")
    out.close()


if __name__ == "__main__":
    main()
