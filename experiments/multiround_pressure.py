#!/usr/bin/env python3
"""Multi-round 3-arm runner WITH sustained GPU0 pressure (phase5-style load).

在 multiround_strategy 基础上加压力层:
  每轮:parent →(retrieve: store 观察)→ 启动 N 个背景槽位(循环续发,复用当轮
  history 前缀,纯解码负载,ignore_eos)→ 屏障(gpu-0 running+waiting ≥ N)
  → 4 child 并发 → child 完成后停背景并排空 → 下一轮

pressure = N 个背景槽位,打在 parent 所在卡(gpu-0/:8000)。
研究问题:gpu-0 忙到什么程度,child 组去另一张卡(retrieve/recompute)开始划算。

label 默认 multiround-p{P}-{strategy}:不同 (压力,策略) 组合天然隔离,
跨 run 零 APC 命中,无需重起 worker。

用法:
  python multiround_pressure.py --strategy pack --pressure 4
  (retrieve 臂需 connector 栈;--server-log 指向 server 日志)
"""
import argparse
import concurrent.futures
import json
import math
import re
import statistics
import threading
import time

from multiround_strategy import (
    LABEL_DEFAULT,
    StoreWatcher,
    build_history,
    chat,
    diff,
    fetch_sources,
    prefix_to,
    scan_stored,
)

RUNNING_RE = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([\d.eE+]+)\s*$")
WAITING_RE = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([\d.eE+]+)\s*$")


def fetch_running(port):
    """Return (running, waiting) gauges, or (None, None)."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None, None
    running = waiting = None
    for line in text.splitlines():
        m = RUNNING_RE.match(line)
        if m:
            running = float(m.group(1))
        m = WAITING_RE.match(line)
        if m:
            waiting = float(m.group(1))
    return running, waiting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["pack", "retrieve", "recompute", "dynamic"], required=True)
    ap.add_argument("--pressure", type=int, required=True,
                    help="gpu-0 背景槽位数(0 = 无压力,等价 multiround_strategy)")
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3.5-4B")
    ap.add_argument("--parent-port", type=int, default=8000)
    ap.add_argument("--child-port", type=int, default=8001)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--initial-tokens", type=int, default=8192)
    ap.add_argument("--append-tokens", type=int, default=2048)
    ap.add_argument("--fanout", type=int, default=4)
    ap.add_argument("--child-max-tokens", type=int, default=256)
    ap.add_argument("--parent-max-tokens", type=int, default=8)
    ap.add_argument("--background-output-tokens", type=int, default=2048)
    ap.add_argument("--barrier-timeout-s", type=float, default=30.0)
    ap.add_argument("--server-log", default="logs/lmcache_server_formal_20260912.log")
    ap.add_argument("--settle-s", type=float, default=0.25)
    ap.add_argument("--store-timeout-s", type=float, default=60.0)
    ap.add_argument("--store-quiet-s", type=float, default=1.5)
    ap.add_argument("--label", default=None,
                    help="默认自动 multiround-p{pressure}-{strategy}")
    ap.add_argument("--child-salt", default=None,
                    help="统一栈 recompute 动作:child 请求携带的 cache_salt,"
                         "默认自动 'recompute-<label>';传 'none' 显式关闭")
    ap.add_argument("--dynamic-threshold", type=float, default=5.0,
                    help="dynamic 策略:parent 卡 pressure(running+waiting) >= 该值则 retrieve,否则 pack")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    label = args.label or f"multiround-p{args.pressure}-{args.strategy}"
    child_salt = args.child_salt
    if child_salt == "none":
        child_salt = None
    elif child_salt is None and args.strategy == "recompute":
        child_salt = f"recompute-{label}"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    suffixes = [f"\nBranch {i}: summarize the branch-specific findings above in one short paragraph."
                for i in range(args.fanout)]
    child_target_port = args.parent_port if args.strategy == "pack" else args.child_port
    recompute_salt = f"recompute-{label}"
    store_healthy = True  # dynamic 的 store 健康兜底:上次 retrieve 若零 transfer 则降级 recompute

    out_path = args.out or (
        f"artifacts/multiround_p{args.pressure}_{args.strategy}_"
        + time.strftime("%Y%m%d_%H%M%S") + ".json")
    out = open(out_path, "w", buffering=1)

    def rec(o):
        out.write(json.dumps(o, ensure_ascii=False) + "\n")

    max_needed = args.initial_tokens + (args.rounds - 1) * args.append_tokens + 512
    t_build = time.time()
    blocks = build_history(tok, max_needed, label)
    rec({"type": "header", "strategy": args.strategy, "pressure": args.pressure,
         "label": label, "child_salt": child_salt, "stack": "unified-connector",
         "dynamic_threshold": args.dynamic_threshold,
         "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
         "dispatch_policy": "optimistic_after_parent",
         "rounds": args.rounds, "initial_tokens": args.initial_tokens,
         "append_tokens": args.append_tokens, "fanout": args.fanout,
         "child_max_tokens": args.child_max_tokens,
         "background_output_tokens": args.background_output_tokens,
         "parent_port": args.parent_port, "child_port": args.child_port,
         "child_target_port": child_target_port,
         "suffix_tokens": [len(tok.encode(s, add_special_tokens=False)) for s in suffixes],
         "build_s": round(time.time() - t_build, 2)})

    arm_start = time.perf_counter()
    # 背景负载锚定在轮0前缀:固定 token 数(prompt+2048 永不超 16K),
    # 且 gpu-0 上 APC 热命中,保持纯解码压力(phase5 语义)
    bg_base, _ = prefix_to(blocks, tok, args.initial_tokens)
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

        # ---- 1. parent (clean of pressure: background starts after parent)
        p_before = fetch_sources(args.parent_port)
        parent = chat(args.parent_port, history, args.parent_max_tokens, f"r{r}-parent")
        parent_after = fetch_sources(args.parent_port)
        parent_src = diff(p_before, parent_after)

        # ---- 2. store watch (retrieve/dynamic only): 旁路观察,默认不阻塞派发
        watcher = None
        if args.strategy in ("retrieve", "dynamic") and server_offset >= 0:
            watcher = StoreWatcher(args.server_log, server_offset,
                                   quiet_s=args.store_quiet_s,
                                   timeout_s=args.store_timeout_s)
            watcher.start()
        time.sleep(args.settle_s)

        # ---- 3. sustained background pressure on parent worker (phase5 pattern)
        bg_stats = {"submitted": 0, "completed": 0, "failed": 0}
        bg_lock = threading.Lock()
        bg_stop = threading.Event()
        bg_executor = None
        if args.pressure > 0:
            bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.pressure)

            def run_background(slot):
                sequence = 0
                while not bg_stop.is_set():
                    suffix = (f"\nBackground slot {slot} sequence {sequence}: "
                              "generate a long deterministic response.")
                    with bg_lock:
                        bg_stats["submitted"] += 1
                    try:
                        chat(args.parent_port, bg_base + suffix,
                             args.background_output_tokens, f"r{r}-bg{slot}-{sequence}",
                             min_tokens=args.background_output_tokens, ignore_eos=True)
                        with bg_lock:
                            bg_stats["completed"] += 1
                    except Exception:
                        with bg_lock:
                            bg_stats["failed"] += 1
                        time.sleep(0.2)
                    sequence += 1

            for i in range(args.pressure):
                bg_executor.submit(run_background, i)

            # barrier: gpu-0 running+waiting >= pressure
            pressure_ready = False
            p_running = p_waiting = 0.0
            barrier_ms = 0.0
            t_barrier = time.perf_counter()
            deadline = t_barrier + args.barrier_timeout_s
            while time.perf_counter() < deadline:
                p_running, p_waiting = fetch_running(args.parent_port)
                if (p_running is not None and p_waiting is not None
                        and p_running + p_waiting >= args.pressure):
                    pressure_ready = True
                    break
                time.sleep(0.05)
            barrier_ms = (time.perf_counter() - t_barrier) * 1000
        else:
            pressure_ready, p_running, p_waiting, barrier_ms = True, 0.0, 0.0, 0.0

        # ---- dynamic decision (per round, reads telemetry the router "sees")
        observed_pressure = (p_running + p_waiting
                             if p_running is not None and p_waiting is not None else 0.0)
        action = args.strategy
        round_target = child_target_port
        round_child_salt = child_salt
        if args.strategy == "dynamic":
            if not store_healthy:
                action = "recompute"
            elif observed_pressure >= args.dynamic_threshold:
                action = "retrieve"
            else:
                action = "pack"
            if action == "pack":
                round_target, round_child_salt = args.parent_port, None
            elif action == "retrieve":
                round_target, round_child_salt = args.child_port, None
            else:  # recompute
                round_target, round_child_salt = args.child_port, recompute_salt

        # ---- 4. children fan-out
        c_before = {args.parent_port: fetch_sources(args.parent_port),
                    args.child_port: fetch_sources(args.child_port)}
        children = [None] * args.fanout

        def run_child(i):
            children[i] = chat(round_target, history + suffixes[i],
                               args.child_max_tokens, f"r{r}-c{i}",
                               cache_salt=round_child_salt)

        threads = [threading.Thread(target=run_child, args=(i,)) for i in range(args.fanout)]
        t_children = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        children_done = time.perf_counter()

        c_after = {args.parent_port: fetch_sources(args.parent_port),
                   args.child_port: fetch_sources(args.child_port)}
        with bg_lock:
            bg_done_at_child = bg_stats["completed"]
            bg_submitted_at_child = bg_stats["submitted"]
            bg_failed_at_child = bg_stats["failed"]

        # ---- 5. stop + drain background before next round
        bg_stop.set()
        drain_ms = 0.0
        if bg_executor is not None:
            t_drain = time.perf_counter()
            bg_executor.shutdown(wait=True)
            drain_ms = (time.perf_counter() - t_drain) * 1000
        with bg_lock:
            bg_total = dict(bg_stats)

        store_rec = None
        if watcher is not None:
            watcher.join(args.store_timeout_s)
            store_rec = watcher.result()
            _, _, _, server_offset = scan_stored(args.server_log, server_offset)

        g0_d = diff(c_before[args.parent_port], c_after[args.parent_port])
        g1_d = diff(c_before[args.child_port], c_after[args.child_port])
        if args.strategy == "dynamic" and action == "retrieve":
            tr = g1_d.get("src:external_kv_transfer", 0)
            if not isinstance(tr, (int, float)) or tr <= 0:
                store_healthy = False

        lat = sorted(c["latency_ms"] for c in children if c)
        n = len(lat)
        p95_idx = max(0, int(math.ceil(0.95 * n)) - 1) if n else None
        rec({
            "type": "round", "round_id": r,
            "history_tokens": history_tokens,
            "append_actual": history_tokens - prev_history_tokens,
            "parent_latency_ms": parent["latency_ms"],
            "parent_sources": parent_src,
            "store": store_rec,
            "pressure": args.pressure,
            "observed_pressure": observed_pressure,
            "action": action,
            "store_healthy": store_healthy,
            "pressure_ready": pressure_ready,
            "pressure_running_at_barrier": p_running,
            "pressure_waiting_at_barrier": p_waiting,
            "barrier_ms": round(barrier_ms, 1),
            "background_done_at_child_finish": bg_done_at_child,
            "background_submitted_at_child_finish": bg_submitted_at_child,
            "background_failed_at_child_finish": bg_failed_at_child,
            "background_total": bg_total,
            "drain_ms": round(drain_ms, 1),
            "children": children,
            "child_makespan_ms": round((children_done - t_children) * 1000, 3),
            "child_latency_p50_ms": round(statistics.median(lat), 3) if n else None,
            "child_latency_p95_ms": round(lat[p95_idx], 3) if n and p95_idx is not None else None,
            "child_latency_max_ms": round(lat[-1], 3) if n else None,
            "round_wall_ms": round((time.perf_counter() - round0) * 1000, 3),
            "cumulative_makespan_ms": round((time.perf_counter() - arm_start) * 1000, 3),
            "sources_delta": {"gpu-0": g0_d, "gpu-1": g1_d},
            "note": "gpu-0 sources_delta includes background prefill hits; gpu-1 is child-clean",
        })
        print(f"[r{r}] hist={history_tokens} parent={parent['latency_ms']:.0f}ms "
              f"barrier={pressure_ready}({p_running:.0f}+{p_waiting:.0f}) "
              f"action={action} makespan={(children_done - t_children) * 1000:.0f}ms "
              f"drain={drain_ms:.0f}ms", flush=True)
        prev_history_tokens = history_tokens

    rounds = [json.loads(l) for l in open(out_path) if l.strip()]
    rounds = [x for x in rounds if x.get("type") == "round"]
    rec({"type": "summary", "strategy": args.strategy, "pressure": args.pressure,
         "rounds_done": len(rounds),
         "actions": [x.get("action") for x in rounds],
         "observed_pressures": [x.get("observed_pressure") for x in rounds],
         "child_makespans_ms": [x["child_makespan_ms"] for x in rounds],
         "store_healthy_final": store_healthy,
         "valid": all(x["pressure_ready"] and x["background_done_at_child_finish"] == 0
                      for x in rounds),
         "total_wall_ms": round((time.perf_counter() - arm_start) * 1000, 3)})
    print(f"[done] {out_path} valid="
          f"{all(x['pressure_ready'] and x['background_done_at_child_finish'] == 0 for x in rounds)}")
    out.close()


if __name__ == "__main__":
    main()
