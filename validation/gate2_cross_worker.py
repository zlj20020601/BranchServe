#!/usr/bin/env python3
"""Gate 2: 跨 Worker 成对对照 Parent@GPU0(store) -> Child@GPU1。

每对(pair)流程（一个 pair 一次调用，由外部调度交替 A/B 顺序）：
  1. parent(:8000, 带 MP connector) 请求 prefix_i+suffixA —— 触发 store
  2. 轮询 lmcache server 日志，等本 pair 的新增 "Stored" 块（默认 16 块）
  3. child(:8001) 请求同一 prefix_i+suffixA —— child worker 由外部按臂启动：
       recompute 臂 = 无 connector 的 worker（纯重算，无外部 KV）
       retrieve  臂 = 带 MP connector 的 worker（从 server 检索）
  4. 校验 child 输出 hash == parent 输出 hash（不同计算路径 greedy 一致）

记录：parent/child 各自 TTFT、hash、metrics 差分、child 日志新增行、store 等待时长。
prefix 每对唯一（seed=base+pair_index），杜绝本地/服务端跨对污染。

用法：
  python gate2_cross_worker.py --child-mode recompute --pair-index 1 --order A_first
"""

import argparse
import hashlib
import json
import math
import random
import re
import time
import urllib.request
import urllib.error

METRIC_RE = re.compile(
    r"prefix.?cache|lmcache|kv.?transfer|retriev|external|"
    r"prefill.*(computed|tokens|source)|computed.*prefill|"
    r"prompt.*(source|compute)|local.*compute",
    re.I,
)
LOG_RE = re.compile(r"lmcache|prefix.?cache|kv.?transfer", re.I)


def http_json(url, payload=None, timeout=600):
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode("utf-8", "replace"), e.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def scrape_metrics(base):
    try:
        status, text = http_json(base + "/metrics", timeout=30)
        if status != 200:
            return None
    except Exception:
        return None
    out = {}
    pat = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-\d.eE+]+)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if m and METRIC_RE.search(m.group(1)):
            try:
                out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
            except ValueError:
                pass
    return out


def diff_metrics(before, after):
    if before is None or after is None:
        return {"error": "metrics unavailable"}
    d = {}
    for k in set(before) | set(after):
        b, a = before.get(k), after.get(k)
        if b is None or a is None:
            d[k] = {"before": b, "after": a}
        elif a != b:
            d[k] = a - b
    return d


class LogTail:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                self.offset = f.tell()
        except OSError:
            self.path = None

    def new_lines(self):
        if not self.path:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
        except OSError:
            return []
        return [l.decode("utf-8", "replace").rstrip()
                for l in data.splitlines() if LOG_RE.search(l.decode("utf-8", "replace"))]


def stored_after(log_path, offset):
    """Return LMCache store record and token counts beyond a byte offset."""
    try:
        with open(log_path, "rb") as f:
            f.seek(offset)
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return -1, -1
    token_counts = [int(m.group(1)) for m in re.finditer(r"Stored (\d+) tokens", data)]
    return len(token_counts), sum(token_counts)


def numeric_store_delta(before, after):
    """Return newly observed store/write/offload counter units."""
    if before is None or after is None:
        return 0.0
    delta = diff_metrics(before, after)
    total = 0.0
    for key, value in delta.items():
        if not re.search(r"store|write|put|offload", key, re.I):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            total += max(0.0, float(value))
    return total


def metric_delta_for_source(delta, source):
    """Sum positive prompt-token deltas for one vLLM source label."""
    total = 0.0
    for key, value in delta.items():
        if f'source="{source}"' not in key and f"source={source}" not in key:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            total += max(0.0, float(value))
    return total


def stream_completion(base, model, prompt_ids, max_tokens, tag):
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, chunks, usage, finish = None, [], None, None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                txt = ch[0].get("text") or ""
                if txt:
                    chunks.append(txt)
                    if ttft is None:
                        ttft = time.time() - t0
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
    text = "".join(chunks)
    return {"tag": tag, "ttft_s": round(ttft, 4) if ttft else None,
            "total_s": round(time.time() - t0, 4), "finish_reason": finish,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "output_preview": text[:60], "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-url", default="http://127.0.0.1:8000")
    ap.add_argument("--child-url", default="http://127.0.0.1:8001")
    ap.add_argument("--child-mode", choices=["recompute", "retrieve"], required=True)
    ap.add_argument("--pair-index", type=int, required=True)
    ap.add_argument("--order", default=None, help="e.g. A_first / B_first (label)")
    ap.add_argument("--base-seed", type=int, default=970000)
    ap.add_argument("--model-config",
                    default="/root/autodl-tmp/models/Qwen3.5-4B/config.json")
    ap.add_argument("--child-log", default="/root/autodl-tmp/branchserve/logs/worker1.log")
    ap.add_argument("--store-log", "--server-log", dest="store_log",
                    default="/root/autodl-tmp/branchserve/logs/lmcache_server_gate1.log",
                    help="parent/LMCache log containing 'Stored N tokens' records")
    ap.add_argument("--store-metrics-url", default=None,
                    help="optional LMCache server base URL; polls <url>/metrics for store counters")
    ap.add_argument("--out",
                    default="/root/autodl-tmp/branchserve/artifacts/lmcache_cross_worker.jsonl")
    ap.add_argument("--prefix-tokens", type=int, default=8448)
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=528)
    ap.add_argument("--store-chunks", type=int, default=0,
                    help="expected newly stored chunks; default is ceil(prefix/chunk-size)")
    ap.add_argument("--store-wait-s", type=float, default=60.0)
    ap.add_argument("--store-settle-s", type=float, default=2.0,
                    help="fallback settle time when no store log/metrics endpoint is available")
    ap.add_argument("--model-id", default=None,
                    help="override the model id returned by /v1/models")
    args = ap.parse_args()

    if args.chunk_size <= 0:
        ap.error("--chunk-size must be positive")
    expected_store_chunks = args.store_chunks or math.ceil(args.prefix_tokens / args.chunk_size)

    with open(args.model_config) as f:
        mcfg = json.load(f)
    vocab = mcfg.get("vocab_size") or mcfg.get("text_config", {}).get("vocab_size")
    if not vocab:
        raise RuntimeError("vocab_size not found in model config")
    lo, hi = 1000, vocab - 1000

    seed = args.base_seed + args.pair_index
    rng = random.Random(seed)
    prefix = [rng.randrange(lo, hi) for _ in range(args.prefix_tokens)]
    suffix = [random.Random(seed + 1).randrange(lo, hi) for _ in range(args.suffix_tokens)]
    prompt = prefix + suffix

    hstat, _ = http_json(args.parent_url + "/health", timeout=10)
    cstat, _ = http_json(args.child_url + "/health", timeout=10)
    if hstat != 200 or cstat != 200:
        raise RuntimeError(f"workers not healthy: parent={hstat} child={cstat}")
    mstat, mobj = http_json(args.parent_url + "/v1/models", timeout=10)
    cmstat, cmobj = http_json(args.child_url + "/v1/models", timeout=10)
    model_id = args.model_id or (
        (mobj.get("data") or [{}])[0].get("id") if isinstance(mobj, dict) else None
    )
    if not model_id:
        raise RuntimeError("could not determine model id; pass --model-id")
    child_model_ids = [item.get("id") for item in cmobj.get("data", [])] \
        if isinstance(cmobj, dict) else []
    if mstat != 200 or cmstat != 200 or model_id not in child_model_ids:
        raise RuntimeError(
            f"model mismatch: parent_status={mstat} child_status={cmstat} "
            f"parent_model={model_id!r} child_models={child_model_ids!r}")

    out = open(args.out, "a", buffering=1)
    def rec(o):
        out.write(json.dumps(o, ensure_ascii=False) + "\n")

    rec({
        "type": "header", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed, "pair": args.pair_index,
        "prefix_tokens": args.prefix_tokens, "suffix_tokens": args.suffix_tokens,
        "max_tokens": args.max_tokens, "chunk_size": args.chunk_size,
        "expected_store_chunks": expected_store_chunks,
        "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:16],
        "model_id": model_id, "child_mode": args.child_mode,
        "order": args.order, "parent_url": args.parent_url,
        "child_url": args.child_url, "parent_health_status": hstat,
        "child_health_status": cstat, "parent_models_status": mstat,
        "child_models_status": cmstat,
        "note": "same deterministic prefix+suffix is used on both workers; temperature=0",
    })

    server_off = 0
    try:
        with open(args.store_log, "rb") as f:
            f.seek(0, 2)
            server_off = f.tell()
    except OSError:
        pass

    # 1. parent request (triggers store)
    store_before = scrape_metrics(args.store_metrics_url) if args.store_metrics_url else None
    p_before = scrape_metrics(args.parent_url)
    p = stream_completion(args.parent_url, model_id, prompt, args.max_tokens, f"p{args.pair_index}_A")
    p_after = scrape_metrics(args.parent_url)
    p_rec = dict(p, type="req", role="parent", pair=args.pair_index,
                 mode=args.child_mode, order=args.order,
                 metrics_diff=diff_metrics(p_before, p_after))
    rec(p_rec)
    print(f"[pair{args.pair_index} parent] ttft={p['ttft_s']} hash={p['output_sha256'][:10]}")

    # 2. wait for store chunks in server log and, when available, server metrics
    t0 = time.time()
    stored_records, stored_tokens = 0, 0
    metric_delta = 0.0
    while time.time() - t0 < args.store_wait_s:
        stored_records, stored_tokens = stored_after(args.store_log, server_off)
        store_after = scrape_metrics(args.store_metrics_url) if args.store_metrics_url else None
        metric_delta = numeric_store_delta(store_before, store_after)
        if (stored_records >= expected_store_chunks or
                stored_tokens >= args.prefix_tokens or
                metric_delta >= expected_store_chunks):
            break
        time.sleep(0.5)
    store_wait = round(time.time() - t0, 3)
    store_after = scrape_metrics(args.store_metrics_url) if args.store_metrics_url else None
    metric_delta = numeric_store_delta(store_before, store_after)
    store_verified = (
        stored_records >= expected_store_chunks or
        stored_tokens >= args.prefix_tokens or
        metric_delta >= expected_store_chunks
    )
    if not store_verified:
        time.sleep(max(0.0, args.store_settle_s))
    rec({"type": "store", "pair": args.pair_index,
         "store_log": args.store_log, "records_seen": stored_records,
         "tokens_seen": stored_tokens,
         "expected_chunks": expected_store_chunks, "wait_s": store_wait,
         "timeout": not store_verified, "store_verified": store_verified,
         "store_metric_delta": metric_delta,
         "settle_s": args.store_settle_s if not store_verified else 0.0})
    print(f"[pair{args.pair_index} store] records={stored_records}/{expected_store_chunks} "
          f"tokens={stored_tokens}/{args.prefix_tokens} "
          f"metric_delta={metric_delta:g} verified={store_verified} wait={store_wait}s")

    # 3. child request
    ctail = LogTail(args.child_log)
    c_before = scrape_metrics(args.child_url)
    c = stream_completion(args.child_url, model_id, prompt, args.max_tokens,
                          f"c{args.pair_index}_{args.child_mode}")
    clogs = ctail.new_lines()
    c_after = scrape_metrics(args.child_url)
    c_metrics_diff = diff_metrics(c_before, c_after)
    external_tokens = metric_delta_for_source(c_metrics_diff, "external_kv_transfer")
    local_hit_tokens = metric_delta_for_source(c_metrics_diff, "local_cache_hit")
    local_compute_tokens = metric_delta_for_source(c_metrics_diff, "local_compute")
    recompute_verified = (
        args.child_mode == "recompute" and external_tokens == 0 and
        local_hit_tokens == 0 and local_compute_tokens >= args.prefix_tokens
    )
    retrieve_verified = (
        args.child_mode == "retrieve" and
        external_tokens >= args.prefix_tokens
    )
    c_rec = dict(c, type="req", role="child", pair=args.pair_index,
                 mode=args.child_mode, order=args.order,
                 metrics_diff=c_metrics_diff,
                 hash_matches_parent=c["output_sha256"] == p["output_sha256"],
                 external_kv_transfer_tokens=external_tokens,
                 local_cache_hit_tokens=local_hit_tokens,
                 local_compute_tokens=local_compute_tokens,
                 recompute_verified=recompute_verified,
                 retrieve_verified=retrieve_verified,
                 lmcache_log_new=len(clogs), lmcache_log_sample=clogs[:30])
    rec(c_rec)
    print(f"[pair{args.pair_index} child/{args.child_mode}] ttft={c['ttft_s']} "
          f"hash={c['output_sha256'][:10]} match={c_rec['hash_matches_parent']} "
          f"lognew={len(clogs)} metrics_changed={len(c_rec['metrics_diff'])}")
    rec({
        "type": "summary", "pair": args.pair_index, "mode": args.child_mode,
        "hash_matches": c_rec["hash_matches_parent"],
        "store_verified": store_verified,
        "recompute_verified": recompute_verified,
        "retrieve_verified": retrieve_verified,
        "verdict": bool(c_rec["hash_matches_parent"] and
                        (recompute_verified if args.child_mode == "recompute"
                         else retrieve_verified)),
    })
    out.close()


if __name__ == "__main__":
    main()
