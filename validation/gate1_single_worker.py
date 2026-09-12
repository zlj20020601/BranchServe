#!/usr/bin/env python3
"""Gate 1: 单 Worker LMCache Store -> Retrieve 证据采集器（不改 BranchServe）。

实验设计（固定）：
  cold      : 唯一 8192-token prefix + suffixA   冷 prefill，KV store 到 LMCache 磁盘后端
  local     : 同 prefix + suffixC               预期 vLLM GPU 本地 APC 命中
  reset     : POST /reset_prefix_cache           驱逐 GPU 本地 prefix cache（磁盘 LMCache 不受影响）
  retrieve  : 同 prefix + suffixA 重放           预期 LMCache 外部 retrieve；输出须与 cold 逐 token 一致
  retrieve2 : 同 prefix + suffixB                第二次 retrieve，证据稳定性
  retrieve_restart : 手动重启 worker 后重放 suffixA（--seed 复用同一 prefix）

每个请求采集：TTFT(流式首 token)、总延迟、输出 sha256、usage、
  /metrics 计数器差分（仅 prefix-cache/LMCache/KV-transfer 相关名字，正则匹配、不硬编码指标名）、
  worker 日志新增的 LMCache 相关行（字节偏移跟踪）。

仅用 stdlib。产物 jsonl：首行 header，其后每请求/每动作一行，末行 summary。

用法（nmb1, branchserve env）：
  python gate1_single_worker.py --phases cold,local,reset,retrieve,retrieve2
  # 重启 worker 后：
  python gate1_single_worker.py --phases retrieve_restart --seed <上次seed>
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
import urllib.error

METRIC_RE = re.compile(r"prefix.?cache|lmcache|kv.?transfer|retriev|external", re.I)
LOG_RE = re.compile(r"lmcache|prefix.?cache|kv.?transfer", re.I)


# ---------------------------------------------------------------- utilities
def http_json(url, payload=None, timeout=600):
    """POST json (or GET if payload None). Returns (status, obj|text)."""
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def scrape_metrics(worker):
    """Filtered /metrics -> {name_with_labels: float}."""
    try:
        status, text = http_json(worker + "/metrics", timeout=30)
        if status != 200:
            return None
    except Exception:
        return None
    out = {}
    pat = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-\d.eE+]+)\s*$")
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        name = m.group(1) + (m.group(2) or "")
        if METRIC_RE.search(name):
            try:
                out[name] = float(m.group(3))
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
    """Track worker log byte offset; return new LMCache-ish lines."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                self.offset = f.tell()
        except OSError:
            self.path = None  # log unavailable -> no-op

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
        lines = [l.decode("utf-8", "replace").rstrip() for l in data.splitlines()]
        return [l for l in lines if LOG_RE.search(l)]


# ---------------------------------------------------------------- request
def gen_ids(rng, n, lo, hi):
    return [rng.randrange(lo, hi) for _ in range(n)]


def stream_completion(worker, model, prompt_ids, max_tokens, tag):
    """Streaming completion; returns dict with ttft/total/out hash/usage."""
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        worker + "/v1/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = []
    usage = None
    finish = None
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
    total = time.time() - t0
    text = "".join(chunks)
    return {
        "tag": tag, "n_prompt_tokens": len(prompt_ids),
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 4), "finish_reason": finish,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_preview": text[:80], "usage": usage,
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model-config",
                    default="/root/autodl-tmp/models/Qwen3.5-4B/config.json")
    ap.add_argument("--worker-log",
                    default="/root/autodl-tmp/branchserve/logs/worker0.log")
    ap.add_argument("--out", default=None, help="default: artifacts/lmcache_single_worker_<ts>.jsonl")
    ap.add_argument("--phases",
                    default="cold,local,reset,retrieve,retrieve2",
                    help="subset of: cold,local,reset,retrieve,retrieve2,retrieve_restart")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: epoch sec; reuse for retrieve_restart")
    ap.add_argument("--prefix-tokens", type=int, default=8192)
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    allowed = {"cold", "local", "reset", "retrieve", "retrieve2", "retrieve_restart"}
    bad = set(phases) - allowed
    if bad:
        sys.exit(f"unknown phases: {bad}")

    seed = args.seed if args.seed is not None else int(time.time())
    with open(args.model_config) as f:
        mcfg = json.load(f)
    vocab = mcfg.get("vocab_size") or mcfg.get("text_config", {}).get("vocab_size")
    if not vocab:
        sys.exit("vocab_size not found in model config (top-level or text_config)")
    lo, hi = 1000, vocab - 1000

    rng = random.Random(seed)
    prefix = gen_ids(rng, args.prefix_tokens, lo, hi)
    suffix = {s: gen_ids(random.Random(seed + i), args.suffix_tokens, lo, hi)
              for i, s in enumerate(("A", "B", "C"), start=1)}

    # health + model id + cache-related routes
    hstat, _ = http_json(args.worker_url + "/health", timeout=10)
    mstat, mobj = http_json(args.worker_url + "/v1/models", timeout=10)
    model_id = (mobj.get("data") or [{}])[0].get("id") if isinstance(mobj, dict) else None
    rstat, routes = http_json(args.worker_url + "/routes", timeout=10)
    cache_routes = [r for r in (routes if isinstance(routes, list) else [])
                    if "cache" in str(r).lower() or "reset" in str(r).lower()]

    out_path = args.out or (
        "/root/autodl-tmp/branchserve/artifacts/lmcache_single_worker_"
        + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    out = open(out_path, "a", buffering=1)

    def rec(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    rec({
        "type": "header", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed, "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens, "max_tokens": args.max_tokens,
        "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:16],
        "vocab_size": vocab, "model_id": model_id,
        "health_status": hstat, "models_status": mstat,
        "cache_routes_seen": cache_routes, "phases": phases,
        "worker_url": args.worker_url,
        "note": "prefix/suffix are raw token-id sequences; greedy temperature=0",
    })
    print(f"[header] seed={seed} model={model_id} health={hstat} out={out_path}")

    tail = LogTail(args.worker_log)
    cold_hash = None
    metrics = scrape_metrics(args.worker_url)

    def run_req(phase, tag, sfx):
        nonlocal cold_hash, metrics
        tail.new_lines()  # advance offset up to now
        before = scrape_metrics(args.worker_url)
        r = stream_completion(args.worker_url, model_id,
                              prefix + suffix[sfx], args.max_tokens, tag)
        logs = tail.new_lines()
        after = scrape_metrics(args.worker_url)
        r.update({
            "type": "req", "phase": phase,
            "metrics_diff": diff_metrics(before, after),
            "lmcache_log_new": len(logs),
            "lmcache_log_sample": logs[:40],
        })
        if phase == "cold" and tag.endswith("A"):
            cold_hash = r["output_sha256"]
        if phase in ("retrieve", "retrieve_restart") and tag.endswith("A"):
            r["hash_matches_cold"] = (r["output_sha256"] == cold_hash) if cold_hash else None
        rec(r)
        print(f"[{phase}] tag={tag} ttft={r['ttft_s']}s total={r['total_s']}s "
              f"hash={r['output_sha256'][:10]} lognew={len(logs)} "
              f"metrics_changed={len(r['metrics_diff'])}")
        return r

    for ph in phases:
        if ph == "cold":
            run_req("cold", "cold_A", "A")
        elif ph == "local":
            run_req("local", "localhit_C", "C")
        elif ph == "reset":
            results = {}
            for path in ("/reset_prefix_cache", "/v1/reset_prefix_cache"):
                st, body = http_json(args.worker_url + path, payload={}, timeout=30)
                results[path] = {"status": st, "body": str(body)[:200]}
            logs = tail.new_lines()
            rec({"type": "action", "phase": "reset", "attempts": results,
                 "lmcache_log_new": len(logs), "lmcache_log_sample": logs[:40]})
            ok = [p for p, v in results.items() if v["status"] == 200]
            print(f"[reset] ok_paths={ok} all={ {p: v['status'] for p, v in results.items()} }")
        elif ph == "retrieve":
            run_req("retrieve", "lmcache_A_replay", "A")
        elif ph == "retrieve2":
            run_req("retrieve2", "lmcache_B", "B")
        elif ph == "retrieve_restart":
            run_req("retrieve_restart", "lmcache_A_after_restart", "A")

    rec({"type": "summary", "seed": seed, "note": "inspect metrics_diff + lmcache_log_sample for external-hit evidence"})
    out.close()
    print(f"[done] artifact={out_path}")


if __name__ == "__main__":
    main()
