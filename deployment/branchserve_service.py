#!/usr/bin/env python3
"""BranchServe router service — 独立三动作动态路由服务.

真实 Agent 流量的入口:标准 /v1/chat/completions 请求 + branchserve 元数据;
router 维护会话注册表,轮询 worker 遥测(running+waiting),按三动作路由:
  PACK      child → parent 所在 worker,无 salt(本地 APC 命中)
  RETRIEVE  child → 另一 worker,无 salt(lmcache 仓库取 KV)
  RECOMPUTE child → 另一 worker,注入 per-session cache_salt(缓存键落空,真实重算)
store 健康追踪:retrieve 派发后 transfer 计数不动 → 记 miss,连续 miss → 降级 RECOMPUTE。

零第三方依赖(标准库)。决策逻辑为可替换函数(decide),阈值模式先行,
成本模型模式预留(mode=cost_model)。

用法:
  python branchserve_service.py --port 9000 --worker0 http://127.0.0.1:8000 \
      --worker1 http://127.0.0.1:8001 --threshold 5
客户端示例(分支元数据放在 body 的 "branchserve" 字段,转发前剥离):
  {"model":"qwen3.5-4b","messages":[...],"branchserve":{"session_id":"s1"},
   ...}                                                       # parent
  {"...","branchserve":{"session_id":"c1","parent_session_id":"s1",
   "branch_id":"b0"}}                                         # child
"""
import argparse
import json
import re
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNNING_RE = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([\d.eE+]+)\s*$")
WAITING_RE = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([\d.eE+]+)\s*$")
TRANSFER_RE = re.compile(
    r"^vllm:prompt_tokens_by_source_total\{[^}]*source=\"?external_kv_transfer\"?[^}]*\}"
    r"\s+([\d.eE+]+)\s*$")

STATE = {
    "workers": ["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
    "threshold": 5.0,
    "mode": "threshold",
    "start_ts": time.time(),
}


# ---------------------------------------------------------------- session registry
LOCK = threading.Lock()
SESSIONS = {}          # session_id -> {"worker": idx, "prompt_tokens": n, "ts": ...}
DECISIONS = []         # ring of last N decisions
STORE = {"healthy": True, "misses": 0, "last_transfer": None}


def register(session_id, worker_idx, prompt_tokens):
    with LOCK:
        SESSIONS[session_id] = {"worker": worker_idx, "prompt_tokens": prompt_tokens,
                                "ts": time.time()}


def lookup_parent(parent_id):
    with LOCK:
        return SESSIONS.get(parent_id)


def log_decision(rec):
    with LOCK:
        DECISIONS.append(rec)
        del DECISIONS[:-500]


# ---------------------------------------------------------------- telemetry
TELEMETRY = {"ts": 0.0, "pressure": [0.0, 0.0], "transfer": [None, None]}


def fetch_worker_metrics(url, timeout=15.0):
    """Return (running, waiting, external_transfer_total) or (None,)*3."""
    try:
        req = urllib.request.Request(url + "/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return None, None, None
    running = waiting = transfer = None
    for line in text.splitlines():
        m = RUNNING_RE.match(line)
        if m:
            running = float(m.group(1))
            continue
        m = WAITING_RE.match(line)
        if m:
            waiting = float(m.group(1))
            continue
        m = TRANSFER_RE.match(line)
        if m:
            transfer = float(m.group(1))
    return running, waiting, transfer


def telemetry_refresh(ttl=0.3):
    now = time.time()
    with LOCK:
        if now - TELEMETRY["ts"] < ttl:
            return
    vals = [fetch_worker_metrics(w) for w in STATE["workers"]]
    with LOCK:
        TELEMETRY["ts"] = now
        for i, (r, w, t) in enumerate(vals):
            if r is not None:
                TELEMETRY["pressure"][i] = r + (w or 0.0)
            if t is not None:
                TELEMETRY["transfer"][i] = t


def pressure_of(worker_idx):
    telemetry_refresh()
    with LOCK:
        return TELEMETRY["pressure"][worker_idx]


def transfer_counter(worker_idx):
    telemetry_refresh()
    with LOCK:
        return TELEMETRY["transfer"][worker_idx]


# ---------------------------------------------------------------- decision core
def decide(parent_worker_idx, session_id):
    """三动作决策(可替换:threshold 模式先行,cost_model 模式预留)。"""
    pressure = pressure_of(parent_worker_idx)
    if not STORE["healthy"]:
        action = "recompute"
    elif pressure >= STATE["threshold"]:
        action = "retrieve"
    else:
        action = "pack"
    return action, pressure


def apply_action(action, parent_worker_idx):
    """动作 → (目标 worker, cache_salt 或 None)。"""
    other = 1 - parent_worker_idx
    if action == "pack":
        return parent_worker_idx, None
    if action == "retrieve":
        return other, None
    return other, "bs-recompute"  # caller 附上 session 维度


# ---------------------------------------------------------------- forwarding
def forward(worker_idx, body, timeout=600):
    url = STATE["workers"][worker_idx] + "/v1/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - 基类签名
        pass  # 静默访问日志,决策日志走 /stats

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "uptime_s": round(time.time() - STATE["start_ts"], 1)})
        elif self.path == "/stats":
            telemetry_refresh()  # 先刷新(TTL 保护),避免展示陈旧值
            with LOCK:
                self._send(200, {
                    "sessions": len(SESSIONS),
                    "store": dict(STORE),
                    "pressure": list(TELEMETRY["pressure"]),
                    "decisions_recent": DECISIONS[-50:],
                })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except Exception as exc:
            self._send(400, {"error": f"bad body: {exc!r}"})
            return
        bs = body.pop("branchserve", None)
        if not bs or not bs.get("session_id"):
            self._send(400, {"error": "missing branchserve.session_id"})
            return

        parent_id = bs.get("parent_session_id")
        if parent_id is None:
            # ---- parent 请求:固定 parent worker(与实验协议一致)
            target_idx = 0
            try:
                t0 = time.perf_counter()
                resp = forward(target_idx, body)
                latency = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                self._send(502, {"error": f"worker: {exc!r}"})
                return
            register(bs["session_id"], target_idx,
                     (resp.get("usage") or {}).get("prompt_tokens") or 0)
            resp["branchserve"] = {"role": "parent", "worker": target_idx,
                                   "latency_ms": round(latency, 1)}
            log_decision({"ts": time.strftime("%H:%M:%S"), "role": "parent",
                          "session": bs["session_id"], "worker": target_idx,
                          "latency_ms": round(latency, 1)})
            self._send(200, resp)
            return

        # ---- child 请求:读遥测 → 决策 → 路由
        parent = lookup_parent(parent_id)
        if parent is None:
            self._send(400, {"error": f"unknown parent session {parent_id}"})
            return
        pw = parent["worker"]
        action, pressure = decide(pw, bs["session_id"])
        target_idx, salt = apply_action(action, pw)
        if salt:
            body["cache_salt"] = f"{salt}-{parent_id}"
        tr_before = transfer_counter(target_idx) if action == "retrieve" else None
        try:
            t0 = time.perf_counter()
            resp = forward(target_idx, body)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            self._send(502, {"error": f"worker: {exc!r}"})
            return
        # store 健康追踪:retrieve 派发后 transfer 计数应有增量
        store_note = None
        if action == "retrieve" and tr_before is not None:
            tr_after = transfer_counter(target_idx)
            if tr_after is not None and tr_after <= tr_before:
                with LOCK:
                    STORE["misses"] += 1
                    if STORE["misses"] >= 2:
                        STORE["healthy"] = False
                store_note = "transfer_miss"
            else:
                with LOCK:
                    STORE["misses"] = 0
                    STORE["healthy"] = True
                store_note = "transfer_ok"
        resp["branchserve"] = {"role": "child", "worker": target_idx,
                               "action": action, "pressure": pressure,
                               "latency_ms": round(latency, 1),
                               "store_note": store_note}
        log_decision({"ts": time.strftime("%H:%M:%S"), "role": "child",
                      "session": bs["session_id"], "parent": parent_id,
                      "action": action, "worker": target_idx,
                      "pressure": pressure, "latency_ms": round(latency, 1),
                      "store": store_note})
        self._send(200, resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--worker0", default="http://127.0.0.1:8000")
    ap.add_argument("--worker1", default="http://127.0.0.1:8001")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--mode", choices=["threshold", "cost_model"], default="threshold")
    args = ap.parse_args()
    STATE["workers"] = [args.worker0, args.worker1]
    STATE["threshold"] = args.threshold
    STATE["mode"] = args.mode
    telemetry_refresh(ttl=0)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"branchserve router on :{args.port} mode={args.mode} "
          f"threshold={args.threshold} workers={STATE['workers']}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
