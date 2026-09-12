#!/usr/bin/env python3
"""统一栈压力矩阵分析:3 动作 × 压力 0/2/4/6/8 × 4 轮 → crossover 表。"""
import json
from pathlib import Path

ART = Path("artifacts")
STRATS = ["pack", "retrieve", "recompute"]
PRESSURES = [0, 2, 4, 6, 8]


def load(strategy, pressure):
    for name in (f"unified_v3_{strategy}_p{pressure}_20260912.json",
                 f"unified_v2_{strategy}_p{pressure}_20260912.json",
                 f"unified_{strategy}_p{pressure}_20260912.json"):
        p = ART / name
        if p.exists():
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            rounds = [r for r in rows if r.get("type") == "round"]
            summary = [r for r in rows if r.get("type") == "summary"][0]
            return rounds, summary, name
    return None, None, None


def mean(v):
    return sum(v) / len(v) if v else float("nan")


table = {}
for s in STRATS:
    for p in PRESSURES:
        rounds, summary, name = load(s, p)
        if not rounds:
            print(f"MISSING {s} p{p}")
            continue
        mk = [r["child_makespan_ms"] for r in rounds]
        p95 = [r["child_latency_p95_ms"] for r in rounds]
        valid = summary["valid"]
        # gpu-1 external transfer + local hit per round (retrieve/recompute amortization evidence)
        tr = []
        for r in rounds:
            g1 = r.get("sources_delta", {}).get("gpu-1", {})
            def num(k):
                v = g1.get(k, 0)
                return v if isinstance(v, (int, float)) else 0
            tr.append((num("src:external_kv_transfer"), num("src:local_cache_hit"),
                       num("src:local_compute")))
        table[(s, p)] = {"mk": mk, "p95": p95, "valid": valid, "tr": tr, "file": name}

print("=" * 96)
print(f"{'strategy':<10}{'p':>3} {'r0':>7} {'r1':>7} {'r2':>7} {'r3':>7} {'mean':>8} {'r1-3均':>8} {'p95均':>8}  valid")
print("-" * 96)
for s in STRATS:
    for p in PRESSURES:
        if (s, p) not in table:
            continue
        d = table[(s, p)]
        m = d["mk"]
        print(f"{s:<10}{p:>3} {m[0]:>7.0f} {m[1]:>7.0f} {m[2]:>7.0f} {m[3]:>7.0f} "
              f"{mean(m):>8.0f} {mean(m[1:]):>8.0f} {mean(d['p95']):>8.0f}  {d['valid']}")

print()
print("best strategy per pressure (by mean / by r1-3 amortized):")
for p in PRESSURES:
    by_mean = {s: mean(table[(s, p)]["mk"]) for s in STRATS if (s, p) in table}
    by_amort = {s: mean(table[(s, p)]["mk"][1:]) for s in STRATS if (s, p) in table}
    bm = min(by_mean, key=by_mean.get)
    ba = min(by_amort, key=by_amort.get)
    print(f"  p{p}: mean→{bm}({by_mean[bm]:.0f}ms, 次优差{sorted(by_mean.values())[1]-by_mean[bm]:.0f}ms)"
          f" | amort→{ba}({by_amort[ba]:.0f}ms, 次优差{sorted(by_amort.values())[1]-by_amort[ba]:.0f}ms)")

print()
print("retrieve gpu-1 per round (transfer / local_hit / compute tokens):")
for p in PRESSURES:
    if ("retrieve", p) in table:
        print(f"  p{p}: " + " | ".join(f"r{i}:{t[0]:.0f}/{t[1]:.0f}/{t[2]:.0f}"
                                       for i, t in enumerate(table[("retrieve", p)]["tr"])))

print()
print("recompute gpu-1 per round (transfer / local_hit / compute tokens):")
for p in PRESSURES:
    if ("recompute", p) in table:
        print(f"  p{p}: " + " | ".join(f"r{i}:{t[0]:.0f}/{t[1]:.0f}/{t[2]:.0f}"
                                       for i, t in enumerate(table[("recompute", p)]["tr"])))
