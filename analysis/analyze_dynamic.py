#!/usr/bin/env python3
"""Dynamic regret 分析:统一栈上 dynamic 臂 vs 静态最优 / 逐轮 oracle。"""
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


def load_dynamic(pressure):
    """Any unified_*_dynamic_p*_20260912.json (v4 naming)."""
    cands = sorted(ART.glob(f"unified_*_dynamic_p{pressure}_20260912.json"))
    if not cands:
        return None, None, None
    p = cands[-1]
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rounds = [r for r in rows if r.get("type") == "round"]
    summary = [r for r in rows if r.get("type") == "summary"][0]
    return rounds, summary, p.name


def mean(v):
    return sum(v) / len(v) if v else float("nan")


# static data: mk[s][p] = list of 4 round makespans
mk = {}
for s in STRATS:
    for p in PRESSURES:
        rounds, summary, name = load(s, p)
        if rounds:
            mk[(s, p)] = [r["child_makespan_ms"] for r in rounds]

print(f"{'p':>2} | {'dyn_mean':>8} {'dyn_act':>10} | {'oracle_fixed':>12} {'regret_fixed':>12} | {'oracle_round':>12} {'regret_round':>12}")
print("-" * 90)
for p in PRESSURES:
    drows, dsum, dname = load_dynamic(p)
    if not drows:
        print(f"{p:>2} | MISSING dynamic artifact")
        continue
    dyn = [r["child_makespan_ms"] for r in drows]
    acts = [r.get("action") for r in drows]
    dm = mean(dyn)

    # oracle: best fixed strategy (per-pressure mean)
    fixed = {s: mean(mk[(s, p)]) for s in STRATS if (s, p) in mk}
    oracle_fixed = min(fixed.values())
    # oracle: round-aware (can switch per round)
    oracle_round = mean([min(mk[(s, p)][r] for s in STRATS if (s, p) in mk)
                         for r in range(len(dyn))])
    print(f"{p:>2} | {dm:>8.0f} {str(acts):>10} | {oracle_fixed:>12.0f} "
          f"{dm - oracle_fixed:>+12.0f} | {oracle_round:>12.0f} {dm - oracle_round:>+12.0f}")

    # r1-3 steady-state regret (excludes cold-server first-transfer r0)
    if len(dyn) >= 2:
        dm13 = mean(dyn[1:])
        o_fixed13 = min(mean(mk[(s, p)][1:]) for s in STRATS if (s, p) in mk)
        o_round13 = mean([min(mk[(s, p)][r] for s in STRATS if (s, p) in mk)
                          for r in range(1, len(dyn))])
        print(f"    r1-3: dyn={dm13:.0f}  regret_fixed={dm13-o_fixed13:+.0f}  regret_round={dm13-o_round13:+.0f}")

print()
print("动态每轮动作(pressure=背景槽位数,阈值默认 5):")
for p in PRESSURES:
    drows, _, dname = load_dynamic(p)
    if drows:
        acts = [r.get("action") for r in drows]
        obs = [r.get("observed_pressure") for r in drows]
        print(f"  p{p}: acts={acts} obs={obs}")
