#!/usr/bin/env python3
"""重复实验聚合:pack/retrieve 各压力点,base + r2 + r3 → mean±std + 最优判定。"""
import json
from pathlib import Path
import statistics

ART = Path("artifacts")
STRATS = ["pack", "retrieve", "recompute"]
PRESSURES = [0, 2, 4, 6, 8]


def mks(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rounds = [r for r in rows if r.get("type") == "round"]
    summaries = [r for r in rows if r.get("type") == "summary"]
    if not summaries or len(rounds) != 4:
        return None, False  # 未完成的 run(缺 summary 或轮数不足)跳过
    return [r["child_makespan_ms"] for r in rounds], summaries[0].get("valid")


def collect(strategy, pressure):
    """All repeats for a cell: base (v2/v3) + r2/r3; recompute only base."""
    files = []
    if strategy == "recompute":
        for n in (f"unified_v2_{strategy}_p{pressure}_20260912.json",
                  f"unified_{strategy}_p{pressure}_20260912.json"):
            if (ART / n).exists():
                files.append(ART / n)
                break
    else:
        # retrieve 的候选:仅 v3(暖 server,采用版)+ r2/r3;
        # v1/v2 除外——两批的 store 均已死,children 实际走本地重算,不是 retrieve 样本
        reps = ["base_v3", "r2", "r3"] if strategy == "retrieve" \
            else ["base_v2", "base_v1", "r2", "r3"]
        for rep in reps:
            names = {
                "base_v3": f"unified_v3_{strategy}_p{pressure}_20260912.json",
                "base_v2": f"unified_v2_{strategy}_p{pressure}_20260912.json",
                "base_v1": f"unified_{strategy}_p{pressure}_20260912.json",
                "r2": f"unified_r2_{strategy}_p{pressure}_20260912.json",
                "r3": f"unified_r3_{strategy}_p{pressure}_20260912.json",
            }[rep]
            p = ART / names
            if p.exists():
                files.append(p)
    out = []
    for p in files:
        mk, valid = mks(p)
        if mk is not None and valid:
            out.append((mk, p.name))
    return out


print(f"{'cell':<18}{'n':>3} {'mean':>7} {'std':>6} {'r0均':>7} {'r1-3均':>7}  per-run means")
print("-" * 92)
cell_mean = {}
for s in STRATS:
    for p in PRESSURES:
        reps = collect(s, p)
        if not reps:
            print(f"{s+' p'+str(p):<18}MISSING")
            continue
        run_means = [statistics.mean(mk) for mk, _ in reps]
        r0s = [mk[0] for mk, _ in reps]
        r13s = [statistics.mean(mk[1:]) for mk, _ in reps]
        m = statistics.mean(run_means)
        sd = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
        cell_mean[(s, p)] = (m, sd, len(reps))
        pm = "/".join(f"{x:.0f}" for x in run_means)
        print(f"{s+' p'+str(p):<18}{len(reps):>3} {m:>7.0f} {sd:>6.0f} "
              f"{statistics.mean(r0s):>7.0f} {statistics.mean(r13s):>7.0f}  {pm}")

print()
print("最优动作判定(按 mean;±std 重叠的格标注 'tie?'):")
for p in PRESSURES:
    avail = {s: cell_mean[(s, p)] for s in STRATS if (s, p) in cell_mean}
    best = min(avail, key=lambda s: avail[s][0])
    ties = [s for s in avail
            if s != best and abs(avail[s][0] - avail[best][0]) <= avail[best][1] + avail[s][1]]
    note = f" (tie? vs {'/'.join(ties)})" if ties else ""
    others = " ".join(f"{s}={avail[s][0]:.0f}±{avail[s][1]:.0f}" for s in avail if s != best)
    print(f"  p{p}: {best} {avail[best][0]:.0f}±{avail[best][1]:.0f}{note}  |  {others}")
