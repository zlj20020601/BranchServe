"""Summarize paired pressure crossover repeats."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _valid(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    return bool(
        row.get("pressure_ready")
        and row.get("pressure_active_at_barrier", 0) >= row.get("pressure", 0)
        and row.get("background_done_at_child_finish", 1) == 0
    )


def _metric_delta(row: dict[str, Any], key: str) -> float:
    total = 0.0
    for worker in row.get("child_metrics_delta", {}).values():
        total += float((worker.get("values", {}).get(key) or {}).get("delta") or 0)
    return total


def summarize(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if "placement" in row:
            pressure = int(row.get("pressure_group", row.get("pressure", 0)))
            groups[pressure][row["placement"]].append(row)

    summaries: list[dict[str, Any]] = []
    for pressure in sorted(groups):
        group = groups[pressure]
        valid_by_placement = {
            placement: [row for row in rows_for_placement if _valid(row)]
            for placement, rows_for_placement in group.items()
        }
        makespans = {
            placement: [float(row["branch_makespan_ms"]) for row in rows_for_placement]
            for placement, rows_for_placement in valid_by_placement.items()
        }
        for placement, values in makespans.items():
            placement_rows = valid_by_placement.get(placement, [])
            regret_values = [float(row["oracle_regret_ms"]) for row in placement_rows if row.get("oracle_regret_ms") is not None]
            regret_ratios = [float(row["oracle_regret_ratio"]) for row in placement_rows if row.get("oracle_regret_ratio") is not None]
            summaries.append(
                {
                    "pressure": pressure,
                    "placement": placement,
                    "n_total": len(group.get(placement, [])),
                    "n_valid": len(values),
                    "mean_branch_makespan_ms": statistics.mean(values) if values else None,
                    "stdev_branch_makespan_ms": statistics.stdev(values) if len(values) >= 2 else None,
                    "p95_branch_makespan_ms": _quantile(values, 0.95) if values else None,
                    "mean_kv_computed_tokens": statistics.mean(
                        [_metric_delta(row, "vllm:request_prefill_kv_computed_tokens_sum") for row in valid_by_placement.get(placement, [])]
                    ) if values else None,
                    "mean_oracle_regret_ms": statistics.mean(regret_values) if regret_values else None,
                    "mean_oracle_regret_ratio": statistics.mean(regret_ratios) if regret_ratios else None,
                    "predicted_strategies": ",".join(
                        f"{strategy}:{count}"
                        for strategy, count in sorted(
                            Counter(
                                row.get("predicted_strategy") for row in placement_rows if row.get("predicted_strategy")
                            ).items()
                        )
                    ),
                    "actual_strategies": ",".join(
                        f"{strategy}:{count}"
                        for strategy, count in sorted(
                            Counter(
                                row.get("actual_strategy") for row in placement_rows if row.get("actual_strategy")
                            ).items()
                        )
                    ),
                }
            )

        packed = {
            (int(row["repeat"]), row.get("spread_permutation")): row
            for row in valid_by_placement.get("parent_affinity", [])
            if "repeat" in row
        }
        spread = {
            (int(row["repeat"]), row.get("spread_permutation")): row
            for row in valid_by_placement.get("fixed_spread", [])
            if "repeat" in row
        }
        paired = []
        for pair_key in sorted(set(packed) & set(spread), key=lambda key: (key[0], str(key[1]))):
            pack_ms = float(packed[pair_key]["branch_makespan_ms"])
            spread_ms = float(spread[pair_key]["branch_makespan_ms"])
            paired.append(spread_ms - pack_ms)
        summaries.append(
            {
                "pressure": pressure,
                "placement": "spread_minus_pack",
                "n_total": min(len(group.get("parent_affinity", [])), len(group.get("fixed_spread", []))),
                "n_valid": len(paired),
                "mean_branch_makespan_ms": statistics.mean(paired) if paired else None,
                "stdev_branch_makespan_ms": statistics.stdev(paired) if len(paired) >= 2 else None,
                "p95_branch_makespan_ms": _quantile(paired, 0.95) if paired else None,
                "spread_wins": sum(delta < 0 for delta in paired),
                "pack_wins": sum(delta > 0 for delta in paired),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print("pressure,placement,n_total,n_valid,mean_branch_makespan_ms,stdev_branch_makespan_ms,p95_branch_makespan_ms,mean_kv_computed_tokens,mean_oracle_regret_ms,mean_oracle_regret_ratio,predicted_strategies,actual_strategies,spread_wins,pack_wins")
    for row in summarize(args.artifact):
        print(",".join(str(row.get(key, "")) for key in [
            "pressure", "placement", "n_total", "n_valid", "mean_branch_makespan_ms",
            "stdev_branch_makespan_ms", "p95_branch_makespan_ms", "mean_kv_computed_tokens",
            "mean_oracle_regret_ms", "mean_oracle_regret_ratio", "predicted_strategies",
            "actual_strategies", "spread_wins", "pack_wins",
        ]))


if __name__ == "__main__":
    main()
