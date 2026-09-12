"""Repeat paired Parent Affinity/Spread pressure experiments."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .cost_model import DynamicPackSpreadCostModel, fit_model
from .pressure_crossover import run_placement


STATIC_PLACEMENTS = ("parent_affinity", "fixed_spread")
DYNAMIC_PLACEMENT = "dynamic_pack_spread"
SPREAD_PERMUTATIONS = {
    "gpu0_first": ("gpu-0", "gpu-1"),
    "gpu1_first": ("gpu-1", "gpu-0"),
}
DEFAULT_SPREAD_PERMUTATIONS = tuple(SPREAD_PERMUTATIONS)


def run_sweep(
    tokenizer: Any,
    pressures: list[int],
    repeats: int,
    shared_prefix: int,
    fanout: int,
    background_output_tokens: int,
    child_output_tokens: int,
    worker0: str,
    worker1: str,
    run_id: str,
    cost_model: DynamicPackSpreadCostModel | None = None,
    telemetry_ttl_ms: float = 1000.0,
    spread_permutations: Sequence[str] = DEFAULT_SPREAD_PERMUTATIONS,
) -> list[dict[str, object]]:
    if cost_model is None:
        raise ValueError("Phase 5 pressure sweep requires a fitted cost_model")
    requested_permutations = tuple(spread_permutations)
    if not requested_permutations:
        raise ValueError("at least one spread permutation is required")
    if len(set(requested_permutations)) != len(requested_permutations):
        raise ValueError("spread permutations cannot contain duplicates")
    unknown = sorted(set(requested_permutations) - set(SPREAD_PERMUTATIONS))
    if unknown:
        raise ValueError(f"unknown spread permutations: {unknown}")

    results: list[dict[str, object]] = []
    permutation_indices = {name: index for index, name in enumerate(requested_permutations)}
    for pressure_index, pressure in enumerate(pressures):
        for repeat in range(1, repeats + 1):
            # Counterbalance which permutation block runs first over pressure/repeat.
            permutation_order = list(requested_permutations)
            if (pressure_index + repeat) % 2 == 0:
                permutation_order.reverse()
            for permutation_name in permutation_order:
                spread_worker_order = SPREAD_PERMUTATIONS[permutation_name]
                placements = ["parent_affinity", "fixed_spread", DYNAMIC_PLACEMENT]
                rotation = (repeat - 1 + permutation_indices[permutation_name]) % len(placements)
                placements = placements[rotation:] + placements[:rotation]
                block_results: list[dict[str, object]] = []
                for placement in placements:
                    placement_run_id = (
                        f"{run_id}-p{pressure}-r{repeat}-{permutation_name}-{placement}"
                    )
                    workload_id = f"{run_id}-p{pressure}-r{repeat}-{permutation_name}"
                    try:
                        result = run_placement(
                            placement,
                            tokenizer,
                            shared_prefix,
                            fanout,
                            pressure,
                            background_output_tokens,
                            child_output_tokens,
                            worker0,
                            worker1,
                            placement_run_id,
                            cost_model=cost_model,
                            workload_id=workload_id,
                            auto_refresh_telemetry=placement == DYNAMIC_PLACEMENT,
                            telemetry_ttl_ms=telemetry_ttl_ms,
                            spread_permutation=permutation_name,
                            spread_worker_order=spread_worker_order,
                        )
                        result["repeat"] = repeat
                        result["pressure_group"] = pressure
                        block_results.append(result)
                    except Exception as exc:
                        block_results.append(
                            {
                                "run_id": placement_run_id,
                                "placement": placement,
                                "repeat": repeat,
                                "pressure_group": pressure,
                                "spread_permutation": permutation_name,
                                "spread_worker_order": list(spread_worker_order),
                                "error": repr(exc),
                            }
                        )
                _attach_oracle_regret(block_results)
                for result in block_results:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                results.extend(block_results)
    return results


def _attach_oracle_regret(results: list[dict[str, object]]) -> None:
    """Compare each dynamic arm with static arms from the same repeat."""
    groups: dict[tuple[object, object, object], dict[str, dict[str, object]]] = {}
    for row in results:
        if "error" in row:
            continue
        key = (
            row.get("pressure_group", row.get("pressure")),
            row.get("repeat"),
            row.get("spread_permutation"),
        )
        placement = str(row.get("placement"))
        groups.setdefault(key, {})[placement] = row
    for arms in groups.values():
        dynamic = arms.get(DYNAMIC_PLACEMENT)
        if dynamic is None:
            continue
        static = [arms[name] for name in STATIC_PLACEMENTS if name in arms and "branch_makespan_ms" in arms[name]]
        if not static or "branch_makespan_ms" not in dynamic:
            continue
        oracle_row = min(static, key=lambda row: float(row["branch_makespan_ms"]))
        oracle = float(oracle_row["branch_makespan_ms"])
        actual = float(dynamic["branch_makespan_ms"])
        dynamic["oracle_placement"] = oracle_row["placement"]
        dynamic["oracle_makespan_ms"] = oracle
        dynamic["oracle_regret_ms"] = round(actual - oracle, 3)
        dynamic["oracle_regret_ratio"] = round((actual - oracle) / oracle, 6) if oracle else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker0", default="http://127.0.0.1:8000")
    parser.add_argument("--worker1", default="http://127.0.0.1:8001")
    parser.add_argument("--pressures", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shared-prefix", type=int, default=8192)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--background-output-tokens", type=int, default=2048)
    parser.add_argument("--child-output-tokens", type=int, default=256)
    parser.add_argument("--cost-model-artifact", type=Path, default=None)
    parser.add_argument("--cost-model-train-pressures", nargs=2, type=int, default=[4, 8])
    parser.add_argument("--telemetry-ttl-ms", type=float, default=1000.0)
    parser.add_argument(
        "--spread-permutations",
        nargs="+",
        choices=tuple(SPREAD_PERMUTATIONS),
        default=list(DEFAULT_SPREAD_PERMUTATIONS),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if args.cost_model_artifact is None:
        parser.error("pressure_sweep Phase 5 requires --cost-model-artifact")
    cost_model = fit_model(args.cost_model_artifact, args.cost_model_train_pressures)
    run_id = args.run_id or str(time.time_ns())
    header = {
        "run_id": run_id,
        "pressures": args.pressures,
        "repeats": args.repeats,
        "shared_prefix": args.shared_prefix,
        "fanout": args.fanout,
        "background_output_tokens": args.background_output_tokens,
        "child_output_tokens": args.child_output_tokens,
        "placements": [*STATIC_PLACEMENTS, DYNAMIC_PLACEMENT],
        "cost_model_train_pressures": args.cost_model_train_pressures,
        "telemetry_ttl_ms": args.telemetry_ttl_ms,
        "spread_permutations": args.spread_permutations,
        "spread_worker_orders": {
            name: list(SPREAD_PERMUTATIONS[name]) for name in args.spread_permutations
        },
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    print(lines[0], flush=True)
    results = run_sweep(
        tokenizer,
        args.pressures,
        args.repeats,
        args.shared_prefix,
        args.fanout,
        args.background_output_tokens,
        args.child_output_tokens,
        args.worker0,
        args.worker1,
        run_id,
        cost_model,
        args.telemetry_ttl_ms,
        args.spread_permutations,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(lines[0] + "\n")
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
