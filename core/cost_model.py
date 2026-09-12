"""Minimal offline Pack/Spread cost model for the pressure crossover experiment.

The first model is intentionally limited to one frozen workload shape.  It
learns a linear queue-pressure term for each placement and keeps the measured
prefix/prefill/decode work in the intercept.  This makes the prediction
auditable before adding a general DAG predictor.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PLACEMENTS = ("parent_affinity", "fixed_spread")


@dataclass(frozen=True)
class WorkloadFingerprint:
    """Workload dimensions that must stay fixed for this first model."""

    shared_prefix: int
    fanout: int
    child_output_tokens: int
    background_output_tokens: int


@dataclass(frozen=True)
class PressureState:
    """Queue state observed at the child dispatch barrier."""

    active_requests: float
    waiting_requests: float = 0.0

    @property
    def pressure(self) -> float:
        return self.active_requests + self.waiting_requests


@dataclass(frozen=True)
class LinearCost:
    intercept_ms: float
    pressure_slope_ms: float

    def estimate(self, state: PressureState | float) -> float:
        pressure = state if isinstance(state, (int, float)) else state.pressure
        return self.intercept_ms + self.pressure_slope_ms * float(pressure)


@dataclass(frozen=True)
class DynamicPackSpreadCostModel:
    """Pressure-only model calibrated for one workload fingerprint."""

    workload: WorkloadFingerprint
    pack: LinearCost
    spread: LinearCost

    def estimate(self, placement: str, state: PressureState | float) -> float:
        if placement == "parent_affinity":
            return self.pack.estimate(state)
        if placement == "fixed_spread":
            return self.spread.estimate(state)
        raise ValueError(f"unknown placement {placement!r}")

    def predict(self, state: PressureState | float) -> str:
        costs = {placement: self.estimate(placement, state) for placement in PLACEMENTS}
        return min(costs, key=costs.get)


def _valid(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    return bool(
        row.get("pressure_ready")
        and row.get("pressure_active_at_barrier", 0) >= row.get("pressure", 0)
        and row.get("background_done_at_child_finish", 1) == 0
    )


def _read_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = next((row for row in rows if "placement" not in row), {})
    runs = [row for row in rows if "placement" in row and _valid(row)]
    return header, runs


def fingerprint_from_header(header: dict[str, Any]) -> WorkloadFingerprint:
    keys = {
        "shared_prefix": "shared_prefix",
        "fanout": "fanout",
        "child_output_tokens": "child_output_tokens",
        "background_output_tokens": "background_output_tokens",
    }
    missing = [key for key in keys.values() if key not in header]
    if missing:
        raise ValueError(f"artifact header is missing workload fields: {missing}")
    return WorkloadFingerprint(**{target: int(header[source]) for source, target in keys.items()})


def _same_fingerprint(expected: WorkloadFingerprint, actual: WorkloadFingerprint) -> None:
    if expected != actual:
        raise ValueError(f"workload mismatch: expected {expected}, got {actual}")


def _mean_by_pressure(runs: Iterable[dict[str, Any]], placement: str) -> dict[float, float]:
    grouped: dict[float, list[float]] = {}
    for row in runs:
        if row.get("placement") != placement:
            continue
        pressure = float(row.get("pressure_active_at_barrier", row.get("pressure", 0))) + float(
            row.get("pressure_waiting_at_barrier", 0)
        )
        grouped.setdefault(pressure, []).append(float(row["branch_makespan_ms"]))
    return {pressure: statistics.mean(values) for pressure, values in grouped.items()}


def _fit_line(points: dict[float, float]) -> LinearCost:
    if len(points) < 2:
        raise ValueError("at least two pressure points are required to fit a cost line")
    x_bar = statistics.mean(points)
    y_bar = statistics.mean(points.values())
    denominator = sum((x - x_bar) ** 2 for x in points)
    if denominator == 0:
        raise ValueError("training pressure points must be distinct")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in points.items()) / denominator
    return LinearCost(intercept_ms=y_bar - slope * x_bar, pressure_slope_ms=slope)


def fit_model(path: Path, train_pressures: Iterable[int]) -> DynamicPackSpreadCostModel:
    header, runs = _read_artifact(path)
    fingerprint = fingerprint_from_header(header)
    wanted = {float(value) for value in train_pressures}
    fitted: dict[str, LinearCost] = {}
    for placement in PLACEMENTS:
        means = _mean_by_pressure(runs, placement)
        points = {pressure: means[pressure] for pressure in wanted if pressure in means}
        missing = sorted(wanted - points.keys())
        if missing:
            raise ValueError(f"{placement} has no valid runs at pressure {missing}")
        fitted[placement] = _fit_line(points)
    return DynamicPackSpreadCostModel(fingerprint, fitted["parent_affinity"], fitted["fixed_spread"])


def summarize_actual(path: Path) -> dict[str, dict[str, Any]]:
    header, runs = _read_artifact(path)
    result: dict[str, dict[str, Any]] = {}
    for placement in PLACEMENTS:
        values = [float(row["branch_makespan_ms"]) for row in runs if row.get("placement") == placement]
        if not values:
            continue
        result[placement] = {
            "n_valid": len(values),
            "mean_ms": statistics.mean(values),
            "placement": placement,
        }
    if result:
        result["faster"] = min(result.values(), key=lambda value: value["mean_ms"])
    result["fingerprint"] = asdict(fingerprint_from_header(header))
    return result


def compare(model: DynamicPackSpreadCostModel, path: Path, pressure: int) -> dict[str, Any]:
    header, runs = _read_artifact(path)
    _same_fingerprint(model.workload, fingerprint_from_header(header))
    state = PressureState(active_requests=float(pressure))
    predicted_costs = {placement: model.estimate(placement, state) for placement in PLACEMENTS}
    predicted = min(predicted_costs, key=predicted_costs.get)
    actual_by_placement = {
        placement: [
            float(row["branch_makespan_ms"])
            for row in runs
            if row.get("placement") == placement and int(row.get("pressure_group", row.get("pressure", -1))) == pressure
        ]
        for placement in PLACEMENTS
    }
    actual_means = {
        placement: statistics.mean(values)
        for placement, values in actual_by_placement.items()
        if values
    }
    actual = min(actual_means, key=actual_means.get) if actual_means else None
    return {
        "pressure": pressure,
        "predicted_costs_ms": predicted_costs,
        "predicted_placement": predicted,
        "actual_means_ms": actual_means,
        "actual_faster_placement": actual,
        "match": actual == predicted if actual is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_artifact", type=Path)
    parser.add_argument("--train-pressures", nargs=2, type=int, default=[4, 8])
    parser.add_argument("--heldout-artifact", type=Path)
    parser.add_argument("--heldout-pressure", type=int, default=6)
    args = parser.parse_args()

    model = fit_model(args.training_artifact, args.train_pressures)
    print(json.dumps({"model": asdict(model)}, ensure_ascii=False))
    if args.heldout_artifact:
        print(json.dumps(compare(model, args.heldout_artifact, args.heldout_pressure), ensure_ascii=False))


if __name__ == "__main__":
    main()
