"""Measure Parent Affinity vs fixed 2+2 Spread under real GPU0 pressure."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .metrics import (
    EXPERIMENT_METRIC_KEYS,
    capture_worker_metrics,
    diff_snapshots,
    select_metrics,
    snapshot_to_dict,
)
from .cost_model import DynamicPackSpreadCostModel, fit_model
from .models import BranchRequest
from .policies import DynamicPackSpreadPolicy
from .router import BranchServeRouter, WorkerEndpoint


def make_text(tokenizer: Any, target_tokens: int, label: str) -> tuple[str, int]:
    parts: list[str] = []
    index = 0
    while True:
        unit = (
            f"Repository fact block {index:05d} for {label}: preserve this unique "
            f"context block and its schema key {index * 7919} during replay.\n"
        )
        candidate = "".join(parts) + unit
        if len(tokenizer.encode(candidate, add_special_tokens=False)) >= target_tokens:
            return candidate, len(tokenizer.encode(candidate, add_special_tokens=False))
        parts.append(unit)
        index += 1


def new_router(
    worker0: str,
    worker1: str,
    *,
    policy: str = "round_robin",
    cost_model: DynamicPackSpreadCostModel | None = None,
    auto_refresh_telemetry: bool = False,
    telemetry_ttl_ms: float = 1000.0,
    spread_worker_order: Sequence[str] | None = None,
) -> BranchServeRouter:
    policy_impl: str | DynamicPackSpreadPolicy = policy
    if policy == "dynamic_pack_spread" and spread_worker_order is not None:
        policy_impl = DynamicPackSpreadPolicy(
            cost_model=cost_model,
            telemetry_ttl_ms=telemetry_ttl_ms,
            spread_worker_order=spread_worker_order,
        )
    return BranchServeRouter(
        [
            WorkerEndpoint("gpu-0", worker0, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
            WorkerEndpoint("gpu-1", worker1, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
        ],
        policy=policy_impl,
        model="qwen3.5-4b",
        cost_model=cost_model,
        auto_refresh_telemetry=auto_refresh_telemetry,
        telemetry_ttl_ms=telemetry_ttl_ms,
    )


def run_placement(
    placement: str,
    tokenizer: Any,
    shared_prefix: int,
    fanout: int,
    pressure: int,
    background_output_tokens: int,
    child_output_tokens: int,
    worker0: str,
    worker1: str,
    run_id: str,
    *,
    cost_model: DynamicPackSpreadCostModel | None = None,
    workload_id: str | None = None,
    auto_refresh_telemetry: bool = False,
    telemetry_ttl_ms: float = 1000.0,
    spread_permutation: str | None = None,
    spread_worker_order: Sequence[str] | None = None,
) -> dict[str, object]:
    if placement not in {"parent_affinity", "fixed_spread", "dynamic_pack_spread"}:
        raise ValueError(f"unsupported placement {placement!r}")
    controlled_spread_order = tuple(spread_worker_order) if spread_worker_order is not None else None
    if controlled_spread_order is not None:
        if len(controlled_spread_order) != 2 or set(controlled_spread_order) != {"gpu-0", "gpu-1"}:
            raise ValueError(
                "spread_worker_order must be a permutation of ('gpu-0', 'gpu-1')"
            )
    policy = "dynamic_pack_spread" if placement == "dynamic_pack_spread" else "round_robin"
    router = new_router(
        worker0,
        worker1,
        policy=policy,
        cost_model=cost_model,
        auto_refresh_telemetry=auto_refresh_telemetry,
        telemetry_ttl_ms=telemetry_ttl_ms,
        spread_worker_order=controlled_spread_order,
    )
    endpoints = {"gpu-0": worker0, "gpu-1": worker1}
    metrics_start = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(endpoints).items()
    }
    shared_text, shared_tokens = make_text(tokenizer, shared_prefix, workload_id or run_id)
    workflow_id = f"{run_id}-{placement}-workflow"
    parent_request = BranchRequest(
        workflow_id,
        f"{workflow_id}-parent",
        None,
        "parent",
        shared_tokens,
        0,
        expected_output_tokens=8,
    )
    job_start = time.perf_counter()
    parent_start = time.perf_counter()
    parent_result = router.chat(
        parent_request,
        [{"role": "user", "content": shared_text}],
        worker_id="gpu-0",
        max_tokens=8,
        temperature=0,
        enable_thinking=False,
    )
    parent_finish = time.perf_counter()
    router.register_parent(parent_request.session_id, "gpu-0", shared_tokens)
    metrics_after_parent = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(endpoints).items()
    }

    background_executor: concurrent.futures.ThreadPoolExecutor | None = None
    background_stop = threading.Event()
    background_lock = threading.Lock()
    background_stats = {"submitted": 0, "completed": 0, "failed": 0}
    if pressure:
        background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=pressure)

        def run_background(slot: int) -> None:
            # Reuse the warmed Parent prefix so pressure consumes execution
            # slots without introducing a large new KV working set.
            sequence = 0
            while not background_stop.is_set():
                suffix = (
                    f"\nBackground slot {slot} sequence {sequence}: "
                    "generate a long deterministic response."
                )
                request = BranchRequest(
                    workflow_id,
                    f"{workflow_id}-background-{slot}-{sequence}",
                    parent_request.session_id,
                    f"background-{slot}-{sequence}",
                    shared_tokens + len(suffix.split()),
                    shared_tokens,
                    expected_output_tokens=background_output_tokens,
                )
                with background_lock:
                    background_stats["submitted"] += 1
                try:
                    router.chat(
                        request,
                        [{"role": "user", "content": shared_text + suffix}],
                        worker_id="gpu-0",
                        max_tokens=background_output_tokens,
                        min_tokens=background_output_tokens,
                        ignore_eos=True,
                        temperature=0,
                        enable_thinking=False,
                    )
                    with background_lock:
                        background_stats["completed"] += 1
                except Exception:
                    with background_lock:
                        background_stats["failed"] += 1
                sequence += 1

        for index in range(pressure):
            background_executor.submit(run_background, index)

    pressure_ready = False
    pressure_running = 0.0
    pressure_deadline = time.perf_counter() + 30.0
    pressure_waiting = 0.0
    while pressure and time.perf_counter() < pressure_deadline:
        current = capture_worker_metrics(endpoints)["gpu-0"]
        pressure_running = current.value("vllm:num_requests_running")
        pressure_waiting = current.value("vllm:num_requests_waiting")
        if pressure_running + pressure_waiting >= pressure:
            pressure_ready = True
            break
        time.sleep(0.05)
    if not pressure:
        pressure_ready = True

    metrics_before_children = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(endpoints).items()
    }
    child_results: list[dict[str, object]] = []
    child_requests: list[BranchRequest] = []
    child_messages: list[list[dict[str, str]]] = []
    for index in range(fanout):
        suffix = f"\nChild branch {index}: summarize branch-specific findings only."
        child_requests.append(
            BranchRequest(
                workflow_id,
                f"{workflow_id}-child-{index}",
                parent_request.session_id,
                f"branch-{index}",
                shared_tokens + len(suffix.split()),
                shared_tokens,
                expected_output_tokens=child_output_tokens,
            )
        )
        child_messages.append([{"role": "user", "content": shared_text + suffix}])

    def run_child(index: int) -> dict[str, object]:
        request = child_requests[index]
        static_spread_order = controlled_spread_order or ("gpu-0", "gpu-1")
        target = (
            "gpu-0"
            if placement == "parent_affinity"
            else static_spread_order[index % len(static_spread_order)]
        )
        start = time.perf_counter()
        result = router.chat(
            request,
            child_messages[index],
            worker_id=target,
            max_tokens=child_output_tokens,
            temperature=0,
            enable_thinking=False,
        )
        finish = time.perf_counter()
        return {
            "branch_id": request.branch_id,
            "worker_id": target,
            "start_ms": round((start - job_start) * 1000, 3),
            "finish_ms": round((finish - job_start) * 1000, 3),
            "latency_ms": round((finish - start) * 1000, 3),
            "prompt_tokens": result.response.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": result.response.get("usage", {}).get("completion_tokens"),
        }

    dynamic_plan = None
    if placement == "dynamic_pack_spread":
        group_start_perf = time.perf_counter()
        group_start_epoch_ms = time.time() * 1000
        route_results = router.dispatch_group(
            child_requests,
            child_messages,
            max_tokens=child_output_tokens,
            temperature=0,
            enable_thinking=False,
        )
        dynamic_plan = router.routing_decisions()[-1]
        for request, result in zip(child_requests, route_results):
            finish_epoch_ms = next(
                (event.timestamp_ms for event in reversed(result.events) if event.event == "finish"),
                group_start_epoch_ms,
            )
            group_start_rel_ms = (group_start_perf - job_start) * 1000
            finish_rel_ms = group_start_rel_ms + (finish_epoch_ms - group_start_epoch_ms)
            child_results.append(
                {
                    "branch_id": request.branch_id,
                    "worker_id": result.worker_id,
                    "start_ms": round(group_start_rel_ms, 3),
                    "finish_ms": round(finish_rel_ms, 3),
                    "latency_ms": round(finish_rel_ms - group_start_rel_ms, 3),
                    "prompt_tokens": result.response.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": result.response.get("usage", {}).get("completion_tokens"),
                }
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=fanout) as executor:
            child_results = list(executor.map(run_child, range(fanout)))
    children_finish = max((float(child["finish_ms"]) for child in child_results), default=0.0)
    metrics_after_children = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(endpoints).items()
    }
    with background_lock:
        background_done_at_child_finish = background_stats["completed"]
        background_submitted_at_child_finish = background_stats["submitted"]
        background_failed_at_child_finish = background_stats["failed"]
    background_stop.set()
    if background_executor is not None:
        background_executor.shutdown(wait=True)
    metrics_end = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(endpoints).items()
    }
    return {
        "run_id": run_id,
        "placement": placement,
        "spread_permutation": spread_permutation,
        "spread_worker_order": list(controlled_spread_order) if controlled_spread_order else None,
        "actual_worker_ids": [str(child.get("worker_id")) for child in child_results],
        "shared_prefix_target": shared_prefix,
        "shared_prefix_actual": shared_tokens,
        "fanout": fanout,
        "pressure": pressure,
        "background_output_tokens": background_output_tokens,
        "child_output_tokens": child_output_tokens,
        "pressure_ready": pressure_ready,
        "pressure_running_at_barrier": pressure_running,
        "pressure_waiting_at_barrier": pressure_waiting,
        "pressure_active_at_barrier": pressure_running + pressure_waiting,
        "background_done_at_child_finish": background_done_at_child_finish,
        "background_submitted_at_child_finish": background_submitted_at_child_finish,
        "background_failed_at_child_finish": background_failed_at_child_finish,
        "parent_worker": parent_result.worker_id,
        "children": child_results,
        "predicted_strategy": dynamic_plan.strategy if dynamic_plan else placement,
        "actual_strategy": (
            "parent_affinity"
            if all(child.get("worker_id") == parent_result.worker_id for child in child_results)
            else "fixed_spread"
        ),
        "telemetry_age_ms": dynamic_plan.telemetry_age_ms if dynamic_plan else None,
        "telemetry_source": dynamic_plan.telemetry_source if dynamic_plan else "none",
        "predicted_costs_ms": dict(dynamic_plan.predicted_costs_ms) if dynamic_plan else {},
        "routing_reason": dynamic_plan.reason if dynamic_plan else "forced_static_arm",
        "parent_latency_ms": round((parent_finish - parent_start) * 1000, 3),
        "branch_makespan_ms": round(children_finish - (parent_finish - job_start) * 1000, 3),
        "dag_makespan_ms": round(children_finish, 3),
        "metrics_start": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_start.items()},
        "metrics_after_parent": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_after_parent.items()},
        "metrics_before_children": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_before_children.items()},
        "metrics_after_children": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_after_children.items()},
        "metrics_end": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_end.items()},
        "child_metrics_delta": diff_snapshots(metrics_before_children, metrics_after_children, keys=EXPERIMENT_METRIC_KEYS),
        "total_metrics_delta": diff_snapshots(metrics_start, metrics_end, keys=EXPERIMENT_METRIC_KEYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker0", default="http://127.0.0.1:8000")
    parser.add_argument("--worker1", default="http://127.0.0.1:8001")
    parser.add_argument("--shared-prefix", type=int, default=8192)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--pressure", type=int, default=0)
    parser.add_argument("--background-output-tokens", type=int, default=2048)
    parser.add_argument("--child-output-tokens", type=int, default=256)
    parser.add_argument("--cost-model-artifact", type=Path, default=None)
    parser.add_argument("--cost-model-train-pressures", nargs=2, type=int, default=[4, 8])
    parser.add_argument("--telemetry-ttl-ms", type=float, default=1000.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--placements", nargs="+", default=["parent_affinity", "fixed_spread"])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if "dynamic_pack_spread" in args.placements and args.cost_model_artifact is None:
        parser.error("dynamic_pack_spread requires --cost-model-artifact")
    cost_model = (
        fit_model(args.cost_model_artifact, args.cost_model_train_pressures)
        if args.cost_model_artifact
        else None
    )
    run_id = args.run_id or str(time.time_ns())
    print(json.dumps({"run_id": run_id, "shared_prefix": args.shared_prefix, "fanout": args.fanout, "pressure": args.pressure}))
    for placement in args.placements:
        try:
            result = run_placement(
                placement,
                tokenizer,
                args.shared_prefix,
                args.fanout,
                args.pressure,
                args.background_output_tokens,
                args.child_output_tokens,
                args.worker0,
                args.worker1,
                run_id,
                cost_model=cost_model,
                auto_refresh_telemetry=placement == "dynamic_pack_spread",
                telemetry_ttl_ms=args.telemetry_ttl_ms,
            )
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"run_id": run_id, "placement": placement, "error": repr(exc)}))


if __name__ == "__main__":
    main()
