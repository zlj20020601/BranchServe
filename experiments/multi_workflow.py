"""Run a multi-workflow Parent/fan-out benchmark against vLLM workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import (
    EXPERIMENT_METRIC_KEYS,
    capture_worker_metrics,
    diff_snapshots,
    select_metrics,
    snapshot_to_dict,
)
from .cost_model import DynamicPackSpreadCostModel, fit_model
from .models import AgentEvent, BranchRequest
from .router import BranchServeRouter, WorkerEndpoint


def make_shared_text(tokenizer: Any, target_tokens: int, label: str) -> tuple[str, int]:
    parts: list[str] = []
    index = 0
    while True:
        unit = (
            f"Repository fact block {index:05d} for {label}: BranchServe keeps "
            f"workflow context block {index:05d} with tool schema key "
            f"{index * 7919} for deterministic replay and branch reuse.\n"
        )
        candidate = "".join(parts) + unit
        if len(tokenizer.encode(candidate, add_special_tokens=False)) >= target_tokens:
            return candidate, len(tokenizer.encode(candidate, add_special_tokens=False))
        parts.append(unit)
        index += 1


def new_router(
    policy: str,
    worker0: str,
    worker1: str,
    *,
    cost_model: DynamicPackSpreadCostModel | None = None,
    auto_refresh_telemetry: bool = False,
) -> BranchServeRouter:
    return BranchServeRouter(
        [
            WorkerEndpoint("gpu-0", worker0, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
            WorkerEndpoint("gpu-1", worker1, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
        ],
        policy=policy,
        model="qwen3.5-4b",
        cost_model=cost_model,
        auto_refresh_telemetry=auto_refresh_telemetry,
    )


def run_policy(
    policy: str,
    tokenizer: Any,
    model_context: int,
    workflows: int,
    fanout: int,
    worker0: str,
    worker1: str,
    run_id: str,
    force_parent_worker: str | None = None,
    child_output_tokens: int = 16,
    tool_wait_ms: float = 0.0,
    cost_model: DynamicPackSpreadCostModel | None = None,
) -> dict[str, object]:
    router = new_router(
        policy,
        worker0,
        worker1,
        cost_model=cost_model,
        auto_refresh_telemetry=policy == "dynamic_pack_spread",
    )
    worker_endpoints = {"gpu-0": worker0, "gpu-1": worker1}
    metrics_before = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(worker_endpoints).items()
    }
    workload: dict[str, tuple[str, int]] = {}
    for workflow_index in range(workflows):
        workflow_id = f"{run_id}-{policy}-wf-{workflow_index:03d}"
        workload[workflow_id] = make_shared_text(tokenizer, model_context, workflow_id)

    experiment_start = time.perf_counter()
    epoch_origin_ms = time.time() * 1000 - experiment_start * 1000
    parent_results: dict[str, dict[str, Any]] = {}
    workflow_events: dict[str, list[dict[str, object]]] = {workflow_id: [] for workflow_id in workload}

    def normalized_agent_event(event: AgentEvent) -> dict[str, object]:
        rendered = asdict(event)
        rendered["timestamp_ms"] = round(float(event.timestamp_ms) - epoch_origin_ms, 3)
        return rendered

    def run_parent(item: tuple[str, tuple[str, int]]) -> dict[str, Any]:
        workflow_id, (shared_text, shared_tokens) = item
        session_id = f"{workflow_id}-parent"
        request = BranchRequest(
            workflow_id,
            session_id,
            None,
            "parent",
            shared_tokens,
            0,
            expected_output_tokens=8,
        )
        start = time.perf_counter()
        result = router.chat(
            request,
            [{"role": "user", "content": shared_text}],
            worker_id=force_parent_worker,
            max_tokens=8,
            temperature=0,
            enable_thinking=False,
        )
        finish = time.perf_counter()
        router.register_parent(session_id, result.worker_id, shared_tokens)
        workflow_events[workflow_id].extend(normalized_agent_event(event) for event in result.agent_events)
        workflow_events[workflow_id].extend(
            [
                asdict(AgentEvent(workflow_id, session_id, "parent_start", (start - experiment_start) * 1000)),
                asdict(AgentEvent(workflow_id, session_id, "parent_finish", (finish - experiment_start) * 1000, worker_id=result.worker_id)),
            ]
        )
        return {
            "workflow_id": workflow_id,
            "worker_id": result.worker_id,
            "start_ms": round((start - experiment_start) * 1000, 3),
            "finish_ms": round((finish - experiment_start) * 1000, 3),
            "latency_ms": round((finish - start) * 1000, 3),
            "prompt_tokens": result.response.get("usage", {}).get("prompt_tokens"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workflows) as executor:
        for parent in executor.map(run_parent, workload.items()):
            parent_results[parent["workflow_id"]] = parent

    if tool_wait_ms > 0:
        wait_start = time.perf_counter()
        for workflow_id in workload:
            workflow_events[workflow_id].append(
                asdict(AgentEvent(workflow_id, f"{workflow_id}-tool", "tool_wait_start", (wait_start - experiment_start) * 1000))
            )
        time.sleep(tool_wait_ms / 1000.0)
        wait_end = time.perf_counter()
        for workflow_id in workload:
            workflow_events[workflow_id].append(
                asdict(AgentEvent(workflow_id, f"{workflow_id}-tool", "tool_wait_end", (wait_end - experiment_start) * 1000))
            )

    def run_child_group(item: tuple[str, tuple[str, int]]) -> dict[str, Any]:
        workflow_id, (shared_text, shared_tokens) = item
        requests: list[BranchRequest] = []
        messages: list[list[dict[str, str]]] = []
        for branch_index in range(fanout):
            suffix = f"\nChild branch {branch_index}: summarize branch-specific findings only."
            request = BranchRequest(
                workflow_id,
                f"{workflow_id}-child-{branch_index:03d}",
                f"{workflow_id}-parent",
                f"branch-{branch_index:03d}",
                shared_tokens + len(suffix.split()),
                shared_tokens,
                expected_output_tokens=child_output_tokens,
            )
            requests.append(request)
            messages.append([{"role": "user", "content": shared_text + suffix}])
            workflow_events[workflow_id].append(
                asdict(
                    AgentEvent(
                        workflow_id,
                        request.session_id,
                        "child_ready",
                        (time.perf_counter() - experiment_start) * 1000,
                        parent_node_id=request.parent_session_id,
                        branch_id=request.branch_id,
                    )
                )
            )
        results = router.dispatch_group(
            requests,
            messages,
            max_tokens=child_output_tokens,
            temperature=0,
            enable_thinking=False,
        )
        children: list[dict[str, Any]] = []
        for request, result in zip(requests, results):
            workflow_events[workflow_id].extend(normalized_agent_event(event) for event in result.agent_events)
            dispatch_events = [event for event in result.agent_events if event.event == "dispatch"]
            finish_events = [event for event in result.agent_events if event.event == "finish"]
            start_ms = (dispatch_events[0].timestamp_ms - epoch_origin_ms) if dispatch_events else None
            finish_ms = (finish_events[-1].timestamp_ms - epoch_origin_ms) if finish_events else (time.perf_counter() - experiment_start) * 1000
            children.append(
                {
                    "workflow_id": workflow_id,
                    "branch_id": request.branch_id,
                    "worker_id": result.worker_id,
                    "start_ms": round(start_ms, 3) if start_ms is not None else None,
                    "finish_ms": round(finish_ms, 3),
                    "latency_ms": round(finish_ms - start_ms, 3) if start_ms is not None else None,
                    "prompt_tokens": result.response.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": result.response.get("usage", {}).get("completion_tokens"),
                }
            )
        join_ms = (time.perf_counter() - experiment_start) * 1000
        workflow_events[workflow_id].append(
            asdict(AgentEvent(workflow_id, f"{workflow_id}-join", "join_wait", join_ms))
        )
        workflow_events[workflow_id].append(
            asdict(AgentEvent(workflow_id, f"{workflow_id}-join", "join", join_ms))
        )
        return {"workflow_id": workflow_id, "children": children}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workflows)) as executor:
        grouped_children = list(executor.map(run_child_group, workload.items()))
    children = [child for group in grouped_children for child in group["children"]]

    workflow_results: list[dict[str, Any]] = []
    for workflow_id, parent in parent_results.items():
        workflow_children = [child for child in children if child["workflow_id"] == workflow_id]
        finish_ms = max([float(parent["finish_ms"])] + [float(child["finish_ms"]) for child in workflow_children])
        workflow_results.append(
            {
                "workflow_id": workflow_id,
                "parent": parent,
                "children": workflow_children,
                "makespan_ms": round(finish_ms - float(parent["start_ms"]), 3),
                "agent_events": workflow_events[workflow_id],
            }
        )

    metrics_after = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(worker_endpoints).items()
    }
    return {
        "run_id": run_id,
        "policy": policy,
        "model_context": model_context,
        "workflows": workflows,
        "fanout": fanout,
        "wall_time_ms": round((time.perf_counter() - experiment_start) * 1000, 3),
        "workflow_results": workflow_results,
        "agent_events": [event for workflow_id in workload for event in workflow_events[workflow_id]],
        "metrics_before": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_before.items()},
        "metrics_after": {worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in metrics_after.items()},
        "metrics_delta": diff_snapshots(metrics_before, metrics_after, keys=EXPERIMENT_METRIC_KEYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker0", default="http://127.0.0.1:8000")
    parser.add_argument("--worker1", default="http://127.0.0.1:8001")
    parser.add_argument("--shared-prefix", type=int, default=4096)
    parser.add_argument("--workflows", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force-parent-worker", choices=["gpu-0", "gpu-1"], default=None)
    parser.add_argument("--child-output-tokens", type=int, default=16)
    parser.add_argument("--tool-wait-ms", type=float, default=0.0)
    parser.add_argument("--cost-model-artifact", type=Path, default=None)
    parser.add_argument("--cost-model-train-pressures", nargs=2, type=int, default=[4, 8])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["round_robin", "least_load", "parent_affinity", "prefix_aware", "dynamic_pack_spread"],
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if "dynamic_pack_spread" in args.policies and args.cost_model_artifact is None:
        parser.error("dynamic_pack_spread requires --cost-model-artifact")
    cost_model = (
        fit_model(args.cost_model_artifact, args.cost_model_train_pressures)
        if args.cost_model_artifact
        else None
    )
    run_id = args.run_id or str(time.time_ns())
    print(
        json.dumps(
            {
                "run_id": run_id,
                "shared_prefix_target": args.shared_prefix,
                "workflows": args.workflows,
                "fanout": args.fanout,
            }
        )
    )
    for policy in args.policies:
        try:
            result = run_policy(
                policy,
                tokenizer,
                args.shared_prefix,
                args.workflows,
                args.fanout,
                args.worker0,
                args.worker1,
                run_id,
                args.force_parent_worker,
                args.child_output_tokens,
                args.tool_wait_ms,
                cost_model,
            )
            print(json.dumps(result, ensure_ascii=False))
        except Exception as exc:  # Keep one failed policy from hiding other baselines.
            print(json.dumps({"run_id": run_id, "policy": policy, "error": repr(exc)}))


if __name__ == "__main__":
    main()
