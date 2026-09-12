"""Run a small Parent + concurrent Child experiment against vLLM workers."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .cost_model import DynamicPackSpreadCostModel, fit_model
from .metrics import (
    EXPERIMENT_METRIC_KEYS,
    capture_worker_metrics,
    diff_snapshots,
    select_metrics,
    snapshot_to_dict,
)
from .models import BranchRequest
from .router import BranchServeRouter, WorkerEndpoint


def make_shared_text(tokenizer: object, target_tokens: int) -> tuple[str, int]:
    parts: list[str] = []
    index = 0
    while True:
        # Every block is unique; otherwise prefix caching can deduplicate
        # repeated text inside the same prompt and hide cross-request reuse.
        unit = (
            f"Repository fact block {index:05d}: BranchServe keeps shared agent "
            f"context block {index:05d} with tool schema key {index * 7919} "
            "for deterministic replay and child-branch reuse.\n"
        )
        candidate = "".join(parts) + unit
        if len(tokenizer.encode(candidate, add_special_tokens=False)) >= target_tokens:
            text = candidate
            break
        parts.append(unit)
        index += 1
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def new_router(
    policy: str,
    worker0: str,
    worker1: str,
    cost_model: DynamicPackSpreadCostModel | None = None,
    hysteresis_ms: float = 0.0,
    telemetry_ttl_ms: float = 1000.0,
) -> BranchServeRouter:
    return BranchServeRouter(
        [
            WorkerEndpoint("gpu-0", worker0, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
            WorkerEndpoint("gpu-1", worker1, prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
        ],
        policy=policy,
        model="qwen3.5-4b",
        cost_model=cost_model,
        hysteresis_ms=hysteresis_ms,
        auto_refresh_telemetry=policy == "dynamic_pack_spread",
        telemetry_ttl_ms=telemetry_ttl_ms,
    )


def run_policy(
    policy: str,
    shared_text: str,
    shared_tokens: int,
    worker0: str,
    worker1: str,
    fanout: int,
    run_id: str,
    child_output_tokens: int = 16,
    cost_model: DynamicPackSpreadCostModel | None = None,
    hysteresis_ms: float = 0.0,
    shared_prefix_target: int | None = None,
    telemetry_ttl_ms: float = 1000.0,
) -> dict[str, object]:
    router = new_router(policy, worker0, worker1, cost_model, hysteresis_ms, telemetry_ttl_ms)
    worker_endpoints = {"gpu-0": worker0, "gpu-1": worker1}
    metrics_before = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(worker_endpoints).items()
    }
    parent = BranchRequest("workflow-0", "parent", None, "parent", shared_tokens, 0, expected_output_tokens=8)
    parent_start = time.perf_counter()
    parent_result = router.chat(
        parent,
        [{"role": "user", "content": shared_text}],
        max_tokens=8,
        temperature=0,
        enable_thinking=False,
    )
    parent_ms = (time.perf_counter() - parent_start) * 1000
    router.register_parent("parent", parent_result.worker_id, shared_tokens)

    child_requests: list[BranchRequest] = []
    child_messages: list[list[dict[str, str]]] = []
    child_shared_prefix = shared_prefix_target if shared_prefix_target is not None else shared_tokens
    for index in range(fanout):
        suffix = f"\nChild branch {index}: summarize only branch-specific findings."
        child_requests.append(BranchRequest(
            "workflow-0",
            f"child-{index}",
            "parent",
            f"branch-{index}",
            shared_tokens + len(suffix.split()),
            child_shared_prefix,
            expected_output_tokens=child_output_tokens,
        ))
        child_messages.append([{"role": "user", "content": shared_text + suffix}])

    group_start_ms = time.time() * 1000
    child_results = router.dispatch_group(
        child_requests,
        child_messages,
        max_tokens=child_output_tokens,
        temperature=0,
        enable_thinking=False,
    )
    children = []
    for request, result in zip(child_requests, child_results):
        finish_ms = next(
            (event.timestamp_ms for event in reversed(result.events) if event.event == "finish"),
            group_start_ms,
        )
        children.append(
            {
                "branch_id": request.branch_id,
                "worker_id": result.worker_id,
                "latency_ms": round(finish_ms - group_start_ms, 3),
                "prompt_tokens": result.response.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": result.response.get("usage", {}).get("completion_tokens"),
            }
        )
    metrics_after = {
        worker_id: select_metrics(snapshot)
        for worker_id, snapshot in capture_worker_metrics(worker_endpoints).items()
    }
    return {
        "run_id": run_id,
        "policy": policy,
        "parent_worker": parent_result.worker_id,
        "parent_latency_ms": round(parent_ms, 3),
        "children": children,
        "metrics_before": {
            worker_id: snapshot_to_dict(snapshot)
            for worker_id, snapshot in metrics_before.items()
        },
        "metrics_after": {
            worker_id: snapshot_to_dict(snapshot)
            for worker_id, snapshot in metrics_after.items()
        },
        "metrics_delta": diff_snapshots(metrics_before, metrics_after, keys=EXPERIMENT_METRIC_KEYS),
        "routing_decisions": [asdict(plan) for plan in router.routing_decisions()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker0", default="http://127.0.0.1:8000")
    parser.add_argument("--worker1", default="http://127.0.0.1:8001")
    parser.add_argument("--shared-prefix", type=int, default=2048)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--child-output-tokens", type=int, default=16)
    parser.add_argument("--cost-model-artifact", type=Path, default=None)
    parser.add_argument("--cost-model-train-pressures", nargs=2, type=int, default=[4, 8])
    parser.add_argument("--hysteresis-ms", type=float, default=0.0)
    parser.add_argument("--telemetry-ttl-ms", type=float, default=1000.0)
    parser.add_argument("--run-id", default=None, help="unique namespace for this run (default: wall-clock timestamp)")
    parser.add_argument("--policies", nargs="+", default=["round_robin", "least_load", "parent_affinity", "prefix_aware", "dynamic_pack_spread"])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    shared_text, shared_tokens = make_shared_text(tokenizer, args.shared_prefix)
    run_id = args.run_id or str(time.time_ns())
    cost_model = (
        fit_model(args.cost_model_artifact, args.cost_model_train_pressures)
        if args.cost_model_artifact
        else None
    )
    print(json.dumps({"run_id": run_id, "target_shared_prefix": args.shared_prefix, "actual_shared_prefix": shared_tokens}))
    for policy in args.policies:
        try:
            # Isolate policies so a previous policy cannot warm the next one's prefix.
            policy_text = shared_text + f"\nBenchmark namespace: {run_id}-{policy}.\n"
            policy_tokens = len(tokenizer.encode(policy_text, add_special_tokens=False))
            print(json.dumps(run_policy(
                policy,
                policy_text,
                policy_tokens,
                args.worker0,
                args.worker1,
                args.fanout,
                run_id,
                args.child_output_tokens,
                cost_model,
                args.hysteresis_ms,
                args.shared_prefix,
                args.telemetry_ttl_ms,
            ), ensure_ascii=False))
        except Exception as exc:  # Keep one failed policy from hiding other baselines.
            print(json.dumps({"policy": policy, "error": repr(exc)}))


if __name__ == "__main__":
    main()
