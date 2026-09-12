"""Summarize BranchServe smoke/formal JSONL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COUNTERS = {
    "local_compute": "vllm:prompt_tokens_by_source_total{source=local_compute}",
    "cache_hit": "vllm:prompt_tokens_by_source_total{source=local_cache_hit}",
    "prompt_total": "vllm:prompt_tokens_total",
    "kv_computed": "vllm:request_prefill_kv_computed_tokens_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "prefill_n": "vllm:request_prefill_time_seconds_count",
    "queue_s": "vllm:request_queue_time_seconds_sum",
    "ttft_s": "vllm:time_to_first_token_seconds_sum",
    "ttft_n": "vllm:time_to_first_token_seconds_count",
    "e2e_s": "vllm:e2e_request_latency_seconds_sum",
    "e2e_n": "vllm:e2e_request_latency_seconds_count",
    "errors": "vllm:request_success_total{finished_reason=error}",
}


def _delta(row: dict[str, Any], metric: str) -> float:
    total = 0.0
    for worker in row.get("metrics_delta", {}).values():
        value = (worker.get("values", {}).get(metric) or {}).get("delta")
        total += float(value or 0)
    return total


def _workflow_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten formal nested workflow results without losing event metadata."""
    workflow_results = row.get("workflow_results")
    if not workflow_results:
        return [row]
    flattened: list[dict[str, Any]] = []
    for workflow in workflow_results:
        parent = workflow.get("parent") or {}
        children = workflow.get("children") or []
        flattened.append(
            {
                **row,
                "run_id": f"{row.get('run_id')}:{workflow.get('workflow_id')}",
                "parent_worker": parent.get("worker_id"),
                "parent_latency_ms": parent.get("latency_ms"),
                "children": children,
                "agent_events": workflow.get("agent_events") or row.get("agent_events") or [],
            }
        )
    return flattened


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {}
    ordered = sorted(events, key=lambda event: float(event.get("timestamp_ms", 0.0)))
    timestamps = [float(event.get("timestamp_ms", 0.0)) for event in ordered]
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    child_nodes: set[str] = set()
    parent_duration = 0.0
    tool_wait_ms = 0.0
    join_at: float | None = None
    for event in ordered:
        kind = event.get("event")
        node_id = str(event.get("node_id") or "")
        timestamp = float(event.get("timestamp_ms", 0.0))
        if kind in {"parent_start", "dispatch", "tool_wait_start"}:
            starts.setdefault(node_id, timestamp)
        if kind in {"parent_finish", "finish", "tool_wait_end"}:
            finishes[node_id] = timestamp
        if event.get("parent_node_id") is not None and kind in {"dispatch", "finish"}:
            child_nodes.add(node_id)
        if kind == "tool_wait_start":
            starts[node_id] = timestamp
        elif kind == "tool_wait_end" and node_id in starts:
            tool_wait_ms += max(0.0, timestamp - starts[node_id])
        elif kind == "join":
            join_at = timestamp
    parent_starts = [float(event["timestamp_ms"]) for event in ordered if event.get("event") == "parent_start"]
    parent_finishes = [float(event["timestamp_ms"]) for event in ordered if event.get("event") == "parent_finish"]
    if parent_starts and parent_finishes:
        parent_duration = max(0.0, max(parent_finishes) - min(parent_starts))
    child_durations = [
        max(0.0, finishes[node_id] - starts[node_id])
        for node_id in child_nodes
        if node_id in starts and node_id in finishes
    ]
    child_finishes = [finishes[node_id] for node_id in child_nodes if node_id in finishes]
    last_child = max(child_finishes, default=None)
    join_wait = max(0.0, join_at - last_child) if join_at is not None and last_child is not None else 0.0
    workflow_start = min(timestamps)
    workflow_end = max(timestamps)
    critical_path = parent_duration + tool_wait_ms + max(child_durations, default=0.0) + join_wait
    return {
        "event_count": len(events),
        "jct_ms": round(workflow_end - workflow_start, 3),
        "dag_makespan_ms": round(workflow_end - workflow_start, 3),
        "critical_path_ms": round(critical_path, 3),
        "join_wait_ms": round(join_wait, 3),
        "tool_wait_ms": round(tool_wait_ms, 3),
    }


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    values = {name: _delta(row, metric) for name, metric in COUNTERS.items()}
    children = row.get("children", [])
    parent_ms = float(row.get("parent_latency_ms", 0.0))
    child_ms = max((float(child.get("latency_ms") or 0.0) for child in children), default=0.0)
    event_summary = _event_summary(row.get("agent_events") or [])
    values.update(
        {
            "run_id": row.get("run_id"),
            "policy": row.get("policy"),
            "parent_worker": row.get("parent_worker"),
            "children_workers": "|".join(child.get("worker_id", "?") for child in children),
            "fanout": len(children),
            "parent_latency_ms": round(parent_ms, 3),
            "dag_makespan_ms": event_summary.get("dag_makespan_ms", round(parent_ms + child_ms, 3)),
            "jct_ms": event_summary.get("jct_ms", round(parent_ms + child_ms, 3)),
            "critical_path_ms": event_summary.get("critical_path_ms"),
            "join_wait_ms": event_summary.get("join_wait_ms"),
            "tool_wait_ms": event_summary.get("tool_wait_ms"),
            "event_count": event_summary.get("event_count", 0),
            "predicted_strategy": row.get("predicted_strategy"),
            "actual_strategy": row.get("actual_strategy"),
            "oracle_regret_ms": row.get("oracle_regret_ms"),
            "oracle_regret_ratio": row.get("oracle_regret_ratio"),
            "ttft_mean_ms": round(1000 * values["ttft_s"] / values["ttft_n"], 3) if values["ttft_n"] else None,
            "e2e_mean_ms": round(1000 * values["e2e_s"] / values["e2e_n"], 3) if values["e2e_n"] else None,
            "prefill_mean_ms": round(1000 * values["prefill_s"] / values["prefill_n"], 3)
            if values["prefill_n"]
            else None,
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = [summarize_row(workflow_row) for row in rows if row.get("policy") for workflow_row in _workflow_rows(row)]
    fields = [
        "run_id", "policy", "parent_worker", "children_workers", "fanout",
        "parent_latency_ms", "jct_ms", "dag_makespan_ms", "critical_path_ms",
        "join_wait_ms", "tool_wait_ms", "event_count", "predicted_strategy",
        "actual_strategy", "oracle_regret_ms", "oracle_regret_ratio",
        "local_compute", "cache_hit", "prompt_total", "kv_computed",
        "ttft_mean_ms", "e2e_mean_ms", "errors",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summaries)
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            file_writer.writeheader()
            file_writer.writerows(summaries)


if __name__ == "__main__":
    main()
