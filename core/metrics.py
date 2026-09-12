"""Small Prometheus reader for vLLM worker metrics.

The controller keeps a lightweight locality estimate, while this module reads
the engine counters used to validate that estimate after an experiment.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Mapping


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|\+Inf|-Inf)(?:\s+\S+)?$"
)
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')

EXPERIMENT_METRIC_KEYS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_by_source_total{source=local_compute}",
    "vllm:prompt_tokens_by_source_total{source=local_cache_hit}",
    "vllm:prompt_tokens_by_source_total{source=external_kv_transfer}",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total{finished_reason=stop}",
    "vllm:request_success_total{finished_reason=length}",
    "vllm:request_success_total{finished_reason=abort}",
    "vllm:request_success_total{finished_reason=error}",
    "vllm:request_success_total{finished_reason=repetition}",
    "vllm:request_prompt_tokens_count",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_count",
    "vllm:request_generation_tokens_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_inference_time_seconds_count",
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_prefill_kv_computed_tokens_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
)


def _canonical_key(name: str, raw_labels: str | None) -> str:
    """Remove process-specific labels while retaining useful dimensions."""

    if not raw_labels:
        return name
    labels = {
        match.group("key"): match.group("value")
        for match in _LABEL_RE.finditer(raw_labels)
    }
    labels.pop("engine", None)
    labels.pop("model_name", None)
    if not labels:
        return name
    suffix = ",".join(f'{key}={labels[key]}' for key in sorted(labels))
    return f"{name}{{{suffix}}}"


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into stable metric keys.

    Histogram buckets are intentionally retained, since they can be used for
    later quantile estimation. The usual ``*_count`` and ``*_sum`` samples
    become ordinary keys after label canonicalization.
    """

    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE_RE.match(line.strip())
        if not match:
            continue
        raw_value = match.group("value")
        if raw_value == "NaN":
            continue
        if raw_value == "+Inf":
            value = float("inf")
        elif raw_value == "-Inf":
            value = float("-inf")
        else:
            value = float(raw_value)
        values[_canonical_key(match.group("name"), match.group("labels"))] = value
    return values


@dataclass(frozen=True)
class MetricSnapshot:
    worker_id: str
    endpoint: str
    captured_at_ms: float
    values: dict[str, float]

    def value(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


def capture_worker_metrics(
    workers: Mapping[str, str], *, timeout_s: float = 5.0
) -> dict[str, MetricSnapshot]:
    """Fetch ``/metrics`` from each worker endpoint."""

    snapshots: dict[str, MetricSnapshot] = {}
    for worker_id, endpoint in workers.items():
        metrics_url = endpoint.rstrip("/") + "/metrics"
        request = urllib.request.Request(metrics_url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8")
        snapshots[worker_id] = MetricSnapshot(
            worker_id=worker_id,
            endpoint=endpoint,
            captured_at_ms=time.time() * 1000,
            values=parse_prometheus(text),
        )
    return snapshots


def snapshot_to_dict(snapshot: MetricSnapshot) -> dict[str, object]:
    return {
        "worker_id": snapshot.worker_id,
        "endpoint": snapshot.endpoint,
        "captured_at_ms": round(snapshot.captured_at_ms, 3),
        "values": snapshot.values,
    }


def select_metrics(snapshot: MetricSnapshot, keys: tuple[str, ...] = EXPERIMENT_METRIC_KEYS) -> MetricSnapshot:
    """Keep only stable counters/gauges needed for experiment reports."""

    return MetricSnapshot(
        worker_id=snapshot.worker_id,
        endpoint=snapshot.endpoint,
        captured_at_ms=snapshot.captured_at_ms,
        values={key: snapshot.values[key] for key in keys if key in snapshot.values},
    )


def diff_snapshots(
    before: Mapping[str, MetricSnapshot],
    after: Mapping[str, MetricSnapshot],
    *,
    keys: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    """Return before/after/delta values for all keys seen in either snapshot."""

    result: dict[str, dict[str, object]] = {}
    for worker_id in sorted(set(before) | set(after)):
        before_snapshot = before.get(worker_id)
        after_snapshot = after.get(worker_id)
        before_values = before_snapshot.values if before_snapshot else {}
        after_values = after_snapshot.values if after_snapshot else {}
        selected_keys = set(keys) if keys is not None else set(before_values) | set(after_values)
        worker_delta: dict[str, object] = {
            "before_captured_at_ms": before_snapshot.captured_at_ms if before_snapshot else None,
            "after_captured_at_ms": after_snapshot.captured_at_ms if after_snapshot else None,
            "values": {},
        }
        values = worker_delta["values"]
        assert isinstance(values, dict)
        for key in sorted(selected_keys):
            old = before_values.get(key)
            new = after_values.get(key)
            delta = None if old is None or new is None else new - old
            values[key] = {"before": old, "after": new, "delta": delta}
        result[worker_id] = worker_delta
    return result


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Capture vLLM Prometheus metrics")
    parser.add_argument(
        "--worker",
        action="append",
        required=True,
        metavar="ID=URL",
        help="worker endpoint, for example gpu-0=http://127.0.0.1:8000",
    )
    args = parser.parse_args()
    workers: dict[str, str] = {}
    for item in args.worker:
        worker_id, separator, endpoint = item.partition("=")
        if not separator or not worker_id or not endpoint:
            parser.error(f"invalid --worker {item!r}; expected ID=URL")
        workers[worker_id] = endpoint
    snapshots = capture_worker_metrics(workers)
    print(json.dumps({worker_id: snapshot_to_dict(snapshot) for worker_id, snapshot in snapshots.items()}))


if __name__ == "__main__":
    main()
