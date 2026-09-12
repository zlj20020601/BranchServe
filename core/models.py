"""Shared data structures for Agent DAG scheduling and event metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Mapping


BranchState = Literal["ready", "running", "waiting", "done", "cancelled"]


@dataclass(frozen=True)
class BranchRequest:
    workflow_id: str
    session_id: str
    parent_session_id: str | None
    branch_id: str
    prompt_tokens: int
    shared_prefix_tokens: int
    expected_output_tokens: int = 128
    priority: int = 0
    state: BranchState = "ready"

    @property
    def suffix_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.shared_prefix_tokens)


@dataclass
class WorkerSnapshot:
    worker_id: str
    queue_tokens: int = 0
    running_requests: int = 0
    prefill_tokens_per_second: float = 1.0
    decode_tokens_per_second: float = 1.0
    cached_prefixes: dict[str, int] = field(default_factory=dict)
    waiting_requests: int = 0
    observed_running_requests: int | None = None
    observed_waiting_requests: int | None = None
    telemetry_captured_at_ms: float | None = None

    def cached_tokens(self, session_id: str | None) -> int:
        if session_id is None:
            return 0
        return self.cached_prefixes.get(session_id, 0)

    @property
    def pressure_requests(self) -> int:
        """Router-local requests currently reserved on this worker."""
        return max(0, int(self.running_requests)) + max(0, int(self.waiting_requests))

    def observed_pressure(self, *, now_ms: float | None = None, ttl_ms: float = 1000.0) -> tuple[float, float] | None:
        """Return engine pressure and telemetry age when the observation is fresh."""
        if self.observed_running_requests is None or self.telemetry_captured_at_ms is None:
            return None
        captured_at = float(self.telemetry_captured_at_ms)
        age_ms = max(0.0, (time.time() * 1000 if now_ms is None else now_ms) - captured_at)
        if age_ms > max(0.0, float(ttl_ms)):
            return None
        waiting = self.observed_waiting_requests or 0
        pressure = max(0, int(self.observed_running_requests)) + max(0, int(waiting))
        return float(pressure), age_ms


@dataclass(frozen=True)
class Placement:
    request: BranchRequest
    worker_id: str
    estimated_queue_ms: float
    estimated_prefill_ms: float
    estimated_decode_ms: float
    policy: str

    @property
    def estimated_completion_ms(self) -> float:
        return self.estimated_queue_ms + self.estimated_prefill_ms + self.estimated_decode_ms


@dataclass(frozen=True)
class RequestEvent:
    workflow_id: str
    session_id: str
    branch_id: str
    worker_id: str
    event: str
    timestamp_ms: float
    prompt_tokens: int
    cached_prefix_tokens: int = 0


@dataclass(frozen=True)
class RoutingPlan:
    """A single strategy decision for one workflow fan-out group."""

    strategy: Literal["parent_affinity", "fixed_spread", "fallback"]
    worker_ids: tuple[str, ...]
    pressure: float | None = None
    predicted_costs_ms: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    telemetry_age_ms: float | None = None
    telemetry_source: Literal["engine", "local", "none"] = "none"


@dataclass(frozen=True)
class AgentEvent:
    """Normalized event emitted by a workflow or serving request."""

    workflow_id: str
    node_id: str
    event: Literal[
        "parent_start",
        "parent_finish",
        "tool_wait_start",
        "tool_wait_end",
        "child_ready",
        "dispatch",
        "queue_start",
        "response_received",
        "finish",
        "join_wait",
        "join",
        "early_exit",
        "cancel",
    ]
    timestamp_ms: float
    parent_node_id: str | None = None
    branch_id: str | None = None
    worker_id: str | None = None
    request_id: str | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
