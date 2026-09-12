"""A small HTTP router for a pool of vLLM OpenAI-compatible workers."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .cost_model import DynamicPackSpreadCostModel
from .metrics import MetricSnapshot
from .models import AgentEvent, BranchRequest, RequestEvent, RoutingPlan, WorkerSnapshot
from .policies import PlacementPolicy, build_policy, estimate_placement


@dataclass
class WorkerEndpoint:
    worker_id: str
    base_url: str
    prefill_tokens_per_second: float = 1.0
    decode_tokens_per_second: float = 1.0
    queue_tokens: int = 0
    running_requests: int = 0
    cached_prefixes: dict[str, int] = field(default_factory=dict)
    waiting_requests: int = 0
    observed_running_requests: int | None = None
    observed_waiting_requests: int | None = None
    telemetry_captured_at_ms: float | None = None

    def snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            worker_id=self.worker_id,
            queue_tokens=self.queue_tokens,
            running_requests=self.running_requests,
            prefill_tokens_per_second=self.prefill_tokens_per_second,
            decode_tokens_per_second=self.decode_tokens_per_second,
            cached_prefixes=dict(self.cached_prefixes),
            waiting_requests=self.waiting_requests,
            observed_running_requests=self.observed_running_requests,
            observed_waiting_requests=self.observed_waiting_requests,
            telemetry_captured_at_ms=self.telemetry_captured_at_ms,
        )


@dataclass(frozen=True)
class RouteResult:
    request: BranchRequest
    worker_id: str
    response: dict[str, Any]
    events: tuple[RequestEvent, ...]
    routing_plan: RoutingPlan | None = None
    agent_events: tuple[AgentEvent, ...] = ()


class BranchServeRouter:
    """Route requests and maintain a lightweight locality directory.

    vLLM remains responsible for the physical prefix cache. The directory is
    an estimate used to make a placement decision; it is updated only after a
    successful request and should be calibrated with engine metrics later.
    """

    def __init__(
        self,
        workers: list[WorkerEndpoint],
        policy: str | PlacementPolicy = "round_robin",
        model: str = "",
        *,
        cost_model: DynamicPackSpreadCostModel | None = None,
        hysteresis_ms: float = 0.0,
        telemetry_ttl_ms: float = 1000.0,
        auto_refresh_telemetry: bool = False,
        telemetry_timeout_s: float = 2.0,
    ) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        self.workers = {worker.worker_id: worker for worker in workers}
        self.model = model
        self.telemetry_ttl_ms = max(0.0, float(telemetry_ttl_ms))
        self.auto_refresh_telemetry = bool(auto_refresh_telemetry)
        self.telemetry_timeout_s = max(0.05, float(telemetry_timeout_s))
        if isinstance(policy, str):
            self.policy = build_policy(
                policy,
                cost_model=cost_model,
                hysteresis_ms=hysteresis_ms,
                telemetry_ttl_ms=self.telemetry_ttl_ms,
            )
        else:
            self.policy = policy
        self._lock = threading.Lock()
        self._decision_history: list[RoutingPlan] = []
        self._last_telemetry_error: str | None = None

    def register_parent(self, session_id: str, worker_id: str, prefix_tokens: int) -> None:
        with self._lock:
            self._worker(worker_id).cached_prefixes[session_id] = prefix_tokens

    def snapshots(self) -> list[WorkerSnapshot]:
        with self._lock:
            return [worker.snapshot() for worker in self.workers.values()]

    def update_worker_telemetry(
        self,
        worker_id: str,
        *,
        queue_tokens: int | None = None,
        running_requests: int | None = None,
        waiting_requests: int | None = None,
        observed_running_requests: int | None = None,
        observed_waiting_requests: int | None = None,
        captured_at_ms: float | None = None,
        prefill_tokens_per_second: float | None = None,
        decode_tokens_per_second: float | None = None,
    ) -> None:
        """Update the scheduler's latest worker observation.

        A serving metrics adapter can call this before planning a fan-out. The
        values are advisory; reservations made by :meth:`chat` remain tracked
        by the router itself.
        """
        with self._lock:
            worker = self._worker(worker_id)
            if queue_tokens is not None:
                worker.queue_tokens = max(0, int(queue_tokens))
            if running_requests is not None:
                worker.running_requests = max(0, int(running_requests))
            if waiting_requests is not None:
                worker.waiting_requests = max(0, int(waiting_requests))
            if observed_running_requests is not None:
                worker.observed_running_requests = max(0, int(observed_running_requests))
            if observed_waiting_requests is not None:
                worker.observed_waiting_requests = max(0, int(observed_waiting_requests))
            if captured_at_ms is not None:
                worker.telemetry_captured_at_ms = float(captured_at_ms)
            if prefill_tokens_per_second is not None:
                worker.prefill_tokens_per_second = max(1e-9, float(prefill_tokens_per_second))
            if decode_tokens_per_second is not None:
                worker.decode_tokens_per_second = max(1e-9, float(decode_tokens_per_second))

    def refresh_worker_telemetry(self, *, timeout_s: float = 5.0) -> dict[str, MetricSnapshot]:
        """Pull active/waiting request pressure from every worker's metrics endpoint."""
        from .metrics import capture_worker_metrics
        endpoints = {worker.worker_id: worker.base_url for worker in self.workers.values()}
        snapshots = capture_worker_metrics(endpoints, timeout_s=timeout_s)
        for worker_id, snapshot in snapshots.items():
            updates: dict[str, int] = {}
            if "vllm:num_requests_running" in snapshot.values:
                updates["observed_running_requests"] = round(snapshot.value("vllm:num_requests_running"))
            if "vllm:num_requests_waiting" in snapshot.values:
                updates["observed_waiting_requests"] = round(snapshot.value("vllm:num_requests_waiting"))
            if updates:
                self.update_worker_telemetry(
                    worker_id,
                    captured_at_ms=snapshot.captured_at_ms,
                    **updates,
                )
        return snapshots

    def plan_children(self, requests: Sequence[BranchRequest]) -> RoutingPlan:
        """Choose one placement strategy for a complete fan-out group."""
        if not requests:
            raise ValueError("request group is empty")
        if self.auto_refresh_telemetry and self.policy.name == "dynamic_pack_spread":
            try:
                self.refresh_worker_telemetry(timeout_s=self.telemetry_timeout_s)
                self._last_telemetry_error = None
            except Exception as exc:
                # A stale observation is safer than blocking the request path;
                # DynamicPackSpreadPolicy will explicitly fall back when its
                # TTL check rejects the observation.
                self._last_telemetry_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            snapshots = [worker.snapshot() for worker in self.workers.values()]
            plan = self.policy.plan_group(requests, snapshots)
            if self._last_telemetry_error and plan.reason == "telemetry_stale":
                plan = RoutingPlan(
                    strategy=plan.strategy,
                    worker_ids=plan.worker_ids,
                    pressure=plan.pressure,
                    predicted_costs_ms=plan.predicted_costs_ms,
                    reason=f"{plan.reason}:{self._last_telemetry_error}",
                    telemetry_age_ms=plan.telemetry_age_ms,
                    telemetry_source=plan.telemetry_source,
                )
            for worker_id in plan.worker_ids:
                self._worker(worker_id)
            self._decision_history.append(plan)
            return plan

    def routing_decisions(self) -> tuple[RoutingPlan, ...]:
        with self._lock:
            return tuple(self._decision_history)

    def dispatch_group(
        self,
        requests: Sequence[BranchRequest],
        messages: Sequence[list[dict[str, str]]],
        *,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        ignore_eos: bool | None = None,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        timeout_s: float = 300.0,
    ) -> tuple[RouteResult, ...]:
        """Dispatch all children using one atomic PACK or SPREAD plan."""
        if len(requests) != len(messages):
            raise ValueError("requests and messages must have the same length")
        if not requests:
            return ()
        plan = self.plan_children(requests)
        self._reserve_group(requests, plan.worker_ids)

        def send(item: tuple[BranchRequest, list[dict[str, str]], str]) -> RouteResult:
            request, prompt, worker_id = item
            return self.chat(
                request,
                prompt,
                worker_id=worker_id,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
                temperature=temperature,
                enable_thinking=enable_thinking,
                timeout_s=timeout_s,
                routing_plan=plan,
                pre_reserved=True,
            )

        items = list(zip(requests, messages, plan.worker_ids))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = [executor.submit(send, item) for item in items]
            return tuple(future.result() for future in futures)

    def choose(self, request: BranchRequest) -> str:
        with self._lock:
            return self.policy.choose(request, [worker.snapshot() for worker in self.workers.values()])

    def chat(
        self,
        request: BranchRequest,
        messages: list[dict[str, str]],
        *,
        worker_id: str | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        ignore_eos: bool | None = None,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        timeout_s: float = 300.0,
        routing_plan: RoutingPlan | None = None,
        pre_reserved: bool = False,
    ) -> RouteResult:
        worker_id = worker_id or self.choose(request)
        worker = self._worker(worker_id)
        placement = estimate_placement(request, worker.snapshot(), self.policy.name)
        start_ms = time.time() * 1000
        request_id = f"{request.workflow_id}:{request.session_id}:{request.branch_id}"
        if not pre_reserved:
            self._reserve(worker_id, request.prompt_tokens)
        events = [
            RequestEvent(request.workflow_id, request.session_id, request.branch_id, worker_id, "dispatch", start_ms, request.prompt_tokens),
        ]
        agent_events = [
            AgentEvent(
                request.workflow_id,
                request.session_id,
                "dispatch",
                start_ms,
                parent_node_id=request.parent_session_id,
                branch_id=request.branch_id,
                worker_id=worker_id,
                request_id=request_id,
                metadata={"prompt_tokens": request.prompt_tokens},
            ),
            AgentEvent(
                request.workflow_id,
                request.session_id,
                "queue_start",
                start_ms,
                parent_node_id=request.parent_session_id,
                branch_id=request.branch_id,
                worker_id=worker_id,
                request_id=request_id,
            ),
        ]
        try:
            response = self._post_chat(
                worker,
                messages,
                max_tokens or request.expected_output_tokens,
                min_tokens,
                ignore_eos,
                temperature,
                enable_thinking,
                timeout_s,
            )
        except Exception:
            self._release(worker_id, request.prompt_tokens)
            raise
        finish_ms = time.time() * 1000
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or request.prompt_tokens)
        cached = min(request.shared_prefix_tokens, worker.cached_prefixes.get(request.parent_session_id or "", 0))
        events.extend(
            [
                RequestEvent(request.workflow_id, request.session_id, request.branch_id, worker_id, "first_token", finish_ms, prompt_tokens, cached),
                RequestEvent(request.workflow_id, request.session_id, request.branch_id, worker_id, "finish", finish_ms, prompt_tokens, cached),
            ]
        )
        agent_events.extend(
            [
                AgentEvent(
                    request.workflow_id,
                    request.session_id,
                    "response_received",
                    finish_ms,
                    parent_node_id=request.parent_session_id,
                    branch_id=request.branch_id,
                    worker_id=worker_id,
                    request_id=request_id,
                    metadata={"prompt_tokens": prompt_tokens},
                ),
                AgentEvent(
                    request.workflow_id,
                    request.session_id,
                    "finish",
                    finish_ms,
                    parent_node_id=request.parent_session_id,
                    branch_id=request.branch_id,
                    worker_id=worker_id,
                    request_id=request_id,
                    metadata={"prompt_tokens": prompt_tokens},
                ),
            ]
        )
        with self._lock:
            worker.queue_tokens = max(0, worker.queue_tokens - request.prompt_tokens)
            worker.running_requests = max(0, worker.running_requests - 1)
            if request.parent_session_id:
                worker.cached_prefixes[request.parent_session_id] = max(
                    request.shared_prefix_tokens,
                    worker.cached_prefixes.get(request.parent_session_id, 0),
                )
        return RouteResult(request, worker_id, response, tuple(events), routing_plan, tuple(agent_events))

    def _reserve(self, worker_id: str, prompt_tokens: int) -> None:
        with self._lock:
            worker = self._worker(worker_id)
            worker.queue_tokens += prompt_tokens
            worker.running_requests += 1

    def _reserve_group(self, requests: Sequence[BranchRequest], worker_ids: Sequence[str]) -> None:
        if len(requests) != len(worker_ids):
            raise ValueError("requests and worker_ids must have the same length")
        with self._lock:
            for request, worker_id in zip(requests, worker_ids):
                worker = self._worker(worker_id)
                worker.queue_tokens += request.prompt_tokens
                worker.running_requests += 1

    def _release(self, worker_id: str, prompt_tokens: int) -> None:
        with self._lock:
            worker = self._worker(worker_id)
            worker.queue_tokens = max(0, worker.queue_tokens - prompt_tokens)
            worker.running_requests = max(0, worker.running_requests - 1)

    def _worker(self, worker_id: str) -> WorkerEndpoint:
        try:
            return self.workers[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker {worker_id!r}") from exc

    def _post_chat(
        self,
        worker: WorkerEndpoint,
        messages: list[dict[str, str]],
        max_tokens: int,
        min_tokens: int | None,
        ignore_eos: bool | None,
        temperature: float,
        enable_thinking: bool,
        timeout_s: float,
    ) -> dict[str, Any]:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if min_tokens is not None:
            payload["min_tokens"] = min_tokens
        if ignore_eos is not None:
            payload["ignore_eos"] = ignore_eos
        if self.model:
            payload["model"] = self.model
        request = urllib.request.Request(
            worker.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker {worker.worker_id} returned HTTP {exc.code}: {detail[:500]}") from exc
