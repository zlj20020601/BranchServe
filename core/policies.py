"""Worker placement policies used by the controller and simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import Lock
import time

from .cost_model import DynamicPackSpreadCostModel, PressureState
from .models import BranchRequest, Placement, RoutingPlan, WorkerSnapshot


class PlacementPolicy(ABC):
    name = "abstract"

    @abstractmethod
    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        raise NotImplementedError

    def plan_group(
        self,
        requests: Sequence[BranchRequest],
        workers: Sequence[WorkerSnapshot],
    ) -> RoutingPlan:
        """Plan a fan-out group while preserving legacy per-request policies."""
        if not requests:
            raise ValueError("request group is empty")
        assignments = tuple(self.choose(request, workers) for request in requests)
        return RoutingPlan(
            strategy="fallback",
            worker_ids=assignments,
            reason=f"policy:{self.name}",
        )


class RoundRobinPolicy(PlacementPolicy):
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0

    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        if not workers:
            raise ValueError("worker pool is empty")
        worker = workers[self._next % len(workers)]
        self._next += 1
        return worker.worker_id


class LeastLoadPolicy(PlacementPolicy):
    name = "least_load"

    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        if not workers:
            raise ValueError("worker pool is empty")
        return min(workers, key=lambda worker: (worker.queue_tokens, worker.running_requests)).worker_id


class ParentAffinityPolicy(PlacementPolicy):
    name = "parent_affinity"

    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        if not workers:
            raise ValueError("worker pool is empty")
        preferred = [worker for worker in workers if request.parent_session_id in worker.cached_prefixes]
        if preferred:
            return min(preferred, key=lambda worker: (worker.queue_tokens, worker.running_requests)).worker_id
        return min(workers, key=lambda worker: (worker.queue_tokens, worker.running_requests)).worker_id


class PrefixAwarePolicy(PlacementPolicy):
    name = "prefix_aware"

    def __init__(self, queue_weight: float = 1.0) -> None:
        self.queue_weight = queue_weight

    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        if not workers:
            raise ValueError("worker pool is empty")

        def cost(worker: WorkerSnapshot) -> float:
            cached = min(request.shared_prefix_tokens, worker.cached_tokens(request.parent_session_id))
            missing = request.prompt_tokens - cached
            queue_ms = self.queue_weight * worker.queue_tokens / max(worker.prefill_tokens_per_second, 1e-9) * 1000
            prefill_ms = missing / max(worker.prefill_tokens_per_second, 1e-9) * 1000
            return queue_ms + prefill_ms

        return min(workers, key=cost).worker_id


class DynamicPackSpreadPolicy(PrefixAwarePolicy):
    """Request-level approximation of DAG-aware Pack/Spread placement.

    The policy compares the best locality-preserving worker with the best
    alternative worker. A spread placement is selected only when its queue
    advantage exceeds the extra prefix prefill cost. A later controller layer
    can replace this local decision with a full DAG critical-path model.
    """

    name = "dynamic_pack_spread"

    def __init__(
        self,
        queue_weight: float = 1.0,
        spread_margin_ms: float = 0.0,
        cost_model: DynamicPackSpreadCostModel | None = None,
        hysteresis_ms: float = 0.0,
        model_prefix_tolerance_tokens: int = 128,
        telemetry_ttl_ms: float = 1000.0,
        spread_worker_order: Sequence[str] | None = None,
    ) -> None:
        super().__init__(queue_weight)
        self.spread_margin_ms = spread_margin_ms
        self.cost_model = cost_model
        self.hysteresis_ms = max(0.0, float(hysteresis_ms))
        self.model_prefix_tolerance_tokens = max(0, int(model_prefix_tolerance_tokens))
        self.telemetry_ttl_ms = max(0.0, float(telemetry_ttl_ms))
        self.spread_worker_order = tuple(spread_worker_order) if spread_worker_order is not None else None
        if self.spread_worker_order is not None:
            if not self.spread_worker_order:
                raise ValueError("spread_worker_order cannot be empty")
            if len(set(self.spread_worker_order)) != len(self.spread_worker_order):
                raise ValueError("spread_worker_order cannot contain duplicate workers")
        self._last_strategy: dict[str | None, str] = {}
        self._strategy_lock = Lock()

    def choose(self, request: BranchRequest, workers: Sequence[WorkerSnapshot]) -> str:
        return self.plan_group([request], workers).worker_ids[0]

    def plan_group(
        self,
        requests: Sequence[BranchRequest],
        workers: Sequence[WorkerSnapshot],
    ) -> RoutingPlan:
        """Choose PACK or SPREAD once, then assign the whole fan-out group."""
        if not requests:
            raise ValueError("request group is empty")
        if not workers:
            raise ValueError("worker pool is empty")
        parent_session_id = requests[0].parent_session_id
        if any(request.parent_session_id != parent_session_id for request in requests):
            raise ValueError("all requests in a group must share parent_session_id")

        affinity = [worker for worker in workers if parent_session_id in worker.cached_prefixes]
        if not affinity:
            assignments = tuple(PrefixAwarePolicy.choose(self, request, workers) for request in requests)
            return RoutingPlan(
                strategy="fallback",
                worker_ids=assignments,
                reason="parent_prefix_not_registered",
            )

        representative = requests[0]
        packed = min(affinity, key=lambda worker: self._cost(worker, representative))
        spread_candidates = [worker for worker in workers if worker.worker_id != packed.worker_id]
        if not spread_candidates:
            return RoutingPlan(
                strategy="parent_affinity",
                worker_ids=tuple(packed.worker_id for _ in requests),
                pressure=float(packed.pressure_requests),
                reason="single_affinity_worker",
                telemetry_source="local",
            )

        strategy, pressure, predicted_costs, reason, telemetry_age_ms, telemetry_source = self._select_strategy(
            requests, packed, spread_candidates
        )
        if strategy == "fixed_spread":
            ordered = self._spread_workers(workers, representative)
            assignments = tuple(ordered[index % len(ordered)].worker_id for index in range(len(requests)))
        else:
            assignments = tuple(packed.worker_id for _ in requests)
        return RoutingPlan(
            strategy=strategy,  # type: ignore[arg-type]
            worker_ids=assignments,
            pressure=pressure,
            predicted_costs_ms=predicted_costs,
            reason=reason,
            telemetry_age_ms=telemetry_age_ms,
            telemetry_source=telemetry_source,
        )

    def _spread_workers(
        self,
        workers: Sequence[WorkerSnapshot],
        representative: BranchRequest,
    ) -> list[WorkerSnapshot]:
        if self.spread_worker_order is None:
            # The production default starts with the least expensive worker.
            return sorted(
                workers,
                key=lambda worker: (self._cost(worker, representative), worker.worker_id),
            )

        by_id = {worker.worker_id: worker for worker in workers}
        configured = self.spread_worker_order
        if len(configured) != len(workers) or set(configured) != set(by_id):
            raise ValueError(
                "spread_worker_order must contain every worker exactly once; "
                f"configured={configured!r}, available={tuple(by_id)!r}"
            )
        return [by_id[worker_id] for worker_id in configured]

    def _select_strategy(
        self,
        requests: Sequence[BranchRequest],
        packed: WorkerSnapshot,
        spread_candidates: Sequence[WorkerSnapshot],
    ) -> tuple[str, float | None, dict[str, float], str, float | None, str]:
        representative = requests[0]
        model = self.cost_model
        if model is not None and self._model_applies(requests, model):
            observed = packed.observed_pressure(ttl_ms=self.telemetry_ttl_ms)
            telemetry_age_ms = self._telemetry_age_ms(packed)
            if observed is None:
                strategy = self._heuristic_strategy(representative, packed, spread_candidates)
                return strategy, None, {}, "telemetry_stale", telemetry_age_ms, "local"
            pressure, telemetry_age_ms = observed
            state = PressureState(active_requests=pressure)
            costs = {
                "parent_affinity": model.estimate("parent_affinity", state),
                "fixed_spread": model.estimate("fixed_spread", state),
            }
            strategy = model.predict(state)
            strategy = self._apply_hysteresis(representative.parent_session_id, strategy, costs)
            return strategy, pressure, costs, "cost_model", telemetry_age_ms, "engine"

        strategy = self._heuristic_strategy(representative, packed, spread_candidates)
        return strategy, float(packed.pressure_requests), {}, "prefix_queue_heuristic", None, "local"

    def _heuristic_strategy(
        self,
        request: BranchRequest,
        packed: WorkerSnapshot,
        spread_candidates: Sequence[WorkerSnapshot],
    ) -> str:
        spread = min(spread_candidates, key=lambda worker: self._cost(worker, request))
        packed_cost = self._cost(packed, request)
        spread_cost = self._cost(spread, request)
        return "fixed_spread" if spread_cost + self.spread_margin_ms < packed_cost else "parent_affinity"

    @staticmethod
    def _telemetry_age_ms(worker: WorkerSnapshot) -> float | None:
        if worker.telemetry_captured_at_ms is None:
            return None
        return max(0.0, time.time() * 1000 - float(worker.telemetry_captured_at_ms))

    def _model_applies(
        self,
        requests: Sequence[BranchRequest],
        model: DynamicPackSpreadCostModel,
    ) -> bool:
        workload = model.workload
        if len(requests) != workload.fanout:
            return False
        return all(
            abs(request.shared_prefix_tokens - workload.shared_prefix)
            <= self.model_prefix_tolerance_tokens
            and request.expected_output_tokens == workload.child_output_tokens
            for request in requests
        )

    def _apply_hysteresis(
        self,
        parent_session_id: str | None,
        strategy: str,
        costs: dict[str, float],
    ) -> str:
        if not self.hysteresis_ms:
            return strategy
        with self._strategy_lock:
            previous = self._last_strategy.get(parent_session_id)
            if previous in costs and previous != strategy:
                if costs[previous] <= costs[strategy] + self.hysteresis_ms:
                    strategy = previous
            self._last_strategy[parent_session_id] = strategy
            return strategy

    def _cost(self, worker: WorkerSnapshot, request: BranchRequest | None = None) -> float:
        if request is None:
            raise TypeError("request is required")
        cached = min(request.shared_prefix_tokens, worker.cached_tokens(request.parent_session_id))
        missing = request.prompt_tokens - cached
        queue_ms = self.queue_weight * worker.queue_tokens / max(worker.prefill_tokens_per_second, 1e-9) * 1000
        return queue_ms + missing / max(worker.prefill_tokens_per_second, 1e-9) * 1000


def build_policy(
    name: str,
    *,
    cost_model: DynamicPackSpreadCostModel | None = None,
    hysteresis_ms: float = 0.0,
    telemetry_ttl_ms: float = 1000.0,
) -> PlacementPolicy:
    policies: dict[str, PlacementPolicy] = {
        "round_robin": RoundRobinPolicy(),
        "least_load": LeastLoadPolicy(),
        "parent_affinity": ParentAffinityPolicy(),
        "prefix_aware": PrefixAwarePolicy(),
        "dynamic_pack_spread": DynamicPackSpreadPolicy(
            cost_model=cost_model,
            hysteresis_ms=hysteresis_ms,
            telemetry_ttl_ms=telemetry_ttl_ms,
        ),
    }
    try:
        return policies[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(policies)}") from exc


def estimate_placement(request: BranchRequest, worker: WorkerSnapshot, policy: str) -> Placement:
    cached = min(request.shared_prefix_tokens, worker.cached_tokens(request.parent_session_id))
    missing = request.prompt_tokens - cached
    queue_ms = worker.queue_tokens / max(worker.prefill_tokens_per_second, 1e-9) * 1000
    prefill_ms = missing / max(worker.prefill_tokens_per_second, 1e-9) * 1000
    decode_ms = request.expected_output_tokens / max(worker.decode_tokens_per_second, 1e-9) * 1000
    return Placement(request, worker.worker_id, queue_ms, prefill_ms, decode_ms, policy)
