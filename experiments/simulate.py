"""Run a deterministic placement simulation without GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import BranchRequest, WorkerSnapshot
from .policies import build_policy, estimate_placement


def run(workload: Path, policy_name: str) -> dict[str, float | int | str]:
    requests = [BranchRequest(**json.loads(line)) for line in workload.read_text(encoding="utf-8").splitlines() if line]
    workers = [
        WorkerSnapshot("gpu-0", prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
        WorkerSnapshot("gpu-1", prefill_tokens_per_second=20_000, decode_tokens_per_second=2_000),
    ]
    policy = build_policy(policy_name)
    placements = []
    for request in requests:
        worker_id = policy.choose(request, workers)
        worker = next(worker for worker in workers if worker.worker_id == worker_id)
        placement = estimate_placement(request, worker, policy.name)
        placements.append(placement)
        worker.queue_tokens += request.prompt_tokens
        worker.running_requests += 1
        if request.parent_session_id:
            worker.cached_prefixes[request.parent_session_id] = max(
                request.shared_prefix_tokens,
                worker.cached_prefixes.get(request.parent_session_id, 0),
            )
    return {
        "policy": policy.name,
        "requests": len(placements),
        "estimated_makespan_ms": max((placement.estimated_completion_ms for placement in placements), default=0.0),
        "total_prefill_ms": sum(placement.estimated_prefill_ms for placement in placements),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--policy", default="parent_affinity")
    args = parser.parse_args()
    print(json.dumps(run(args.workload, args.policy), indent=2))


if __name__ == "__main__":
    main()
