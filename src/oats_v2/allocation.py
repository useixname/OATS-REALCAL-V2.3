from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Iterable

from .types import AllocationOutcome, AllocationSnapshot, Candidate


def _density(candidate: Candidate, snapshot: AllocationSnapshot) -> Decimal:
    # Paper Eq. (73)-(76): the dual price multiplies the pair's RESERVED
    # payment (bid + expected score bonus), not the uniform public cap. With a
    # per-pair reserve the AV gate has heterogeneous thresholds — raising
    # lambda progressively cuts the lowest-density pairs (lambda acts as a
    # density cutoff) instead of switching the whole market on/off, which is
    # what made every pacing controller bang-bang in V2..V2.2.
    numerator = max(
        Decimal("0"),
        candidate.predicted_value - snapshot.dual_lambda * candidate.estimated_reserve,
    )
    return numerator / (candidate.estimated_reserve + snapshot.epsilon_rank)


def candidate_order(
    candidates: Iterable[Candidate], snapshot: AllocationSnapshot
) -> tuple[Candidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-_density(item, snapshot), item.worker_id, item.task_id),
        )
    )


def validate_single_task_types(candidates: Iterable[Candidate]) -> set[str]:
    tasks_by_worker: dict[str, set[str]] = {}
    for candidate in candidates:
        tasks_by_worker.setdefault(candidate.worker_id, set()).add(candidate.task_id)
    return {worker for worker, tasks in tasks_by_worker.items() if len(tasks) > 1}


def allocate(
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    *,
    av_gate: bool = False,
) -> AllocationOutcome:
    """Reference density allocator.

    ``av_gate`` enforces the paper's Eq. (78) adjusted-value gate: candidates
    whose dual-adjusted value ``vhat - lambda*Abar`` is non-positive are
    rejected instead of being ranked at density zero. Together with the
    per-candidate ``reserve_price`` (platform max willingness to pay), this
    bounds the critical payment by the estimated value instead of the public
    base cap. Both gates are monotone in the candidate's own bid.
    """
    candidates = tuple(candidates)
    multi_task_workers = validate_single_task_types(candidates)
    eligible: list[Candidate] = []
    rejected: dict[str, str] = {}
    for candidate in candidates:
        if candidate.worker_id in multi_task_workers:
            rejected[candidate.key] = "MULTI_TASK_TYPE_UNSUPPORTED"
        elif not candidate.active or candidate.task_id not in snapshot.active_tasks:
            rejected[candidate.key] = "TASK_NOT_ACTIVATED"
        elif not candidate.deadline_ok:
            rejected[candidate.key] = "DEADLINE_OR_DOMAIN"
        elif candidate.reserve_price is not None and candidate.bid > candidate.reserve_price:
            rejected[candidate.key] = "RESERVE_PRICE_EXCEEDED"
        elif av_gate and (
            candidate.predicted_value - snapshot.dual_lambda * candidate.estimated_reserve
        ) <= 0:
            rejected[candidate.key] = "NONPOSITIVE_ADJUSTED_VALUE"
        else:
            eligible.append(candidate)

    ordered = candidate_order(eligible, snapshot)
    task_counts: Counter[str] = Counter()
    worker_counts: Counter[str] = Counter()
    used_cap = Decimal("0")
    winners: list[str] = []
    total_cap = min(snapshot.actual_base_capacity, snapshot.shadow_base_capacity)
    for candidate in ordered:
        worker_cap = snapshot.worker_capacities.get(candidate.worker_id, 1)
        task_cap = snapshot.task_capacities.get(candidate.task_id, 0)
        feasible = (
            worker_counts[candidate.worker_id] < worker_cap
            and task_counts[candidate.task_id] < task_cap
            and used_cap + candidate.base_cap <= total_cap
        )
        if not feasible:
            rejected[candidate.key] = "DOWNWARD_CLOSED_CAPACITY"
            continue
        winners.append(candidate.key)
        worker_counts[candidate.worker_id] += 1
        task_counts[candidate.task_id] += 1
        used_cap += candidate.base_cap
    return AllocationOutcome(
        winners=tuple(winners),
        order=tuple(candidate.key for candidate in ordered),
        rejected=rejected,
    )


def allocation_resources(
    outcome: AllocationOutcome, candidates: Iterable[Candidate]
) -> Decimal:
    by_key = {candidate.key: candidate for candidate in candidates}
    return sum((by_key[key].base_cap for key in outcome.winners), Decimal("0"))
