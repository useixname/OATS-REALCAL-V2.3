from __future__ import annotations

import random
from decimal import Decimal

from src.oats_v2.allocation import allocate
from src.oats_v2.critical_payment import critical_value
from src.oats_v2.types import AllocationSnapshot, Candidate


SEED = 20260715
GRID = (Decimal("0"), Decimal("1"), Decimal("2"), Decimal("3"))


def _case(rng: random.Random) -> tuple[tuple[Candidate, ...], AllocationSnapshot]:
    task_count = rng.randint(1, 4)
    worker_count = rng.randint(1, 6)
    tasks = tuple(f"t{index}" for index in range(task_count))
    active = frozenset(task for task in tasks if rng.choice((False, True)))
    candidates = tuple(
        Candidate(
            worker_id=f"w{index}",
            task_id=tasks[index % task_count],
            bid=rng.choice(GRID),
            base_cap=Decimal("3"),
            predicted_value=Decimal(rng.randint(0, 5)),
            expected_score_at_reference=Decimal(rng.randint(0, 2)) / Decimal("2"),
            deadline_ok=rng.choice((True, True, False)),
        )
        for index in range(worker_count)
    )
    snapshot = AllocationSnapshot(
        active_tasks=active,
        task_capacities={task: rng.randint(1, 3) for task in tasks},
        actual_base_capacity=Decimal(rng.randint(0, 6)),
        shadow_base_capacity=Decimal(rng.randint(0, 6)),
        dual_lambda=Decimal(rng.randint(0, 2)) / Decimal("2"),
        epsilon_rank=Decimal("0.1"),
    )
    return candidates, snapshot


def test_lowering_bid_preserves_every_observed_win() -> None:
    rng = random.Random(SEED)
    checked = 0
    for _ in range(10_000):
        candidates, snapshot = _case(rng)
        outcome = allocate(candidates, snapshot)
        for candidate in candidates:
            if not outcome.won(candidate.key):
                continue
            for lower_bid in GRID:
                if lower_bid > candidate.bid:
                    continue
                changed = tuple(
                    Candidate(
                        worker_id=item.worker_id,
                        task_id=item.task_id,
                        bid=lower_bid if item.key == candidate.key else item.bid,
                        base_cap=item.base_cap,
                        predicted_value=item.predicted_value,
                        expected_score_at_reference=item.expected_score_at_reference,
                        active=item.active,
                        deadline_ok=item.deadline_ok,
                    )
                    for item in candidates
                )
                checked += 1
                assert allocate(changed, snapshot).won(candidate.key), {
                    "candidate": candidate.key,
                    "original_bid": str(candidate.bid),
                    "lower_bid": str(lower_bid),
                    "snapshot": snapshot,
                }
    assert checked > 1_000


def test_critical_value_is_grid_threshold_and_cap_bounded() -> None:
    candidates = (
        Candidate("w0", "t0", Decimal("0"), Decimal("3"), Decimal("5")),
        Candidate("w1", "t0", Decimal("2"), Decimal("3"), Decimal("4")),
    )
    snapshot = AllocationSnapshot(
        active_tasks=frozenset({"t0"}),
        task_capacities={"t0": 1},
        actual_base_capacity=Decimal("3"),
        shadow_base_capacity=Decimal("3"),
        epsilon_rank=Decimal("0.1"),
    )
    winner = allocate(candidates, snapshot).winners[0]
    value = critical_value(winner, candidates, snapshot, Decimal("1"))
    assert value in GRID
    assert Decimal("0") <= value <= Decimal("3")
