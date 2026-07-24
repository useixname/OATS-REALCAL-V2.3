from __future__ import annotations

import random
from decimal import Decimal

from src.oats_v2.critical_payment import build_selection
from src.oats_v2.experiments.fast_selection import build_selection_fast
from src.oats_v2.types import AllocationSnapshot, Candidate


def _sample_candidates(n: int = 40) -> tuple[Candidate, ...]:
    out = []
    for i in range(n):
        out.append(
            Candidate(
                worker_id=f"w{i}",
                task_id=f"t{i % 8}",
                bid=Decimal("0.5") + Decimal("0.01") * i,
                base_cap=Decimal("3.0"),
                predicted_value=Decimal("2.0") + Decimal("0.05") * (n - i),
                expected_score_at_reference=Decimal("0.4"),
            )
        )
    return tuple(out)


def test_fast_selection_matches_reference_critical_values() -> None:
    candidates = _sample_candidates(36)
    snapshot = AllocationSnapshot(
        active_tasks=frozenset(c.task_id for c in candidates),
        task_capacities={f"t{i}": 3 for i in range(8)},
        actual_base_capacity=Decimal("60"),
        shadow_base_capacity=Decimal("60"),
        dual_lambda=Decimal("0.2"),
        epsilon_rank=Decimal("0.001"),
    )
    money = Decimal("0.001")
    ref = build_selection(candidates, snapshot, money)
    fast = build_selection_fast(candidates, snapshot, money, use_density=True)
    assert ref.status == fast.status
    assert set(ref.allocation.winners) == set(fast.allocation.winners)
    for key in ref.allocation.winners:
        assert ref.critical_bases[key] == fast.critical_bases[key]


def _random_case(rng: random.Random) -> tuple[tuple[Candidate, ...], AllocationSnapshot]:
    n = rng.randint(4, 30)
    n_tasks = rng.randint(1, 6)
    candidates = []
    for i in range(n):
        base_cap = Decimal(rng.choice(["1.0", "2.0", "3.0"]))
        # bid must lie on grid and be <= base_cap
        grid_units = rng.randint(0, int(base_cap / Decimal("0.001")))
        bid = (Decimal("0.001") * grid_units)
        candidates.append(
            Candidate(
                worker_id=f"w{i}",
                task_id=f"t{i % n_tasks}",
                bid=bid,
                base_cap=base_cap,
                predicted_value=Decimal(rng.choice(["0", "0.5", "1.0", "1.5", "2.0", "2.5"])),
                expected_score_at_reference=Decimal(rng.choice(["0", "0.2", "0.4", "0.6"])),
            )
        )
    snapshot = AllocationSnapshot(
        active_tasks=frozenset(c.task_id for c in candidates),
        task_capacities={f"t{i}": rng.randint(1, 4) for i in range(n_tasks)},
        actual_base_capacity=Decimal(rng.choice(["3", "6", "10", "20", "50"])),
        shadow_base_capacity=Decimal(rng.choice(["3", "6", "10", "20", "50"])),
        dual_lambda=Decimal(rng.choice(["0", "0.1", "0.3", "0.5"])),
        epsilon_rank=Decimal("0.001"),
    )
    return tuple(candidates), snapshot


def test_fast_selection_differential_random() -> None:
    rng = random.Random(4242)
    money = Decimal("0.001")
    for _ in range(200):
        candidates, snapshot = _random_case(rng)
        ref = build_selection(candidates, snapshot, money)
        fast = build_selection_fast(candidates, snapshot, money, use_density=True)
        assert ref.status == fast.status, (ref.status, fast.status)
        if ref.status.name != "ACTIVE":
            continue
        assert tuple(ref.allocation.winners) == tuple(fast.allocation.winners)
        for key in ref.allocation.winners:
            assert ref.critical_bases[key] == fast.critical_bases[key], key
