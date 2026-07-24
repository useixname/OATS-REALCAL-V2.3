from __future__ import annotations

import hashlib
import itertools
import json
from decimal import Decimal
from pathlib import Path

from src.oats_v2.allocation import allocate
from src.oats_v2.types import AllocationSnapshot, Candidate


GRID = (Decimal("0"), Decimal("1"), Decimal("2"))
SEED = 20260715
MANIFEST = Path("audit_results/t2_exhaustive_manifest.json")


def _assert_monotone(
    candidates: tuple[Candidate, ...], snapshot: AllocationSnapshot
) -> int:
    original = allocate(candidates, snapshot)
    checks = 0
    for target in candidates:
        if not original.won(target.key):
            continue
        for lower_bid in GRID:
            if lower_bid > target.bid:
                continue
            changed = tuple(
                Candidate(
                    item.worker_id,
                    item.task_id,
                    lower_bid if item.key == target.key else item.bid,
                    item.base_cap,
                    item.predicted_value,
                    item.expected_score_at_reference,
                    item.active,
                    item.deadline_ok,
                )
                for item in candidates
            )
            checks += 1
            assert allocate(changed, snapshot).won(target.key), {
                "status": "CLAIM_REFUTED",
                "target": target.key,
                "old_bid": str(target.bid),
                "lower_bid": str(lower_bid),
                "candidates": [item.__dict__ for item in candidates],
                "snapshot": snapshot,
            }
    return checks


def run_exhaustive() -> dict[str, object]:
    allocations = 0
    monotonicity_checks = 0
    tie_cases = 0

    # Full bid-vector product through six workers.  Structural quantities vary
    # independently in the second sweep to keep the declared domain tractable.
    for worker_count in range(1, 7):
        for bids in itertools.product(GRID, repeat=worker_count):
            candidates = tuple(
                Candidate(
                    f"w{index}",
                    "t0",
                    bid,
                    Decimal("2"),
                    Decimal((index % 3) + 1),
                    Decimal("0"),
                )
                for index, bid in enumerate(bids)
            )
            for capacity in (1, 2, worker_count):
                for actual_cap in (Decimal("0"), Decimal("2"), Decimal("4"), Decimal("12")):
                    for shadow_cap in (Decimal("0"), Decimal("2"), Decimal("4"), Decimal("12")):
                        snapshot = AllocationSnapshot(
                            active_tasks=frozenset({"t0"}),
                            task_capacities={"t0": capacity},
                            actual_base_capacity=actual_cap,
                            shadow_base_capacity=shadow_cap,
                            epsilon_rank=Decimal("0.1"),
                        )
                        allocations += 1
                        monotonicity_checks += _assert_monotone(candidates, snapshot)
            if len(set(bids)) < len(bids):
                tie_cases += 1

    # Cross-product sweep covering 1--4 tasks, 1--6 workers, every activation
    # pattern, two task-capacity patterns, multiple actual/shadow caps, and all
    # grid values at every target position.
    for task_count in range(1, 5):
        tasks = tuple(f"t{index}" for index in range(task_count))
        activation_patterns = tuple(
            frozenset(tasks[index] for index, bit in enumerate(bits) if bit)
            for bits in itertools.product((False, True), repeat=task_count)
        )
        for worker_count in range(1, 7):
            for target_position in range(worker_count):
                for bid in GRID:
                    bids = [Decimal((index + task_count) % 3) for index in range(worker_count)]
                    bids[target_position] = bid
                    candidates = tuple(
                        Candidate(
                            f"w{index}",
                            tasks[index % task_count],
                            bids[index],
                            Decimal("2"),
                            Decimal((index + task_count) % 4),
                            Decimal(index % 2),
                        )
                        for index in range(worker_count)
                    )
                    for active in activation_patterns:
                        for capacity_mode in (1, 2):
                            capacities = {
                                task: min(worker_count, capacity_mode + (index % 2))
                                for index, task in enumerate(tasks)
                            }
                            for actual_cap, shadow_cap in (
                                (Decimal("0"), Decimal("12")),
                                (Decimal("2"), Decimal("2")),
                                (Decimal("4"), Decimal("6")),
                                (Decimal("6"), Decimal("4")),
                                (Decimal("12"), Decimal("4")),
                                (Decimal("12"), Decimal("12")),
                            ):
                                snapshot = AllocationSnapshot(
                                    active_tasks=active,
                                    task_capacities=capacities,
                                    actual_base_capacity=actual_cap,
                                    shadow_base_capacity=shadow_cap,
                                    dual_lambda=Decimal(target_position % 2),
                                    epsilon_rank=Decimal("0.1"),
                                )
                                allocations += 1
                                monotonicity_checks += _assert_monotone(candidates, snapshot)

    manifest: dict[str, object] = {
        "claim": "T2 implementation-path finite-domain search",
        "conclusion": "NO_COUNTEREXAMPLE_IN_DECLARED_DOMAIN",
        "proof_substitute": False,
        "seed": SEED,
        "domain": {
            "tasks": [1, 2, 3, 4],
            "workers": [1, 2, 3, 4, 5, 6],
            "bid_grid": ["0", "1", "2"],
            "full_bid_vector_product_workers": [1, 2, 3, 4, 5, 6],
            "task_capacities": "1, 2, worker_count plus cross-task alternating capacities",
            "actual_caps": ["0", "2", "4", "6", "12"],
            "shadow_caps": ["0", "2", "4", "6", "12"],
            "activation_patterns": "all 2^m patterns for m=1..4",
            "tie_break": "(-density, worker_id, task_id)",
        },
        "allocation_instances": allocations,
        "winning_lower-bid_checks": monotonicity_checks,
        "full-vector_tie_profiles": tie_cases,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256_without_hash_field"] = hashlib.sha256(canonical.encode()).hexdigest()
    return manifest


def test_small_domain_has_no_monotonicity_counterexample() -> None:
    manifest = run_exhaustive()
    assert manifest["allocation_instances"] > 70_000
    assert manifest["winning_lower-bid_checks"] > 10_000


if __name__ == "__main__":
    result = run_exhaustive()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
