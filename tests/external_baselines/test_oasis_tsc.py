from __future__ import annotations

from decimal import Decimal

from src.oats_external.adapters.oasis_tsc import (
    OasisCandidate,
    OasisTSCPolicy,
)
from src.oats_external.types import CandidateKey


def _candidate(task: str, worker: str, bid: str, quality: str) -> OasisCandidate:
    return OasisCandidate(
        key=CandidateKey(task_id=task, worker_id=worker),
        bid=Decimal(bid),
        long_term_quality=Decimal(quality),
    )


def test_algorithm_1_hand_calculation_and_basic_budget() -> None:
    policy = OasisTSCPolicy()
    policy.reset(20260735)
    candidates = (
        _candidate("t1", "w1", "1", "0.5"),
        _candidate("t1", "w2", "2", "0.5"),
        _candidate("t1", "w3", "0.5", "0.6"),
        _candidate("t1", "w4", "0.4", "0.6"),
        _candidate("t1", "w5", "0.3", "0.6"),
    )
    selection = policy.select_task(
        task_id="t1",
        task_budget="2",
        capacity=2,
        ordered_candidates=candidates,
        forecast_candidate_count=5,
    )
    # floor(5/e)=1: w1 defines rho*=0.5 and b*=1.  Both w3 and w4
    # satisfy rho>rho*, bid<b*, and the cumulative basic payment stays <=2.
    assert selection.observation_count == 1
    assert selection.winners == (
        CandidateKey("t1", "w3"),
        CandidateKey("t1", "w4"),
    )
    assert selection.basic_payments[CandidateKey("t1", "w3")] == Decimal("1.2")
    # After w3 replaces the sample threshold, w4 is paid at rho*/=1.2.
    assert selection.basic_payments[CandidateKey("t1", "w4")] == Decimal("0.5")
    assert sum(selection.basic_payments.values()) == Decimal("1.7")
    assert not selection.basic_budget_violation


def test_algorithm_1_replaces_sample_even_when_candidate_is_not_recruited() -> None:
    policy = OasisTSCPolicy()
    policy.reset(20260735)
    candidates = (
        _candidate("t1", "w1", "0.5", "0.25"),
        _candidate("t1", "w2", "2", "0.5"),
        # rho improves, but bid is not below the threshold bid.
        _candidate("t1", "w3", "1", "1"),
        # This worker is now compared with the replaced threshold.
        _candidate("t1", "w4", "0.2", "0.4"),
    )
    selection = policy.select_task(
        task_id="t1",
        task_budget="4",
        capacity=2,
        ordered_candidates=candidates,
        forecast_candidate_count=4,
    )
    assert selection.sample_replacements == 2
    assert selection.winners == (CandidateKey("t1", "w4"),)


def test_truth_discovery_identical_reports_uses_symmetric_limit() -> None:
    policy = OasisTSCPolicy()
    reports = {
        CandidateKey("t1", "w1"): Decimal("0.25"),
        CandidateKey("t1", "w2"): Decimal("0.25"),
    }
    result = policy.discover_truth(reports)
    assert result.converged
    assert result.truth == Decimal("0.25")
    assert result.zero_range_normalization
    assert set(result.current_quality.values()) == {Decimal("1")}
    assert set(result.normalized_quality.values()) == {Decimal("1")}


def test_actual_payment_is_not_clipped_and_budget_violation_is_visible() -> None:
    policy = OasisTSCPolicy()
    policy.reset(20260735)
    candidates = (
        _candidate("t1", "w1", "1", "0.5"),
        _candidate("t1", "w2", "0.5", "0.6"),
        _candidate("t1", "w3", "0.4", "0.6"),
    )
    selection = policy.select_task(
        task_id="t1",
        task_budget="2",
        capacity=2,
        ordered_candidates=candidates,
        forecast_candidate_count=3,
    )
    settlement = policy.settle_task(
        selection,
        reports={
            CandidateKey("t1", "w2"): Decimal("0"),
            CandidateKey("t1", "w3"): Decimal("1"),
        },
        bids={
            CandidateKey("t1", "w2"): Decimal("0.5"),
            CandidateKey("t1", "w3"): Decimal("0.4"),
        },
    )
    assert settlement.total_actual_payment == settlement.total_basic_payment
    assert not settlement.actual_budget_violation


def test_same_slot_updates_are_deterministic_and_worker_specific() -> None:
    first = OasisTSCPolicy()
    second = OasisTSCPolicy()
    first.reset(20260735)
    second.reset(20260735)
    updates = (
        (CandidateKey("t2", "w1"), Decimal("0")),
        (CandidateKey("t1", "w1"), Decimal("1")),
    )
    first.apply_quality_updates(updates)
    second.apply_quality_updates(reversed(updates))
    assert first.quality_of("w1") == Decimal("0.32")
    assert second.quality_of("w1") == first.quality_of("w1")
    assert first.audit_state()["quality_state_hash"] == second.audit_state()[
        "quality_state_hash"
    ]


def test_full_task_replay_is_deterministic() -> None:
    def replay() -> tuple[str, str]:
        policy = OasisTSCPolicy()
        policy.reset(20260735)
        candidates = (
            _candidate("t1", "w1", "1", "0.5"),
            _candidate("t1", "w2", "0.5", "0.6"),
            _candidate("t1", "w3", "0.4", "0.6"),
        )
        selection = policy.select_task(
            task_id="t1",
            task_budget="2",
            capacity=2,
            ordered_candidates=candidates,
            forecast_candidate_count=3,
        )
        settlement = policy.settle_task(
            selection,
            reports={
                CandidateKey("t1", "w2"): Decimal("0.2"),
                CandidateKey("t1", "w3"): Decimal("0.8"),
            },
            bids={
                CandidateKey("t1", "w2"): Decimal("0.5"),
                CandidateKey("t1", "w3"): Decimal("0.4"),
            },
        )
        policy.apply_quality_updates(settlement.current_quality.items())
        return selection.decision_hash, settlement.settlement_hash

    assert replay() == replay()
