from __future__ import annotations

from decimal import Decimal

from src.oats_v2.experiments.metrics import compute_trust_metrics
from src.oats_v2.trust import TrustState, trust_feedback_id


def test_two_workers_on_one_task_receive_distinct_trust_transitions() -> None:
    trust = TrustState()
    trust.initialize("w1", Decimal("0.5"))
    trust.initialize("w2", Decimal("0.5"))

    first = trust_feedback_id("cell-a", 12, "task-a", "w1")
    second = trust_feedback_id("cell-a", 12, "task-a", "w2")

    assert first != second
    assert trust.update(
        "w1",
        first,
        Decimal("0.9"),
        Decimal("0.2"),
        available=True,
        independent=True,
    ) == Decimal("0.58")
    assert trust.update(
        "w2",
        second,
        Decimal("0.1"),
        Decimal("0.2"),
        available=True,
        independent=True,
    ) == Decimal("0.42")
    assert trust.transition_count == 2
    assert trust.feedback_submission_count == 2
    assert trust.duplicate_feedback_suppressed_count == 0


def test_identical_worker_specific_feedback_is_idempotent() -> None:
    trust = TrustState()
    trust.initialize("w1", Decimal("0.5"))
    feedback_id = trust_feedback_id("cell-a", 12, "task-a", "w1")

    trust.update(
        "w1",
        feedback_id,
        Decimal("0.9"),
        Decimal("0.2"),
        available=True,
        independent=True,
    )
    first_value = trust.values["w1"]
    trust.update(
        "w1",
        feedback_id,
        Decimal("0.1"),
        Decimal("0.2"),
        available=True,
        independent=True,
    )

    assert trust.values["w1"] == first_value
    assert trust.feedback_submission_count == 2
    assert trust.transition_count == 1
    assert trust.duplicate_feedback_suppressed_count == 1


def test_feedback_identity_binds_cell_slot_task_and_worker() -> None:
    base = trust_feedback_id("cell-a", 12, "task-a", "w1")
    assert base != trust_feedback_id("cell-b", 12, "task-a", "w1")
    assert base != trust_feedback_id("cell-a", 13, "task-a", "w1")
    assert base != trust_feedback_id("cell-a", 12, "task-b", "w1")
    assert base != trust_feedback_id("cell-a", 12, "task-a", "w2")


def test_feedback_and_transition_metrics_are_not_conflated() -> None:
    events = [
        {
            "selected": True,
            "feedback": True,
            "quality": Decimal("0.9"),
            "rho": Decimal("0.5"),
            "trust_transition_applied": True,
            "duplicate_feedback_suppressed": False,
        },
        {
            "selected": True,
            "feedback": True,
            "quality": Decimal("0.1"),
            "rho": Decimal("0.5"),
            "trust_transition_applied": True,
            "duplicate_feedback_suppressed": False,
        },
        {
            "selected": True,
            "feedback": True,
            "quality": Decimal("0.1"),
            "rho": Decimal("0.5"),
            "trust_transition_applied": False,
            "duplicate_feedback_suppressed": True,
        },
        {"selected": True, "feedback": False},
    ]

    metrics = compute_trust_metrics(events, population_size=2)
    assert metrics.selected_count == 4
    assert metrics.feedback_count == 3
    assert metrics.trust_transition_count == 2
    assert metrics.duplicate_feedback_suppressed_count == 1
    assert metrics.feedback_count_definition != metrics.trust_transition_count_definition
    assert metrics.feedback_id_fields == (
        "cell_id",
        "feedback_slot",
        "task_id",
        "worker_id",
    )
