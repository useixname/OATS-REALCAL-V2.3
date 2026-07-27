from __future__ import annotations

from decimal import Decimal

from src.oats_v2.feedback_calendar import (
    FeedbackCalendar,
    QueuedTaskOutcome,
    QueuedWorkerFeedback,
)
from src.oats_v2.experiments.formal_runner import _apply_due_task_outcomes
from src.oats_v2.experiments.run_matrix import RunCell
from src.oats_v2.ledger import LedgerState
from src.oats_v2.shadow_envelope import ShadowEnvelopeState
from src.oats_v2.task_activation import activate_task_atomic
from src.oats_v2.trust import TrustState
from src.oats_v2.types import Task


def _cell() -> RunCell:
    return RunCell(
        cell_id="queue-test",
        family="TEST",
        seed=20260715,
        method_id="V2-FULL",
        gamma=Decimal("0.3"),
        budget_ratio=Decimal("0.25"),
        contamination=Decimal("0"),
        delay=5,
        missing_prob=Decimal("0"),
        arrival_multiplier=Decimal("1"),
        order_index=0,
    )


def _record(
    *,
    worker_id: str,
    task_id: str,
    event_index: int,
    quality: str,
) -> QueuedWorkerFeedback:
    return QueuedWorkerFeedback(
        worker_id=worker_id,
        task_id=task_id,
        contract_id=f"{worker_id}|{task_id}",
        quality=Decimal(quality),
        score_cap=Decimal("1"),
        estimated_value=Decimal("1"),
        realized_value=Decimal("1"),
        trust_event_index=event_index,
    )


def _activate_committed_task(
    task_id: str,
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
) -> str:
    receipt = activate_task_atomic(
        Task(task_id, capacity=2, score_escrow=Decimal("2")),
        1,
        ledger,
        shadow,
    )
    shadow.commit_admission((receipt.task_key,), {})
    return receipt.task_key


def test_calendar_has_predecision_and_postdecision_delay_zero_phases() -> None:
    calendar = FeedbackCalendar()
    assert calendar.pop_due(3) == ()

    delay_zero = QueuedTaskOutcome(
        purchase_slot=3,
        available_slot=3,
        task_id="task-zero",
        task_key="3|task-zero",
        deadline=10,
        feedback_records=(_record(worker_id="w1", task_id="task-zero", event_index=0, quality="1"),),
    )
    calendar.schedule(delay_zero)
    assert calendar.pop_due(3) == (delay_zero,)

    delayed = QueuedTaskOutcome(
        purchase_slot=3,
        available_slot=8,
        task_id="task-five",
        task_key="3|task-five",
        deadline=10,
        feedback_records=(_record(worker_id="w1", task_id="task-five", event_index=1, quality="1"),),
    )
    calendar.schedule(delayed)
    assert calendar.pop_due(7) == ()
    assert calendar.pop_due(8) == (delayed,)


def test_multiworker_task_applies_one_transition_per_feedback_and_closes_escrow() -> None:
    ledger = LedgerState(Decimal("10"))
    shadow = ShadowEnvelopeState(Decimal("10"))
    task_key = _activate_committed_task("task-a", ledger, shadow)
    trust = TrustState()
    trust.initialize("w1", Decimal("0.5"))
    trust.initialize("w2", Decimal("0.5"))
    trust_events = [
        {"selected": True, "feedback": False},
        {"selected": True, "feedback": False},
    ]
    event = QueuedTaskOutcome(
        purchase_slot=1,
        available_slot=6,
        task_id="task-a",
        task_key=task_key,
        deadline=10,
        feedback_records=(
            _record(worker_id="w2", task_id="task-a", event_index=1, quality="0.1"),
            _record(worker_id="w1", task_id="task-a", event_index=0, quality="0.9"),
        ),
    )

    gross, estimated, realized = _apply_due_task_outcomes(
        (event,),
        cell=_cell(),
        trust=trust,
        ledger=ledger,
        shadow=shadow,
        trust_events=trust_events,
        alpha=Decimal("0.2"),
        money_grid=Decimal("0.001"),
    )

    assert trust.values == {"w1": Decimal("0.58"), "w2": Decimal("0.42")}
    assert trust.feedback_submission_count == 2
    assert trust.transition_count == 2
    assert trust.duplicate_feedback_suppressed_count == 0
    assert ledger.locked_score == 0
    assert ledger.paid_score == Decimal("1.000")
    assert shadow.committed == 0
    assert shadow.settled == Decimal("1.000")
    assert gross == Decimal("2")
    assert estimated == [Decimal("1"), Decimal("1")]
    assert realized == [Decimal("1"), Decimal("1")]


def test_same_slot_same_worker_uses_task_then_worker_lexicographic_order() -> None:
    ledger = LedgerState(Decimal("10"))
    shadow = ShadowEnvelopeState(Decimal("10"))
    task_a_key = _activate_committed_task("task-a", ledger, shadow)
    task_b_key = _activate_committed_task("task-b", ledger, shadow)
    trust = TrustState()
    trust.initialize("w1", Decimal("0.5"))
    trust_events = [
        {"selected": True, "feedback": False},
        {"selected": True, "feedback": False},
    ]
    calendar = FeedbackCalendar()
    calendar.schedule(
        QueuedTaskOutcome(
            purchase_slot=1,
            available_slot=6,
            task_id="task-b",
            task_key=task_b_key,
            deadline=10,
            feedback_records=(
                _record(worker_id="w1", task_id="task-b", event_index=1, quality="0"),
            ),
        )
    )
    calendar.schedule(
        QueuedTaskOutcome(
            purchase_slot=1,
            available_slot=6,
            task_id="task-a",
            task_key=task_a_key,
            deadline=10,
            feedback_records=(
                _record(worker_id="w1", task_id="task-a", event_index=0, quality="1"),
            ),
        )
    )

    due = calendar.pop_due(6)
    assert [event.task_id for event in due] == ["task-a", "task-b"]
    _apply_due_task_outcomes(
        due,
        cell=_cell(),
        trust=trust,
        ledger=ledger,
        shadow=shadow,
        trust_events=trust_events,
        alpha=Decimal("0.2"),
        money_grid=Decimal("0.001"),
    )

    # q(task-a)=1 then q(task-b)=0: 0.5 -> 0.6 -> 0.48.
    assert trust.values["w1"] == Decimal("0.48")


def test_drain_preserves_terminal_pending_counts() -> None:
    calendar = FeedbackCalendar()
    event = QueuedTaskOutcome(
        purchase_slot=990,
        available_slot=1010,
        task_id="task-late",
        task_key="990|task-late",
        deadline=1020,
        feedback_records=(
            _record(worker_id="w1", task_id="task-late", event_index=0, quality="0.8"),
        ),
    )
    calendar.schedule(event)
    assert calendar.pending_task_count == 1
    assert calendar.pending_feedback_count == 1
    assert calendar.drain() == (event,)
    assert calendar.pending_task_count == 0
    assert calendar.pending_feedback_count == 0
