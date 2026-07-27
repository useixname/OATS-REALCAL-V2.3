from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class QueuedWorkerFeedback:
    """One purchased worker--task record carried by a task outcome event."""

    worker_id: str
    task_id: str
    contract_id: str
    quality: Decimal
    score_cap: Decimal
    estimated_value: Decimal
    realized_value: Decimal
    trust_event_index: int
    feedback_available: bool = True
    recognize_value_at_feedback: bool = True


@dataclass(frozen=True, slots=True)
class QueuedTaskOutcome:
    """Calendar-time settlement event for one purchased task."""

    purchase_slot: int
    available_slot: int
    task_id: str
    task_key: str
    deadline: int
    feedback_records: tuple[QueuedWorkerFeedback, ...]

    def deterministic_key(self) -> tuple[int, str, str]:
        return (self.available_slot, self.task_id, self.task_key)


@dataclass
class FeedbackCalendar:
    """Deterministic calendar queue for delayed outcome feedback.

    A caller processes the queue twice per slot: once before the slot decision
    (events scheduled by earlier purchases), and once after the current
    decision (which makes delay-zero feedback visible only after selection).
    Within an availability slot, task events are ordered by task id and their
    worker records are ordered separately by the runner.
    """

    _events_by_slot: dict[int, list[QueuedTaskOutcome]] = field(default_factory=dict)
    _scheduled_task_keys: set[str] = field(default_factory=set)

    def schedule(self, event: QueuedTaskOutcome) -> None:
        if event.available_slot < event.purchase_slot:
            raise ValueError("feedback cannot become available before purchase")
        if not event.feedback_records:
            raise ValueError("a queued task outcome must contain a purchased record")
        if any(record.task_id != event.task_id for record in event.feedback_records):
            raise ValueError("feedback task id does not match queued task")
        if event.task_key in self._scheduled_task_keys:
            raise ValueError(f"duplicate queued task outcome: {event.task_key}")
        self._scheduled_task_keys.add(event.task_key)
        self._events_by_slot.setdefault(event.available_slot, []).append(event)

    def pop_due(self, slot: int) -> tuple[QueuedTaskOutcome, ...]:
        due_slots = sorted(key for key in self._events_by_slot if key <= slot)
        due: list[QueuedTaskOutcome] = []
        for due_slot in due_slots:
            due.extend(self._events_by_slot.pop(due_slot))
        due.sort(key=QueuedTaskOutcome.deterministic_key)
        return tuple(due)

    def drain(self) -> tuple[QueuedTaskOutcome, ...]:
        pending: list[QueuedTaskOutcome] = []
        for slot in sorted(self._events_by_slot):
            pending.extend(self._events_by_slot[slot])
        self._events_by_slot.clear()
        pending.sort(key=QueuedTaskOutcome.deterministic_key)
        return tuple(pending)

    @property
    def pending_task_count(self) -> int:
        return sum(len(events) for events in self._events_by_slot.values())

    @property
    def pending_feedback_count(self) -> int:
        return sum(
            len(event.feedback_records)
            for events in self._events_by_slot.values()
            for event in events
        )
