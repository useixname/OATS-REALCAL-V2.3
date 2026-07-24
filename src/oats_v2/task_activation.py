from __future__ import annotations

from dataclasses import dataclass

from .ledger import LedgerState
from .shadow_envelope import ShadowEnvelopeState
from .types import MechanismStatus, Task


@dataclass(frozen=True)
class ActivationReceipt:
    task_key: str
    status: MechanismStatus
    newly_activated: bool


def activate_task_atomic(
    task: Task,
    slot: int,
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
) -> ActivationReceipt:
    task_key = f"{slot}|{task.task_id}"
    already = task_key in ledger.task_escrows
    if already:
        if ledger.task_escrows[task_key] != task.score_escrow:
            raise ValueError("task activation idempotency conflict")
        return ActivationReceipt(task_key, MechanismStatus.ACTIVE, False)
    if not task.active or ledger.free < task.score_escrow or shadow.free < task.score_escrow:
        return ActivationReceipt(task_key, MechanismStatus.TASK_NOT_ACTIVATED, False)
    ledger.activate_task(task_key, task.score_escrow)
    try:
        shadow.hold_task(task_key, task.score_escrow)
    except Exception:
        ledger.return_empty_task(task_key)
        raise
    return ActivationReceipt(task_key, MechanismStatus.ACTIVE, True)


def return_empty_activation(
    task_key: str,
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
) -> None:
    if task_key in shadow.committed_tasks:
        raise ValueError("cannot return committed task activation")
    ledger.return_empty_task(task_key)
    shadow.release_empty_task(task_key)
