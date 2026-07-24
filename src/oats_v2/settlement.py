from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from .ledger import LedgerState
from .types import D, MechanismStatus, Money


COMPLIANT_BASE_STATUSES = {
    MechanismStatus.SCREEN_PASS,
    MechanismStatus.SCREEN_SOFT_PASS,
    MechanismStatus.SCREEN_FAIL_COMPLIANT,
    MechanismStatus.COLD_START,
    MechanismStatus.PROTOCOL_FAULT,
}


@dataclass(frozen=True)
class FailureDisposition:
    status: MechanismStatus
    base_action: str
    task_action: str
    data_trust_action: str


def failure_disposition(status: MechanismStatus, *, effort_started: bool = False) -> FailureDisposition:
    table = {
        MechanismStatus.TASK_NOT_ACTIVATED: ("NONE", "NONE", "NONE"),
        MechanismStatus.MULTI_TASK_TYPE_UNSUPPORTED: ("NONE", "RETURN_EMPTY", "NONE"),
        MechanismStatus.CONTINUATION_TABLE_INVALID: ("NONE", "RETURN_EMPTY", "PRIVACY_GATE_FAIL"),
        MechanismStatus.TYPE_MODEL_UNSUPPORTED: ("NONE", "RETURN_EMPTY", "NONE"),
        MechanismStatus.MONOTONICITY_VIOLATION: ("ROLLBACK_ALL_BASE", "RETURN_NEW_EMPTY", "NONE"),
        MechanismStatus.CAP_INVARIANT_BROKEN: ("ROLLBACK_ALL_BASE", "PRESERVE_PREEXISTING_ONLY", "NONE"),
        MechanismStatus.WORKER_NONCOMPLIANT: ("RETURN", "CLOSE_AT_TASK_END", "NO_TRUST"),
        MechanismStatus.SCREEN_FAIL_COMPLIANT: ("RELEASE", "CLOSE_AT_TASK_END", "NO_TRUST"),
        MechanismStatus.COLD_START: ("TRIAL_POLICY", "TRIAL_POLICY", "EXTERNAL_ONLY"),
        MechanismStatus.PROTOCOL_FAULT: ("RELEASE" if effort_started else "RETURN", "FALLBACK", "NO_PRE_OUTCOME_UPDATE"),
        MechanismStatus.MISSING_OUTCOME: ("ALREADY_SETTLED", "QMISSING_AND_RETURN", "NO_TRUST_OR_MC"),
        MechanismStatus.ENDOGENOUS_OUTCOME: ("ALREADY_SETTLED", "QMISSING_AND_RETURN", "NO_TRUST_OR_MC"),
        MechanismStatus.LEDGER_FATAL: ("HALT", "FREEZE", "NONE"),
    }
    if status not in table:
        raise ValueError(f"status has no failure disposition: {status}")
    return FailureDisposition(status, *table[status])


def settle_base_for_status(
    ledger: LedgerState,
    contract_id: str,
    status: MechanismStatus,
    *,
    cancelled_before_effort: bool = False,
) -> dict[str, str]:
    pay = status in COMPLIANT_BASE_STATUSES and not cancelled_before_effort
    return ledger.settle_base(contract_id, pay=pay)


def settle_task_scores(
    ledger: LedgerState,
    task_key: str,
    qualities: Mapping[str, Decimal],
    per_worker_score_cap: Money,
    *,
    money_grid: Money | None = None,
) -> dict[str, str]:
    """Pay ``cap * quality`` per contract from the task escrow.

    ``qualities`` must be raw quality scores in [0,1]; the per-worker cap is
    multiplied exactly once here. When ``money_grid`` is given, each payment is
    floor-quantized to the grid, which keeps the settled total weakly below the
    exact value and therefore below the escrow whenever K*cap <= escrow.
    """
    cap = D(per_worker_score_cap)
    payments: dict[str, Money] = {}
    for contract_id, quality in qualities.items():
        quality = D(quality)
        if quality < 0 or quality > 1:
            raise ValueError("quality outside [0,1]")
        amount = cap * quality
        if money_grid is not None:
            amount = amount.quantize(D(money_grid), rounding=ROUND_FLOOR)
        payments[contract_id] = amount
    return ledger.close_task(task_key, payments)
