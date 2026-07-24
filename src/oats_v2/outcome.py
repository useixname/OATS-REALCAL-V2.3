from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .types import D, MechanismStatus


@dataclass(frozen=True)
class OutcomeProvenance:
    provider: str
    task_id: str
    cell: str
    time: str
    measurement_process: str
    digest: str
    independence_attestation: bool

    def validate(self) -> MechanismStatus:
        complete = all((self.provider, self.task_id, self.cell, self.time, self.measurement_process, self.digest))
        if not complete:
            return MechanismStatus.MISSING_OUTCOME
        if not self.independence_attestation:
            return MechanismStatus.ENDOGENOUS_OUTCOME
        return MechanismStatus.ACTIVE


def gross_value(
    *, purchased: bool, independent_outcome: bool, task_value: Decimal, task_capacity: int, quality: Decimal
) -> Decimal:
    quality = D(quality)
    if not purchased or not independent_outcome:
        return Decimal("0")
    if task_capacity < 1 or quality < 0 or quality > 1:
        raise ValueError("invalid gross-value input")
    return D(task_value) / task_capacity * quality
