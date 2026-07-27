from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping


def decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class SpecificationStatus(str, Enum):
    READY = "READY"
    CORE_ONLY = "CORE_ONLY"
    EXCLUDED_INCOMPLETE_SPEC = "EXCLUDED_INCOMPLETE_SPEC"


@dataclass(frozen=True, slots=True, order=True)
class CandidateKey:
    task_id: str
    worker_id: str


@dataclass(frozen=True, slots=True)
class ExternalCandidateView:
    """Method-independent information available before a purchase decision."""

    slot: int
    task_id: str
    worker_id: str
    current_bid: Decimal
    public_task_value: Decimal
    method_independent_forecast: Decimal
    capacity: int
    deadline: int
    public_cell: int
    public_coverage_count: int = 0
    completion_probability: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("current_bid", "public_task_value", "method_independent_forecast"):
            object.__setattr__(self, name, decimal(getattr(self, name)))
        if self.completion_probability is not None:
            object.__setattr__(
                self,
                "completion_probability",
                decimal(self.completion_probability),
            )
        if self.slot < 0 or self.capacity < 1 or self.deadline < self.slot:
            raise ValueError("invalid public candidate timing/capacity")
        if self.current_bid < 0 or self.public_task_value < 0:
            raise ValueError("negative public monetary/value field")
        if self.public_coverage_count < 0:
            raise ValueError("negative public coverage count")
        if self.completion_probability is not None and not (
            Decimal("0") <= self.completion_probability <= Decimal("1")
        ):
            raise ValueError("completion probability outside [0, 1]")

    @property
    def key(self) -> CandidateKey:
        return CandidateKey(task_id=self.task_id, worker_id=self.worker_id)


@dataclass(frozen=True, slots=True)
class ExternalSlotView:
    slot: int
    trace_seed: int
    remaining_budget: Decimal
    candidates: tuple[ExternalCandidateView, ...]
    public_coverage_history: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining_budget", decimal(self.remaining_budget))
        if self.slot < 0 or self.remaining_budget < 0:
            raise ValueError("invalid slot view")
        if any(candidate.slot != self.slot for candidate in self.candidates):
            raise ValueError("candidate belongs to a different slot")


@dataclass(frozen=True, slots=True)
class NativePayment:
    key: CandidateKey
    amount: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", decimal(self.amount))
        if self.amount < 0:
            raise ValueError("negative native payment")


@dataclass(frozen=True, slots=True)
class ExternalDecision:
    method_id: str
    winners: tuple[CandidateKey, ...]
    native_payments: tuple[NativePayment, ...] | None
    specification_status: SpecificationStatus
    audit_fields: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExternalFeedback:
    """Outcome revealed only for a previously purchased candidate."""

    key: CandidateKey
    purchase_slot: int
    revealed_slot: int
    realized_external_value: Decimal
    selected_quality: Decimal | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "realized_external_value", decimal(self.realized_external_value)
        )
        if self.selected_quality is not None:
            object.__setattr__(self, "selected_quality", decimal(self.selected_quality))
        if self.purchase_slot < 0 or self.revealed_slot < self.purchase_slot:
            raise ValueError("feedback precedes purchase")
