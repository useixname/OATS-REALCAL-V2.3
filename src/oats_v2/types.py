from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping


Money = Decimal


def D(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class MechanismStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TASK_NOT_ACTIVATED = "TASK_NOT_ACTIVATED"
    NONPOSITIVE_ADJUSTED_VALUE = "NONPOSITIVE_ADJUSTED_VALUE"
    RESERVE_PRICE_EXCEEDED = "RESERVE_PRICE_EXCEEDED"
    SCREEN_SOFT_PASS = "SCREEN_SOFT_PASS"
    MULTI_TASK_TYPE_UNSUPPORTED = "MULTI_TASK_TYPE_UNSUPPORTED"
    CONTINUATION_TABLE_INVALID = "CONTINUATION_TABLE_INVALID"
    TYPE_MODEL_UNSUPPORTED = "TYPE_MODEL_UNSUPPORTED"
    MONOTONICITY_VIOLATION = "MONOTONICITY_VIOLATION"
    CAP_INVARIANT_BROKEN = "CAP_INVARIANT_BROKEN"
    WORKER_NONCOMPLIANT = "WORKER_NONCOMPLIANT"
    SCREEN_PASS = "SCREEN_PASS"
    SCREEN_FAIL_COMPLIANT = "SCREEN_FAIL_COMPLIANT"
    COLD_START = "COLD_START"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    PROTOCOL_FAULT = "PROTOCOL_FAULT"
    MISSING_OUTCOME = "MISSING_OUTCOME"
    ENDOGENOUS_OUTCOME = "ENDOGENOUS_OUTCOME"
    LEDGER_FATAL = "LEDGER_FATAL"
    SETTLED = "SETTLED"


class ReportMode(str, Enum):
    ATTESTED_POINT_ESTIMATE = "ATTESTED_POINT_ESTIMATE"


@dataclass(frozen=True)
class Candidate:
    worker_id: str
    task_id: str
    bid: Money
    base_cap: Money
    predicted_value: Decimal
    expected_score_at_reference: Money = Decimal("0")
    active: bool = True
    deadline_ok: bool = True
    public_role: str = "default"
    # Platform reserve price (max willingness to pay for the base payment).
    # None disables the reserve gate. A candidate whose bid exceeds the reserve
    # is rejected, and the critical payment is bounded by the reserve, which
    # gives the platform per-trade individual rationality (paper §V-C).
    reserve_price: Money | None = None

    def __post_init__(self) -> None:
        for name in ("bid", "base_cap", "predicted_value", "expected_score_at_reference"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                object.__setattr__(self, name, D(value))
        if self.reserve_price is not None and not isinstance(self.reserve_price, Decimal):
            object.__setattr__(self, "reserve_price", D(self.reserve_price))
        if self.bid < 0 or self.base_cap < 0 or self.predicted_value < 0:
            raise ValueError("candidate monetary/value fields must be nonnegative")
        if self.bid > self.base_cap:
            raise ValueError("bid exceeds public base cap")

    @property
    def key(self) -> str:
        return f"{self.worker_id}|{self.task_id}"

    @property
    def estimated_reserve(self) -> Money:
        return self.bid + self.expected_score_at_reference


@dataclass(frozen=True)
class Task:
    task_id: str
    capacity: int
    score_escrow: Money
    active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.score_escrow, Decimal):
            object.__setattr__(self, "score_escrow", D(self.score_escrow))
        if self.capacity < 1 or self.score_escrow < 0:
            raise ValueError("invalid task")


@dataclass(frozen=True)
class AllocationSnapshot:
    active_tasks: frozenset[str]
    task_capacities: Mapping[str, int]
    worker_capacities: Mapping[str, int] = field(default_factory=dict)
    actual_base_capacity: Money = Decimal("0")
    shadow_base_capacity: Money = Decimal("0")
    dual_lambda: Decimal = Decimal("0")
    epsilon_rank: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        for name in ("actual_base_capacity", "shadow_base_capacity", "dual_lambda", "epsilon_rank"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                object.__setattr__(self, name, D(value))
        if self.actual_base_capacity < 0 or self.shadow_base_capacity < 0:
            raise ValueError("negative capacity")
        if self.dual_lambda < 0 or self.epsilon_rank <= 0:
            raise ValueError("invalid ranking parameters")


@dataclass(frozen=True)
class AllocationOutcome:
    winners: tuple[str, ...]
    order: tuple[str, ...]
    rejected: Mapping[str, str]

    def won(self, candidate_key: str) -> bool:
        return candidate_key in self.winners


@dataclass(frozen=True)
class SelectionOutcome:
    allocation: AllocationOutcome
    critical_bases: Mapping[str, Money]
    status: MechanismStatus

