from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .events import EventLog
from .types import D


@dataclass
class TrustState:
    values: dict[str, Decimal] = field(default_factory=dict)
    processed_outcomes: set[str] = field(default_factory=set)
    event_log: EventLog = field(default_factory=EventLog, repr=False)

    def initialize(self, worker_id: str, rho0: Decimal) -> None:
        rho0 = D(rho0)
        if rho0 < 0 or rho0 > 1:
            raise ValueError("rho0 outside [0,1]")
        if worker_id not in self.values:
            self.values[worker_id] = rho0
            self.event_log.emit("TRUST_INITIALIZED", {"worker_id": worker_id, "rho": rho0})

    def update(
        self,
        worker_id: str,
        outcome_key: str,
        quality: Decimal,
        alpha0: Decimal,
        *,
        available: bool,
        independent: bool,
    ) -> Decimal:
        if worker_id not in self.values:
            raise KeyError(worker_id)
        if outcome_key in self.processed_outcomes or not available or not independent:
            return self.values[worker_id]
        quality, alpha0 = D(quality), D(alpha0)
        if quality < 0 or quality > 1 or alpha0 <= 0 or alpha0 > 1:
            raise ValueError("trust input outside canonical domain")
        updated = (Decimal("1") - alpha0) * self.values[worker_id] + alpha0 * quality
        if updated < 0 or updated > 1:
            raise AssertionError("trust-domain invariant broken")
        self.values[worker_id] = updated
        self.processed_outcomes.add(outcome_key)
        self.event_log.emit(
            "TRUST_UPDATED",
            {"worker_id": worker_id, "outcome_key": outcome_key, "rho": updated, "alpha": alpha0},
        )
        return updated


def clipped_adaptive_alpha(raw_alpha: Decimal, alpha_min: Decimal, alpha_max: Decimal) -> Decimal:
    raw_alpha, alpha_min, alpha_max = D(raw_alpha), D(alpha_min), D(alpha_max)
    if not (Decimal("0") < alpha_min <= alpha_max <= Decimal("1")):
        raise ValueError("invalid adaptive-alpha bounds")
    return min(alpha_max, max(alpha_min, raw_alpha))
