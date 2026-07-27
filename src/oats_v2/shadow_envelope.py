from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from .events import EventLog, stable_hash
from .types import D, Money


@dataclass
class ShadowEnvelopeState:
    """Non-monetary R3 resource account using public worst-case caps.

    Lifecycle of a reservation (OATS / paper return-settlement semantics):
      * task activation HOLDS the escrow (worst-case bonus obligation);
      * admission COMMITS the escrow and the per-winner base caps;
      * settlement SETTLES each obligation at its realized amount and releases
        the worst-case slack back to ``free``.
    Earlier versions kept commitments forever, which made the shadow account
    strictly more conservative than the paper (unused escrow was never
    returned) and turned the envelope into a hard market-capacity cap.
    """

    budget: Money
    free: Money | None = None
    held_tasks: dict[str, Money] = field(default_factory=dict)
    committed_tasks: dict[str, Money] = field(default_factory=dict)
    committed_bases: dict[str, Money] = field(default_factory=dict)
    event_log: EventLog = field(default_factory=EventLog, repr=False)
    _held_total: Money = Decimal("0")
    _committed_total: Money = Decimal("0")
    # Permanently consumed capacity = realized settlements (actual payments).
    _settled_total: Money = Decimal("0")

    def __post_init__(self) -> None:
        self.budget = D(self.budget)
        self.free = self.budget if self.free is None else D(self.free)
        # Running totals so identity()/assert_invariants stay O(1) instead of
        # re-summing the committed dicts every call. Logic-preserving: the
        # totals equal the dict sums by construction.
        self._held_total = sum(self.held_tasks.values(), Decimal("0"))
        self._committed_total = sum(self.committed_tasks.values(), Decimal("0")) + sum(
            self.committed_bases.values(), Decimal("0")
        )
        self._settled_total = D(self._settled_total)
        self.assert_invariants()

    @property
    def held(self) -> Money:
        return self._held_total

    @property
    def committed(self) -> Money:
        return self._committed_total

    @property
    def settled(self) -> Money:
        return self._settled_total

    def identity(self) -> Money:
        # O(1) hot-path identity using running totals (numerically equal to the
        # dict sums by construction).
        return self.free + self._held_total + self._committed_total + self._settled_total

    @staticmethod
    def _fmt_amount(amount: Money) -> str:
        # Normalize exact zeros so serialized snapshots stay "0" (not "0E-9")
        # after increment/decrement of running totals.
        return "0" if amount == 0 else str(amount)

    def snapshot(self) -> dict[str, str]:
        return {
            "budget": str(self.budget),
            "free": str(self.free),
            "held": self._fmt_amount(self._held_total),
            "committed": self._fmt_amount(self._committed_total),
            "settled": self._fmt_amount(self._settled_total),
        }

    def _payload(self, pre: dict[str, str], **extra: object) -> dict[str, object]:
        post = self.snapshot()
        return {
            "contract_version": "oats-v2-core-20260715",
            "pre_hash": stable_hash(pre),
            "post_hash": stable_hash(post),
            "delta_shadow_free": D(post["free"]) - D(pre["free"]),
            "delta_shadow_consumed": (D(post["held"]) + D(post["committed"])) - (D(pre["held"]) + D(pre["committed"])),
            **extra,
            **post,
        }

    def assert_invariants(self) -> None:
        if (
            self.budget < 0
            or self.free < 0
            or self._held_total < 0
            or self._committed_total < 0
            or self._settled_total < 0
        ):
            raise RuntimeError("CAP_INVARIANT_BROKEN: negative shadow state")
        if self.identity() != self.budget:
            raise RuntimeError("CAP_INVARIANT_BROKEN: shadow identity broken")

    def hold_task(self, escrow_key: str, amount: Money) -> bool:
        amount = D(amount)
        if escrow_key in self.held_tasks or escrow_key in self.committed_tasks:
            existing = self.held_tasks.get(escrow_key, self.committed_tasks.get(escrow_key))
            if existing != amount:
                raise ValueError("shadow task idempotency conflict")
            return False
        if amount < 0 or amount > self.free:
            raise ValueError("insufficient shadow task capacity")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free -= amount
        self.held_tasks[escrow_key] = amount
        self._held_total += amount
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "SHADOW_TASK_HELD",
                self._payload(pre, escrow_key=escrow_key, amount=amount),
            )
        return True

    def release_empty_task(self, escrow_key: str) -> Money:
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        amount = self.held_tasks.pop(escrow_key, None)
        if amount is None:
            return Decimal("0")
        self.free += amount
        self._held_total -= amount
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "SHADOW_EMPTY_TASK_RELEASED",
                self._payload(pre, escrow_key=escrow_key, amount=amount),
            )
        return amount

    def settle_base(self, contract_id: str, realized: Money) -> Money:
        """Settle a committed per-winner base cap at its realized payment.

        The worst-case slack (cap - realized) returns to ``free``; the realized
        amount is accounted as permanently consumed. Mirrors the ledger's
        ``settle_base`` (paper Eq. (91): unused reserves are returned).
        """
        cap = self.committed_bases.pop(contract_id, None)
        if cap is None:
            raise KeyError(f"unknown shadow base commitment: {contract_id}")
        realized = D(realized)
        if realized < 0 or realized > cap:
            self.committed_bases[contract_id] = cap
            raise ValueError("realized base payment outside [0, cap]")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free += cap - realized
        self._committed_total -= cap
        self._settled_total += realized
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "SHADOW_BASE_SETTLED",
                self._payload(pre, contract_id=contract_id, amount=realized, slack=cap - realized),
            )
        return cap - realized

    def settle_task(self, task_key: str, realized: Money) -> Money:
        """Settle a committed task escrow at its realized score payout.

        The unused escrow (escrow - realized) returns to ``free``; the realized
        amount is accounted as permanently consumed. Mirrors the ledger's
        ``close_task`` (paper Eq. (93): unused escrow is returned).
        """
        escrow = self.committed_tasks.pop(task_key, None)
        if escrow is None:
            raise KeyError(f"unknown shadow task commitment: {task_key}")
        realized = D(realized)
        if realized < 0 or realized > escrow:
            self.committed_tasks[task_key] = escrow
            raise ValueError("realized task payout outside [0, escrow]")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free += escrow - realized
        self._committed_total -= escrow
        self._settled_total += realized
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "SHADOW_TASK_SETTLED",
                self._payload(pre, task_key=task_key, amount=realized, slack=escrow - realized),
            )
        return escrow - realized

    def commit_admission(self, task_keys: tuple[str, ...], base_caps: Mapping[str, Money]) -> bool:
        normalized = {str(key): D(value) for key, value in base_caps.items()}
        if set(normalized).intersection(self.committed_bases):
            if all(self.committed_bases.get(key) == value for key, value in normalized.items()):
                return False
            raise ValueError("shadow base idempotency conflict")
        for key in task_keys:
            if key not in self.held_tasks and key not in self.committed_tasks:
                raise ValueError(f"task not shadow-activated: {key}")
        base_total = sum(normalized.values(), Decimal("0"))
        if base_total > self.free:
            raise ValueError("insufficient shadow base capacity")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free -= base_total
        for key in task_keys:
            if key in self.held_tasks:
                moved = self.held_tasks.pop(key)
                self.committed_tasks[key] = moved
                self._held_total -= moved
                self._committed_total += moved
        self.committed_bases.update(normalized)
        self._committed_total += base_total
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "SHADOW_ADMISSION_COMMITTED",
                self._payload(
                    pre,
                    task_keys=sorted(task_keys),
                    contract_ids=sorted(normalized),
                    base_cap_total=base_total,
                ),
            )
        return True


@dataclass
class DualController:
    value: Decimal
    maximum: Decimal

    def __post_init__(self) -> None:
        self.value, self.maximum = D(self.value), D(self.maximum)
        if self.value < 0 or self.maximum < self.value:
            raise ValueError("invalid dual interval")

    def update(self, envelope_consumption: Money, pacing_budget: Money, step_size: Decimal) -> Decimal:
        gradient = D(envelope_consumption) - D(pacing_budget)
        self.value = min(self.maximum, max(Decimal("0"), self.value + D(step_size) * gradient))
        return self.value
