from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from .events import EventLog, stable_hash
from .types import D, Money


@dataclass
class LedgerState:
    """Pathwise money ledger.

    The only conserved quantity is ``free + locked_base + locked_score + paid``.
    Estimated reserves and shadow-envelope amounts never enter this object.
    """

    budget: Money
    free: Money | None = None
    locked_base: Money = Decimal("0")
    locked_score: Money = Decimal("0")
    paid: Money = Decimal("0")
    # Component running totals (paid == paid_base + paid_score by construction).
    paid_base: Money = Decimal("0")
    paid_score: Money = Decimal("0")
    base_locks: dict[str, Money] = field(default_factory=dict)
    task_escrows: dict[str, Money] = field(default_factory=dict)
    receipts: dict[str, dict[str, str]] = field(default_factory=dict)
    event_log: EventLog = field(default_factory=EventLog, repr=False)

    def __post_init__(self) -> None:
        self.budget = D(self.budget)
        self.free = self.budget if self.free is None else D(self.free)
        self.locked_base = D(self.locked_base)
        self.locked_score = D(self.locked_score)
        self.paid = D(self.paid)
        self.paid_base = D(self.paid_base)
        self.paid_score = D(self.paid_score)
        self.assert_invariants()

    def identity(self) -> Money:
        return self.free + self.locked_base + self.locked_score + self.paid

    def snapshot(self) -> dict[str, str]:
        return {
            "budget": str(self.budget),
            "free": str(self.free),
            "locked_base": str(self.locked_base),
            "locked_score": str(self.locked_score),
            "paid": str(self.paid),
        }

    @staticmethod
    def _delta(pre: dict[str, str], post: dict[str, str], key: str) -> Money:
        return D(post[key]) - D(pre[key])

    def _event_payload(
        self,
        pre: dict[str, str],
        *,
        slot: int | None = None,
        task: str | None = None,
        worker: str | None = None,
        status: str = "ACTIVE",
        **extra: object,
    ) -> dict[str, object]:
        post = self.snapshot()
        return {
            "slot": slot,
            "task": task,
            "worker": worker,
            "contract_version": "oats-v2-core-20260715",
            "pre_hash": stable_hash(pre),
            "post_hash": stable_hash(post),
            "delta_free": self._delta(pre, post, "free"),
            "delta_locked_base": self._delta(pre, post, "locked_base"),
            "delta_locked_score": self._delta(pre, post, "locked_score"),
            "delta_paid": self._delta(pre, post, "paid"),
            "status": status,
            **extra,
            **post,
        }

    def assert_invariants(self) -> None:
        values = (self.budget, self.free, self.locked_base, self.locked_score, self.paid)
        if any(value < 0 for value in values):
            raise RuntimeError("LEDGER_FATAL: negative ledger state")
        if self.identity() != self.budget:
            drift = abs(self.identity() - self.budget)
            if drift > Decimal("1e-9"):
                raise RuntimeError("LEDGER_FATAL: conservation identity broken")
        # Registry sums are O(n) in the number of live locks; skip on the formal
        # hot path (event_log disabled) where updates maintain them by construction.
        if not self.event_log.enabled:
            return
        if sum(self.base_locks.values(), Decimal("0")) != self.locked_base:
            raise RuntimeError("LEDGER_FATAL: base registry mismatch")
        if sum(self.task_escrows.values(), Decimal("0")) != self.locked_score:
            raise RuntimeError("LEDGER_FATAL: task registry mismatch")

    def can_lock(self, amount: Money) -> bool:
        amount = D(amount)
        return amount >= 0 and self.free >= amount

    def activate_task(self, escrow_key: str, amount: Money) -> bool:
        amount = D(amount)
        if escrow_key in self.task_escrows:
            if self.task_escrows[escrow_key] != amount:
                raise ValueError("idempotency key reused with different task escrow")
            return False
        if amount < 0 or self.free < amount:
            raise ValueError("insufficient task score budget")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free -= amount
        self.locked_score += amount
        self.task_escrows[escrow_key] = amount
        self.assert_invariants()
        if log:
            slot_text, _, task = escrow_key.partition("|")
            self.event_log.emit(
                "TASK_ESCROW_LOCKED",
                self._event_payload(pre, slot=int(slot_text) if slot_text.isdigit() else None, task=task or None, escrow_key=escrow_key, amount=amount),
            )
        return True

    def return_empty_task(self, escrow_key: str) -> Money:
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        amount = self.task_escrows.pop(escrow_key, None)
        if amount is None:
            return Decimal("0")
        self.locked_score -= amount
        self.free += amount
        self.receipts[f"task-return:{escrow_key}"] = {"amount": str(amount), "kind": "return"}
        self.assert_invariants()
        if log:
            slot_text, _, task = escrow_key.partition("|")
            self.event_log.emit(
                "EMPTY_TASK_RETURNED",
                self._event_payload(pre, slot=int(slot_text) if slot_text.isdigit() else None, task=task or None, escrow_key=escrow_key, amount=amount),
            )
        return amount

    def lock_bases_atomic(self, locks: Mapping[str, Money]) -> bool:
        normalized = {str(key): D(value) for key, value in locks.items()}
        duplicate = set(normalized).intersection(self.base_locks)
        if duplicate:
            existing = {key: self.base_locks[key] for key in duplicate}
            if existing == {key: normalized[key] for key in duplicate} and len(duplicate) == len(normalized):
                return False
            raise ValueError("base-lock idempotency conflict")
        if any(value < 0 for value in normalized.values()):
            raise ValueError("negative base lock")
        total = sum(normalized.values(), Decimal("0"))
        if total > self.free:
            raise ValueError("insufficient actual free budget")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        self.free -= total
        self.locked_base += total
        self.base_locks.update(normalized)
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "BASES_LOCKED_ATOMIC",
                self._event_payload(pre, contract_ids=sorted(normalized), amount=total),
            )
        return True

    def settle_base(self, contract_id: str, *, pay: bool) -> dict[str, str]:
        receipt_key = f"base:{contract_id}"
        if receipt_key in self.receipts:
            return self.receipts[receipt_key]
        if contract_id not in self.base_locks:
            raise KeyError(f"unknown base lock: {contract_id}")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        amount = self.base_locks.pop(contract_id)
        self.locked_base -= amount
        if pay:
            self.paid += amount
            self.paid_base += amount
            kind = "release"
        else:
            self.free += amount
            kind = "return"
        receipt = {"amount": str(amount), "kind": kind}
        self.receipts[receipt_key] = receipt
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "BASE_SETTLED",
                self._event_payload(
                    pre,
                    worker=contract_id.split("|", 1)[0],
                    contract_id=contract_id,
                    amount=amount,
                    disposition=kind,
                ),
            )
        return receipt

    def close_task(self, escrow_key: str, score_payments: Mapping[str, Money]) -> dict[str, str]:
        receipt_key = f"task-close:{escrow_key}"
        if receipt_key in self.receipts:
            return self.receipts[receipt_key]
        if escrow_key not in self.task_escrows:
            raise KeyError(f"unknown task escrow: {escrow_key}")
        log = self.event_log.enabled
        pre = self.snapshot() if log else None
        escrow = self.task_escrows.pop(escrow_key)
        payments = {str(key): D(value) for key, value in score_payments.items()}
        if any(value < 0 for value in payments.values()):
            raise ValueError("negative score payment")
        released = sum(payments.values(), Decimal("0"))
        if released > escrow:
            raise ValueError("score payment exceeds task escrow")
        returned = escrow - released
        self.locked_score -= escrow
        self.paid += released
        self.paid_score += released
        self.free += returned
        receipt = {"released": str(released), "returned": str(returned), "kind": "close"}
        self.receipts[receipt_key] = receipt
        self.assert_invariants()
        if log:
            self.event_log.emit(
                "TASK_ESCROW_CLOSED",
                self._event_payload(
                    pre,
                    slot=int(escrow_key.partition("|")[0]) if escrow_key.partition("|")[0].isdigit() else None,
                    task=escrow_key.partition("|")[2] or None,
                    escrow_key=escrow_key,
                    released=released,
                    returned=returned,
                ),
            )
        return receipt
