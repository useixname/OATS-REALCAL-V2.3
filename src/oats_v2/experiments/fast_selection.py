from __future__ import annotations

"""Fast critical-payment selection for formal runs.

Supports the OATS density rule plus baseline ranking modes, the paper's
Eq. (78) adjusted-value gate, and the platform reserve price. All gates are
monotone in the candidate's own bid, so the grid binary search still returns
the Myerson critical payment and the non-prefix witness check remains valid.
"""

import hashlib
from decimal import Decimal
from typing import Iterable

from ..critical_payment import MonotonicityViolation, _grid
from ..types import AllocationOutcome, AllocationSnapshot, Candidate, MechanismStatus, Money, SelectionOutcome


RANK_MODES = ("density", "surplus", "cost", "value", "quality", "random")


def _wins(
    order: list[int],
    *,
    worker_ids: list[str],
    task_ids: list[str],
    base_caps_m: list[int],
    skip: list[bool],
    worker_caps: dict[str, int],
    task_caps: dict[str, int],
    total_cap: Decimal,
    money_grid: Decimal,
    target: int,
) -> bool:
    # Base caps are grid-aligned; accumulate in milli-ticks. Total capacity
    # comes from ledger/shadow free and may carry extra precision, so compare
    # against it in Decimal (same as the reference allocator).
    used = 0
    worker_used: dict[str, int] = {}
    task_used: dict[str, int] = {}
    for idx in order:
        if skip[idx]:
            continue
        w = worker_ids[idx]
        t = task_ids[idx]
        cap = base_caps_m[idx]
        if (
            worker_used.get(w, 0) >= worker_caps.get(w, 1)
            or task_used.get(t, 0) >= task_caps.get(t, 0)
            or Decimal(used + cap) * money_grid > total_cap
        ):
            continue
        if idx == target:
            return True
        worker_used[w] = worker_used.get(w, 0) + 1
        task_used[t] = task_used.get(t, 0) + 1
        used += cap
    return False


def _allocate_full(
    order: list[int],
    *,
    keys: list[str],
    worker_ids: list[str],
    task_ids: list[str],
    base_caps_m: list[int],
    skip: list[bool],
    skip_reason: list[str],
    worker_caps: dict[str, int],
    task_caps: dict[str, int],
    total_cap: Decimal,
    money_grid: Decimal,
) -> AllocationOutcome:
    used = 0
    worker_used: dict[str, int] = {}
    task_used: dict[str, int] = {}
    winners: list[str] = []
    rejected: dict[str, str] = {}
    for idx in order:
        if skip[idx]:
            rejected[keys[idx]] = skip_reason[idx]
            continue
        w = worker_ids[idx]
        t = task_ids[idx]
        cap = base_caps_m[idx]
        if (
            worker_used.get(w, 0) >= worker_caps.get(w, 1)
            or task_used.get(t, 0) >= task_caps.get(t, 0)
            or Decimal(used + cap) * money_grid > total_cap
        ):
            rejected[keys[idx]] = "DOWNWARD_CLOSED_CAPACITY"
            continue
        winners.append(keys[idx])
        worker_used[w] = worker_used.get(w, 0) + 1
        task_used[t] = task_used.get(t, 0) + 1
        used += cap
    return AllocationOutcome(
        winners=tuple(winners),
        order=tuple(keys[i] for i in order),
        rejected=rejected,
    )


def _random_rank(salt: str, worker_id: str, task_id: str) -> int:
    digest = hashlib.sha256(f"{salt}|{worker_id}|{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class _FastPool:
    __slots__ = (
        "n",
        "keys",
        "worker_ids",
        "task_ids",
        "base_caps",
        "base_caps_m",
        "bids",
        "score_ref_f",
        "vhat_f",
        "dual",
        "rank_mode",
        "av_gate",
        "eps_f",
        "reserve",
        "skip",
        "skip_reason",
        "rand_key",
        "worker_caps",
        "task_caps",
        "total_cap",
        "order",
        "pos",
        "by_key",
        "money_grid",
    )

    def __init__(
        self,
        candidates: tuple[Candidate, ...],
        snapshot: AllocationSnapshot,
        money_grid: Money,
        *,
        rank_mode: str,
        av_gate: bool,
        rank_salt: str = "",
    ) -> None:
        if rank_mode not in RANK_MODES:
            raise ValueError(f"unknown rank mode: {rank_mode}")
        self.n = len(candidates)
        self.keys = [c.key for c in candidates]
        self.worker_ids = [c.worker_id for c in candidates]
        self.task_ids = [c.task_id for c in candidates]
        self.base_caps = [c.base_cap for c in candidates]
        self.base_caps_m = [self._to_milli(c.base_cap, money_grid) for c in candidates]
        self.bids = [c.bid for c in candidates]
        # Decimal ranking inputs, identical to allocation._density / rank_key so
        # the fast path reproduces the reference allocator exactly (Spec §8/§9, T2).
        self.score_ref_f = [c.expected_score_at_reference for c in candidates]
        self.vhat_f = [c.predicted_value for c in candidates]
        self.dual = snapshot.dual_lambda
        self.rank_mode = rank_mode
        self.av_gate = av_gate
        self.eps_f = snapshot.epsilon_rank
        self.reserve = [c.reserve_price for c in candidates]
        self.rand_key = (
            [_random_rank(rank_salt, c.worker_id, c.task_id) for c in candidates]
            if rank_mode == "random"
            else [0] * self.n
        )
        self.skip = [False] * self.n
        self.skip_reason = [""] * self.n
        for i in range(self.n):
            self._refresh_skip(i)
        self.worker_caps = dict(snapshot.worker_capacities)
        self.task_caps = {tid: int(cap) for tid, cap in snapshot.task_capacities.items()}
        self.total_cap = min(snapshot.actual_base_capacity, snapshot.shadow_base_capacity)
        self.money_grid = money_grid
        self.by_key = {key: i for i, key in enumerate(self.keys)}
        self.order = list(range(self.n))
        self.pos = list(range(self.n))
        self._full_sort()

    @staticmethod
    def _to_milli(amount: Decimal, money_grid: Decimal) -> int:
        ticks = amount / money_grid
        as_int = int(ticks)
        if ticks != as_int:
            raise ValueError(f"amount {amount} is not aligned to money grid {money_grid}")
        return as_int

    def _adjusted_value(self, idx: int) -> Decimal:
        # Paper Eq. (75): AV = vhat - lambda * reserved payment where the
        # reserve is this pair's own (bid + expected score bonus). Monotone
        # decreasing in the candidate's own bid, so gates and ranking preserve
        # the prefix property required by the critical-payment search.
        return self.vhat_f[idx] - self.dual * (self.bids[idx] + self.score_ref_f[idx])

    def _refresh_skip(self, idx: int) -> None:
        if self.reserve[idx] is not None and self.bids[idx] > self.reserve[idx]:
            self.skip[idx] = True
            self.skip_reason[idx] = MechanismStatus.RESERVE_PRICE_EXCEEDED.value
            return
        if self.av_gate and self.rank_mode == "density" and self._adjusted_value(idx) <= 0:
            self.skip[idx] = True
            self.skip_reason[idx] = MechanismStatus.NONPOSITIVE_ADJUSTED_VALUE.value
            return
        if self.av_gate and self.rank_mode == "surplus" and (self.vhat_f[idx] - self.bids[idx]) <= 0:
            self.skip[idx] = True
            self.skip_reason[idx] = MechanismStatus.NONPOSITIVE_ADJUSTED_VALUE.value
            return
        self.skip[idx] = False
        self.skip_reason[idx] = ""

    def _rank_key(self, idx: int) -> tuple:
        mode = self.rank_mode
        if mode == "density":
            num = self._adjusted_value(idx)
            if num < 0:
                num = Decimal("0")
            dens = num / (self.bids[idx] + self.score_ref_f[idx] + self.eps_f)
            return (-dens, self.worker_ids[idx], self.task_ids[idx])
        if mode == "surplus":
            return (-(self.vhat_f[idx] - self.bids[idx]), self.worker_ids[idx], self.task_ids[idx])
        if mode == "cost":
            return (self.bids[idx], self.worker_ids[idx], self.task_ids[idx])
        if mode == "value":
            return (-self.vhat_f[idx], self.worker_ids[idx], self.task_ids[idx])
        if mode == "quality":
            return (-self.score_ref_f[idx], self.worker_ids[idx], self.task_ids[idx])
        return (self.rand_key[idx], self.worker_ids[idx], self.task_ids[idx])

    def _full_sort(self) -> None:
        self.order.sort(key=self._rank_key)
        for p, idx in enumerate(self.order):
            self.pos[idx] = p

    def set_bid(self, idx: int, bid: Money) -> None:
        if self.bids[idx] == bid:
            return
        # Remove from current position.
        p = self.pos[idx]
        del self.order[p]
        for j in range(p, len(self.order)):
            self.pos[self.order[j]] = j
        self.bids[idx] = bid
        self._refresh_skip(idx)
        # Re-insert with binary search.
        key = self._rank_key(idx)
        lo, hi = 0, len(self.order)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._rank_key(self.order[mid]) < key:
                lo = mid + 1
            else:
                hi = mid
        self.order.insert(lo, idx)
        for j in range(lo, len(self.order)):
            self.pos[self.order[j]] = j

    def wins(self, idx: int) -> bool:
        if self.skip[idx]:
            return False
        return _wins(
            self.order,
            worker_ids=self.worker_ids,
            task_ids=self.task_ids,
            base_caps_m=self.base_caps_m,
            skip=self.skip,
            worker_caps=self.worker_caps,
            task_caps=self.task_caps,
            total_cap=self.total_cap,
            money_grid=self.money_grid,
            target=idx,
        )

    def outcome(self) -> AllocationOutcome:
        return _allocate_full(
            self.order,
            keys=self.keys,
            worker_ids=self.worker_ids,
            task_ids=self.task_ids,
            base_caps_m=self.base_caps_m,
            skip=self.skip,
            skip_reason=self.skip_reason,
            worker_caps=self.worker_caps,
            task_caps=self.task_caps,
            total_cap=self.total_cap,
            money_grid=self.money_grid,
        )

    def critical_value(self, key: str) -> Money:
        idx = self.by_key[key]
        original = self.bids[idx]
        grid = _grid(self.base_caps[idx], self.money_grid)
        lo, hi = 0, len(grid) - 1
        best: Money | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            bid = grid[mid]
            self.set_bid(idx, bid)
            if self.wins(idx):
                best = bid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            self.set_bid(idx, original)
            raise ValueError("critical value requested for a never-winning contract")
        witness_index = hi + 2
        if witness_index < len(grid):
            self.set_bid(idx, grid[witness_index])
            if self.wins(idx):
                self.set_bid(idx, original)
                raise MonotonicityViolation(
                    f"non-prefix winning set for {key}; witness bid={grid[witness_index]}"
                )
        self.set_bid(idx, original)
        return best


def build_selection_fast(
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    money_grid: Money,
    *,
    use_density: bool = True,
    rank_mode: str | None = None,
    av_gate: bool = False,
    rank_salt: str = "",
) -> SelectionOutcome:
    candidates = tuple(candidates)
    if not candidates:
        return SelectionOutcome(AllocationOutcome((), (), {}), {}, MechanismStatus.ACTIVE)
    mode = rank_mode if rank_mode is not None else ("density" if use_density else "surplus")
    pool = _FastPool(
        candidates,
        snapshot,
        money_grid,
        rank_mode=mode,
        av_gate=av_gate,
        rank_salt=rank_salt,
    )
    outcome = pool.outcome()
    try:
        critical = {key: pool.critical_value(key) for key in outcome.winners}
    except MonotonicityViolation:
        return SelectionOutcome(outcome, {}, MechanismStatus.MONOTONICITY_VIOLATION)
    return SelectionOutcome(outcome, critical, MechanismStatus.ACTIVE)


def build_selection_myopic_fast(
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    money_grid: Money,
) -> SelectionOutcome:
    return build_selection_fast(candidates, snapshot, money_grid, use_density=False)
