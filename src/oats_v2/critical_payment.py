from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Iterable

from .allocation import allocate
from .types import AllocationSnapshot, Candidate, MechanismStatus, Money, SelectionOutcome


class MonotonicityViolation(RuntimeError):
    pass


_GRID_CACHE: dict[tuple[Money, Money], tuple[Money, ...]] = {}


def _grid(cap: Money, step: Money) -> tuple[Money, ...]:
    key = (cap, step)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    count = int(cap / step)
    if step <= 0 or step * count != cap:
        raise ValueError("cap must be an integer multiple of money grid")
    grid = tuple(step * index for index in range(count + 1))
    _GRID_CACHE[key] = grid
    return grid


def rerun_bid(
    target_key: str,
    bid: Money,
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
) -> bool:
    replaced = tuple(
        replace(candidate, bid=bid) if candidate.key == target_key else candidate
        for candidate in candidates
    )
    if not any(candidate.key == target_key for candidate in replaced):
        raise KeyError(target_key)
    return allocate(replaced, snapshot).won(target_key)


def critical_value(
    target_key: str,
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    money_grid: Money,
) -> Money:
    candidates = tuple(candidates)
    target = next((candidate for candidate in candidates if candidate.key == target_key), None)
    if target is None:
        raise KeyError(target_key)
    grid = _grid(target.base_cap, money_grid)
    lo, hi = 0, len(grid) - 1
    best: Money | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        bid = grid[mid]
        if rerun_bid(target_key, bid, candidates, snapshot):
            best = bid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError("critical value requested for a never-winning contract")
    # Witness check: once lost, must stay lost (T2 prefix property).
    witness_index = hi + 2
    if witness_index < len(grid) and rerun_bid(target_key, grid[witness_index], candidates, snapshot):
        raise MonotonicityViolation(
            f"non-prefix winning set for {target_key}; witness bid={grid[witness_index]}"
        )
    return best


def all_critical_values(
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    money_grid: Money,
) -> dict[str, Money]:
    candidates = tuple(candidates)
    outcome = allocate(candidates, snapshot)
    return {
        key: critical_value(key, candidates, snapshot, money_grid)
        for key in outcome.winners
    }


def build_selection(
    candidates: Iterable[Candidate],
    snapshot: AllocationSnapshot,
    money_grid: Money,
) -> SelectionOutcome:
    candidates = tuple(candidates)
    outcome = allocate(candidates, snapshot)
    try:
        critical = {
            key: critical_value(key, candidates, snapshot, money_grid)
            for key in outcome.winners
        }
    except MonotonicityViolation:
        return SelectionOutcome(outcome, {}, MechanismStatus.MONOTONICITY_VIOLATION)
    return SelectionOutcome(outcome, critical, MechanismStatus.ACTIVE)
