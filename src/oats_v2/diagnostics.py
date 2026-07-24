from __future__ import annotations

from decimal import Decimal
from typing import Callable, Sequence

from .types import D


MC_UNAVAILABLE = "MC_UNAVAILABLE"


def marginal_contribution(
    reports: Sequence[Decimal],
    index: int,
    outcome: Decimal | None,
    loss: Callable[[Sequence[Decimal], Decimal], Decimal],
) -> Decimal | str:
    if outcome is None or len(reports) < 2 or index < 0 or index >= len(reports):
        return MC_UNAVAILABLE
    full = D(loss(reports, D(outcome)))
    leave_one_out = D(loss(tuple(report for position, report in enumerate(reports) if position != index), D(outcome)))
    return max(Decimal("0"), leave_one_out - full)
