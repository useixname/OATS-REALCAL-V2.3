from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from typing import Mapping, Sequence

from .types import D, Money, ReportMode


def quantize_ceiling(value: Money, grid: Money) -> Money:
    value, grid = D(value), D(grid)
    if value < 0 or grid <= 0:
        raise ValueError("value must be nonnegative and grid positive")
    units = (value / grid).to_integral_value(rounding=ROUND_CEILING)
    return units * grid


def quantize_half_even(value: Money, precision: Money) -> Money:
    value, precision = D(value), D(precision)
    if precision <= 0:
        raise ValueError("precision must be positive")
    units = (value / precision).to_integral_value(rounding=ROUND_HALF_EVEN)
    return units * precision


@dataclass(frozen=True)
class ContinuationTable:
    values: Mapping[str, Money]
    precision: Money
    version: str
    signed: bool = True

    def __post_init__(self) -> None:
        precision = D(self.precision)
        object.__setattr__(self, "precision", precision)
        normalized = {str(e): quantize_half_even(D(v), precision) for e, v in self.values.items()}
        object.__setattr__(self, "values", normalized)
        if not self.signed or not self.version or precision <= 0:
            raise ValueError("invalid continuation table")
        if any(v < 0 for v in normalized.values()):
            raise ValueError("negative continuation value")


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    effort_levels: tuple[str, ...]
    effort_basis: Mapping[str, Decimal]
    continuation: ContinuationTable
    base_cap: Money
    money_grid: Money
    score_cap: Money
    missing_score: Decimal = Decimal("0")
    report_mode: ReportMode = ReportMode.ATTESTED_POINT_ESTIMATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_cap", D(self.base_cap))
        object.__setattr__(self, "money_grid", D(self.money_grid))
        object.__setattr__(self, "score_cap", D(self.score_cap))
        object.__setattr__(self, "missing_score", D(self.missing_score))
        basis = {str(e): D(v) for e, v in self.effort_basis.items()}
        object.__setattr__(self, "effort_basis", basis)
        if set(self.effort_levels) != set(basis) or set(self.effort_levels) != set(self.continuation.values):
            raise ValueError("effort domains disagree")
        if not self.effort_levels or min(basis.values()) <= 0:
            raise ValueError("finite effort menu requires k_min > 0")
        if self.base_cap < 0 or self.money_grid <= 0 or self.score_cap < 0:
            raise ValueError("invalid caps/grid")
        if self.base_cap % self.money_grid != 0:
            raise ValueError("base cap must lie on money grid")
        if not (Decimal("0") <= self.missing_score <= Decimal("1")):
            raise ValueError("missing score outside [0,1]")

    def effective_cost(self, scalar_cost: Decimal) -> tuple[Money, tuple[str, ...]]:
        scalar_cost = D(scalar_cost)
        if scalar_cost < 0:
            raise ValueError("negative scalar cost")
        utilities = {
            e: scalar_cost * self.effort_basis[e] - self.continuation.values[e]
            for e in self.effort_levels
        }
        d = min(utilities.values())
        best = tuple(e for e in self.effort_levels if utilities[e] == d)
        if d < 0 or d > self.base_cap:
            raise ValueError("TYPE_MODEL_UNSUPPORTED")
        return d, best

    def truthful_bid(self, scalar_cost: Decimal) -> Money:
        d, _ = self.effective_cost(scalar_cost)
        return quantize_ceiling(d, self.money_grid)

    def grid_values(self) -> tuple[Money, ...]:
        count = int(self.base_cap / self.money_grid)
        return tuple(self.money_grid * i for i in range(count + 1))


def best_response(
    scalar_cost: Decimal,
    effort_levels: Sequence[str],
    effort_basis: Mapping[str, Decimal],
    continuation: Mapping[str, Decimal],
) -> tuple[Money, tuple[str, ...]]:
    scalar_cost = D(scalar_cost)
    values = {
        e: scalar_cost * D(effort_basis[e]) - D(continuation[e])
        for e in effort_levels
    }
    d = min(values.values())
    return d, tuple(e for e in effort_levels if values[e] == d)

