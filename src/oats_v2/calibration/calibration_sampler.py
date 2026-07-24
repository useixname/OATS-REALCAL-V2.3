from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from src.oats_v2.data.anchor_history_generator import FrozenAnchorBounds, report_sigma
from src.oats_v2.data.schemas import EFFORTS, FORMAL_SEEDS, TraceConfig


ROLE_IDS = {"honest": 0, "low": 1}
EFFORT_IDS = {Decimal("0"): 0, Decimal("0.5"): 1, Decimal("1"): 2}


@dataclass(frozen=True)
class CalibrationEstimate:
    effort: Decimal
    sample_count: int
    pass_count: int
    pibar_raw: Decimal
    gbar_raw: Decimal


def formal_seed_index(seed: int) -> int:
    if seed not in FORMAL_SEEDS:
        raise ValueError("SYN-V2-1 calibration seed must be in the frozen formal seed list")
    return FORMAL_SEEDS.index(seed)


def calibration_seed(seed: int, cell: int, role: str, effort: Decimal) -> int:
    return (
        800_000_000
        + 1_000_000 * formal_seed_index(seed)
        + 10_000 * cell
        + 100 * ROLE_IDS[role]
        + EFFORT_IDS[effort]
    )


def role_quality(role: str, effort: Decimal) -> Decimal:
    if role == "honest":
        return Decimal("0.6") + Decimal("0.3") * effort
    if role == "low":
        return Decimal("0.4") + Decimal("0.2") * effort
    raise ValueError(role)


def sample_estimate(
    seed: int,
    cell: int,
    role: str,
    effort: Decimal,
    bounds: FrozenAnchorBounds,
    config: TraceConfig,
) -> CalibrationEstimate:
    source = random.Random(calibration_seed(seed, cell, role, effort))
    sigma = float(report_sigma(role_quality(role, effort)))
    lower = float(bounds.lower)
    upper = float(bounds.upper)
    passed = 0
    score_sum = 0.0
    for _ in range(config.calibration_n):
        theta = source.betavariate(2.0, 2.0)
        report = min(1.0, max(0.0, theta + sigma * source.gauss(0.0, 1.0)))
        holdout = min(1.0, max(0.0, theta + 0.01 * source.gauss(0.0, 1.0)))
        if lower <= report <= upper:
            passed += 1
            score_sum += 1.0 - (report - holdout) ** 2
    return CalibrationEstimate(
        effort=effort,
        sample_count=config.calibration_n,
        pass_count=passed,
        pibar_raw=Decimal(repr(passed / config.calibration_n)),
        gbar_raw=Decimal(repr(score_sum / config.calibration_n)),
    )


def sample_table(
    seed: int, cell: int, role: str, bounds: FrozenAnchorBounds, config: TraceConfig
) -> tuple[CalibrationEstimate, ...]:
    return tuple(sample_estimate(seed, cell, role, effort, bounds, config) for effort in EFFORTS)
