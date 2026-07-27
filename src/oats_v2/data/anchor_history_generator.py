from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from .schemas import TraceConfig, canonical_json, decimal_from_float
from .substreams import beta22, normal
from .task_generator import clip01


@dataclass(frozen=True)
class FrozenAnchorBounds:
    cell: int
    version: str
    center: Decimal
    mad: Decimal
    sigma: Decimal
    lower: Decimal
    upper: Decimal


def honest_quality(effort: Decimal) -> Decimal:
    return Decimal("0.6") + Decimal("0.3") * effort


def report_sigma(quality: Decimal) -> Decimal:
    return Decimal("0.25") * (Decimal("1") - quality)


def generate_anchor_history(
    seed: int, config: TraceConfig
) -> tuple[list[dict[str, object]], dict[int, FrozenAnchorBounds]]:
    rows: list[dict[str, object]] = []
    bounds: dict[int, FrozenAnchorBounds] = {}
    version = f"syn-v2-1-anchor-{seed}-v1"
    effort = Decimal("0.5")
    sigma = report_sigma(honest_quality(effort))
    for cell in range(1, config.cell_count + 1):
        cell_reports: list[Decimal] = []
        for index in range(1, config.anchor_count_per_cell + 1):
            theta = decimal_from_float(beta22(seed, "anchor_theta", cell, index))
            report = clip01(theta + sigma * decimal_from_float(normal(seed, "anchor_report_normal", cell, index)))
            holdout = clip01(theta + Decimal("0.01") * decimal_from_float(normal(seed, "anchor_holdout_normal", cell, index)))
            commitment_input = {
                "seed": seed,
                "cell": cell,
                "anchor_index": index,
                "report": report,
                "z": holdout,
                "anchor_version": version,
            }
            commitment = hashlib.sha256(canonical_json(commitment_input).encode("utf-8")).hexdigest()
            rows.append(
                {
                    "seed": seed,
                    "cell": cell,
                    "anchor_index": index,
                    "report": report,
                    "z": holdout,
                    "eligible": True,
                    "anchor_version": version,
                    "commitment": commitment,
                }
            )
            cell_reports.append(report)
        center = median(cell_reports)
        mad = median(abs(value - center) for value in cell_reports)
        scale = max(mad, Decimal("0.01"))
        bounds[cell] = FrozenAnchorBounds(
            cell=cell,
            version=version,
            center=center,
            mad=mad,
            sigma=scale,
            lower=center - Decimal("3") * scale,
            upper=center + Decimal("3") * scale,
        )
    return rows, bounds
