"""REAL-CAL-V1 Gbar calibration.

Produces continuation tables (pibar, gbar) per (cell, role, effort) using the SAME
report model and score function as the REAL-CAL generator, so the published Gbar is
a faithful expectation of the realized report distribution (unlike the SYN reuse,
which mixed the SYN report model with the realcal generator). The discriminating
Gaussian score makes Gbar rise meaningfully with effort, restoring the effort
incentive that the mechanism is designed to create.

Same continuation-table schema and epsilon-certificate machinery as SYN-V2-1, so
the existing trace_loader / contract builder consume it unchanged.
"""

from __future__ import annotations

import hashlib
import random
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping

from ..calibration.error_certificate import build_certificate, hoeffding_radius
from ..calibration.signing import placeholder_signature
from ..data.anchor_history_generator import FrozenAnchorBounds
from ..data.schemas import (
    EFFORTS,
    GBAR_PRECISION,
    TraceConfig,
    canonical_json,
    decimal_text,
)
from .config import RealCalConfig
from .score import gaussian_score, report_sigma

ROLE_IDS = {"honest": 0, "low": 1}
EFFORT_IDS = {Decimal("0"): 0, Decimal("0.5"): 1, Decimal("1"): 2}
# Disjoint calibration RNG namespace from SYN (which starts at 800_000_000).
REALCAL_CAL_BASE = 900_000_000


def _beta_params(mean: float, std: float) -> tuple[float, float]:
    var = std * std
    max_var = mean * (1.0 - mean)
    if var <= 0 or var >= max_var:
        var = max_var * 0.5
    k = mean * (1.0 - mean) / var - 1.0
    return max(1e-3, mean * k), max(1e-3, (1.0 - mean) * k)


def _cal_seed(seed_index: int, cell: int, role: str, effort: Decimal) -> int:
    return (
        REALCAL_CAL_BASE
        + 1_000_000 * seed_index
        + 10_000 * cell
        + 100 * ROLE_IDS[role]
        + EFFORT_IDS[effort]
    )


def _q(value: Decimal) -> Decimal:
    return value.quantize(GBAR_PRECISION, rounding=ROUND_HALF_EVEN)


def _role_base_cv(cfg: RealCalConfig, role: str) -> Decimal:
    return cfg.low_quality_report_cv if role == "low" else cfg.honest_report_cv


def _sample_estimate(
    cal_seed: int,
    role: str,
    effort: Decimal,
    bounds: FrozenAnchorBounds,
    cfg: RealCalConfig,
    a: float,
    b: float,
    n: int,
) -> tuple[int, Decimal, Decimal]:
    source = random.Random(cal_seed)
    cv = float(_role_base_cv(cfg, role))
    reduction = 1.0 - 0.6 * float(effort)
    if reduction < 0.0:
        reduction = 0.0
    lower = float(bounds.lower)
    upper = float(bounds.upper)
    passed = 0
    score_sum = 0.0
    for _ in range(n):
        theta = source.betavariate(a, b)
        sigma = cv * theta * reduction
        report = min(1.0, max(0.0, theta + sigma * source.gauss(0.0, 1.0)))
        holdout = min(1.0, max(0.0, theta + 0.01 * source.gauss(0.0, 1.0)))
        if lower <= report <= upper:
            passed += 1
            score_sum += float(gaussian_score(Decimal(repr(report)), Decimal(repr(holdout))))
    pibar = Decimal(repr(passed / n))
    gbar = Decimal(repr(score_sum / n))
    return passed, pibar, gbar


def calibrate_realcal_seed(
    seed: int,
    seed_index: int,
    anchor_bounds: Mapping[int, FrozenAnchorBounds],
    cfg: RealCalConfig,
    calib_config: TraceConfig,
    *,
    data_hash: str,
    code_hash: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    a, b = _beta_params(float(cfg.theta_mean), float(cfg.theta_std))
    n = calib_config.calibration_n
    radius = hoeffding_radius(calib_config)
    rows: list[dict[str, object]] = []
    certificates: list[dict[str, object]] = []

    for cell in range(1, cfg.cell_count + 1):
        anchor = anchor_bounds[cell]
        for role in ("honest", "low"):
            estimates: list[tuple[Decimal, int, Decimal, Decimal]] = []
            for effort in EFFORTS:
                cal_seed = _cal_seed(seed_index, cell, role, effort)
                passed, pibar, gbar = _sample_estimate(
                    cal_seed, role, effort, anchor, cfg, a, b, n
                )
                estimates.append((effort, passed, pibar, gbar))

            base_rows = [
                {
                    "seed": seed,
                    "cell": cell,
                    "public_signal_role": role,
                    "effort": effort,
                    "pibar": _q(pibar),
                    "gbar": _q(gbar),
                    "Gbar_precision": GBAR_PRECISION,
                    "anchor_version": anchor.version,
                }
                for effort, _passed, pibar, gbar in estimates
            ]
            table_hash = hashlib.sha256(canonical_json(base_rows).encode("utf-8")).hexdigest()
            rows.extend({**row, "table_hash": table_hash} for row in base_rows)

            role_version = f"real-cal-v1-gbar-{seed}-{cell}-{role}-v1"
            signature = placeholder_signature(table_hash, role_version, 1)
            intervals = {
                decimal_text(effort): {
                    "pibar_lower": max(Decimal("0"), pibar - radius),
                    "pibar_upper": min(Decimal("1"), pibar + radius),
                    "gbar_lower": max(Decimal("0"), gbar - radius),
                    "gbar_upper": min(Decimal("1"), gbar + radius),
                }
                for effort, _passed, pibar, gbar in estimates
            }
            certificate = build_certificate(
                seed=seed,
                cell=cell,
                role=role,
                table_hash=table_hash,
                data_hash=data_hash,
                code_hash=code_hash,
                sample_counts=(passed for _e, passed, _p, _g in estimates),
                intervals=intervals,
                config=calib_config,
                anchor_version=anchor.version,
                signature=signature,
            )
            certificate.update(
                {
                    "role_version": role_version,
                    "anti_rollback_counter": 1,
                    "score_rule": "S(r,z)=exp(-(r-z)^2/(2*tau^2)), tau=0.1",
                    "report_model": "realcal multiplicative cv noise, effort noise reduction 0.6*e",
                    "rounding_rule": "ROUND_HALF_EVEN",
                    "dataset_id": "REAL-CAL-V1",
                }
            )
            certificates.append(certificate)
    return rows, certificates
