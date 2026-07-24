from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping

from src.oats_v2.data.anchor_history_generator import FrozenAnchorBounds
from src.oats_v2.data.schemas import GBAR_PRECISION, TraceConfig, canonical_json, decimal_text

from .calibration_sampler import CalibrationEstimate, calibration_seed, sample_table
from .error_certificate import build_certificate, hoeffding_radius
from .signing import placeholder_signature


@dataclass(frozen=True)
class CalibratedRoleTable:
    seed: int
    cell: int
    role: str
    rows: tuple[dict[str, object], ...]
    table_hash: str
    certificate: dict[str, object]


def _q(value: Decimal) -> Decimal:
    return value.quantize(GBAR_PRECISION, rounding=ROUND_HALF_EVEN)


def _interval(point: Decimal, radius: Decimal) -> dict[str, Decimal]:
    return {
        "lower": max(Decimal("0"), point - radius),
        "upper": min(Decimal("1"), point + radius),
    }


def build_role_table(
    seed: int,
    cell: int,
    role: str,
    anchor: FrozenAnchorBounds,
    estimates: tuple[CalibrationEstimate, ...],
    config: TraceConfig,
    *,
    data_hash: str,
    code_hash: str,
) -> CalibratedRoleTable:
    base_rows = [
        {
            "seed": seed,
            "cell": cell,
            "public_signal_role": role,
            "effort": estimate.effort,
            "pibar": _q(estimate.pibar_raw),
            "gbar": _q(estimate.gbar_raw),
            "Gbar_precision": GBAR_PRECISION,
            "anchor_version": anchor.version,
        }
        for estimate in estimates
    ]
    table_hash = hashlib.sha256(canonical_json(base_rows).encode("utf-8")).hexdigest()
    sufficient_statistics = {
        "seed": seed,
        "cell": cell,
        "role": role,
        "estimates": [
            {
                "effort": decimal_text(estimate.effort),
                "calibration_seed": calibration_seed(seed, cell, role, estimate.effort),
                "sample_count": estimate.sample_count,
                "pass_count": estimate.pass_count,
                "pibar_raw": decimal_text(estimate.pibar_raw),
                "gbar_raw": decimal_text(estimate.gbar_raw),
            }
            for estimate in estimates
        ],
    }
    calibration_data_hash = hashlib.sha256(
        canonical_json(sufficient_statistics).encode("utf-8")
    ).hexdigest()
    rows = tuple({**row, "table_hash": table_hash} for row in base_rows)
    role_version = f"syn-v2-1-gbar-{seed}-{cell}-{role}-v1"
    signature = placeholder_signature(table_hash, role_version, 1)
    radius = hoeffding_radius(config)
    intervals = {
        decimal_text(estimate.effort): {
            "pibar_lower": _interval(estimate.pibar_raw, radius)["lower"],
            "pibar_upper": _interval(estimate.pibar_raw, radius)["upper"],
            "gbar_lower": _interval(estimate.gbar_raw, radius)["lower"],
            "gbar_upper": _interval(estimate.gbar_raw, radius)["upper"],
        }
        for estimate in estimates
    }
    certificate = build_certificate(
        seed=seed,
        cell=cell,
        role=role,
        table_hash=table_hash,
        data_hash=calibration_data_hash,
        code_hash=code_hash,
        sample_counts=(estimate.sample_count for estimate in estimates),
        intervals=intervals,
        config=config,
        anchor_version=anchor.version,
        signature=signature,
    )
    certificate.update(
        {
            "calibration_input_contract_hash": data_hash,
            "data_hash_semantics": "SHA256 of deterministic calibration seeds and exact sufficient statistics; raw samples are reproducible from the locked runtime",
            "role_version": role_version,
            "anti_rollback_counter": 1,
            "valid_from": "2026-07-15T00:00:00Z",
            "valid_until": "2027-07-15T00:00:00Z",
            "update_schedule": "version bump before use; no retrospective table replacement",
            "score_rule": "S(r,z)=1-(r-z)^2",
            "pass_missing_fallback": {"missing_probabilities": ["0", "0.1", "0.3"], "qmiss": "0"},
            "rounding_rule": "ROUND_HALF_EVEN",
        }
    )
    return CalibratedRoleTable(seed, cell, role, rows, table_hash, certificate)


def calibrate_seed(
    seed: int,
    anchor_bounds: Mapping[int, FrozenAnchorBounds],
    config: TraceConfig,
    *,
    data_hash: str,
    code_hash: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    certificates: list[dict[str, object]] = []
    for cell in range(1, config.cell_count + 1):
        for role in ("honest", "low"):
            estimates = sample_table(seed, cell, role, anchor_bounds[cell], config)
            table = build_role_table(
                seed,
                cell,
                role,
                anchor_bounds[cell],
                estimates,
                config,
                data_hash=data_hash,
                code_hash=code_hash,
            )
            rows.extend(table.rows)
            certificates.append(table.certificate)
    return rows, certificates
