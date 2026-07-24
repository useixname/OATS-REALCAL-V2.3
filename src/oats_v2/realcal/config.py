"""Load the frozen REAL-CAL-V1 profile into a generator-ready config.

This is the bridge between the offline calibration profile (JSON) and a future
REAL-CAL trace generator. It keeps all frozen-by-design constants identical to
SYN-V2-1's ``TraceConfig`` while exposing the real-calibrated distributions as
plain Python objects the generator can sample from.

Deterministic: the profile hash is carried through so any generated trace can be
bound back to the exact calibration inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class RealCalConfig:
    profile_version: str
    profile_hash: str
    # Real-calibrated
    availability_probability: Decimal
    arrival_intensity: tuple[Decimal, ...]  # length == horizon, mean ~= 1
    theta_mean: Decimal
    theta_std: Decimal
    missing_rate: Decimal
    anchor_mad_median: Decimal
    honest_report_cv: Decimal
    low_quality_report_cv: Decimal
    bias_sign_positive_fraction: Decimal
    contamination_anchor: tuple[Decimal, ...]
    stratum_fractions: dict
    # Frozen-by-design (mirrors TraceConfig)
    horizon: int = 1000
    worker_count: int = 500
    cell_count: int = 100
    public_base_cap: Decimal = Decimal("3.0")
    task_value_low: Decimal = Decimal("0.5")
    task_value_high: Decimal = Decimal("1.5")
    rho0: Decimal = Decimal("0.5")
    raw: dict = field(default_factory=dict)

    def arrival_multiplier(self, slot: int) -> Decimal:
        """Real diurnal arrival multiplier for a 1-indexed slot (mean ~= 1)."""
        if not self.arrival_intensity:
            return Decimal("1")
        idx = min(max(slot - 1, 0), len(self.arrival_intensity) - 1)
        return self.arrival_intensity[idx]

    def report_cv(self, stratum: str) -> Decimal:
        if stratum == "low-quality":
            return self.low_quality_report_cv
        return self.honest_report_cv


def load_realcal_config(profile_path: Path) -> RealCalConfig:
    doc = json.loads(profile_path.read_text(encoding="utf-8"))
    avail = doc["availability"]
    task = doc["task_and_anchor"]
    report = doc["report_model"]

    def dec(x) -> Decimal:
        return Decimal(str(x))

    return RealCalConfig(
        profile_version=doc["profile_version"],
        profile_hash=doc.get("profile_hash", ""),
        availability_probability=dec(avail["availability_probability"]),
        arrival_intensity=tuple(dec(x) for x in avail["arrival_intensity"]),
        theta_mean=dec(task["theta_mean"]),
        theta_std=dec(task["theta_std"]),
        missing_rate=dec(task["missing_rate"]),
        anchor_mad_median=dec(task["anchor_mad_median"]),
        honest_report_cv=dec(report["honest_report_cv"]),
        low_quality_report_cv=dec(report["low_quality_report_cv"]),
        bias_sign_positive_fraction=dec(report["bias_sign_positive_fraction"]),
        contamination_anchor=tuple(dec(x) for x in report["contamination_anchor"]),
        stratum_fractions={
            k: dec(v)
            for k, v in doc["stratum_fractions"].items()
            if k in {"honest", "low-quality", "malicious", "camouflage"}
        },
        raw=doc,
    )
