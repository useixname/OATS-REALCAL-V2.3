from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.oats_v2.realcal.config import load_realcal_config

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data_real" / "REAL-CAL-V1" / "calibration_profile.json"


requires_profile = pytest.mark.skipif(
    not PROFILE.exists(), reason="REAL-CAL-V1 profile not built yet"
)


@requires_profile
def test_profile_loads_and_is_nondegenerate() -> None:
    cfg = load_realcal_config(PROFILE)
    assert cfg.profile_hash
    assert Decimal("0.01") <= cfg.availability_probability <= Decimal("0.95")
    assert len(cfg.arrival_intensity) == cfg.horizon
    assert Decimal("0") < cfg.theta_mean < Decimal("1")
    assert cfg.honest_report_cv > 0
    assert cfg.low_quality_report_cv > 0
    # low-quality sensors should be noisier than honest ones
    assert cfg.low_quality_report_cv >= cfg.honest_report_cv


@requires_profile
def test_arrival_intensity_mean_normalised() -> None:
    cfg = load_realcal_config(PROFILE)
    mean = sum(cfg.arrival_intensity, Decimal("0")) / Decimal(len(cfg.arrival_intensity))
    assert abs(mean - Decimal("1")) < Decimal("0.05")


@requires_profile
def test_frozen_constants_match_syn() -> None:
    cfg = load_realcal_config(PROFILE)
    assert cfg.worker_count == 500
    assert cfg.horizon == 1000
    assert cfg.cell_count == 100
    assert cfg.public_base_cap == Decimal("3.0")
    assert cfg.task_value_low == Decimal("0.5")
    assert cfg.task_value_high == Decimal("1.5")


@requires_profile
def test_stratum_fractions_sum_to_one() -> None:
    cfg = load_realcal_config(PROFILE)
    total = sum(cfg.stratum_fractions.values(), Decimal("0"))
    assert abs(total - Decimal("1")) < Decimal("0.001")


@requires_profile
def test_arrival_multiplier_bounds() -> None:
    cfg = load_realcal_config(PROFILE)
    vals = [cfg.arrival_multiplier(s) for s in range(1, cfg.horizon + 1)]
    assert min(vals) > 0
    assert max(vals) >= min(vals)
