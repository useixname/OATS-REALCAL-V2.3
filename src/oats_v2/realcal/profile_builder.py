"""Combine the three real datasets into one frozen REAL-CAL-V1 profile.

The profile maps every SYN-V2-1 generator parameter that the experiment depends
on to a real-data-derived value, while KEEPING FROZEN the structural constants
that the mechanism/proofs rely on (money grid, effort menu, gamma grid, budget
ratios, worker/task counts, capacities, deadline structure). Only the *input
distributions* change; the mechanism code is untouched.

Frozen-by-design (NOT recalibrated), to preserve Phase-4A comparability:
  - EFFORTS, GAMMAS, MONEY_GRID, public_base_cap
  - task value band [0.5, 1.5], capacities {3,5,10}
  - worker_count=500, horizon=1000, cell_count=100
  - c_i support {0.5,1.0,2.0}

Recalibrated from real data:
  - availability_probability + per-slot arrival intensity  (T-Drive)
  - theta difficulty distribution + missing rate + anchor spread (Beijing air)
  - report noise sigma per quality tier, bias-sign split, quality-tier fractions,
    contamination anchor (PurpleAir/EPA)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import REALCAL_DATASET_ID, REALCAL_PROFILE_VERSION
from .airquality_preprocess import build_airquality_calibration
from .purpleair_preprocess import build_purpleair_calibration
from .tdrive_preprocess import build_tdrive_calibration


FROZEN_CONSTANTS = {
    "efforts": ["0", "0.5", "1"],
    "gammas": ["0", "0.1", "0.3", "0.5", "0.8", "1.0"],
    "money_grid": "0.001",
    "public_base_cap": "3.0",
    "task_value_band": ["0.5", "1.5"],
    "capacities": [3, 5, 10],
    "worker_count": 500,
    "horizon": 1000,
    "cell_count": 100,
    "c_i_support": ["0.5", "1.0", "2.0"],
    "budget_ratios": ["0.10", "0.25", "0.50"],
    "rationale": "kept identical to SYN-V2-1 so proofs, capacities and comparability hold",
}


@dataclass
class RealCalProfile:
    dataset_id: str
    profile_version: str
    frozen_constants: dict
    availability: dict
    task_and_anchor: dict
    report_model: dict
    stratum_fractions: dict
    profile_hash: str = ""
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("profile_hash", None)
        return d


def _stratum_fractions_from(honest_tier_fraction: float) -> dict:
    """Blend real quality-tier evidence with the frozen 4-way stratum structure.

    PurpleAir gives an honest-vs-low split among *benign* sensors. Malicious /
    camouflage strata have no real ground truth (no labelled attackers), so their
    fractions stay at the frozen designed-stress values and are declared as such.
    """
    benign_mass = 0.80  # honest + low-quality share (frozen)
    honest = round(benign_mass * honest_tier_fraction, 4)
    low = round(benign_mass - honest, 4)
    return {
        "honest": honest,
        "low-quality": low,
        "malicious": 0.10,
        "camouflage": 0.10,
        "note": (
            "honest/low split calibrated from PurpleAir per-sensor dispersion; "
            "malicious/camouflage kept at frozen designed-stress levels (no real "
            "attacker labels exist) and reported as injected stress, not real prevalence"
        ),
    }


def build_profile(
    *,
    tdrive_dir: Path,
    airquality_dir: Path,
    purpleair_csv: Path,
    tdrive_max_files: int | None = None,
) -> RealCalProfile:
    td = build_tdrive_calibration(tdrive_dir, max_files=tdrive_max_files)
    aq = build_airquality_calibration(airquality_dir)
    pa = build_purpleair_calibration(purpleair_csv)

    availability = {
        "availability_probability": round(td.availability_probability, 6),
        "arrival_intensity": [round(x, 6) for x in td.arrival_intensity],
        "peak_ratio": round(td.peak_ratio, 4),
        "trough_ratio": round(td.trough_ratio, 4),
        "raw": td.to_json(),
    }

    task_and_anchor = {
        "theta_mean": round(aq.theta_mean, 6),
        "theta_std": round(aq.theta_std, 6),
        "theta_quantiles_p10_25_50_75_90": [round(x, 6) for x in aq.theta_quantiles],
        "missing_rate": round(aq.missing_rate, 6),
        "anchor_mad_median": round(aq.anchor_mad_median, 6),
        "anchor_mad_quantiles_p10_25_50_75_90": [round(x, 6) for x in aq.anchor_mad_quantiles],
        "value_band": ["0.5", "1.5"],
        "raw": aq.to_json(),
    }

    # Convert relative-error dispersion into report_sigma on the [0,1] score scale.
    # PurpleAir sigma is on the ratio scale; we normalise by the honest ratio mean
    # to obtain a comparable coefficient of variation used as report noise.
    ref = max(pa.rel_error_mean, 1e-6)
    honest_cv = pa.honest_sigma / ref
    low_cv = pa.low_quality_sigma / ref
    report_model = {
        "honest_report_cv": round(honest_cv, 6),
        "low_quality_report_cv": round(low_cv, 6),
        "over_read_fraction": round(pa.over_read_fraction, 6),
        "bias_sign_positive_fraction": round(pa.over_read_fraction, 6),
        "gross_outlier_rate": round(pa.gross_outlier_rate, 6),
        "contamination_anchor": [round(x, 4) for x in pa.contamination_anchor],
        "raw": pa.to_json(),
    }

    stratum = _stratum_fractions_from(pa.honest_tier_fraction)

    profile = RealCalProfile(
        dataset_id=REALCAL_DATASET_ID,
        profile_version=REALCAL_PROFILE_VERSION,
        frozen_constants=FROZEN_CONSTANTS,
        availability=availability,
        task_and_anchor=task_and_anchor,
        report_model=report_model,
        stratum_fractions=stratum,
        provenance={
            "tdrive_files": td.fleet_size,
            "airquality_stations": aq.station_count,
            "purpleair_rows": pa.paired_rows,
            "claim_ceiling": "real-data-calibrated semi-synthetic evidence only",
            "forbidden": [
                "REAL_S2 / privacy / cryptographic security",
                "regret / sqrt(T) / vanishing regret",
                "real-world deployment generalization",
                "real malicious-worker prevalence (attackers are injected stress)",
            ],
        },
    )
    payload = json.dumps(profile.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    profile.profile_hash = hashlib.sha256(payload).hexdigest()
    return profile


def write_profile(profile: RealCalProfile, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = profile.to_json()
    doc["profile_hash"] = profile.profile_hash
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
