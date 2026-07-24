"""PurpleAir vs EPA (FRM/FEM) colocation -> sensor report calibration.

This is the closest real analogue of the mechanism's core object: a crowdsourced
sensor report vs a trusted reference. From the Barkjohn (2021) US-wide colocation
dataset (24-hr averages, ~12k paired rows), we distil the report error model:

  * relative error r = PA / FRM  (PA reading normalised by the reference truth)
  * quality tiers    : split colocated sensors into honest / low-quality tiers by
                       per-sensor error dispersion (real, not assumed).
  * bias sign split  : fraction of sensors that systematically over- vs under-read.
  * report noise sigma: dispersion of the normalised error (drives report_sigma).
  * contamination    : fraction of paired days whose |relative error| exceeds a
                       gross-outlier threshold => natural "bad sensor" rate that
                       anchors contamination levels.

Everything is a summary statistic; no raw PII, and no counterfactuals invented.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PurpleAirCalibration:
    paired_rows: int
    sensor_count: int
    rel_error_mean: float
    rel_error_std: float
    rel_error_quantiles: list[float]  # p05,p25,p50,p75,p95
    over_read_fraction: float
    honest_sigma: float
    low_quality_sigma: float
    honest_tier_fraction: float
    gross_outlier_rate: float
    contamination_anchor: list[float]  # suggested contamination levels from data
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "source": "Barkjohn 2021 US-wide PurpleAir-vs-FRM colocation (EPA 10.23719/1522388)",
            "paired_rows": self.paired_rows,
            "sensor_count": self.sensor_count,
            "relative_error": {
                "mean": round(self.rel_error_mean, 6),
                "std": round(self.rel_error_std, 6),
                "quantiles_p05_25_50_75_95": [round(x, 6) for x in self.rel_error_quantiles],
            },
            "over_read_fraction": round(self.over_read_fraction, 6),
            "report_sigma": {
                "honest": round(self.honest_sigma, 6),
                "low_quality": round(self.low_quality_sigma, 6),
            },
            "honest_tier_fraction": round(self.honest_tier_fraction, 6),
            "gross_outlier_rate": round(self.gross_outlier_rate, 6),
            "contamination_anchor": [round(x, 4) for x in self.contamination_anchor],
            "provenance": self.provenance,
        }


def _quantiles(values: list[float], ps: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0 for _ in ps]
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, max(0, int(p * (len(ordered) - 1))))] for p in ps]


def build_purpleair_calibration(
    csv_path: Path,
    *,
    outlier_threshold: float = 0.5,
    sensor_channel: str = "PM25PARHcor",
) -> PurpleAirCalibration:
    """Calibrate the report model from colocated PurpleAir-vs-FRM pairs.

    ``sensor_channel`` selects which PurpleAir estimate stands in for the worker
    report. Default ``PM25PARHcor`` is the RH-corrected usable output (what a
    reasonable crowdsensor reports); fall back through cf1/cfatm if absent.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"missing PurpleAir colocation csv: {csv_path}")

    channel_priority = [sensor_channel, "PM25PARHcor", "PM25PAlinearcor", "PM25cf1", "PM25cfatm"]

    rel_errors: list[float] = []
    per_sensor: dict[str, list[float]] = {}
    over_read = 0
    gross_outliers = 0
    used_channel = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fm = row.get("PM25FM", "")
            pa = ""
            for ch in channel_priority:
                candidate = row.get(ch, "")
                if candidate not in ("", None, "NA"):
                    pa = candidate
                    used_channel = used_channel or ch
                    break
            sensor = row.get("ID", "?")
            try:
                fm_v = float(fm)
                pa_v = float(pa)
            except (TypeError, ValueError):
                continue
            if fm_v <= 1.0:  # avoid dividing by near-zero reference (clean-air noise floor)
                continue
            rel = pa_v / fm_v
            if rel <= 0 or rel > 10:  # discard non-physical rows
                continue
            rel_errors.append(rel)
            per_sensor.setdefault(sensor, []).append(rel)
            if rel > 1.0:
                over_read += 1
            if abs(rel - 1.0) > outlier_threshold:
                gross_outliers += 1

    if not rel_errors:
        raise ValueError("no usable paired rows in PurpleAir colocation csv")

    rel_mean = statistics.fmean(rel_errors)
    rel_std = statistics.pstdev(rel_errors)
    rel_q = _quantiles(rel_errors, (0.05, 0.25, 0.5, 0.75, 0.95))
    over_frac = over_read / len(rel_errors)
    gross_rate = gross_outliers / len(rel_errors)

    # Per-sensor dispersion -> quality tiers. Honest tier = lower-dispersion half.
    sensor_disp: list[tuple[str, float]] = []
    for sensor, vals in per_sensor.items():
        if len(vals) < 5:
            continue
        sensor_disp.append((sensor, statistics.pstdev(vals)))
    sensor_disp.sort(key=lambda kv: kv[1])
    n = len(sensor_disp)
    if n >= 2:
        split = n // 2
        honest_sensors = sensor_disp[:split]
        low_sensors = sensor_disp[split:]
        honest_sigma = statistics.fmean([d for _, d in honest_sensors]) if honest_sensors else rel_std
        low_sigma = statistics.fmean([d for _, d in low_sensors]) if low_sensors else rel_std
        honest_fraction = split / n
    else:
        honest_sigma = rel_std
        low_sigma = rel_std
        honest_fraction = 0.5

    # Contamination anchors: keep the experiment's frozen grid {0.1,0.3,0.5} but
    # record the real gross-outlier rate as the empirically-supported low point.
    contamination_anchor = sorted({round(gross_rate, 2), 0.1, 0.3, 0.5})

    return PurpleAirCalibration(
        paired_rows=len(rel_errors),
        sensor_count=len(per_sensor),
        rel_error_mean=rel_mean,
        rel_error_std=rel_std,
        rel_error_quantiles=rel_q,
        over_read_fraction=over_frac,
        honest_sigma=honest_sigma,
        low_quality_sigma=low_sigma,
        honest_tier_fraction=honest_fraction,
        gross_outlier_rate=gross_rate,
        contamination_anchor=contamination_anchor,
        provenance={
            "outlier_threshold_rel": outlier_threshold,
            "reference_channel": "PM25FM (FRM/FEM gravimetric/reference)",
            "sensor_channel": used_channel or sensor_channel,
        },
    )
