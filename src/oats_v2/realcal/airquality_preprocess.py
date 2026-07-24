"""Beijing multi-site air quality -> task / anchor calibration.

The 12 nationally-controlled monitoring stations are the natural analogue of the
experiment's "frozen historical anchor" plus the sensing "truth". From the hourly
PM2.5 series we distil:

  * task difficulty theta:   spatial disagreement between stations at a given hour,
                             normalised to [0,1] (harder = stations disagree more).
  * task value V:            information value proxy (variance of the field), mapped
                             into the frozen [0.5, 1.5] band.
  * anchor spread:           per-station robust spread (MAD) of readings, used to
                             calibrate anchor interval half-widths.
  * missingness:             real NA rate in the raw station feed.

Nothing here fabricates counterfactuals; it only summarises the real field.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AirQualityCalibration:
    station_count: int
    hours: int
    theta_mean: float
    theta_std: float
    theta_quantiles: list[float]  # [p10,p25,p50,p75,p90]
    value_low: float
    value_high: float
    anchor_mad_median: float
    anchor_mad_quantiles: list[float]
    missing_rate: float
    pm25_median: float
    pm25_p95: float
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "source": "Beijing Multi-Site Air Quality (UCI 501), 12 stations 2013-2017",
            "station_count": self.station_count,
            "hours": self.hours,
            "theta": {
                "mean": round(self.theta_mean, 6),
                "std": round(self.theta_std, 6),
                "quantiles_p10_25_50_75_90": [round(x, 6) for x in self.theta_quantiles],
            },
            "value_band": [round(self.value_low, 6), round(self.value_high, 6)],
            "anchor_mad": {
                "median": round(self.anchor_mad_median, 6),
                "quantiles_p10_25_50_75_90": [round(x, 6) for x in self.anchor_mad_quantiles],
            },
            "missing_rate": round(self.missing_rate, 6),
            "pm25_median": round(self.pm25_median, 4),
            "pm25_p95": round(self.pm25_p95, 4),
            "provenance": self.provenance,
        }


def _quantiles(values: list[float], ps: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0 for _ in ps]
    ordered = sorted(values)
    out = []
    for p in ps:
        idx = min(len(ordered) - 1, max(0, int(p * (len(ordered) - 1))))
        out.append(ordered[idx])
    return out


def build_airquality_calibration(station_dir: Path) -> AirQualityCalibration:
    files = sorted(station_dir.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(f"no PRSA station files under {station_dir}")

    # hour_key -> {station: pm25}
    by_hour: dict[tuple[int, int, int, int], dict[str, float]] = {}
    total_cells = 0
    missing_cells = 0
    all_pm25: list[float] = []
    station_series: dict[str, list[float]] = {}

    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                station = row.get("station", path.stem)
                total_cells += 1
                raw = row.get("PM2.5", "NA")
                if raw in ("NA", "", None):
                    missing_cells += 1
                    continue
                try:
                    pm = float(raw)
                except ValueError:
                    missing_cells += 1
                    continue
                key = (int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"]))
                by_hour.setdefault(key, {})[station] = pm
                all_pm25.append(pm)
                station_series.setdefault(station, []).append(pm)

    # theta = normalised cross-station disagreement per hour (higher => harder task).
    thetas: list[float] = []
    field_vars: list[float] = []
    for readings in by_hour.values():
        vals = list(readings.values())
        if len(vals) < 3:
            continue
        spread = statistics.pstdev(vals)
        mean_val = statistics.fmean(vals)
        field_vars.append(spread)
        # coefficient of variation squashed to [0,1] via a soft cap.
        cv = spread / mean_val if mean_val > 0 else 0.0
        thetas.append(min(1.0, cv))

    theta_mean = statistics.fmean(thetas) if thetas else 0.0
    theta_std = statistics.pstdev(thetas) if len(thetas) > 1 else 0.0
    theta_q = _quantiles(thetas, (0.1, 0.25, 0.5, 0.75, 0.9))

    # value proxy: map field variance quantiles into frozen [0.5, 1.5].
    var_q = _quantiles(field_vars, (0.1, 0.9)) if field_vars else [0.0, 1.0]
    value_low, value_high = 0.5, 1.5  # keep frozen band; V within it via generator

    # anchor MAD per station (robust spread), normalised by median PM to be scale-free.
    station_mads: list[float] = []
    for series in station_series.values():
        if len(series) < 10:
            continue
        med = statistics.median(series)
        mad = statistics.median([abs(x - med) for x in series])
        norm = mad / med if med > 0 else 0.0
        station_mads.append(norm)
    anchor_mad_median = statistics.median(station_mads) if station_mads else 0.0
    anchor_mad_q = _quantiles(station_mads, (0.1, 0.25, 0.5, 0.75, 0.9))

    missing_rate = missing_cells / total_cells if total_cells else 0.0
    pm_median = statistics.median(all_pm25) if all_pm25 else 0.0
    pm_p95 = _quantiles(all_pm25, (0.95,))[0] if all_pm25 else 0.0

    return AirQualityCalibration(
        station_count=len(files),
        hours=len(by_hour),
        theta_mean=theta_mean,
        theta_std=theta_std,
        theta_quantiles=theta_q,
        value_low=value_low,
        value_high=value_high,
        anchor_mad_median=anchor_mad_median,
        anchor_mad_quantiles=anchor_mad_q,
        missing_rate=missing_rate,
        pm25_median=pm_median,
        pm25_p95=pm_p95,
        provenance={
            "field_variance_p10_p90": [round(v, 4) for v in var_q],
            "stations": [p.stem.replace("PRSA_Data_", "").split("_")[0] for p in files],
        },
    )
