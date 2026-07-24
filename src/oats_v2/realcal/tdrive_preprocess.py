"""T-Drive taxi GPS -> worker availability / arrival calibration.

Maps the one-week Beijing taxi trace onto the experiment's HORIZON=1000 slot
grid inside the central-Beijing bounding box. For each slot we count how many
distinct taxis emit at least one GPS ping (i.e. are "active"). Two calibration
targets come out of this:

  * availability_probability : mean fraction of the fleet active per slot
  * arrival_intensity_grid   : per-slot active-taxi counts, normalised, giving a
                               realistic (non-uniform) diurnal arrival curve.

Only edge statistics are exported; raw trajectories are never copied into a
trace. All heavy lifting stays in this offline preprocessing step.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Central Beijing bounding box (roughly inside the 5th ring road), matching the
# Beijing air-quality station footprint so the two datasets share geography.
LON_MIN, LON_MAX = 116.20, 116.55
LAT_MIN, LAT_MAX = 39.80, 40.05

HORIZON = 1000
# One week of taxi data (Feb 2-8 2008) compressed onto HORIZON slots.
WEEK_START = datetime(2008, 2, 2, 0, 0, 0)
WEEK_END = datetime(2008, 2, 9, 0, 0, 0)


@dataclass
class TDriveCalibration:
    fleet_size: int
    slots: int
    availability_probability: float
    active_counts: list[int]
    arrival_intensity: list[float]  # normalised so mean == 1.0
    peak_ratio: float
    trough_ratio: float
    total_pings: int
    in_box_pings: int
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "source": "T-Drive (Microsoft Research), Beijing taxis Feb 2008",
            "fleet_size": self.fleet_size,
            "slots": self.slots,
            "availability_probability": round(self.availability_probability, 6),
            "arrival_intensity": [round(x, 6) for x in self.arrival_intensity],
            "active_counts_summary": {
                "min": min(self.active_counts),
                "max": max(self.active_counts),
                "mean": round(statistics.fmean(self.active_counts), 4),
                "median": statistics.median(self.active_counts),
            },
            "peak_ratio": round(self.peak_ratio, 4),
            "trough_ratio": round(self.trough_ratio, 4),
            "total_pings": self.total_pings,
            "in_box_pings": self.in_box_pings,
            "provenance": self.provenance,
        }


def _iter_pings(taxi_file: Path) -> Iterator[tuple[datetime, float, float]]:
    with taxi_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 4:
                continue
            _, ts, lon_s, lat_s = row
            try:
                lon = float(lon_s)
                lat = float(lat_s)
                when = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
            except (ValueError, IndexError):
                continue
            yield when, lon, lat


def _slot_for(when: datetime) -> int | None:
    if when < WEEK_START or when >= WEEK_END:
        return None
    span = (WEEK_END - WEEK_START).total_seconds()
    frac = (when - WEEK_START).total_seconds() / span
    slot = int(frac * HORIZON) + 1
    return min(max(slot, 1), HORIZON)


def build_tdrive_calibration(extracted_dir: Path, *, max_files: int | None = None) -> TDriveCalibration:
    files = sorted(extracted_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no T-Drive taxi files under {extracted_dir}")

    slot_active: list[set[int]] = [set() for _ in range(HORIZON + 1)]
    total_pings = 0
    in_box_pings = 0
    for taxi_file in files:
        taxi_id = int(taxi_file.stem) if taxi_file.stem.isdigit() else hash(taxi_file.stem)
        for when, lon, lat in _iter_pings(taxi_file):
            total_pings += 1
            if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
                continue
            slot = _slot_for(when)
            if slot is None:
                continue
            in_box_pings += 1
            slot_active[slot].add(taxi_id)

    active_counts = [len(slot_active[s]) for s in range(1, HORIZON + 1)]
    fleet_size = len(files)
    mean_active = statistics.fmean(active_counts) if active_counts else 0.0
    availability = mean_active / fleet_size if fleet_size else 0.0

    # Normalised arrival intensity (mean == 1.0), flooring empty slots gently so a
    # slot is never impossible (keeps online policy from starving).
    floor = max(1.0, mean_active * 0.05)
    smoothed = [max(float(c), floor) for c in active_counts]
    grid_mean = statistics.fmean(smoothed)
    arrival_intensity = [x / grid_mean for x in smoothed]

    peak = max(arrival_intensity)
    trough = min(arrival_intensity)

    return TDriveCalibration(
        fleet_size=fleet_size,
        slots=HORIZON,
        availability_probability=availability,
        active_counts=active_counts,
        arrival_intensity=arrival_intensity,
        peak_ratio=peak,
        trough_ratio=trough,
        total_pings=total_pings,
        in_box_pings=in_box_pings,
        provenance={
            "bbox": [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX],
            "files_used": fleet_size,
            "week_start": WEEK_START.isoformat(),
            "week_end": WEEK_END.isoformat(),
        },
    )
