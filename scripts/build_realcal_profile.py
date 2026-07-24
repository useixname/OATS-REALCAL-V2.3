#!/usr/bin/env python3
"""Preprocess the three real datasets into the frozen REAL-CAL-V1 profile.

Reads:
  data_real/raw/tdrive/extracted/*.txt
  data_real/raw/beijing_air/extracted/PRSA_Data_20130301-20170228/*.csv
  data_real/raw/purpleair_epa/extracted/Data_DevelopmentUSPAcorrection_210408/Full24hrdataset.csv

Writes:
  data_real/REAL-CAL-V1/calibration_profile.json
  data_real/REAL-CAL-V1/CALIBRATION_VALIDATION.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_v2.realcal.profile_builder import build_profile, write_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data_real" / "raw")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data_real" / "REAL-CAL-V1")
    parser.add_argument("--tdrive-max-files", type=int, default=0, help="0 = use all taxi files")
    args = parser.parse_args()

    tdrive_dir = args.raw_root / "tdrive" / "extracted"
    airquality_dir = args.raw_root / "beijing_air" / "extracted" / "PRSA_Data_20130301-20170228"
    purpleair_csv = (
        args.raw_root
        / "purpleair_epa"
        / "extracted"
        / "Data_DevelopmentUSPAcorrection_210408"
        / "Full24hrdataset.csv"
    )

    max_files = None if args.tdrive_max_files == 0 else args.tdrive_max_files
    print(f"[realcal] building profile: tdrive={tdrive_dir} air={airquality_dir} pa={purpleair_csv}", flush=True)
    profile = build_profile(
        tdrive_dir=tdrive_dir,
        airquality_dir=airquality_dir,
        purpleair_csv=purpleair_csv,
        tdrive_max_files=max_files,
    )
    out_path = args.out_root / "calibration_profile.json"
    write_profile(profile, out_path)
    print(f"[realcal] wrote {out_path} (hash={profile.profile_hash[:16]}...)", flush=True)
    print(json.dumps(profile.to_json(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
