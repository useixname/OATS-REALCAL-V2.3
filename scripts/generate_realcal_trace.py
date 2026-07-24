#!/usr/bin/env python3
"""Generate REAL-CAL semi-synthetic traces from the frozen calibration profile.

Usage:
  python scripts/generate_realcal_trace.py --seeds 20260715              # one seed
  python scripts/generate_realcal_trace.py --all                         # all 30 formal seeds

Writes data/<dataset>/<seed>/ with the same file layout as SYN-V2-1 and a
per-run trace_hashes_realcal.json manifest. Refuses to overwrite existing seed
directories (version bump required), mirroring the SYN safety rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_v2.data.schemas import FORMAL_SEEDS
from src.oats_v2.realcal.trace_builder import generate_realcal_trace


def _generate_one(seed: int, out_root: str, profile: str, dataset_version: int) -> dict:
    out_dir = Path(out_root) / str(seed)
    result = generate_realcal_trace(
        seed=seed,
        output_directory=out_dir,
        root=ROOT,
        profile_path=Path(profile),
        dataset_version=dataset_version,
    )
    return {
        "seed": seed,
        "files": {f["name"]: f["sha256"] for f in result["files"]},
        "task_count": result["task_count"],
        "available_mapped": result["available_mapped"],
        "profile_hash": result["profile_hash"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--profile", type=Path, default=ROOT / "data_real" / "REAL-CAL-V1" / "calibration_profile.json")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--dataset-version",
        type=int,
        default=1,
        choices=(1, 2),
        help="1 = legacy REAL-CAL calibration, 2 = current REAL-CAL benchmark",
    )
    parser.add_argument("--workers", type=int, default=10, help="Parallel processes; 0 = os.cpu_count()")
    args = parser.parse_args()

    dataset_id = "REAL-CAL-V1" if args.dataset_version == 1 else "REAL-CAL"
    if args.out_root is None:
        args.out_root = ROOT / "data" / dataset_id

    seeds = list(FORMAL_SEEDS) if args.all else args.seeds
    if not seeds:
        print("nothing to do: pass --seeds <s...> or --all", file=sys.stderr)
        return 2

    workers = max(1, os.cpu_count() or 1) if args.workers == 0 else int(args.workers)
    manifest_path = args.out_root / "trace_hashes_realcal.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("dataset_id", dataset_id)
    manifest.setdefault("seed_file_hashes", {})

    # Skip seeds already present (resumable).
    pending = [s for s in seeds if not (args.out_root / str(s)).exists()]
    print(f"[realcal] seeds={len(seeds)} pending={len(pending)} workers={workers}", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_generate_one, s, str(args.out_root), str(args.profile), args.dataset_version): s
            for s in pending
        }
        for future in as_completed(futures):
            res = future.result()
            manifest["seed_file_hashes"][str(res["seed"])] = res["files"]
            done += 1
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"[realcal] seed {res['seed']} done ({done}/{len(pending)}) "
                f"tasks={res['task_count']} avail={res['available_mapped']}",
                flush=True,
            )

    print(f"[realcal] wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

