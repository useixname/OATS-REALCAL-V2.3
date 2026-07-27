from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_external.realcal_bridge import (
    canonical_hash,
    load_trace_hash_manifest,
    sha256_file,
    verify_seed_files,
)


DEFAULT_DATA = ROOT / "data" / "REAL-CAL-V2"
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "published_baseline_oasis_20260727"
    / "arrival_count_forecast.json"
)
TRAINING_SEEDS = tuple(range(20260725, 20260735))


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _mean(total: int, count: int) -> str:
    if count < 1:
        raise ValueError("empty forecast group")
    return str(Decimal(total) / Decimal(count))


def fit_forecast(data_root: Path, seeds: tuple[int, ...]) -> dict[str, object]:
    manifest = load_trace_hash_manifest(data_root)
    by_slot_cell: dict[tuple[int, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    by_cell: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    global_count = [0, 0]
    seed_receipts: dict[str, object] = {}

    for seed in seeds:
        verified = verify_seed_files(
            data_root,
            seed,
            ("tasks.jsonl", "eligibility.jsonl"),
            manifest,
        )
        task_features: dict[str, tuple[int, int]] = {}
        task_counts: dict[str, int] = {}
        for row in _rows(data_root / str(seed) / "tasks.jsonl"):
            task_id = str(row["task_id"])
            task_features[task_id] = (int(row["slot"]), int(row["cell"]))
            task_counts[task_id] = 0
        for row in _rows(data_root / str(seed) / "eligibility.jsonl"):
            if bool(row["available"]) and row["mapped_task_id"] is not None:
                task_id = str(row["mapped_task_id"])
                if task_id not in task_counts:
                    raise RuntimeError(
                        f"eligibility maps to missing task: seed={seed}, task={task_id}"
                    )
                task_counts[task_id] += 1

        for task_id, count in task_counts.items():
            slot, cell = task_features[task_id]
            by_slot_cell[(slot, cell)][0] += count
            by_slot_cell[(slot, cell)][1] += 1
            by_cell[cell][0] += count
            by_cell[cell][1] += 1
            global_count[0] += count
            global_count[1] += 1
        seed_receipts[str(seed)] = {
            "verified_file_hashes": verified,
            "task_count": len(task_counts),
            "available_mapped_count": sum(task_counts.values()),
        }

    payload: dict[str, object] = {
        "model_id": "OASIS-ARRIVAL-COUNT-TRAINING-ONLY-20260727-R1",
        "method": "arithmetic mean candidate count",
        "target": "available mapped candidates per task",
        "features": ["slot", "public_cell"],
        "fallback_order": ["slot_cell", "cell", "global"],
        "rounding": "ROUND_HALF_UP at prediction time",
        "training_seeds": list(seeds),
        "data_root_id": "REAL-CAL-V2",
        "slot_cell_mean": {
            f"{slot}|{cell}": _mean(total, count)
            for (slot, cell), (total, count) in sorted(by_slot_cell.items())
        },
        "cell_mean": {
            str(cell): _mean(total, count)
            for cell, (total, count) in sorted(by_cell.items())
        },
        "global_mean": _mean(global_count[0], global_count[1]),
        "training_receipts": seed_receipts,
    }
    payload["model_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen forecast: {output}")
    payload = fit_forecast(data_root, TRAINING_SEEDS)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "model_hash": payload["model_hash"],
                "training_seed_count": len(TRAINING_SEEDS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
