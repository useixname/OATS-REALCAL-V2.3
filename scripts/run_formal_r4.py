#!/usr/bin/env python3
"""Execute Phase 4A formal R4 / IDEAL_S2 mechanism experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_v2.experiments.audit import write_audit_reports
from src.oats_v2.experiments.formal_runner import FormalRunner
from src.oats_v2.experiments.run_matrix import count_run_cells, count_run_cells_v2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "SYN-V2-1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "formal_r4_ideal_s2")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--run-version",
        default="formal-r4-familyid-v1.1.0",
        help="Run/checkpoint version recorded in audit artifacts.",
    )
    parser.add_argument(
        "--matrix",
        choices=("v1", "v2"),
        default="v1",
        help="v1 = legacy F1-F5 matrix; v2 = REAL-CAL E1-E7 matrix (实验.md)",
    )
    parser.add_argument(
        "--trace-hashes",
        type=Path,
        default=ROOT / "trace_hashes.json",
        help="Trace hash manifest; use data/REAL-CAL/trace_hashes_realcal.json for REAL-CAL runs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel worker processes; 0 = max(os.cpu_count(), 1); default 10",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Restrict the run to these seeds (e.g. REAL-CAL 10-seed set). Default: all.",
    )
    parser.add_argument(
        "--lp-refresh",
        action="store_true",
        help="Discard stored LP results and recompute the comparator (post-pass only).",
    )
    args = parser.parse_args()

    if args.output_root.resolve() == (ROOT / "results" / "formal_r4_ideal_s2").resolve() and (
        args.data_root.name != "SYN-V2-1"
    ):
        parser.error(
            "refusing to write non-SYN-V2-1 data into the Phase-4A output root; "
            "pass a separate --output-root (e.g. results/formal_r4_realcal)"
        )

    workers = max(1, os.cpu_count() or 1) if args.workers == 0 else int(args.workers)
    trace_hashes = json.loads(args.trace_hashes.read_text(encoding="utf-8"))
    write_audit_reports(ROOT / "audit_results")

    runner = FormalRunner(
        data_root=args.data_root,
        output_root=args.output_root,
        trace_hashes=trace_hashes,
        workers=workers,
        seeds=tuple(args.seeds) if args.seeds else None,
        run_version=args.run_version,
        matrix_version=args.matrix,
        lp_refresh=args.lp_refresh,
    )
    summary = runner.run_all(resume=not args.no_resume)
    summary_path = args.output_root / "audit" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    counts = count_run_cells_v2() if args.matrix == "v2" else count_run_cells()
    print(json.dumps({"matrix": counts, **summary}, indent=2))
    return 0 if summary.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

