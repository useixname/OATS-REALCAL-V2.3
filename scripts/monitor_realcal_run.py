#!/usr/bin/env python3
"""Lightweight progress monitor for the REAL-CAL formal run.

Prints a heartbeat every 10 minutes and a sentinel line when the run finishes or
appears stalled (no worker processes but not all cells done).
"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "formal_r4_realcal" / "raw"
TARGET = 2280


def _python_procs() -> int:
    n = 0
    for p in psutil.process_iter(["name"]):
        try:
            if "python" in (p.info["name"] or "").lower():
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def main() -> int:
    while True:
        cells = len(glob.glob(str(RAW / "*.json")))
        procs = _python_procs()
        vm = psutil.virtual_memory()
        print(
            f"PROGRESS cells={cells}/{TARGET} python_procs={procs} "
            f"ram_avail_gb={vm.available/1e9:.1f} ram_pct={vm.percent}",
            flush=True,
        )
        if cells >= TARGET:
            print("RUN_COMPLETE", flush=True)
            return 0
        # This monitor is itself a python process, so <=2 means the run's workers
        # are gone while cells remain -> stalled/crashed.
        if procs <= 2 and cells < TARGET:
            print("RUN_STALLED_NO_WORKERS", flush=True)
            return 2
        time.sleep(600)


if __name__ == "__main__":
    raise SystemExit(main())
