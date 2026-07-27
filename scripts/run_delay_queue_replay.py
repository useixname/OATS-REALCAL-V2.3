#!/usr/bin/env python3
"""Run and freeze the 30-cell calendar-time delayed-feedback replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oats_v2.experiments.formal_runner import simulate_cell  # noqa: E402
from src.oats_v2.experiments.lp_comparator import LPComparatorCache  # noqa: E402
from src.oats_v2.experiments.run_matrix import RunCell  # noqa: E402
from src.oats_v2.experiments.trace_loader import load_trace  # noqa: E402
from src.oats_v2.realcal import REALCAL_SEEDS  # noqa: E402


RUN_VERSION = "formal-realcal-delay-queue-1.0.0"
DELAYS = (0, 5, 20)
DEFAULT_DATA = ROOT / "data" / "REAL-CAL-V2"
DEFAULT_MANIFEST = DEFAULT_DATA / "trace_hashes_realcal.json"
DEFAULT_OUTPUT = ROOT / "results" / "delay_queue_replay_20260727"
PREREG = (
    ROOT
    / "docs"
    / "delay_queue_replay_20260727"
    / "DELAY_QUEUE_PREREGISTRATION.md"
)
SOURCE_PATHS = (
    "src/oats_v2/trust.py",
    "src/oats_v2/feedback_calendar.py",
    "src/oats_v2/experiments/formal_runner.py",
    "src/oats_v2/experiments/metrics.py",
    "src/oats_v2/experiments/result_schema.py",
    "tests/implementation/test_trust_feedback_contract.py",
    "tests/implementation/test_feedback_calendar.py",
    "scripts/run_delay_queue_replay.py",
    "docs/delay_queue_replay_20260727/DELAY_QUEUE_PREREGISTRATION.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cells() -> tuple[RunCell, ...]:
    cells: list[RunCell] = []
    order = 0
    for seed in REALCAL_SEEDS:
        for delay in DELAYS:
            cells.append(
                RunCell(
                    cell_id=f"E9_DELAY_QUEUE_s{seed}_d{delay}",
                    family="E9_DELAY_QUEUE",
                    seed=int(seed),
                    method_id="V2-FULL",
                    gamma=Decimal("0.3"),
                    budget_ratio=Decimal("0.25"),
                    contamination=Decimal("0"),
                    delay=delay,
                    missing_prob=Decimal("0"),
                    arrival_multiplier=Decimal("1"),
                    order_index=order,
                )
            )
            order += 1
    if len(cells) != 30 or len({cell.cell_id for cell in cells}) != 30:
        raise RuntimeError("delay queue matrix must contain 30 unique cells")
    return tuple(cells)


def _source_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in SOURCE_PATHS}


def _freeze_payload(
    *,
    data_root: Path,
    trace_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    trace_manifest = _json_load(trace_manifest_path)
    return {
        "schema_version": "oats-delay-queue-prerun-freeze-1.0.0",
        "status": "FROZEN_BEFORE_EXECUTION",
        "run_version": RUN_VERSION,
        "matrix": [
            {key: str(value) if isinstance(value, Decimal) else value for key, value in asdict(cell).items()}
            for cell in _cells()
        ],
        "data_root": str(data_root.relative_to(ROOT)).replace("\\", "/"),
        "trace_manifest": str(trace_manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "trace_manifest_sha256": _sha256(trace_manifest_path),
        "dataset_id": trace_manifest.get("dataset_id"),
        "source_hashes": _source_hashes(),
        "output_root": str(output_root.relative_to(ROOT)).replace("\\", "/"),
        "bootstrap_resamples": 10000,
        "old_formal_roots_are_read_only": [
            "results/formal_realcal_trustfix_20260726",
            "results/formal_realcal_pre_repair",
        ],
    }


def _prepare_freeze(
    *,
    data_root: Path,
    trace_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    freeze_path = output_root / "audit" / "PRERUN_FREEZE.json"
    expected = _freeze_payload(
        data_root=data_root,
        trace_manifest_path=trace_manifest_path,
        output_root=output_root,
    )
    if freeze_path.exists():
        current = _json_load(freeze_path)
        if current != expected:
            raise RuntimeError("existing delay-queue freeze does not match current sources or matrix")
        return current
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("nonempty output root has no matching preregistration freeze")
    _json_dump(freeze_path, expected)
    return expected


def _cell_result_path(output_root: Path, cell: RunCell) -> Path:
    return output_root / "raw" / f"{cell.cell_id}.json"


def _validate_result(payload: dict[str, Any], cell: RunCell) -> None:
    if payload["invariant_status"] != "PASS":
        raise RuntimeError(f"invariant failure: {cell.cell_id}")
    trust = payload["trust"]
    if trust["feedback_count"] != trust["trust_transition_count"]:
        raise RuntimeError(f"feedback/transition mismatch: {cell.cell_id}")
    if trust["duplicate_feedback_suppressed_count"] != 0:
        raise RuntimeError(f"unexpected duplicate suppression: {cell.cell_id}")
    if Decimal(payload["final_ledger"]["locked_base"]) != 0:
        raise RuntimeError(f"terminal base lock remains: {cell.cell_id}")
    if Decimal(payload["final_ledger"]["locked_score"]) != 0:
        raise RuntimeError(f"terminal score lock remains: {cell.cell_id}")
    if Decimal(payload["final_shadow"]["held"]) != 0:
        raise RuntimeError(f"terminal shadow hold remains: {cell.cell_id}")
    if Decimal(payload["final_shadow"]["committed"]) != 0:
        raise RuntimeError(f"terminal shadow commitment remains: {cell.cell_id}")
    if not payload["feedback_queue_mode"].startswith("calendar-time"):
        raise RuntimeError(f"wrong feedback mode: {cell.cell_id}")


def _run_seed(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    data_root = Path(job["data_root"])
    output_root = Path(job["output_root"])
    manifest = _json_load(Path(job["trace_manifest"]))
    trace = load_trace(
        seed,
        data_root,
        manifest,
        verify_hashes=False,
        load_eligibility_index=False,
        load_gammas=frozenset({Decimal("0.3")}),
    )
    cells = [cell for cell in _cells() if cell.seed == seed]
    receipts: list[dict[str, Any]] = []
    started = time.time()
    for cell in cells:
        result_path = _cell_result_path(output_root, cell)
        if result_path.exists():
            payload = _json_load(result_path)
            _validate_result(payload, cell)
            receipts.append(
                {
                    "cell_id": cell.cell_id,
                    "delay": cell.delay,
                    "status": "ADOPTED",
                    "result_sha256": _sha256(result_path),
                }
            )
            continue
        result = simulate_cell(cell, trace, LPComparatorCache(), compute_lp=False)
        payload = result.to_dict()
        payload["run_version"] = RUN_VERSION
        payload["order_index"] = cell.order_index
        _validate_result(payload, cell)
        _json_dump(result_path, payload)
        receipts.append(
            {
                "cell_id": cell.cell_id,
                "delay": cell.delay,
                "status": "COMPLETED",
                "result_sha256": _sha256(result_path),
                "runtime_seconds": payload["runtime_seconds"],
            }
        )
        print(
            f"[delay-queue] seed={seed} delay={cell.delay} "
            f"runtime={payload['runtime_seconds']:.1f}s",
            flush=True,
        )
    receipt = {
        "schema_version": "oats-delay-queue-seed-receipt-1.0.0",
        "status": "PASS",
        "seed": seed,
        "cells": receipts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _json_dump(output_root / "audit" / f"seed_{seed}.json", receipt)
    return receipt


def _number(value: Any) -> float:
    return float(value) if value is not None else math.nan


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    trajectory = payload["trust_trajectory"].get("1000", {})
    composition = payload["worker_type_composition"]
    trust = payload["trust"]
    return {
        "seed": int(payload["seed"]),
        "delay": int(payload["cell_id"].rsplit("_d", 1)[1]),
        "purchased_count": int(payload["purchased_count"]),
        "selection_honest": _number(composition.get("honest", 0)),
        "selection_low_quality": _number(composition.get("low-quality", 0)),
        "selection_malicious": _number(composition.get("malicious", 0)),
        "selection_camouflage": _number(composition.get("camouflage", 0)),
        "gross_external_value": _number(payload["gross_external_value"]),
        "platform_net_value": _number(payload["platform_net_value"]),
        "final_trust_honest": _number(trajectory.get("honest")),
        "final_trust_malicious": _number(trajectory.get("malicious")),
        "final_trust_camouflage": _number(trajectory.get("camouflage")),
        "trust_auc": _number(trust["auc"]),
        "feedback_count": int(trust["feedback_count"]),
        "trust_transition_count": int(trust["trust_transition_count"]),
        "duplicate_feedback_suppressed_count": int(
            trust["duplicate_feedback_suppressed_count"]
        ),
        "deadline_satisfaction": _number(payload["deadline_satisfaction"]),
        "mean_outstanding_score_escrow": _number(
            payload["mean_outstanding_score_escrow"]
        ),
        "peak_outstanding_score_escrow": _number(
            payload["peak_outstanding_score_escrow"]
        ),
        "terminal_outstanding_score_escrow": _number(
            payload["terminal_outstanding_score_escrow"]
        ),
        "terminal_pending_task_count": int(payload["terminal_pending_task_count"]),
        "terminal_pending_feedback_count": int(
            payload["terminal_pending_feedback_count"]
        ),
    }


METRICS = (
    "purchased_count",
    "selection_honest",
    "selection_low_quality",
    "selection_malicious",
    "selection_camouflage",
    "gross_external_value",
    "platform_net_value",
    "final_trust_honest",
    "final_trust_malicious",
    "final_trust_camouflage",
    "trust_auc",
    "feedback_count",
    "deadline_satisfaction",
    "mean_outstanding_score_escrow",
    "peak_outstanding_score_escrow",
    "terminal_outstanding_score_escrow",
    "terminal_pending_task_count",
    "terminal_pending_feedback_count",
)


def _bootstrap_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    indices = rng.integers(0, values.size, size=(10000, values.size))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _analyze(output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_seed_delay: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in _cells():
        payload = _json_load(_cell_result_path(output_root, cell))
        _validate_result(payload, cell)
        row = _flatten(payload)
        rows.append(row)
        by_seed_delay[(cell.seed, cell.delay)] = row
    _write_csv(output_root / "analysis" / "cell_metrics.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        for delay in DELAYS:
            values = np.asarray(
                [float(by_seed_delay[(seed, delay)][metric]) for seed in REALCAL_SEEDS],
                dtype=float,
            )
            mean, low, high = _bootstrap_mean(
                values,
                rng=np.random.default_rng(2026072700 + metric_index * 10 + delay),
            )
            summary_rows.append(
                {
                    "metric": metric,
                    "delay": delay,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        base = np.asarray(
            [float(by_seed_delay[(seed, 0)][metric]) for seed in REALCAL_SEEDS],
            dtype=float,
        )
        for delay in (5, 20):
            current = np.asarray(
                [float(by_seed_delay[(seed, delay)][metric]) for seed in REALCAL_SEEDS],
                dtype=float,
            )
            paired = current - base
            mean, low, high = _bootstrap_mean(
                paired,
                rng=np.random.default_rng(2026072800 + metric_index * 10 + delay),
            )
            contrast_rows.append(
                {
                    "metric": metric,
                    "contrast": f"delay_{delay}_minus_delay_0",
                    "paired_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "all_seed_deltas_nonnegative": bool(np.all(paired >= 0)),
                    "all_seed_deltas_nonpositive": bool(np.all(paired <= 0)),
                }
            )
    _write_csv(output_root / "analysis" / "delay_summary.csv", summary_rows)
    _write_csv(output_root / "analysis" / "paired_delay_contrasts.csv", contrast_rows)

    totals = {
        "feedback_count": sum(row["feedback_count"] for row in rows),
        "trust_transition_count": sum(row["trust_transition_count"] for row in rows),
        "duplicate_feedback_suppressed_count": sum(
            row["duplicate_feedback_suppressed_count"] for row in rows
        ),
    }
    if totals["feedback_count"] != totals["trust_transition_count"]:
        raise RuntimeError("full matrix feedback/transition totals differ")
    if totals["duplicate_feedback_suppressed_count"] != 0:
        raise RuntimeError("full matrix contains unexpected duplicate suppressions")

    result = {
        "schema_version": "oats-delay-queue-analysis-1.0.0",
        "status": "PASS",
        "cell_count": len(rows),
        "seed_count": len(REALCAL_SEEDS),
        "delays": list(DELAYS),
        "totals": totals,
        "summary_csv": "analysis/delay_summary.csv",
        "paired_contrasts_csv": "analysis/paired_delay_contrasts.csv",
    }
    _json_dump(output_root / "analysis" / "ANALYSIS_COMPLETE.json", result)
    return result


def _write_report(output_root: Path) -> None:
    summary = list(csv.DictReader((output_root / "analysis" / "delay_summary.csv").open(encoding="utf-8-sig")))
    contrasts = list(csv.DictReader((output_root / "analysis" / "paired_delay_contrasts.csv").open(encoding="utf-8-sig")))
    by_metric_delay = {(row["metric"], int(row["delay"])): row for row in summary}
    by_metric_contrast = {(row["metric"], row["contrast"]): row for row in contrasts}

    def fmt(value: Any, digits: int = 4) -> str:
        return f"{float(value):.{digits}f}"

    table_metrics = (
        ("platform_net_value", "Net value", 2),
        ("purchased_count", "Purchases", 1),
        ("final_trust_honest", "Final honest trust", 4),
        ("final_trust_camouflage", "Final camouflage trust", 4),
        ("trust_auc", "Trust AUC", 4),
        ("deadline_satisfaction", "Deadline satisfaction", 4),
        ("mean_outstanding_score_escrow", "Mean outstanding escrow", 2),
        ("peak_outstanding_score_escrow", "Peak outstanding escrow", 2),
    )
    lines = [
        "# Calendar-Time Delayed-Feedback Replay",
        "",
        "All values come from the frozen 30-cell REAL-CAL replay. Intervals use "
        "10,000 seed-level paired bootstrap resamples.",
        "",
        "| Metric | Delay 0 mean | Delay 5 mean | Delay 20 mean | D5-D0 (95% CI) | D20-D0 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, label, digits in table_metrics:
        values = [by_metric_delay[(metric, delay)]["mean"] for delay in DELAYS]
        d5 = by_metric_contrast[(metric, "delay_5_minus_delay_0")]
        d20 = by_metric_contrast[(metric, "delay_20_minus_delay_0")]
        lines.append(
            f"| {label} | {fmt(values[0], digits)} | {fmt(values[1], digits)} | "
            f"{fmt(values[2], digits)} | {fmt(d5['paired_mean'], digits)} "
            f"[{fmt(d5['ci_low'], digits)}, {fmt(d5['ci_high'], digits)}] | "
            f"{fmt(d20['paired_mean'], digits)} "
            f"[{fmt(d20['ci_low'], digits)}, {fmt(d20['ci_high'], digits)}] |"
        )
    lines.extend(
        [
            "",
            "The raw cell metrics, all selection-composition fields, terminal "
            "pending counts, and every paired contrast are retained in the CSV files.",
            "",
        ]
    )
    (output_root / "DELAY_QUEUE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_manifest(output_root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name in {"ARTIFACT_MANIFEST.json", "RUN_COMPLETE.json"}:
            continue
        relative = str(path.relative_to(output_root)).replace("\\", "/")
        artifacts[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "oats-delay-queue-artifact-manifest-1.0.0",
        "status": "COMPLETE",
        "run_version": RUN_VERSION,
        "prerun_freeze_sha256": _sha256(output_root / "audit" / "PRERUN_FREEZE.json"),
        "source_hashes": freeze["source_hashes"],
        "artifacts": artifacts,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root.resolve()
    trace_manifest_path = args.trace_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root in {
        (ROOT / "results" / "formal_realcal_trustfix_20260726").resolve(),
        (ROOT / "results" / "formal_realcal_pre_repair").resolve(),
    }:
        raise RuntimeError("refusing to overwrite a frozen formal result root")
    if not PREREG.is_file():
        raise RuntimeError(f"missing preregistration: {PREREG}")
    freeze = _prepare_freeze(
        data_root=data_root,
        trace_manifest_path=trace_manifest_path,
        output_root=output_root,
    )
    complete_path = output_root / "RUN_COMPLETE.json"
    if complete_path.exists():
        complete = _json_load(complete_path)
        if complete.get("prerun_freeze_sha256") == _sha256(
            output_root / "audit" / "PRERUN_FREEZE.json"
        ):
            print(json.dumps(complete, indent=2, ensure_ascii=False))
            return 0
        raise RuntimeError("completion receipt belongs to a different freeze")

    workers = max(1, min(int(args.workers), len(REALCAL_SEEDS)))
    jobs = [
        {
            "seed": int(seed),
            "data_root": str(data_root),
            "trace_manifest": str(trace_manifest_path),
            "output_root": str(output_root),
        }
        for seed in REALCAL_SEEDS
    ]
    print(
        f"[delay-queue] run={RUN_VERSION} cells=30 workers={workers} output={output_root}",
        flush=True,
    )
    started = time.time()
    receipts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_seed, job): job["seed"] for job in jobs}
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(
                f"[delay-queue] seed={receipt['seed']} PASS "
                f"elapsed={receipt['elapsed_seconds']}s",
                flush=True,
            )
    if len(receipts) != len(REALCAL_SEEDS):
        raise RuntimeError("not all seed receipts completed")

    analysis = _analyze(output_root)
    _write_report(output_root)
    manifest = _artifact_manifest(output_root, freeze)
    _json_dump(output_root / "ARTIFACT_MANIFEST.json", manifest)
    complete = {
        "schema_version": "oats-delay-queue-run-completion-1.0.0",
        "status": "COMPLETE",
        "run_version": RUN_VERSION,
        "cell_count": 30,
        "invalid_cells": 0,
        "seed_count": len(REALCAL_SEEDS),
        "delays": list(DELAYS),
        "workers": workers,
        "elapsed_seconds": round(time.time() - started, 3),
        "prerun_freeze_sha256": _sha256(output_root / "audit" / "PRERUN_FREEZE.json"),
        "analysis_complete_sha256": _sha256(
            output_root / "analysis" / "ANALYSIS_COMPLETE.json"
        ),
        "artifact_manifest_sha256": _sha256(output_root / "ARTIFACT_MANIFEST.json"),
        "trust_counter_totals": analysis["totals"],
        "frozen_formal_roots_modified": False,
    }
    _json_dump(complete_path, complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
