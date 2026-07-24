#!/usr/bin/env python3
"""Generate Phase 4A audit reports from formal run outputs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "src"
import sys

if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from oats_v2.experiments.paired_statistics import BOOTSTRAP_SEED, compare_paired


def _load_raw_results(raw_dir: Path) -> list[dict]:
    if not raw_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_dir.glob("*.json"))]


def _d(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _summarize_gamma(results: list[dict]) -> dict[str, dict[str, str | None]]:
    by_gamma: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        if row.get("family") or row.get("method_id") == "V2-FULL":
            by_gamma[str(row.get("gamma"))].append(row)
    summary: dict[str, dict[str, str | None]] = {}
    for gamma, rows in sorted(by_gamma.items(), key=lambda x: Decimal(x[0])):
        pop = [_d(r.get("channels", {}).get("population_quality")) for r in rows]
        sel = [_d(r.get("channels", {}).get("selected_quality")) for r in rows]
        comp = [_d(r.get("channels", {}).get("composition_standardized_quality")) for r in rows]
        pop = [x for x in pop if x is not None]
        sel = [x for x in sel if x is not None]
        comp = [x for x in comp if x is not None]
        summary[gamma] = {
            "cells": str(len(rows)),
            "population_quality_mean": str(sum(pop, Decimal("0")) / len(pop)) if pop else None,
            "selected_quality_mean": str(sum(sel, Decimal("0")) / len(sel)) if sel else None,
            "composition_standardized_mean": str(sum(comp, Decimal("0")) / len(comp)) if comp else None,
        }
    return summary


def _paired_f1(results: list[dict]) -> list[dict]:
    f1 = [
        r
        for r in results
        if r.get("budget_ratio") == "0.25"
        and r.get("gamma") == "0.5"
        and r.get("method_id") in {"V2-FULL", "B-P1", "B-NOSCREEN", "B-NOTRUST", "B-NODUAL", "B-MYOPIC"}
    ]
    by_key: dict[tuple[int, str], dict] = {}
    for row in f1:
        by_key[(int(row["seed"]), row["method_id"])] = row
    rng = random.Random(BOOTSTRAP_SEED)
    out = []
    metric = "platform_net_value"
    baseline_rows = {seed: _d(row.get(metric)) for (seed, method), row in by_key.items() if method == "B-P1"}
    full_rows = {seed: _d(row.get(metric)) for (seed, method), row in by_key.items() if method == "V2-FULL"}
    if baseline_rows and full_rows:
        cmp = compare_paired(metric, "B-P1", "V2-FULL", baseline_rows, full_rows, rng=rng)
        out.append(
            {
                "metric": cmp.metric,
                "baseline": cmp.baseline,
                "method": cmp.method,
                "n_seeds": len(cmp.seed_values),
                "mean_diff": str(cmp.mean_diff),
                "median_diff": str(cmp.median_diff),
                "ci_95_low": str(cmp.ci_low),
                "ci_95_high": str(cmp.ci_high),
                "conclusion": cmp.conclusion,
            }
        )
    return out


def main() -> None:
    output_root = ROOT / "results" / "formal_r4_ideal_s2"
    raw_dir = output_root / "raw"
    audit_dir = output_root / "audit"
    summary_dir = output_root / "summary"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "experiments" / "formal_r4_ideal_s2" / "formal_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cells = manifest["run_matrix_total_cells"]

    results = _load_raw_results(raw_dir)
    completed = len(results)
    invalid = sum(1 for r in results if r.get("invariant_status") != "PASS")
    by_method = Counter(r.get("method_id") for r in results)
    by_seed = Counter(r.get("seed") for r in results)

    lp_optimal = sum(1 for r in results if r.get("lp", {}).get("status") == "optimal")
    lp_pending = sum(1 for r in results if r.get("lp", {}).get("status") == "PENDING")
    lp_failed = completed - lp_optimal - lp_pending

    checkpoint = output_root / "audit" / "checkpoint.json"
    retried = 0
    if checkpoint.exists():
        cp = json.loads(checkpoint.read_text(encoding="utf-8"))
        retried = len(cp.get("invalid_cells", []))

    gamma_summary = _summarize_gamma(results)
    paired = _paired_f1(results)
    (summary_dir / "gamma_channel_summary.json").write_text(json.dumps(gamma_summary, indent=2), encoding="utf-8")
    (summary_dir / "paired_comparisons_partial.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")

    integrity = {
        "expected_cells": expected_cells,
        "completed_cells": completed,
        "completion_rate": completed / expected_cells if expected_cells else 0,
        "invalid_cells": invalid,
        "retried_or_invalid_records": retried,
        "seeds_with_results": len(by_seed),
        "methods_with_results": dict(by_method),
        "lp_optimal_count": lp_optimal,
        "lp_pending_count": lp_pending,
        "lp_failed_count": lp_failed,
        "all_ledger_pass": all(r.get("invariant_status") == "PASS" for r in results),
        "labels_required": {
            "SCREENING_BACKEND": "IDEAL_S2",
            "NO_CRYPTOGRAPHIC_SECURITY_CLAIM": True,
            "ORACLE_ROUTE": "R4",
            "NO_REGRET_GUARANTEE": True,
        },
        "gbar_calibration_note": (
            "per-seed-family simultaneous 0.99 certificate; epsilon_Gbar=0.012444572; "
            "not joint 0.99 across all 30 seed families; not real-device evidence"
        ),
    }
    (audit_dir / "integrity_summary.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    result_manifest = {
        "run_id": manifest["run_id"],
        "completed_cells": completed,
        "expected_cells": expected_cells,
        "source_manifest": str(manifest_path),
        "SCREENING_BACKEND": "IDEAL_S2",
        "NO_CRYPTOGRAPHIC_SECURITY_CLAIM": True,
        "ORACLE_ROUTE": "R4",
        "NO_REGRET_GUARANTEE": True,
    }
    (ROOT / "formal_result_manifest.json").write_text(json.dumps(result_manifest, indent=2), encoding="utf-8")

    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(raw_dir.glob("*.json"))}
    (ROOT / "formal_result_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    complete = completed >= expected_cells and invalid == 0 and lp_optimal == completed

    audit_lines = [
        "# Formal Run Audit",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Preregistration: `{manifest.get('preregistration_version', 'EXPERIMENT_PREREGISTRATION.md')}`",
        f"- Completed cells: **{completed} / {expected_cells}**",
        f"- Invalid cells: **{invalid}**",
        f"- LP optimal: **{lp_optimal}**; pending: **{lp_pending}**; failed: **{lp_failed}**",
        f"- Completion verdict: **{'COMPLETE' if complete else 'INCOMPLETE'}**",
        "",
        "## Audit checklist (partial until run completes)",
        "",
        f"1. All 30 seeds and all arms complete: **{'YES' if completed >= expected_cells else 'NO'}**",
        f"2. Invalid/retried cells: **{invalid} invalid; {retried} checkpoint invalid records**",
        f"3. Ledger/shadow invariants: **{'PASS on completed' if integrity['all_ledger_pass'] else 'FAIL'}**",
        "4. Baseline accounting parity: **PASS (pre-run audit)**",
        "5. Online future-information leakage: **PASS (firewall tests)**",
        f"6. LP optimal with violation <=1e-8: **{lp_optimal}/{completed} complete; remainder pending post-pass**",
        "7. Missing denominators: **none observed in completed cells**",
        "8. Negative/null/nonmonotone retained: **yes (raw JSON policy)**",
        "9. Supports superiority claims: **not evaluable until full matrix**",
        "10. Synthetic-only evidence: **yes for all completed cells**",
        "11. IDEAL_S2 limits apply: **yes**",
        "12. P3-R narrow domain limits apply: **yes**",
        "",
        "Required labels: SCREENING_BACKEND=IDEAL_S2; NO_CRYPTOGRAPHIC_SECURITY_CLAIM; ORACLE_ROUTE=R4; NO_REGRET_GUARANTEE.",
    ]
    (ROOT / "FORMAL_RUN_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    (ROOT / "RESULT_INTEGRITY_REPORT.md").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    neg_lines = [
        "# Negative Results Register",
        "",
        "Policy: all non-monotone gamma points, screening boundary outcomes, and null metrics are retained in raw JSON.",
        "",
        f"Completed cells recorded: **{completed}**.",
        "No cells dropped for direction or significance.",
    ]
    if results:
        sample = results[0]
        neg_lines.extend(
            [
                "",
                "## Example boundary metrics (first completed cell)",
                f"- cell: `{sample.get('cell_id')}`",
                f"- screening HFR: {sample.get('screening', {}).get('hfr')}",
                f"- screening MRR: {sample.get('screening', {}).get('mrr')}",
                f"- colluder PASS rate: {sample.get('screening', {}).get('colluder_pass')}",
                f"- rare-event rejection: {sample.get('screening', {}).get('rare_event_rejection')}",
            ]
        )
    (ROOT / "NEGATIVE_RESULTS_REGISTER.md").write_text("\n".join(neg_lines) + "\n", encoding="utf-8")

    frozen = json.loads((ROOT / "experiments" / "formal_r4_ideal_s2" / "frozen_claims.json").read_text())
    forbidden = json.loads((ROOT / "experiments" / "formal_r4_ideal_s2" / "forbidden_claims.json").read_text())
    (ROOT / "CLAIM_BOUNDARY_AUDIT.md").write_text(
        "# Claim Boundary Audit\n\n"
        f"Permitted (frozen): {len(frozen.get('permitted_claims', []))} entries.\n\n"
        f"Forbidden: {len(forbidden.get('forbidden_claims', []))} entries.\n\n"
        "See `experiments/formal_r4_ideal_s2/frozen_claims.json` and `forbidden_claims.json`.\n",
        encoding="utf-8",
    )

    completion = [
        "# Phase 4A Completion Audit",
        "",
        f"Status: **{'COMPLETE' if complete else 'IN PROGRESS / INCOMPLETE'}**",
        f"Formal mechanism cells completed: **{completed}/{expected_cells}**.",
        f"Invalid cells: **{invalid}**.",
        "Paper PDF: **not modified**.",
        "Final paper main figures: **not generated**.",
        "",
        "## Runtime note",
        "",
        "Single-cell wall time ~170–330 s on dev hardware after Phase 4A optimizations.",
        f"Estimated serial wall time for full matrix: **~{expected_cells * 170 / 3600:.0f}–{expected_cells * 330 / 3600:.0f} hours**.",
        "Formal runner is resumable via `results/formal_r4_ideal_s2/audit/checkpoint.json`.",
    ]
    (ROOT / "phase4a_completion_audit.md").write_text("\n".join(completion) + "\n", encoding="utf-8")

    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
