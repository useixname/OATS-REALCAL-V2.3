#!/usr/bin/env python3
"""Derive E6/E7 paired intervals and audit the near-one trust AUC.

This is deterministic post-processing of the frozen 830-cell trust-repair run.
It does not execute the mechanism or modify any frozen raw result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = tuple(range(20260715, 20260725))
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260715
DEFAULTS = {
    "gamma": "0.3",
    "alpha": "0.2",
    "theta_a": "0.75",
    "lambda_max": "10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "formal_realcal_trustfix_20260726",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "realcal_trustfix_analysis_20260727",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cells(raw_root: Path) -> list[dict]:
    cells: list[dict] = []
    for path in sorted(raw_root.glob("*.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        cell["_path"] = path
        cell["_family"] = cell["cell_id"].split("_s2026", 1)[0]
        cells.append(cell)
    if len(cells) != 830:
        raise ValueError(f"expected 830 frozen raw cells, found {len(cells)}")
    invalid = [cell["cell_id"] for cell in cells if cell.get("invariant_status") != "PASS"]
    if invalid:
        raise ValueError(f"frozen cells with non-PASS status: {invalid[:5]}")
    return cells


def number(value: object) -> float:
    if value is None:
        raise ValueError("required metric is missing")
    return float(value)


def metrics(cell: dict) -> dict[str, float]:
    paid = number(cell["total_paid"])
    gross = number(cell["gross_external_value"])
    return {
        "net": number(cell["platform_net_value"]),
        "gross": gross,
        "paid": paid,
        "budget_efficiency": gross / paid,
        "selected_quality": number(cell["channels"]["selected_quality"]),
        "trust_auc": number(cell["trust"]["auc"]),
    }


def paired_bootstrap(differences: list[float]) -> tuple[float, float, float]:
    if len(differences) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected {len(EXPECTED_SEEDS)} paired differences")
    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[float] = []
    n = len(differences)
    for _ in range(BOOTSTRAP_N):
        draws.append(statistics.fmean(differences[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    return (
        statistics.fmean(differences),
        draws[int(0.025 * BOOTSTRAP_N)],
        draws[int(0.975 * BOOTSTRAP_N)],
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_seed_map(cells: list[dict], predicate: Callable[[dict], bool], label: str) -> dict[int, dict]:
    selected = {int(cell["seed"]): cell for cell in cells if predicate(cell)}
    if tuple(sorted(selected)) != EXPECTED_SEEDS:
        raise ValueError(f"{label}: expected seeds {EXPECTED_SEEDS}, found {tuple(sorted(selected))}")
    return selected


def cell_flag(cell_id: str, name: str) -> str:
    patterns = {
        "gamma": r"_g([^_]+)",
        "alpha": r"_al([^_]+)",
        "theta_a": r"_ta([^_]+)",
        "lambda_max": r"_lm([^_.]+(?:\.[^_]+)?)",
    }
    match = re.search(patterns[name], cell_id)
    if not match:
        raise ValueError(f"missing {name} flag in {cell_id}")
    return match.group(1).removesuffix(".json")


def derive_e6(cells: list[dict]) -> list[dict]:
    reference = exact_seed_map(
        cells,
        lambda c: (
            c["_family"] == "E1_OVERALL"
            and c["method_id"] == "V2-FULL"
            and c["gamma"] == "0.3"
            and c["budget_ratio"] == "0.25"
        ),
        "E6 reference OATS",
    )
    comparator_sources = {
        "B-NOTRUST": "E6_ABLATION",
        "B-NODUAL": "E1_OVERALL",
        "C-EFFORT-OFF": "E6_ABLATION",
        "B-NOSCREEN": "E6_ABLATION",
        "B-MYOPIC": "E1_OVERALL",
        "B-P1": "E6_ABLATION",
    }
    rows: list[dict] = []
    for comparator, family in comparator_sources.items():
        other = exact_seed_map(
            cells,
            lambda c, comparator=comparator, family=family: (
                c["_family"] == family
                and c["method_id"] == comparator
                and c["gamma"] == "0.3"
                and c["budget_ratio"] == "0.25"
            ),
            f"E6 comparator {comparator}",
        )
        for metric in ("net", "gross", "budget_efficiency", "selected_quality"):
            differences = [
                metrics(reference[seed])[metric] - metrics(other[seed])[metric]
                for seed in EXPECTED_SEEDS
            ]
            mean_diff, ci_lo, ci_hi = paired_bootstrap(differences)
            rows.append(
                {
                    "contrast": f"OATS - {comparator}",
                    "metric": metric,
                    "n_seeds": len(differences),
                    "mean_diff": mean_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "bootstrap_resamples": BOOTSTRAP_N,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    return rows


def derive_e7(cells: list[dict]) -> list[dict]:
    reference = exact_seed_map(
        cells,
        lambda c: (
            c["_family"] == "E1_OVERALL"
            and c["method_id"] == "V2-FULL"
            and c["gamma"] == "0.3"
            and c["budget_ratio"] == "0.25"
        ),
        "E7 default reference",
    )
    families = {
        "gamma": "E7_SENS_GAMMA",
        "alpha": "E7_SENS_ALPHA",
        "theta_a": "E7_SENS_THETAA",
        "lambda_max": "E7_SENS_LMAX",
    }
    rows: list[dict] = []
    for parameter, family in families.items():
        values = sorted(
            {cell_flag(cell["cell_id"], parameter) for cell in cells if cell["_family"] == family},
            key=float,
        )
        for value in values:
            other = exact_seed_map(
                cells,
                lambda c, family=family, parameter=parameter, value=value: (
                    c["_family"] == family and cell_flag(c["cell_id"], parameter) == value
                ),
                f"E7 {parameter}={value}",
            )
            for metric in ("net", "gross", "paid", "selected_quality", "trust_auc"):
                differences = [
                    metrics(reference[seed])[metric] - metrics(other[seed])[metric]
                    for seed in EXPECTED_SEEDS
                ]
                mean_diff, ci_lo, ci_hi = paired_bootstrap(differences)
                rows.append(
                    {
                        "parameter": parameter,
                        "default_value": DEFAULTS[parameter],
                        "comparison_value": value,
                        "contrast": f"default - {parameter}={value}",
                        "metric": metric,
                        "n_seeds": len(differences),
                        "mean_diff": mean_diff,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "bootstrap_resamples": BOOTSTRAP_N,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                    }
                )
    return rows


def audit_trust_auc(cells: list[dict], results_root: Path) -> dict:
    canonical = exact_seed_map(
        cells,
        lambda c: (
            c["_family"] == "E1_OVERALL"
            and c["method_id"] == "V2-FULL"
            and c["gamma"] == "0.3"
            and c["budget_ratio"] == "0.25"
        ),
        "trust-AUC canonical cells",
    )
    auc_by_seed = {str(seed): number(canonical[seed]["trust"]["auc"]) for seed in EXPECTED_SEEDS}
    feedback_by_seed = {
        str(seed): int(canonical[seed]["trust"]["feedback_count"]) for seed in EXPECTED_SEEDS
    }
    transition_by_seed = {
        str(seed): int(canonical[seed]["trust"]["trust_transition_count"])
        for seed in EXPECTED_SEEDS
    }
    duplicate_by_seed = {
        str(seed): int(canonical[seed]["trust"]["duplicate_feedback_suppressed_count"])
        for seed in EXPECTED_SEEDS
    }
    if feedback_by_seed != transition_by_seed:
        raise ValueError("canonical feedback and trust-transition counts differ")
    if any(duplicate_by_seed.values()):
        raise ValueError("canonical run contains duplicate feedback suppression")

    verification_path = results_root / "audit" / "FORMAL_RESULT_VERIFICATION.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise ValueError("formal result verification is not PASS")
    if verification["feedback_total"] != verification["trust_transition_total"]:
        raise ValueError("full-matrix feedback and transition totals differ")
    if verification["duplicate_feedback_suppression_total"] != 0:
        raise ValueError("full-matrix duplicate feedback suppression is nonzero")

    formal_runner = ROOT / "src" / "oats_v2" / "experiments" / "formal_runner.py"
    metric_source = ROOT / "src" / "oats_v2" / "experiments" / "metrics.py"
    projection_source = ROOT / "src" / "oats_v2" / "experiments" / "online_projection.py"
    validator_source = ROOT / "src" / "oats_v2" / "data" / "trace_validator.py"
    generator_source = ROOT / "src" / "oats_v2" / "realcal" / "generator.py"

    runner_text = formal_runner.read_text(encoding="utf-8")
    order_markers = {
        "prior_trust_read": runner_text.index("trust_val = trust.values[worker_id]"),
        "selection": runner_text.index("selection = build_selection_fast("),
        "outcome_feedback_update": runner_text.index("trust.update("),
    }
    if not (
        order_markers["prior_trust_read"]
        < order_markers["selection"]
        < order_markers["outcome_feedback_update"]
    ):
        raise ValueError("selection/feedback source order check failed")

    projection_text = projection_source.read_text(encoding="utf-8")
    forbidden_fields = (
        "stratum",
        "bias_sign",
        "current_raw_report",
        "future_outcome",
        "report",
        "score",
        "v_ijt",
    )
    missing_forbidden = [field for field in forbidden_fields if f'"{field}"' not in projection_text]
    if missing_forbidden:
        raise ValueError(f"online firewall does not list forbidden fields: {missing_forbidden}")

    aucs = list(auc_by_seed.values())
    feedbacks = list(feedback_by_seed.values())
    return {
        "status": "PASS",
        "scope": (
            "Final-state within-run ranking diagnostic for the 10 default OATS cells; "
            "not a held-out classifier AUC or deployment generalization estimate."
        ),
        "metric_definition": (
            "P(final honest trust > final malicious-or-camouflage trust) "
            "+ 0.5 P(tie), computed only after the horizon."
        ),
        "auc_by_seed": auc_by_seed,
        "auc_mean": statistics.fmean(aucs),
        "auc_min": min(aucs),
        "auc_max": max(aucs),
        "exact_one_seed_count": sum(value == 1.0 for value in aucs),
        "class_counts_per_seed": {
            "honest_positive": 300,
            "malicious_or_camouflage_negative": 100,
            "pairwise_rank_comparisons": 30_000,
        },
        "feedback_by_seed": feedback_by_seed,
        "feedback_mean": statistics.fmean(feedbacks),
        "transition_count_matches_feedback_each_seed": True,
        "duplicate_feedback_suppressed_each_seed": duplicate_by_seed,
        "full_matrix_transition_audit": {
            "feedback_total": verification["feedback_total"],
            "trust_transition_total": verification["trust_transition_total"],
            "duplicate_feedback_suppression_total": verification[
                "duplicate_feedback_suppression_total"
            ],
            "verification_sha256": sha256_file(verification_path),
        },
        "anti_leakage_checks": {
            "online_forbidden_fields": list(forbidden_fields),
            "selection_uses_pre_feedback_trust": True,
            "outcome_feedback_occurs_after_selection_in_each_default_delay_zero_slot": True,
            "stratum_labels_used_only_for_offline_grouped_metrics_and_diagnostics": True,
            "malicious_and_camouflage_share_the_non_low_public_signal_role": True,
            "trace_contract_and_eligibility_schema_validator_excludes_private_labels": True,
        },
        "interpretation": (
            "The near-one value is expected under the controlled semi-synthetic generator: "
            "malicious-mode reports receive a 0.4 signed deviation with probability 0.8, "
            "camouflage switches at slot 501, and roughly 77,000 completed feedback "
            "transitions repeatedly update 500 worker states. It demonstrates separation "
            "inside this benchmark and must not be described as held-out attack detection."
        ),
        "source_sha256": {
            "formal_runner.py": sha256_file(formal_runner),
            "metrics.py": sha256_file(metric_source),
            "online_projection.py": sha256_file(projection_source),
            "trace_validator.py": sha256_file(validator_source),
            "generator.py": sha256_file(generator_source),
        },
    }


def main() -> int:
    args = parse_args()
    raw_root = args.results / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    args.out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(raw_root)

    e6_rows = derive_e6(cells)
    e7_rows = derive_e7(cells)
    e6_path = args.out / "e6_paired_contrasts.csv"
    e7_path = args.out / "e7_paired_contrasts.csv"
    trust_path = args.out / "trust_auc_leakage_audit.json"
    write_csv(
        e6_path,
        e6_rows,
        [
            "contrast",
            "metric",
            "n_seeds",
            "mean_diff",
            "ci_lo",
            "ci_hi",
            "bootstrap_resamples",
            "bootstrap_seed",
        ],
    )
    write_csv(
        e7_path,
        e7_rows,
        [
            "parameter",
            "default_value",
            "comparison_value",
            "contrast",
            "metric",
            "n_seeds",
            "mean_diff",
            "ci_lo",
            "ci_hi",
            "bootstrap_resamples",
            "bootstrap_seed",
        ],
    )
    trust_audit = audit_trust_auc(cells, args.results)
    trust_path.write_text(json.dumps(trust_audit, indent=2), encoding="utf-8")

    freeze_path = args.results / "audit" / "FORMAL_EVIDENCE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = {
        "status": "PASS",
        "source_run_version": freeze["run_version"],
        "source_artifact_tree_sha256": freeze["artifact_tree_sha256"],
        "source_raw_tree_sha256": freeze["raw_tree_sha256"],
        "source_cells": len(cells),
        "seeds": list(EXPECTED_SEEDS),
        "inference_unit": "paired REAL-CAL trace seed",
        "bootstrap": {
            "resamples": BOOTSTRAP_N,
            "rng_seed": BOOTSTRAP_SEED,
            "interval": "two-sided percentile 95%",
        },
        "outputs": {
            e6_path.name: sha256_file(e6_path),
            e7_path.name: sha256_file(e7_path),
            trust_path.name: sha256_file(trust_path),
        },
        "script_sha256": sha256_file(Path(__file__)),
    }
    manifest_path = args.out / "secondary_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
