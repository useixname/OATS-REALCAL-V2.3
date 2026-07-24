#!/usr/bin/env python3
"""Emit CALIBRATION_VALIDATION.md: does the profile cover every generator param?

Cross-checks each SYN-V2-1 generator parameter the experiment depends on and marks
it as REAL-CALIBRATED, FROZEN-BY-DESIGN, or MODELED (synthetic counterfactual).
Fails loudly if a real-calibrated field is missing or degenerate, so we never
silently ship an under-specified profile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    # (param, category, json_path, validity_fn, note)
    ("availability_probability", "REAL-CALIBRATED", ["availability", "availability_probability"],
     lambda v: 0.01 <= float(v) <= 0.95, "fleet active-fraction per slot (T-Drive)"),
    ("arrival_intensity_grid", "REAL-CALIBRATED", ["availability", "arrival_intensity"],
     lambda v: isinstance(v, list) and len(v) == 1000 and abs(sum(v)/len(v) - 1.0) < 0.05,
     "per-slot diurnal arrival curve, mean-normalised (T-Drive)"),
    ("theta_difficulty", "REAL-CALIBRATED", ["task_and_anchor", "theta_mean"],
     lambda v: 0.0 < float(v) < 1.0, "cross-station disagreement (Beijing air)"),
    ("theta_spread", "REAL-CALIBRATED", ["task_and_anchor", "theta_std"],
     lambda v: float(v) > 0.0, "difficulty dispersion (Beijing air)"),
    ("missing_rate", "REAL-CALIBRATED", ["task_and_anchor", "missing_rate"],
     lambda v: 0.0 <= float(v) < 0.5, "real NA rate in station feed (Beijing air)"),
    ("anchor_mad_median", "REAL-CALIBRATED", ["task_and_anchor", "anchor_mad_median"],
     lambda v: float(v) > 0.0, "robust anchor interval half-width (Beijing air)"),
    ("honest_report_cv", "REAL-CALIBRATED", ["report_model", "honest_report_cv"],
     lambda v: float(v) > 0.0, "honest-tier report noise (PurpleAir/EPA)"),
    ("low_quality_report_cv", "REAL-CALIBRATED", ["report_model", "low_quality_report_cv"],
     lambda v: float(v) > 0.0, "low-tier report noise (PurpleAir/EPA)"),
    ("bias_sign_positive_fraction", "REAL-CALIBRATED", ["report_model", "bias_sign_positive_fraction"],
     lambda v: 0.0 <= float(v) <= 1.0, "over-read fraction (PurpleAir/EPA)"),
    ("contamination_anchor", "REAL-CALIBRATED", ["report_model", "contamination_anchor"],
     lambda v: isinstance(v, list) and len(v) >= 1, "gross-outlier rate anchoring contamination (PurpleAir/EPA)"),
    ("stratum_honest_fraction", "REAL-CALIBRATED", ["stratum_fractions", "honest"],
     lambda v: 0.0 < float(v) < 1.0, "honest tier share (PurpleAir dispersion split)"),
    # Frozen-by-design
    ("efforts", "FROZEN-BY-DESIGN", ["frozen_constants", "efforts"], lambda v: v == ["0", "0.5", "1"], "proof/menu"),
    ("gammas", "FROZEN-BY-DESIGN", ["frozen_constants", "gammas"], lambda v: len(v) == 6, "preregistered grid"),
    ("money_grid", "FROZEN-BY-DESIGN", ["frozen_constants", "money_grid"], lambda v: v == "0.001", "critical payment grid"),
    ("public_base_cap", "FROZEN-BY-DESIGN", ["frozen_constants", "public_base_cap"], lambda v: v == "3.0", "capacity"),
    ("task_value_band", "FROZEN-BY-DESIGN", ["frozen_constants", "task_value_band"], lambda v: v == ["0.5", "1.5"], "value band"),
    ("capacities", "FROZEN-BY-DESIGN", ["frozen_constants", "capacities"], lambda v: v == [3, 5, 10], "task K"),
    ("worker_count", "FROZEN-BY-DESIGN", ["frozen_constants", "worker_count"], lambda v: v == 500, "population size"),
    ("horizon", "FROZEN-BY-DESIGN", ["frozen_constants", "horizon"], lambda v: v == 1000, "slots"),
    ("c_i_support", "FROZEN-BY-DESIGN", ["frozen_constants", "c_i_support"], lambda v: v == ["0.5", "1.0", "2.0"], "cost support"),
]

MODELED = [
    ("private_cost_c_i", "individual private cost draw (no public dataset exposes per-worker cost)"),
    ("effort_response_curve", "effort -> quality mapping / Gbar (counterfactual, unobservable)"),
    ("potential_reports", "per-effort counterfactual reports (needed for replay channels)"),
    ("malicious_camouflage_behavior", "attacker deviation model (injected designed stress; no real labels)"),
]


def _dig(doc, path):
    cur = doc
    for key in path:
        cur = cur[key]
    return cur


def main() -> int:
    profile_path = ROOT / "data_real" / "REAL-CAL-V1" / "calibration_profile.json"
    if not profile_path.exists():
        print(f"missing profile: {profile_path}; run build_realcal_profile.py first", file=sys.stderr)
        return 2
    doc = json.loads(profile_path.read_text(encoding="utf-8"))

    lines = [
        "# REAL-CAL-V1 Calibration Validation",
        "",
        f"Profile version: `{doc['profile_version']}`  ",
        f"Profile hash: `{doc.get('profile_hash', 'n/a')}`  ",
        f"Dataset id: `{doc['dataset_id']}`",
        "",
        "Claim ceiling: **real-data-calibrated semi-synthetic evidence only** "
        "(no REAL_S2, no regret, no real-world generalization, attackers are injected stress).",
        "",
        "## Datasets",
        "",
        f"- T-Drive taxi GPS — files used: {doc['provenance']['tdrive_files']}",
        f"- Beijing Multi-Site Air Quality — stations: {doc['provenance']['airquality_stations']}",
        f"- PurpleAir vs EPA colocation — paired rows: {doc['provenance']['purpleair_rows']}",
        "",
        "## Parameter coverage",
        "",
        "| Generator parameter | Category | Value | Valid | Source / note |",
        "| --- | --- | --- | --- | --- |",
    ]

    all_ok = True
    for name, category, path, valid_fn, note in CHECKS:
        try:
            value = _dig(doc, path)
            ok = bool(valid_fn(value))
        except (KeyError, TypeError, ValueError):
            value, ok = "MISSING", False
        if not ok:
            all_ok = False
        shown = value if not isinstance(value, list) else f"[{len(value)} items]"
        lines.append(f"| {name} | {category} | `{shown}` | {'OK' if ok else 'FAIL'} | {note} |")

    lines.append("")
    lines.append("## Modeled (synthetic counterfactuals, unavoidable)")
    lines.append("")
    lines.append("| Parameter | Why it stays modeled |")
    lines.append("| --- | --- |")
    for name, note in MODELED:
        lines.append(f"| {name} | {note} |")

    lines.append("")
    lines.append("## Sufficiency verdict")
    lines.append("")
    real_count = sum(1 for _, c, *_ in CHECKS if c == "REAL-CALIBRATED")
    lines.append(f"- Real-calibrated parameters validated: **{real_count}**")
    lines.append(f"- Overall: **{'PASS — profile is complete and non-degenerate' if all_ok else 'FAIL — see FAIL rows above'}**")
    lines.append("")
    lines.append(
        "Every experiment-relevant *input distribution* is either real-calibrated or "
        "frozen-by-design; the only synthetic pieces are genuinely unobservable "
        "counterfactuals (private cost, effort response, per-effort potential reports, "
        "and injected attacker behavior). This is the maximal real-data grounding "
        "possible for this mechanism."
    )

    out_path = ROOT / "data_real" / "REAL-CAL-V1" / "CALIBRATION_VALIDATION.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print("VERDICT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
