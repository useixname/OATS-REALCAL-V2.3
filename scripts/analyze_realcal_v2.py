#!/usr/bin/env python3
"""REAL-CAL-V2 analysis: produce E1-E8 tables (实验.md) + hypothesis verdicts.

Reads results/formal_realcal_v2/raw/*.json and writes CSV/JSON artifacts to
reports/realcal_v2_analysis/. Pure post-processing; no simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_parser = argparse.ArgumentParser()
_parser.add_argument("--results", type=Path, default=ROOT / "results" / "formal_realcal_v2")
_parser.add_argument("--out", type=Path, default=ROOT / "reports" / "realcal_v2_analysis")
_args = _parser.parse_args()
RAW = _args.results / "raw"
OUT = _args.out

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260715


def fnum(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_cells() -> list[dict]:
    cells = []
    for path in sorted(RAW.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_family"] = payload["cell_id"].split("_s2026")[0]
        cells.append(payload)
    return cells


def parse_flags(cell_id: str) -> dict:
    # ..._g{gamma}_b{budget}_c{cont}_d{delay}_m{missing}_a{arrival}[_al{a}_ta{t}_lm{l}]
    parts = cell_id.split("_")
    flags = {}
    for part in parts:
        for prefix in ("g", "b", "c", "d", "m", "a", "al", "ta", "lm"):
            if part.startswith(prefix) and len(part) > len(prefix):
                rest = part[len(prefix):]
                try:
                    float(rest)
                except ValueError:
                    continue
                flags.setdefault(prefix, rest)
    return flags


def cell_metrics(c: dict) -> dict:
    paid = fnum(c.get("total_paid"), 0.0)
    gross = fnum(c.get("gross_external_value"), 0.0)
    return {
        "net": fnum(c.get("platform_net_value"), 0.0),
        "gross": gross,
        "paid": paid,
        "base_paid": fnum(c.get("base_paid"), 0.0),
        "score_paid": fnum(c.get("score_paid"), 0.0),
        "budget_efficiency": (gross / paid) if paid else None,
        "purchased": c.get("purchased_count", 0),
        "deadline": fnum(c.get("deadline_satisfaction")),
        "selected_quality": fnum((c.get("channels") or {}).get("selected_quality")),
        "population_quality": fnum((c.get("channels") or {}).get("population_quality")),
        "effort_effect": fnum((c.get("channels") or {}).get("effort_effect")),
        "mrr": fnum((c.get("screening") or {}).get("mrr")),
        "hfr": fnum((c.get("screening") or {}).get("hfr")),
        "honest_fail": fnum((c.get("screening") or {}).get("honest_fail")),
        "colluder_pass": fnum((c.get("screening") or {}).get("colluder_pass")),
        "soft_pass": fnum((c.get("screening") or {}).get("soft_pass_rate")),
        "brier": fnum((c.get("trust") or {}).get("brier")),
        "auc": fnum((c.get("trust") or {}).get("auc")),
        "mc_rho": fnum(c.get("mc_correlation")),
        "mc_topk": fnum(c.get("mc_top_k_overlap")),
        "mc_undefined": fnum(c.get("mc_undefined_rate")),
    }


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def paired_bootstrap(diffs: list[float]) -> tuple[float, float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = []
    for _ in range(BOOTSTRAP_N):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return (
        statistics.fmean(diffs),
        means[int(0.025 * BOOTSTRAP_N)],
        means[int(0.975 * BOOTSTRAP_N)],
    )


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print(f"loaded {len(cells)} raw cells")
    invalid = [c["cell_id"] for c in cells if c.get("invariant_status") != "PASS"]
    summary: dict = {"total_cells": len(cells), "invalid_cells": invalid}

    # ---------------- E1: overall comparison ----------------
    e1 = [c for c in cells if c["_family"] == "E1_OVERALL"]
    by_mb: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in e1:
        by_mb[(c["method_id"], c["budget_ratio"])].append(cell_metrics(c))
    rows = []
    for (method, budget), ms in sorted(by_mb.items()):
        rows.append(
            [
                method,
                budget,
                len(ms),
                mean([m["net"] for m in ms]),
                mean([m["gross"] for m in ms]),
                mean([m["paid"] for m in ms]),
                mean([m["budget_efficiency"] for m in ms]),
                mean([m["deadline"] for m in ms]),
                mean([m["selected_quality"] for m in ms]),
                mean([m["purchased"] for m in ms]),
            ]
        )
    write_csv(
        OUT / "e1_method_budget_summary.csv",
        ["method", "budget", "n", "net", "gross", "paid", "budget_efficiency", "deadline", "selected_quality", "purchased"],
        rows,
    )

    # paired V2-FULL vs baselines at each budget
    paired_rows = []
    h1_pass = {}
    for budget in ("0.10", "0.25", "0.50"):
        base_cells = {c["seed"]: cell_metrics(c) for c in e1 if c["method_id"] == "V2-FULL" and c["budget_ratio"] == budget}
        for method in ("B-RANDOM", "B-COST", "B-TRUST", "B-QUALITY", "B-NODUAL", "B-MYOPIC"):
            other = {c["seed"]: cell_metrics(c) for c in e1 if c["method_id"] == method and c["budget_ratio"] == budget}
            seeds = sorted(set(base_cells) & set(other))
            if not seeds:
                continue
            for metric in ("net", "gross", "budget_efficiency"):
                diffs = [base_cells[s][metric] - other[s][metric] for s in seeds if base_cells[s][metric] is not None and other[s][metric] is not None]
                if not diffs:
                    continue
                m, lo, hi = paired_bootstrap(diffs)
                paired_rows.append([budget, f"V2-FULL - {method}", metric, len(diffs), m, lo, hi])
                if budget == "0.25" and metric == "net" and method in ("B-RANDOM", "B-COST", "B-TRUST", "B-QUALITY"):
                    h1_pass[method] = lo > 0
    write_csv(
        OUT / "e1_paired_contrasts.csv",
        ["budget", "contrast", "metric", "n_seeds", "mean_diff", "ci_lo", "ci_hi"],
        paired_rows,
    )
    summary["H1_net_beats_naive_baselines_b025"] = h1_pass
    summary["H1_pass"] = bool(h1_pass) and all(h1_pass.values())

    # ---------------- E1B (V2.1): tight budgets — dual's intended regime ----
    e1b = [c for c in cells if c["_family"] == "E1B_TIGHT"]
    if e1b:
        rows = []
        by_mb_t: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for c in e1b:
            by_mb_t[(c["method_id"], c["budget_ratio"])].append(cell_metrics(c))
        for (method, budget), ms in sorted(by_mb_t.items()):
            rows.append(
                [
                    method,
                    budget,
                    len(ms),
                    mean([m["net"] for m in ms]),
                    mean([m["gross"] for m in ms]),
                    mean([m["paid"] for m in ms]),
                    mean([m["budget_efficiency"] for m in ms]),
                    mean([m["purchased"] for m in ms]),
                ]
            )
        write_csv(
            OUT / "e1b_tight_budget_summary.csv",
            ["method", "budget", "n", "net", "gross", "paid", "budget_efficiency", "purchased"],
            rows,
        )
        tight_rows = []
        h1b = {}
        for budget in ("0.03", "0.05"):
            base_cells = {c["seed"]: cell_metrics(c) for c in e1b if c["method_id"] == "V2-FULL" and c["budget_ratio"] == budget}
            for method in ("B-NODUAL", "B-COST", "B-MYOPIC"):
                other = {c["seed"]: cell_metrics(c) for c in e1b if c["method_id"] == method and c["budget_ratio"] == budget}
                seeds = sorted(set(base_cells) & set(other))
                for metric in ("net", "gross", "budget_efficiency"):
                    diffs = [base_cells[s][metric] - other[s][metric] for s in seeds if base_cells[s][metric] is not None and other[s][metric] is not None]
                    if not diffs:
                        continue
                    m, lo, hi = paired_bootstrap(diffs)
                    tight_rows.append([budget, f"V2-FULL - {method}", metric, len(diffs), m, lo, hi])
                    if metric == "net":
                        h1b[f"{method}@b{budget}"] = lo > 0
        write_csv(
            OUT / "e1b_tight_paired_contrasts.csv",
            ["budget", "contrast", "metric", "n_seeds", "mean_diff", "ci_lo", "ci_hi"],
            tight_rows,
        )
        summary["H1b_dual_advantage_tight_budget"] = h1b

    # ---------------- E2: value identification ----------------
    rows = []
    h2 = {}
    for method in ("V2-FULL", "B-NOTRUST", "B-NODUAL", "B-MYOPIC", "B-COST", "B-RANDOM"):
        source = e1 if method != "B-NOTRUST" else [c for c in cells if c["_family"] == "E6_ABLATION"]
        ms = [cell_metrics(c) for c in source if c["method_id"] == method and c["budget_ratio"] == "0.25"]
        if not ms:
            continue
        rows.append([method, len(ms), mean([m["mc_rho"] for m in ms]), mean([m["mc_topk"] for m in ms]), mean([m["mc_undefined"] for m in ms])])
        h2[method] = mean([m["mc_rho"] for m in ms])
    write_csv(OUT / "e2_value_identification.csv", ["method", "n", "spearman_rho", "top10_overlap", "zero_value_rate"], rows)
    v2_rho = h2.get("V2-FULL")
    summary["H2_pass"] = bool(v2_rho and v2_rho > 0.5 and (h2.get("B-NOTRUST") is None or v2_rho > h2["B-NOTRUST"]))
    summary["H2_spearman"] = h2

    # ---------------- E3: screening under contamination ----------------
    e3 = [c for c in cells if c["_family"] == "E3_SCREENING"]
    rows = []
    h3_ok = True
    for method in ("V2-FULL", "B-NOSCREEN"):
        for cont in ("0.1", "0.2", "0.3", "0.4", "0.5"):
            ms = [cell_metrics(c) for c in e3 if c["method_id"] == method and parse_flags(c["cell_id"]).get("c") == cont]
            if not ms:
                continue
            purchased = mean([m["purchased"] for m in ms])
            rows.append(
                [
                    method,
                    cont,
                    len(ms),
                    mean([m["mrr"] for m in ms]),
                    mean([m["hfr"] for m in ms]),
                    mean([m["soft_pass"] for m in ms]),
                    purchased,
                    mean([m["gross"] for m in ms]),
                    mean([m["net"] for m in ms]),
                ]
            )
            if method == "V2-FULL":
                hfr = mean([m["hfr"] for m in ms]) or 0
                if hfr > 0.5 or (purchased or 0) < 1000:
                    h3_ok = False  # "reject everything" collapse
    write_csv(
        OUT / "e3_screening_contamination.csv",
        ["method", "contamination", "n", "mrr", "hfr", "soft_pass_rate", "purchased", "gross", "net"],
        rows,
    )
    summary["H3_no_collapse"] = h3_ok

    # ---------------- E4: trust evolution ----------------
    canonical = [
        c
        for c in e1
        if c["method_id"] == "V2-FULL" and c["budget_ratio"] == "0.25"
    ]
    traj_acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in canonical:
        for slot_str, strata in (c.get("trust_trajectory") or {}).items():
            for stratum, rho in strata.items():
                traj_acc[int(slot_str)][stratum].append(float(rho))
    rows = []
    for slot in sorted(traj_acc):
        row = [slot]
        for stratum in ("honest", "low-quality", "malicious", "camouflage"):
            row.append(mean(traj_acc[slot].get(stratum, [])))
        rows.append(row)
    write_csv(OUT / "e4_trust_trajectory.csv", ["slot", "honest", "low_quality", "malicious", "camouflage"], rows)
    aucs = [cell_metrics(c)["auc"] for c in canonical]
    summary["H5_trust_auc_mean"] = mean(aucs)
    camo_drop = None
    camo_500 = mean(traj_acc.get(500, {}).get("camouflage", []))
    camo_600 = mean(traj_acc.get(600, {}).get("camouflage", []))
    if camo_500 is not None and camo_600 is not None:
        camo_drop = camo_500 - camo_600
    summary["H5_camouflage_drop_500_600"] = camo_drop
    summary["H5_pass"] = bool((mean(aucs) or 0) >= 0.8 and (camo_drop or 0) >= 0.05)

    # ---------------- E5: adaptability + empirical regret ----------------
    e5 = [c for c in cells if c["_family"] == "E5_ADAPT"]
    rows = []
    for c in sorted(e5, key=lambda x: x["cell_id"]):
        flags = parse_flags(c["cell_id"])
        m = cell_metrics(c)
        rows.append(
            [
                flags.get("a"),
                flags.get("d"),
                flags.get("m"),
                c["seed"],
                m["net"],
                m["gross"],
                m["purchased"],
                m["deadline"],
                m["brier"],
            ]
        )
    write_csv(
        OUT / "e5_adaptability_cells.csv",
        ["arrival", "delay", "missing", "seed", "net", "gross", "purchased", "deadline", "brier"],
        rows,
    )
    # aggregate
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for c in e5:
        flags = parse_flags(c["cell_id"])
        agg[(flags.get("a"), flags.get("d"), flags.get("m"))].append(cell_metrics(c))
    rows = [
        [a, d, mi, len(ms), mean([m["net"] for m in ms]), mean([m["gross"] for m in ms]), mean([m["purchased"] for m in ms]), mean([m["deadline"] for m in ms])]
        for (a, d, mi), ms in sorted(agg.items())
    ]
    write_csv(
        OUT / "e5_adaptability_summary.csv",
        ["arrival", "delay", "missing", "n", "net", "gross", "purchased", "deadline"],
        rows,
    )

    # regret curve from lp_prefix on canonical E1 cells
    reg_acc: dict[int, list[float]] = defaultdict(list)
    gap_signs = []
    for c in canonical:
        prefix = c.get("lp_prefix") or {}
        for t_str, lp in prefix.items():
            opt = fnum(lp.get("opt_lp"))
            u = fnum(lp.get("u_online"))
            if opt is None or u is None:
                continue
            reg_acc[int(t_str)].append((opt - u) / int(t_str))
        lp_final = c.get("lp") or {}
        gap = fnum(lp_final.get("lp_gap"))
        if gap is not None:
            gap_signs.append(gap >= 0)
    rows = [[t, len(v), mean(v)] for t, v in sorted(reg_acc.items())]
    write_csv(OUT / "e5_regret_curve.csv", ["T", "n", "mean_regret_over_T"], rows)
    # H4: nonneg gap share across ALL cells with LP
    all_gaps = []
    for c in cells:
        gap = fnum((c.get("lp") or {}).get("lp_gap"))
        if gap is not None:
            all_gaps.append(gap >= -1e-6)
    summary["H4_nonneg_gap_share"] = (sum(all_gaps) / len(all_gaps)) if all_gaps else None
    reg_means = [mean(v) for _, v in sorted(reg_acc.items())]
    summary["H4_regret_over_T_curve"] = reg_means
    # Prereg H4'': Reg(T)/T non-increasing AFTER the warm-up phase (trust and
    # dual-price learning accumulate estimation regret early — the E_T term in
    # the paper's Theorem 7). Warm-up = up to the curve peak, which must occur
    # in the first 60% of the horizon; afterwards the average gap must decline
    # monotonically (5% noise tolerance).
    decreasing = None
    if len(reg_means) >= 3:
        peak = max(range(len(reg_means)), key=lambda i: reg_means[i])
        summary["H4_warmup_peak_T"] = (peak + 1) * 100
        decreasing = peak <= int(0.6 * len(reg_means)) and all(
            reg_means[i + 1] <= reg_means[i] * 1.05 for i in range(peak, len(reg_means) - 1)
        )
    summary["H4_pass"] = bool(((summary["H4_nonneg_gap_share"] or 0) >= 0.95) and decreasing)

    # ---------------- E6: ablation ----------------
    ablation_methods = {
        "V2-FULL": e1,
        "B-NODUAL": e1,
        "B-MYOPIC": e1,
        "B-NOSCREEN": cells,
        "B-NOTRUST": cells,
        "B-P1": cells,
        "C-EFFORT-OFF": cells,
    }
    rows = []
    for method, pool in ablation_methods.items():
        ms = [
            cell_metrics(c)
            for c in pool
            if c["method_id"] == method
            and c["budget_ratio"] == "0.25"
            and c["_family"] in ("E1_OVERALL", "E6_ABLATION")
            and c["gamma"] == "0.3"
        ]
        if not ms:
            continue
        rows.append(
            [
                method,
                len(ms),
                mean([m["net"] for m in ms]),
                mean([m["gross"] for m in ms]),
                mean([m["budget_efficiency"] for m in ms]),
                mean([m["selected_quality"] for m in ms]),
                mean([m["auc"] for m in ms]),
                mean([m["mrr"] for m in ms]),
            ]
        )
    write_csv(
        OUT / "e6_ablation.csv",
        ["method", "n", "net", "gross", "budget_efficiency", "selected_quality", "trust_auc", "mrr"],
        rows,
    )

    # ---------------- E7: sensitivity ----------------
    for fam, param_key, out_name in (
        ("E7_SENS_GAMMA", "g", "e7_gamma.csv"),
        ("E7_SENS_ALPHA", "al", "e7_alpha.csv"),
        ("E7_SENS_THETAA", "ta", "e7_theta_a.csv"),
        ("E7_SENS_LMAX", "lm", "e7_lambda_max.csv"),
    ):
        pool = [c for c in cells if c["_family"] == fam]
        # include the default point from E1 (V2-FULL, gamma 0.3, b 0.25)
        agg2: dict[str, list[dict]] = defaultdict(list)
        for c in pool:
            flags = parse_flags(c["cell_id"])
            agg2[flags.get(param_key, "?")].append(cell_metrics(c))
        default_map = {"g": "0.3", "al": "0.2", "ta": "0.75", "lm": "10"}
        for c in canonical:
            agg2[default_map[param_key]].append(cell_metrics(c))
        rows = []
        for value, ms in sorted(agg2.items(), key=lambda kv: float(kv[0]) if kv[0] != "?" else -1):
            rows.append(
                [
                    value,
                    len(ms),
                    mean([m["net"] for m in ms]),
                    mean([m["gross"] for m in ms]),
                    mean([m["paid"] for m in ms]),
                    mean([m["selected_quality"] for m in ms]),
                    mean([m["effort_effect"] for m in ms]),
                    mean([m["mrr"] for m in ms]),
                    mean([m["hfr"] for m in ms]),
                    mean([m["soft_pass"] for m in ms]),
                    mean([m["auc"] for m in ms]),
                ]
            )
        write_csv(
            OUT / out_name,
            ["value", "n", "net", "gross", "paid", "selected_quality", "effort_effect", "mrr", "hfr", "soft_pass", "trust_auc"],
            rows,
        )

    # ---------------- E8: runtime breakdown ----------------
    rt_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        rb = c.get("runtime_breakdown") or {}
        for key, val in rb.items():
            rt_acc[c["method_id"]][key].append(float(val))
    rows = []
    for method, comps in sorted(rt_acc.items()):
        rows.append(
            [
                method,
                len(comps.get("total", [])),
                mean(comps.get("total", [])),
                mean(comps.get("selection_and_critical_payment", [])),
                mean(comps.get("screening", [])),
                mean(comps.get("settlement", [])),
                mean(comps.get("other", [])),
            ]
        )
    write_csv(
        OUT / "e8_runtime_breakdown.csv",
        ["method", "n", "total_s", "selection_and_critical_payment_s", "screening_s", "settlement_s", "other_s"],
        rows,
    )

    # ---------------- H6 + summary ----------------
    summary["H6_pass"] = not invalid
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("H") or k == "total_cells"}, indent=2, default=str))
    print(f"artifacts -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
