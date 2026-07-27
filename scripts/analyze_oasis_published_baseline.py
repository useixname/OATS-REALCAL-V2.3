from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import statistics
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OASIS_ROOT = ROOT / "results" / "published_baseline_oasis_20260727_r1"
OATS_ROOT = ROOT / "results" / "formal_realcal_trustfix_20260726" / "raw"
REPORT_ROOT = ROOT / "reports" / "published_baseline_oasis_20260727"
BETAS = ("0.03", "0.05", "0.10", "0.25", "0.50")
SEEDS = tuple(range(20260715, 20260725))
ZERO = Decimal("0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _oats_path(seed: int, beta: str) -> Path:
    family = "E1B_TIGHT" if beta in {"0.03", "0.05"} else "E1_OVERALL"
    return (
        OATS_ROOT
        / (
            f"{family}_s{seed}_mV2-FULL_g0.3_b{beta}"
            "_c0_d0_m0_a1.json"
        )
    )


def _oasis_path(seed: int, beta: str) -> Path:
    slug = str(Decimal(beta)).replace(".", "p")
    return OASIS_ROOT / "formal" / "cells" / f"seed_{seed}_beta_{slug}.json"


def _decimal(payload: Mapping[str, object], key: str) -> Decimal | None:
    value = payload.get(key)
    return None if value is None else Decimal(str(value))


def _verify_cell_hash(payload: Mapping[str, object]) -> None:
    unsigned = dict(payload)
    supplied = str(unsigned.pop("cell_hash"))
    if canonical_hash(unsigned) != supplied:
        raise RuntimeError("Oasis cell hash mismatch")


def _bootstrap_ci(
    values: list[Decimal],
    *,
    seed: int,
    draws: int = 10_000,
) -> tuple[Decimal, Decimal]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum((values[rng.randrange(n)] for _ in range(n)), ZERO) / Decimal(n)
        for _ in range(draws)
    )
    lower = means[int(Decimal("0.025") * draws)]
    upper = means[min(draws - 1, int(Decimal("0.975") * draws))]
    return lower, upper


def _exact_sign_flip_p(values: list[Decimal]) -> Decimal:
    observed = abs(sum(values, ZERO) / Decimal(len(values)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        statistic = abs(
            sum(
                (value * Decimal(sign) for value, sign in zip(values, signs)),
                ZERO,
            )
            / Decimal(len(values))
        )
        exceed += int(statistic >= observed)
        total += 1
    return Decimal(exceed) / Decimal(total)


def _holm(raw: Mapping[str, Decimal]) -> dict[str, Decimal]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    m = len(ordered)
    adjusted: dict[str, Decimal] = {}
    running = ZERO
    for rank, (key, value) in enumerate(ordered):
        candidate = min(Decimal("1"), Decimal(m - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _mean(values: Iterable[Decimal]) -> Decimal:
    materialized = list(values)
    return sum(materialized, ZERO) / Decimal(len(materialized))


def main() -> None:
    manifest_path = OASIS_ROOT / "formal" / "STAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cell_count") != 50:
        raise RuntimeError("formal Oasis manifest does not contain 50 cells")
    if manifest.get("truth_nonconvergence_count") != 0:
        raise RuntimeError("formal Oasis truth-discovery gate failed")
    if manifest.get("basic_payment_budget_violation_count") != 0:
        raise RuntimeError("formal Oasis basic-payment budget gate failed")

    paired_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    metric_differences: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
    metrics = {
        "platform_net_value": ("platform_net_value", "platform_net_value"),
        "gross_external_value": ("gross_external_value", "gross_external_value"),
        "actual_payment": ("total_paid", "total_actual_payment"),
        "selected_quality": ("channels.selected_quality", "mean_selected_external_quality"),
        "selected_count": ("purchased_count", "recruited_count"),
        "budget_efficiency": ("derived", "budget_efficiency"),
        "registered_net_per_payment": ("derived", "budget_efficiency"),
    }

    for beta in BETAS:
        for seed in SEEDS:
            oasis_path = _oasis_path(seed, beta)
            oats_path = _oats_path(seed, beta)
            oasis = json.loads(oasis_path.read_text(encoding="utf-8"))
            oats = json.loads(oats_path.read_text(encoding="utf-8"))
            _verify_cell_hash(oasis)
            if (
                int(oasis["trace_seed"]) != seed
                or Decimal(str(oasis["beta"])) != Decimal(beta)
                or oats.get("method_id") != "V2-FULL"
                or int(oats["seed"]) != seed
                or Decimal(str(oats["budget_ratio"])) != Decimal(beta)
                or oats.get("invariant_status") != "PASS"
            ):
                raise RuntimeError(f"source identity mismatch: seed={seed}, beta={beta}")
            trust = dict(oats["trust"])
            if not (
                int(trust["feedback_count"])
                == int(trust["trust_transition_count"])
                and int(trust["duplicate_feedback_suppressed_count"]) == 0
            ):
                raise RuntimeError(f"OATS trust audit mismatch: seed={seed}, beta={beta}")
            oasis_budget = Decimal(str(oasis["run_budget"])).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            oats_budget = Decimal(str(oats["final_ledger"]["budget"]))
            if oasis_budget != oats_budget:
                raise RuntimeError(
                    f"common budget mismatch: seed={seed}, beta={beta}, "
                    f"{oasis_budget} != {oats_budget}"
                )

            economic_eligible = oasis["comparison_status"] == "ELIGIBLE"
            oats_payment = Decimal(str(oats["total_paid"]))
            values: dict[str, tuple[Decimal | None, Decimal | None]] = {
                "platform_net_value": (
                    Decimal(str(oats["platform_net_value"])),
                    (
                        Decimal(str(oasis["platform_net_value"]))
                        if economic_eligible
                        else None
                    ),
                ),
                "gross_external_value": (
                    Decimal(str(oats["gross_external_value"])),
                    (
                        Decimal(str(oasis["gross_external_value"]))
                        if economic_eligible
                        else None
                    ),
                ),
                "actual_payment": (
                    oats_payment,
                    (
                        Decimal(str(oasis["total_actual_payment"]))
                        if economic_eligible
                        else None
                    ),
                ),
                "selected_quality": (
                    Decimal(str(oats["channels"]["selected_quality"])),
                    _decimal(oasis, "mean_selected_external_quality"),
                ),
                "selected_count": (
                    Decimal(int(oats["purchased_count"])),
                    Decimal(int(oasis["recruited_count"])),
                ),
                "budget_efficiency": (
                    Decimal(str(oats["gross_external_value"])) / oats_payment,
                    (
                        Decimal(str(oasis["gross_external_value"]))
                        / Decimal(str(oasis["total_actual_payment"]))
                        if economic_eligible
                        else None
                    ),
                ),
                "registered_net_per_payment": (
                    Decimal(str(oats["platform_net_value"])) / oats_payment,
                    (
                        _decimal(oasis, "budget_efficiency")
                        if economic_eligible
                        else None
                    ),
                ),
            }
            row: dict[str, object] = {
                "seed": seed,
                "beta": beta,
                "oasis_comparison_status": oasis["comparison_status"],
                "economic_comparison_eligible": economic_eligible,
                "oasis_actual_budget_violation_count": oasis[
                    "actual_payment_budget_violation_count"
                ],
                "oasis_ir_violation_count": oasis[
                    "individual_rationality_violation_count"
                ],
            }
            for metric in metrics:
                oats_value, oasis_value = values[metric]
                difference = (
                    None
                    if oats_value is None or oasis_value is None
                    else oats_value - oasis_value
                )
                row[f"oats_{metric}"] = (
                    None if oats_value is None else str(oats_value)
                )
                row[f"oasis_{metric}"] = (
                    None if oasis_value is None else str(oasis_value)
                )
                row[f"difference_oats_minus_oasis_{metric}"] = (
                    None if difference is None else str(difference)
                )
                if difference is not None:
                    metric_differences[(beta, metric)].append(difference)
            paired_rows.append(row)
            provenance_rows.append(
                {
                    "seed": seed,
                    "beta": beta,
                    "oats_path": oats_path.relative_to(ROOT).as_posix(),
                    "oats_sha256": sha256_file(oats_path),
                    "oats_cell_id": oats["cell_id"],
                    "oasis_path": oasis_path.relative_to(ROOT).as_posix(),
                    "oasis_sha256": sha256_file(oasis_path),
                    "oasis_cell_hash": oasis["cell_hash"],
                    "common_budget_rounded_0p001": str(oats_budget),
                    "trace_hash_identity": (
                        canonical_hash(oasis["trace_file_hashes"])
                    ),
                }
            )

    statistics_rows: list[dict[str, object]] = []
    raw_net_p: dict[str, Decimal] = {}
    economic_metrics = {
        "platform_net_value",
        "gross_external_value",
        "actual_payment",
        "budget_efficiency",
        "registered_net_per_payment",
    }
    for beta in BETAS:
        for metric in metrics:
            values = metric_differences.get((beta, metric), [])
            if not values or (metric in economic_metrics and len(values) != 10):
                statistics_rows.append(
                    {
                        "beta": beta,
                        "metric": metric,
                        "paired_seed_count": len(values),
                        "status": (
                            "NA_PAYMENT_BUDGET_GATE_FAILED"
                            if not values
                            else "NA_INCOMPLETE_BETA_PAYMENT_GATE"
                        ),
                    }
                )
                continue
            lower, upper = _bootstrap_ci(
                values,
                seed=20260715,
            )
            p_value = _exact_sign_flip_p(values)
            row = {
                "beta": beta,
                "metric": metric,
                "paired_seed_count": len(values),
                "status": "REPORTED",
                "mean_difference_oats_minus_oasis": str(_mean(values)),
                "median_difference_oats_minus_oasis": str(statistics.median(values)),
                "positive_seed_count": sum(value > ZERO for value in values),
                "negative_seed_count": sum(value < ZERO for value in values),
                "zero_seed_count": sum(value == ZERO for value in values),
                "paired_bootstrap_95_lower": str(lower),
                "paired_bootstrap_95_upper": str(upper),
                "exact_sign_flip_p": str(p_value),
            }
            statistics_rows.append(row)
            if metric == "platform_net_value":
                raw_net_p[beta] = p_value

    holm = _holm(raw_net_p)
    for row in statistics_rows:
        if row["metric"] == "platform_net_value" and row["beta"] in holm:
            row["holm_adjusted_p"] = str(holm[str(row["beta"])])

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paired_csv = REPORT_ROOT / "oats_oasis_paired_cells.csv"
    with paired_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    statistics_csv = REPORT_ROOT / "oats_oasis_paired_statistics.csv"
    with statistics_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted(
            {key for row in statistics_rows for key in row},
            key=lambda key: (
                [
                    "beta",
                    "metric",
                    "paired_seed_count",
                    "status",
                    "mean_difference_oats_minus_oasis",
                    "median_difference_oats_minus_oasis",
                    "positive_seed_count",
                    "negative_seed_count",
                    "zero_seed_count",
                    "paired_bootstrap_95_lower",
                    "paired_bootstrap_95_upper",
                    "exact_sign_flip_p",
                    "holm_adjusted_p",
                ].index(key)
                if key
                in {
                    "beta",
                    "metric",
                    "paired_seed_count",
                    "status",
                    "mean_difference_oats_minus_oasis",
                    "median_difference_oats_minus_oasis",
                    "positive_seed_count",
                    "negative_seed_count",
                    "zero_seed_count",
                    "paired_bootstrap_95_lower",
                    "paired_bootstrap_95_upper",
                    "exact_sign_flip_p",
                    "holm_adjusted_p",
                }
                else 99
            ),
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(statistics_rows)
    provenance_csv = REPORT_ROOT / "oats_oasis_provenance.csv"
    with provenance_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance_rows[0]))
        writer.writeheader()
        writer.writerows(provenance_rows)

    eligible_betas = [
        beta
        for beta in BETAS
        if len(metric_differences.get((beta, "platform_net_value"), [])) == 10
    ]
    failed_betas = [beta for beta in BETAS if beta not in eligible_betas]
    summary = {
        "status": (
            "FULL_ECONOMIC_COMPARISON"
            if not failed_betas
            else (
                "PARTIAL_ECONOMIC_COMPARISON"
                if eligible_betas
                else "CONTRACT_INCOMPATIBILITY_ONLY"
            )
        ),
        "published_baseline": "Oasis",
        "paper_doi": "10.1109/TSC.2024.3354240",
        "parent_freeze_id": "OATS-TRUSTFIX-PRE-PUBLISHED-BASELINE-20260727-R1",
        "formal_oasis_manifest_hash": manifest["manifest_hash"],
        "eligible_betas": eligible_betas,
        "failed_betas": failed_betas,
        "paired_cell_count": len(paired_rows),
        "economic_eligible_cell_count": sum(
            bool(row["economic_comparison_eligible"]) for row in paired_rows
        ),
        "total_oasis_actual_budget_violations": sum(
            int(row["oasis_actual_budget_violation_count"]) for row in paired_rows
        ),
        "total_oasis_ir_violations": sum(
            int(row["oasis_ir_violation_count"]) for row in paired_rows
        ),
        "outputs": {
            "paired_cells": paired_csv.name,
            "paired_statistics": statistics_csv.name,
            "provenance": provenance_csv.name,
        },
    }
    summary["summary_hash"] = canonical_hash(summary)
    summary_path = REPORT_ROOT / "OATS_OASIS_FORMAL_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# 已发表端到端基线正式审计",
        "",
        f"结论：`{summary['status']}`。",
        "",
        "- baseline：Oasis（IEEE TSC 2024，DOI 10.1109/TSC.2024.3354240）；",
        "- 公共 traces：REAL-CAL-V2 evaluation seeds 20260715--20260724；",
        "- 公共预算：每个 seed/beta 的 OATS 绝对预算与 Oasis task budgets 总和",
        "  在 0.001 报告精度严格相等；",
        f"- 正式 paired cells：{len(paired_rows)}；",
        f"- 可进入经济排名的 cells：{summary['economic_eligible_cell_count']}；",
        f"- Oasis actual-payment task-budget violations："
        f"{summary['total_oasis_actual_budget_violations']}；",
        f"- Oasis worker-IR violations：{summary['total_oasis_ir_violations']}。",
        "",
        "## 解释边界",
        "",
        "Oasis 的动态 recruitment、truth discovery、long-term quality 与两级支付均按",
        "论文公式运行；必要但论文未唯一指定的标量距离、到达顺序和同槽更新顺序已在",
        "打开 evaluation results 前冻结。Oasis 不含 OATS 的 effort action、delayed",
        "outcome escrow 或全局跨任务预算定理，因此本轮只比较 delay=0、missing=0",
        "的最小公共端到端子问题。",
        "",
    ]
    if failed_betas:
        report_lines.extend(
            [
                "actual payment 超过 task budget 的 cells 没有被裁剪或事后归一化；",
                "这些 cells 的 net value、payment 和 efficiency 均按预注册规则标为 NA，",
                "不能写成 OATS 相对 Oasis 的经济胜负。selected quality 与 recruited",
                "count 仍作为非经济的最小公共子问题结果保留。",
                "",
            ]
        )
    report_path = REPORT_ROOT / "OATS_OASIS_FORMAL_AUDIT_ZH.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
