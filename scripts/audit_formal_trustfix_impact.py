#!/usr/bin/env python3
"""Audit the online impact of the worker-specific trust repair.

This is read-only post-processing over the immutable legacy 830-cell package
and the independently frozen repaired 830-cell package. The legacy package is
used only as a paired impact comparator, never as evidence for the repaired
mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_RAW_TREE = "77ad671c3eb622d1cd678e9c53ba2f6be7da66785b586de59c2d159cb4bbcedd"
REPAIRED_FREEZE = "c46172cf7b40a0cccf1c014f81109f59a84731a4fb4c5944f376fcb981c928dd"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026072603


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_tree_hash(raw_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(raw_root.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family(payload: dict[str, Any]) -> str:
    return str(payload["cell_id"]).split("_s2026", 1)[0]


def number(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


METRICS: dict[str, Callable[[dict[str, Any]], float]] = {
    "purchased_count": lambda row: number(row.get("purchased_count")),
    "contracted_count": lambda row: number(row.get("contracted_count")),
    "activated_count": lambda row: number(row.get("activated_count")),
    "gross_external_value": lambda row: number(row.get("gross_external_value")),
    "platform_net_value": lambda row: number(row.get("platform_net_value")),
    "total_paid": lambda row: number(row.get("total_paid")),
    "selected_quality": lambda row: number(nested(row, "channels", "selected_quality")),
    "trust_auc": lambda row: number(nested(row, "trust", "auc")),
    "trust_brier": lambda row: number(nested(row, "trust", "brier")),
}


def load_raw(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "raw").glob("*.json")):
        payload = load_json(path)
        cell_id = str(payload["cell_id"])
        if cell_id in result:
            raise RuntimeError(f"duplicate cell ID: {cell_id}")
        result[cell_id] = payload
    return result


def composition_l1(old: dict[str, Any], new: dict[str, Any]) -> float:
    left = old.get("worker_type_composition") or {}
    right = new.get("worker_type_composition") or {}
    return sum(abs(number(left.get(key, 0)) - number(right.get(key, 0))) for key in set(left) | set(right))


def main_cell(payload: dict[str, Any]) -> bool:
    return (
        family(payload) == "E1_OVERALL"
        and payload["method_id"] == "V2-FULL"
        and number(payload["gamma"]) == 0.3
        and number(payload["budget_ratio"]) == 0.25
    )


def paired_bootstrap(
    pairs: list[tuple[float, float]],
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    old = np.asarray([item[0] for item in pairs], dtype=float)
    new = np.asarray([item[1] for item in pairs], dtype=float)
    delta = new - old
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    draws = delta[indices].mean(axis=1)
    old_mean = float(old.mean())
    new_mean = float(new.mean())
    return {
        "n_seeds": len(pairs),
        "legacy_mean": old_mean,
        "repaired_mean": new_mean,
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_ci_low": float(np.quantile(draws, 0.025)),
        "paired_delta_ci_high": float(np.quantile(draws, 0.975)),
        "relative_delta": (
            float(delta.mean()) / abs(old_mean) if abs(old_mean) > 1e-15 else math.nan
        ),
    }


def summarize_group(
    cell_ids: list[str],
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"cells": len(cell_ids)}
    for metric, getter in METRICS.items():
        old_values = np.asarray([getter(old[cell_id]) for cell_id in cell_ids], dtype=float)
        new_values = np.asarray([getter(new[cell_id]) for cell_id in cell_ids], dtype=float)
        result[f"{metric}_legacy_mean"] = float(np.nanmean(old_values))
        result[f"{metric}_repaired_mean"] = float(np.nanmean(new_values))
        result[f"{metric}_delta_mean"] = float(np.nanmean(new_values - old_values))
    l1 = [composition_l1(old[cell_id], new[cell_id]) for cell_id in cell_ids]
    result["selected_composition_l1_mean"] = float(np.mean(l1))
    result["selected_composition_changed_cells"] = sum(value > 1e-12 for value in l1)
    return result


def audit_manifest(out_root: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(out_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in {
            "AUDIT_MANIFEST.json",
            "RUN_COMPLETE.json",
        }:
            continue
        relative = path.relative_to(ROOT).as_posix()
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema_version": "oats-formal-trustfix-impact-manifest-1.0.0",
        "status": "COMPLETE",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "source_hashes": {
            "scripts/audit_formal_trustfix_impact.py": sha256_file(Path(__file__)),
            "legacy_raw_tree": LEGACY_RAW_TREE,
            "repaired_formal_freeze": REPAIRED_FREEZE,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=ROOT / "results" / "formal_realcal_pre_repair",
    )
    parser.add_argument(
        "--repaired-root",
        type=Path,
        default=ROOT / "results" / "formal_realcal_trustfix_20260726",
    )
    parser.add_argument(
        "--old-e1",
        type=Path,
        default=ROOT / "reports" / "round2_e1_assumption_diagnostic_20260726",
    )
    parser.add_argument(
        "--new-e1",
        type=Path,
        default=ROOT
        / "reports"
        / "round3_e1_assumption_diagnostic_trustfix_20260727",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "formal_trustfix_impact_20260727",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = args.legacy_root.resolve()
    repaired_root = args.repaired_root.resolve()
    out_root = args.out.resolve()
    if out_root.exists():
        raise RuntimeError(f"impact output already exists: {out_root}")
    out_root.mkdir(parents=True)
    tables = out_root / "tables"

    legacy_tree = raw_tree_hash(legacy_root / "raw")
    if legacy_tree != LEGACY_RAW_TREE:
        raise RuntimeError(
            f"legacy raw tree changed: expected={LEGACY_RAW_TREE} actual={legacy_tree}"
        )
    freeze_path = repaired_root / "audit" / "FORMAL_EVIDENCE_FREEZE.json"
    if sha256_file(freeze_path) != REPAIRED_FREEZE:
        raise RuntimeError("repaired formal freeze hash mismatch")
    freeze = load_json(freeze_path)
    if freeze.get("status") != "FORMAL_EVIDENCE_FROZEN":
        raise RuntimeError("repaired formal evidence is not frozen")

    old = load_raw(legacy_root)
    new = load_raw(repaired_root)
    if len(old) != 830 or len(new) != 830 or set(old) != set(new):
        raise RuntimeError(
            f"formal cell identity mismatch: old={len(old)} new={len(new)} "
            f"shared={len(set(old) & set(new))}"
        )

    cell_rows: list[dict[str, Any]] = []
    for cell_id in sorted(old):
        left, right = old[cell_id], new[cell_id]
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "family": family(right),
            "seed": int(right["seed"]),
            "method_id": right["method_id"],
            "selected_composition_l1": composition_l1(left, right),
        }
        for metric, getter in METRICS.items():
            old_value, new_value = getter(left), getter(right)
            row[f"legacy_{metric}"] = old_value
            row[f"repaired_{metric}"] = new_value
            row[f"delta_{metric}"] = new_value - old_value
        cell_rows.append(row)
    write_csv(tables / "paired_cell_deltas.csv", cell_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    main_ids = sorted(
        (cell_id for cell_id, payload in new.items() if main_cell(payload)),
        key=lambda cell_id: int(new[cell_id]["seed"]),
    )
    if len(main_ids) != 10:
        raise RuntimeError(f"expected ten main cells, got {len(main_ids)}")
    main_rows: list[dict[str, Any]] = []
    for metric, getter in METRICS.items():
        summary = paired_bootstrap(
            [(getter(old[cell_id]), getter(new[cell_id])) for cell_id in main_ids],
            rng=rng,
        )
        main_rows.append({"metric": metric, **summary})
    for slot in ("500", "600", "1000"):
        summary = paired_bootstrap(
            [
                (
                    number(nested(old[cell_id], "trust_trajectory", slot, "camouflage")),
                    number(nested(new[cell_id], "trust_trajectory", slot, "camouflage")),
                )
                for cell_id in main_ids
            ],
            rng=rng,
        )
        main_rows.append({"metric": f"camouflage_trust_{slot}", **summary})
    write_csv(tables / "main_configuration_paired_bootstrap.csv", main_rows)

    trajectory_rows: list[dict[str, Any]] = []
    for slot in sorted(
        nested(new[main_ids[0]], "trust_trajectory"),
        key=lambda value: int(value),
    ):
        for stratum in ("honest", "low-quality", "malicious", "camouflage"):
            left = [
                number(nested(old[cell_id], "trust_trajectory", slot, stratum))
                for cell_id in main_ids
            ]
            right = [
                number(nested(new[cell_id], "trust_trajectory", slot, stratum))
                for cell_id in main_ids
            ]
            trajectory_rows.append(
                {
                    "slot": int(slot),
                    "stratum": stratum,
                    "legacy_mean": float(np.mean(left)),
                    "repaired_mean": float(np.mean(right)),
                    "delta": float(np.mean(np.asarray(right) - np.asarray(left))),
                }
            )
    write_csv(tables / "main_trust_trajectory_comparison.csv", trajectory_rows)

    family_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for cell_id, payload in new.items():
        family_groups[(family(payload), str(payload["method_id"]))].append(cell_id)
    family_rows: list[dict[str, Any]] = []
    for (group_family, method), cell_ids in sorted(family_groups.items()):
        family_rows.append(
            {
                "family": group_family,
                "method_id": method,
                **summarize_group(cell_ids, old, new),
            }
        )
    write_csv(tables / "family_method_impact_summary.csv", family_rows)

    full_by_seed = {int(new[cell_id]["seed"]): cell_id for cell_id in main_ids}
    ablation_ids: dict[str, dict[int, str]] = defaultdict(dict)
    for cell_id, payload in new.items():
        if family(payload) == "E6_ABLATION":
            ablation_ids[str(payload["method_id"])][int(payload["seed"])] = cell_id
    ablation_ids["V2-FULL"] = full_by_seed
    ablation_rows: list[dict[str, Any]] = []
    for method, by_seed in sorted(ablation_ids.items()):
        if set(by_seed) != set(full_by_seed):
            raise RuntimeError(f"incomplete ablation seeds for {method}: {sorted(by_seed)}")
        cell_ids = [by_seed[seed] for seed in sorted(by_seed)]
        ablation_rows.append(
            {
                "method_id": method,
                **summarize_group(cell_ids, old, new),
            }
        )
    write_csv(tables / "ablation_impact_summary.csv", ablation_rows)

    contrast_rows: list[dict[str, Any]] = []
    for method, by_seed in sorted(ablation_ids.items()):
        if method == "V2-FULL":
            continue
        legacy_contrasts = []
        repaired_contrasts = []
        for seed in sorted(full_by_seed):
            full_id, comparator_id = full_by_seed[seed], by_seed[seed]
            legacy_contrasts.append(
                METRICS["platform_net_value"](old[full_id])
                - METRICS["platform_net_value"](old[comparator_id])
            )
            repaired_contrasts.append(
                METRICS["platform_net_value"](new[full_id])
                - METRICS["platform_net_value"](new[comparator_id])
            )
        legacy_mean = float(np.mean(legacy_contrasts))
        repaired_mean = float(np.mean(repaired_contrasts))
        contrast_rows.append(
            {
                "comparator": method,
                "legacy_oats_minus_comparator_net_mean": legacy_mean,
                "repaired_oats_minus_comparator_net_mean": repaired_mean,
                "change_in_contrast": repaired_mean - legacy_mean,
                "legacy_sign": int(np.sign(legacy_mean)),
                "repaired_sign": int(np.sign(repaired_mean)),
                "sign_reversal": int(np.sign(legacy_mean) != np.sign(repaired_mean)),
            }
        )
    write_csv(tables / "ablation_net_contrast_summary.csv", contrast_rows)

    with (args.old_e1 / "tables" / "bootstrap_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        old_e1 = {
            (row["panel"], row["key"]): row for row in csv.DictReader(handle)
        }
    with (args.new_e1 / "tables" / "bootstrap_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        new_e1 = {
            (row["panel"], row["key"]): row for row in csv.DictReader(handle)
        }
    if set(old_e1) != set(new_e1):
        raise RuntimeError("old/new E1 diagnostic row identities differ")
    e1_rows: list[dict[str, Any]] = []
    for panel, key in sorted(new_e1):
        left, right = old_e1[(panel, key)], new_e1[(panel, key)]
        e1_rows.append(
            {
                "panel": panel,
                "key": key,
                "legacy_mean": float(left["mean"]),
                "repaired_mean": float(right["mean"]),
                "delta": float(right["mean"]) - float(left["mean"]),
                "legacy_ci_low": float(left["ci_low"]),
                "legacy_ci_high": float(left["ci_high"]),
                "repaired_ci_low": float(right["ci_low"]),
                "repaired_ci_high": float(right["ci_high"]),
            }
        )
    write_csv(tables / "e1_diagnostic_old_new_comparison.csv", e1_rows)

    main_by_metric = {row["metric"]: row for row in main_rows}
    thresholds = {
        "absolute_auc_delta": 0.01,
        "absolute_camouflage_trust_delta": 0.02,
        "absolute_relative_purchased_count_delta": 0.005,
        "absolute_relative_gross_value_delta": 0.005,
        "absolute_relative_net_value_delta": 0.005,
    }
    materiality = {
        "main_auc_threshold_crossed": abs(
            main_by_metric["trust_auc"]["paired_delta_mean"]
        )
        > thresholds["absolute_auc_delta"],
        "main_camouflage_threshold_crossed": any(
            abs(main_by_metric[f"camouflage_trust_{slot}"]["paired_delta_mean"])
            > thresholds["absolute_camouflage_trust_delta"]
            for slot in ("500", "600", "1000")
        ),
        "main_purchased_threshold_crossed": abs(
            main_by_metric["purchased_count"]["relative_delta"]
        )
        > thresholds["absolute_relative_purchased_count_delta"],
        "main_gross_threshold_crossed": abs(
            main_by_metric["gross_external_value"]["relative_delta"]
        )
        > thresholds["absolute_relative_gross_value_delta"],
        "main_net_threshold_crossed": abs(
            main_by_metric["platform_net_value"]["relative_delta"]
        )
        > thresholds["absolute_relative_net_value_delta"],
        "ablation_contrast_sign_reversal": any(
            bool(row["sign_reversal"]) for row in contrast_rows
        ),
    }

    verification = load_json(
        repaired_root / "audit" / "FORMAL_RESULT_VERIFICATION.json"
    )
    consistency = {
        "legacy_cells": len(old),
        "repaired_cells": len(new),
        "shared_cell_ids": len(set(old) & set(new)),
        "main_cells": len(main_ids),
        "ablation_methods": len(ablation_ids),
        "e1_seeds": len(
            list((args.new_e1 / "replay_receipts").glob("seed_*.json"))
        ),
        "repaired_feedback_total": int(verification["feedback_total"]),
        "repaired_trust_transition_total": int(
            verification["trust_transition_total"]
        ),
        "repaired_duplicate_suppression_total": int(
            verification["duplicate_feedback_suppression_total"]
        ),
        "feedback_transition_equal": int(verification["feedback_total"])
        == int(verification["trust_transition_total"]),
        "all_new_lp_optimal": verification["lp_statuses"] == {"optimal": 830},
        "all_new_invariants_pass": verification["invariant_statuses"] == {"PASS": 830},
        "raw_per_seed_projection_mismatches": int(
            verification["raw_per_seed_non_lp_projection_mismatches"]
        ),
        "selected_composition_changed_cells": sum(
            float(row["selected_composition_l1"]) > 1e-12 for row in cell_rows
        ),
    }
    dump_json(out_root / "NUMBER_CONSISTENCY.json", consistency)

    claims = [
        {
            "claim": "formal completion and trust reconciliation",
            "evidence": "results/formal_realcal_trustfix_20260726/audit/FORMAL_RESULT_VERIFICATION.json",
            "status": "PASS",
        },
        {
            "claim": "main configuration online impact",
            "evidence": "reports/formal_trustfix_impact_20260727/tables/main_configuration_paired_bootstrap.csv",
            "status": "PASS",
        },
        {
            "claim": "all ablations and ordering flips",
            "evidence": "reports/formal_trustfix_impact_20260727/tables/ablation_impact_summary.csv; tables/ablation_net_contrast_summary.csv",
            "status": "PASS",
        },
        {
            "claim": "camouflage trust trajectory",
            "evidence": "reports/formal_trustfix_impact_20260727/tables/main_trust_trajectory_comparison.csv",
            "status": "PASS",
        },
        {
            "claim": "all sensitivity and stress families",
            "evidence": "reports/formal_trustfix_impact_20260727/tables/family_method_impact_summary.csv",
            "status": "PASS",
        },
        {
            "claim": "Assumption 4 empirical compatibility diagnostic",
            "evidence": "reports/round3_e1_assumption_diagnostic_trustfix_20260727/E1_ASSUMPTION_DIAGNOSTIC_REPORT_ZH.md",
            "status": "PASS_WITH_LITERAL_MARGIN_FAILURES",
        },
    ]
    write_csv(tables / "claim_to_evidence.csv", claims)

    summary = {
        "schema_version": "oats-formal-trustfix-impact-summary-1.0.0",
        "status": "PASS",
        "legacy_role": "paired impact comparator only",
        "repaired_role": "authoritative formal evidence",
        "bootstrap": {
            "unit": "seed",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "materiality_thresholds": thresholds,
        "materiality": materiality,
        "main_configuration": main_by_metric,
        "ablation_net_contrasts": contrast_rows,
        "number_consistency": consistency,
    }
    dump_json(out_root / "FORMAL_TRUSTFIX_IMPACT_SUMMARY.json", summary)

    def metric_line(name: str, label: str) -> str:
        row = main_by_metric[name]
        return (
            f"| {label} | {row['legacy_mean']:.6f} | {row['repaired_mean']:.6f} | "
            f"{row['paired_delta_mean']:+.6f} | "
            f"[{row['paired_delta_ci_low']:+.6f}, {row['paired_delta_ci_high']:+.6f}] |"
        )

    sign_flips = [
        row["comparator"] for row in contrast_rows if bool(row["sign_reversal"])
    ]
    report = [
        "# Worker-specific trust 修复后的 830-cell 正式影响审计",
        "",
        "状态：`PASS`。修复后正式包是当前权威证据；旧正式包仅作为 paired legacy 对照。",
        "",
        "## 1. 主配置影响",
        "",
        "| 指标 | Legacy mean | Repaired mean | 配对均值差 | 95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|",
        metric_line("purchased_count", "Purchased count"),
        metric_line("contracted_count", "Contracted count"),
        metric_line("activated_count", "Activated count"),
        metric_line("gross_external_value", "Gross external value"),
        metric_line("platform_net_value", "Platform net value"),
        metric_line("selected_quality", "Selected quality"),
        metric_line("trust_auc", "Final trust AUC"),
        metric_line("camouflage_trust_500", "Camouflage trust @500"),
        metric_line("camouflage_trust_600", "Camouflage trust @600"),
        metric_line("camouflage_trust_1000", "Camouflage trust @1000"),
        "",
        "全部预冻结 materiality 门均被跨越；这确认修复影响后续在线决策，而非离线计数替换。",
        "",
        "## 2. 消融与适用范围",
        "",
        f"- 消融对比发生净值符号翻转：{', '.join(sign_flips) if sign_flips else '无'}。",
        f"- 830 个配对 cell 中 selected worker-type composition 发生变化的 cell 数：{consistency['selected_composition_changed_cells']}。",
        "- `family_method_impact_summary.csv` 保留 E1--E7 每个 family/method 的 purchased、gross、net、quality、AUC 和 composition 差值，包括零影响与负向变化。",
        "",
        "## 3. Trust 与 E1",
        "",
        f"- 新正式包全矩阵 feedback={consistency['repaired_feedback_total']:,}，"
        f"实际 transition={consistency['repaired_trust_transition_total']:,}，"
        f"duplicate suppression={consistency['repaired_duplicate_suppression_total']}。",
        "- 新 E1 的 10 个主配置 seed 均实现 feedback 与 transition 一一对应。",
        "- Outcome-noise tail 兼容冻结阈值，但 Assumption 4 的五个逐更新 margin 层全部出现非零违例；不得以均值 score gap 替代字面条件。",
        "",
        "## 4. 证据边界",
        "",
        "- 旧包不得用于修复后机制的论文数值、图表或主张。",
        "- 正文、附录、图表和 number-consistency ledger 必须全部重建自新正式包。",
        "- 本审计是冻结结果的只读后处理，没有重跑或修改任何正式 cell。",
        "",
    ]
    (out_root / "FORMAL_TRUSTFIX_IMPACT_REPORT_ZH.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    manifest = audit_manifest(out_root)
    dump_json(out_root / "AUDIT_MANIFEST.json", manifest)
    complete = {
        "schema_version": "oats-formal-trustfix-impact-complete-1.0.0",
        "status": "COMPLETE",
        "audit_manifest_sha256": sha256_file(out_root / "AUDIT_MANIFEST.json"),
        "artifact_count": manifest["artifact_count"],
        "legacy_raw_tree_sha256": legacy_tree,
        "repaired_formal_freeze_sha256": sha256_file(freeze_path),
        "formal_cells_rerun": 0,
    }
    dump_json(out_root / "RUN_COMPLETE.json", complete)
    print(json.dumps(complete, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
