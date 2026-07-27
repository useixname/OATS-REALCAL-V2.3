#!/usr/bin/env python3
"""Post-run integrity audit for the 30-cell delayed-feedback queue replay."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "delay_queue_replay_20260727"
FORMAL_ROOT = ROOT / "results" / "formal_realcal_trustfix_20260726"
EXPECTED_FORMAL_RAW_TREE = (
    "ab91fb6212322b11e7d168ffb53c63127ba024d4e5e26c56a3c2fa45a8dc439b"
)
STABLE_DELAY_ZERO_FIELDS = (
    "purchased_count",
    "base_paid",
    "score_paid",
    "total_paid",
    "final_ledger",
    "final_shadow",
    "effort_histogram",
    "channels",
    "screening",
    "trust",
    "deadline_satisfaction",
    "gross_external_value",
    "platform_net_value",
    "mc_correlation",
    "mc_top_k_overlap",
    "mc_undefined_rate",
    "failure_counts",
    "trust_trajectory",
    "value_prefix",
    "rejection_counts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_tree_hash(raw_root: Path) -> str:
    digest = hashlib.sha256()
    # Sort the filename strings explicitly.  Sorting Path objects is
    # case-insensitive on Windows but case-sensitive on Linux, which would make
    # the same frozen tree produce different aggregate hashes across hosts.
    for path in sorted(raw_root.glob("*.json"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    complete_path = RESULT_ROOT / "RUN_COMPLETE.json"
    if not complete_path.is_file():
        raise RuntimeError("delay queue run is not complete")
    complete = load_json(complete_path)
    if complete.get("status") != "COMPLETE" or complete.get("cell_count") != 30:
        raise RuntimeError("invalid delay queue completion receipt")

    raw_paths = sorted((RESULT_ROOT / "raw").glob("*.json"))
    if len(raw_paths) != 30:
        raise RuntimeError(f"expected 30 raw cells, found {len(raw_paths)}")
    by_seed_delay: dict[tuple[int, int], dict[str, Any]] = {}
    counter_totals = {
        "feedback_count": 0,
        "trust_transition_count": 0,
        "duplicate_feedback_suppressed_count": 0,
    }
    for path in raw_paths:
        payload = load_json(path)
        seed = int(payload["seed"])
        delay = int(payload["cell_id"].rsplit("_d", 1)[1])
        by_seed_delay[(seed, delay)] = payload
        if payload["invariant_status"] != "PASS":
            raise RuntimeError(f"invariant failure: {path.name}")
        for field in counter_totals:
            counter_totals[field] += int(payload["trust"][field])
        composition = payload["worker_type_composition"]
        if "aggregate" in composition:
            raise RuntimeError(f"aggregate leaked into selection composition: {path.name}")
        if abs(sum((Decimal(value) for value in composition.values()), Decimal("0")) - 1) > Decimal("1e-20"):
            raise RuntimeError(f"selection composition does not sum to one: {path.name}")
        if any(
            Decimal(payload["final_ledger"][field]) != 0
            for field in ("locked_base", "locked_score")
        ):
            raise RuntimeError(f"terminal ledger lock remains: {path.name}")
        if any(
            Decimal(payload["final_shadow"][field]) != 0
            for field in ("held", "committed")
        ):
            raise RuntimeError(f"terminal shadow obligation remains: {path.name}")
        if delay == 0 and any(
            Decimal(payload[field]) != 0
            for field in (
                "mean_outstanding_score_escrow",
                "peak_outstanding_score_escrow",
                "terminal_outstanding_score_escrow",
            )
        ):
            raise RuntimeError(f"delay-zero cell has cross-slot score escrow: {path.name}")
        if delay > 0 and Decimal(payload["peak_outstanding_score_escrow"]) <= 0:
            raise RuntimeError(f"delayed cell has no outstanding escrow: {path.name}")

    if len(by_seed_delay) != 30:
        raise RuntimeError("seed-delay keys are not unique")
    if counter_totals["feedback_count"] != counter_totals["trust_transition_count"]:
        raise RuntimeError("matrix feedback and transition totals differ")
    if counter_totals["duplicate_feedback_suppressed_count"] != 0:
        raise RuntimeError("unexpected matrix duplicate suppression")

    delay_zero_checks: list[dict[str, Any]] = []
    for seed in sorted({seed for seed, _ in by_seed_delay}):
        new = by_seed_delay[(seed, 0)]
        old_path = (
            FORMAL_ROOT
            / "raw"
            / (
                f"E1_OVERALL_s{seed}_mV2-FULL_g0.3_b0.25_"
                "c0_d0_m0_a1.json"
            )
        )
        old = load_json(old_path)
        mismatches = [
            field for field in STABLE_DELAY_ZERO_FIELDS if new[field] != old[field]
        ]
        if mismatches:
            raise RuntimeError(
                f"delay-zero replay differs from frozen main cell seed={seed}: {mismatches}"
            )
        delay_zero_checks.append(
            {
                "seed": seed,
                "frozen_main_cell": str(old_path.relative_to(ROOT)).replace("\\", "/"),
                "frozen_main_cell_sha256": sha256_file(old_path),
                "stable_fields_checked": len(STABLE_DELAY_ZERO_FIELDS),
                "mismatches": [],
                "selection_composition_recomputed_from_actual_counts": True,
            }
        )

    formal_tree = raw_tree_hash(FORMAL_ROOT / "raw")
    if formal_tree != EXPECTED_FORMAL_RAW_TREE:
        raise RuntimeError(
            f"frozen repaired formal tree changed: {formal_tree}"
        )

    manifest = load_json(RESULT_ROOT / "ARTIFACT_MANIFEST.json")
    artifact_mismatches: list[str] = []
    for relative, record in manifest["artifacts"].items():
        path = RESULT_ROOT / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            artifact_mismatches.append(relative)
    if artifact_mismatches:
        raise RuntimeError(f"artifact manifest mismatch: {artifact_mismatches}")

    audit = {
        "schema_version": "oats-delay-queue-postrun-audit-1.0.0",
        "status": "PASS",
        "raw_cells": len(raw_paths),
        "unique_seed_delay_keys": len(by_seed_delay),
        "invariant_failures": 0,
        "terminal_lock_failures": 0,
        "selection_composition_failures": 0,
        "trust_counter_totals": counter_totals,
        "delay_zero_frozen_equivalence": delay_zero_checks,
        "frozen_repaired_formal_raw_tree_sha256": formal_tree,
        "frozen_repaired_formal_tree_unchanged": True,
        "artifact_manifest_entries_checked": len(manifest["artifacts"]),
        "artifact_manifest_mismatches": [],
        "run_complete_sha256": sha256_file(complete_path),
        "artifact_manifest_sha256": sha256_file(
            RESULT_ROOT / "ARTIFACT_MANIFEST.json"
        ),
    }
    audit_path = RESULT_ROOT / "POSTRUN_AUDIT.json"
    dump_json(audit_path, audit)
    final_manifest = {
        "schema_version": "oats-delay-queue-final-audit-manifest-1.0.0",
        "status": "PASS",
        "postrun_audit_sha256": sha256_file(audit_path),
        "run_complete_sha256": audit["run_complete_sha256"],
        "artifact_manifest_sha256": audit["artifact_manifest_sha256"],
        "preregistration_sha256": sha256_file(
            ROOT
            / "docs"
            / "delay_queue_replay_20260727"
            / "DELAY_QUEUE_PREREGISTRATION.md"
        ),
    }
    dump_json(RESULT_ROOT / "FINAL_AUDIT_MANIFEST.json", final_manifest)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
