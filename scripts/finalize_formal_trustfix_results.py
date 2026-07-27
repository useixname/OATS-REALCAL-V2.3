#!/usr/bin/env python3
"""Validate and freeze the completed 830-cell trust-fix formal package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from src.oats_v2.experiments.run_matrix import build_run_matrix_v2  # noqa: E402


DEFAULT_AUTHORIZATION = (
    ROOT
    / "docs"
    / "p0_trust_feedback_repair_20260726"
    / "FORMAL_830_AUTHORIZATION.json"
)
DEFAULT_RESULTS = ROOT / "results" / "formal_realcal_trustfix_20260726"
MANIFEST_NAME = "FORMAL_EVIDENCE_MANIFEST.json"
FREEZE_NAME = "FORMAL_EVIDENCE_FREEZE.json"
VERIFICATION_NAME = "FORMAL_RESULT_VERIFICATION.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def raw_tree_hash(raw_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(raw_root.glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def non_lp_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"lp", "lp_prefix"}
    }


def artifact_manifest(
    result_root: Path,
    *,
    authorization_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    excluded = {MANIFEST_NAME, FREEZE_NAME}
    for path in sorted(result_root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(result_root).as_posix()
        value = sha256_file(path)
        size = path.stat().st_size
        files[relative] = {"sha256": value, "bytes": size}
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(value))
    return {
        "schema_version": "oats-formal-trustfix-evidence-manifest-1.0.0",
        "status": "COMPLETE",
        "authorization_sha256": authorization_hash,
        "source_hashes": source_hashes,
        "artifact_count": len(files),
        "artifact_tree_sha256": digest.hexdigest(),
        "artifacts": files,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count_visible": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def validate_results(
    result_root: Path,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    execution = authorization["formal_execution"]
    acceptance = authorization["acceptance"]
    matrix = build_run_matrix_v2()
    expected = {cell.cell_id: cell for cell in matrix}
    if len(expected) != int(execution["expected_unique_cell_ids"]):
        raise RuntimeError("current matrix is not the authorized unique matrix")

    summary_path = result_root / "audit" / "run_summary.json"
    checkpoint_path = result_root / "audit" / "checkpoint.json"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("formal run summary or checkpoint is missing")
    summary = load_json(summary_path)
    checkpoint = load_json(checkpoint_path)
    if summary.get("run_version") != execution["run_version"]:
        raise RuntimeError(f"run summary version mismatch: {summary}")
    if int(summary.get("total_cells", -1)) != int(acceptance["completed_cells"]):
        raise RuntimeError(f"run summary total mismatch: {summary}")
    if int(summary.get("completed", -1)) != int(acceptance["completed_cells"]):
        raise RuntimeError(f"run summary completion mismatch: {summary}")
    if int(summary.get("invalid", -1)) != int(acceptance["invalid_cells"]):
        raise RuntimeError(f"run summary invalid mismatch: {summary}")
    if checkpoint.get("run_version") != execution["run_version"]:
        raise RuntimeError("checkpoint run version mismatch")
    completed_ids = set(checkpoint.get("completed_cells", []))
    if completed_ids != set(expected):
        missing = sorted(set(expected) - completed_ids)[:10]
        extra = sorted(completed_ids - set(expected))[:10]
        raise RuntimeError(f"checkpoint ID mismatch missing={missing} extra={extra}")
    if checkpoint.get("invalid_cells"):
        raise RuntimeError(f"checkpoint has invalid cells: {checkpoint['invalid_cells'][:5]}")

    raw_paths = sorted((result_root / "raw").glob("*.json"))
    per_seed_paths = sorted((result_root / "per_seed").glob("*/*.json"))
    if len(raw_paths) != int(acceptance["completed_cells"]):
        raise RuntimeError(f"raw result count mismatch: {len(raw_paths)}")
    if len(per_seed_paths) != int(acceptance["completed_cells"]):
        raise RuntimeError(f"per-seed result count mismatch: {len(per_seed_paths)}")
    raw_ids = {path.stem for path in raw_paths}
    per_seed_ids = {path.stem for path in per_seed_paths}
    if raw_ids != set(expected):
        raise RuntimeError("raw result IDs do not equal the authorized matrix")
    if per_seed_ids != set(expected):
        raise RuntimeError("per-seed result IDs do not equal the authorized matrix")

    feedback_total = 0
    transition_total = 0
    duplicate_total = 0
    lp_statuses: dict[str, int] = {}
    invariant_statuses: dict[str, int] = {}
    non_lp_projection_mismatches: list[str] = []
    config_hashes: set[str] = set()
    final_nonzero_ledger_locks = 0
    final_nonzero_shadow_active = 0
    negative_free_count = 0

    per_seed_by_id = {path.stem: path for path in per_seed_paths}
    for path in raw_paths:
        payload = load_json(path)
        cell_id = path.stem
        cell = expected[cell_id]
        if payload.get("cell_id") != cell_id:
            raise RuntimeError(f"payload cell ID mismatch: {path}")
        if int(payload.get("seed", -1)) != int(cell.seed):
            raise RuntimeError(f"seed mismatch: {cell_id}")
        if payload.get("method_id") != cell.method_id:
            raise RuntimeError(f"method mismatch: {cell_id}")
        scalar_pairs = {
            "gamma": cell.gamma,
            "budget_ratio": cell.budget_ratio,
        }
        for key, expected_value in scalar_pairs.items():
            if Decimal(str(payload.get(key))) != Decimal(expected_value):
                raise RuntimeError(
                    f"{key} mismatch {cell_id}: "
                    f"{payload.get(key)} != {expected_value}"
                )
        invariant = str(payload.get("invariant_status"))
        invariant_statuses[invariant] = invariant_statuses.get(invariant, 0) + 1
        if invariant != acceptance["invariant_status_all_cells"]:
            raise RuntimeError(f"invariant failure: {cell_id}={invariant}")

        trust = payload.get("trust") or {}
        feedback = int(trust.get("feedback_count", -1))
        transitions = int(trust.get("trust_transition_count", -2))
        duplicates = int(trust.get("duplicate_feedback_suppressed_count", -3))
        if acceptance["feedback_equals_trust_transitions_all_cells"] and (
            feedback != transitions
        ):
            raise RuntimeError(
                f"feedback/transition mismatch {cell_id}: {feedback}!={transitions}"
            )
        if duplicates != int(acceptance["duplicate_feedback_suppressions_all_cells"]):
            raise RuntimeError(f"duplicate suppression is nonzero: {cell_id}={duplicates}")
        feedback_total += feedback
        transition_total += transitions
        duplicate_total += duplicates

        lp_status = str((payload.get("lp") or {}).get("status"))
        lp_statuses[lp_status] = lp_statuses.get(lp_status, 0) + 1
        if lp_status != acceptance["lp_status_all_cells"]:
            raise RuntimeError(f"LP status failure: {cell_id}={lp_status}")

        config_hash = str(payload.get("config_hash") or "")
        if not config_hash:
            raise RuntimeError(f"empty config hash: {cell_id}")
        config_hashes.add(config_hash)
        ledger = payload.get("final_ledger") or {}
        shadow = payload.get("final_shadow") or {}
        if any(Decimal(str(ledger.get(key, "0"))) != 0 for key in ("locked_base", "locked_score")):
            final_nonzero_ledger_locks += 1
        if any(Decimal(str(shadow.get(key, "0"))) != 0 for key in ("held", "committed")):
            final_nonzero_shadow_active += 1
        if Decimal(str(ledger.get("free", "0"))) < 0 or Decimal(str(shadow.get("free", "0"))) < 0:
            negative_free_count += 1

        per_seed_payload = load_json(per_seed_by_id[cell_id])
        if non_lp_projection(payload) != non_lp_projection(per_seed_payload):
            non_lp_projection_mismatches.append(cell_id)
    if non_lp_projection_mismatches:
        raise RuntimeError(
            "raw/per-seed non-LP projection mismatch: "
            f"{non_lp_projection_mismatches[:10]}"
        )
    if final_nonzero_ledger_locks:
        raise RuntimeError(f"nonzero final ledger locks: {final_nonzero_ledger_locks}")
    if final_nonzero_shadow_active:
        raise RuntimeError(f"nonzero final shadow active state: {final_nonzero_shadow_active}")
    if negative_free_count:
        raise RuntimeError(f"negative free-state cells: {negative_free_count}")

    return {
        "schema_version": "oats-formal-trustfix-result-verification-1.0.0",
        "status": "PASS",
        "run_version": execution["run_version"],
        "matrix_cells": len(matrix),
        "unique_matrix_ids": len(expected),
        "raw_files": len(raw_paths),
        "per_seed_files": len(per_seed_paths),
        "completed_checkpoint_ids": len(completed_ids),
        "invalid_checkpoint_entries": len(checkpoint.get("invalid_cells", [])),
        "feedback_total": feedback_total,
        "trust_transition_total": transition_total,
        "duplicate_feedback_suppression_total": duplicate_total,
        "lp_statuses": lp_statuses,
        "invariant_statuses": invariant_statuses,
        "config_hash_count": len(config_hashes),
        "raw_per_seed_non_lp_projection_mismatches": 0,
        "final_nonzero_ledger_locks": final_nonzero_ledger_locks,
        "final_nonzero_shadow_active": final_nonzero_shadow_active,
        "negative_free_count": negative_free_count,
        "raw_tree_sha256": raw_tree_hash(result_root / "raw"),
        "run_summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help="Verify the existing manifest and freeze without rewriting them.",
    )
    return parser.parse_args()


def verify_frozen(result_root: Path) -> None:
    manifest_path = result_root / "audit" / MANIFEST_NAME
    freeze_path = result_root / "audit" / FREEZE_NAME
    manifest = load_json(manifest_path)
    freeze = load_json(freeze_path)
    if freeze.get("status") != "FORMAL_EVIDENCE_FROZEN":
        raise RuntimeError("formal evidence freeze is not complete")
    if freeze.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("formal evidence manifest hash mismatch")
    mismatches: list[str] = []
    for relative, record in manifest["artifacts"].items():
        path = result_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"formal evidence artifact mismatches: {mismatches[:10]}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact_count": len(manifest["artifacts"]),
                "mismatches": 0,
                "manifest_sha256": sha256_file(manifest_path),
                "freeze_sha256": sha256_file(freeze_path),
            },
            indent=2,
        )
    )


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    if args.verify_frozen:
        verify_frozen(result_root)
        return 0

    authorization_path = args.authorization.resolve()
    authorization = load_json(authorization_path)
    if authorization.get("status") != "AUTHORIZED":
        raise RuntimeError("formal authorization is not active")
    verification = validate_results(result_root, authorization)
    verification_path = result_root / "audit" / VERIFICATION_NAME
    dump_json(verification_path, verification)

    source_hashes: dict[str, str] = {}
    for relative, expected in authorization["required_source_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"formal source changed before freeze {relative}: "
                f"expected={expected} actual={actual}"
            )
        source_hashes[relative] = actual
    authorization_hash = sha256_file(authorization_path)
    manifest = artifact_manifest(
        result_root,
        authorization_hash=authorization_hash,
        source_hashes=source_hashes,
    )
    manifest_path = result_root / "audit" / MANIFEST_NAME
    dump_json(manifest_path, manifest)
    freeze = {
        "schema_version": "oats-formal-trustfix-evidence-freeze-1.0.0",
        "status": "FORMAL_EVIDENCE_FROZEN",
        "run_version": authorization["formal_execution"]["run_version"],
        "result_root": str(result_root),
        "authorization_sha256": authorization_hash,
        "verification_sha256": sha256_file(verification_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": manifest["artifact_count"],
        "artifact_tree_sha256": manifest["artifact_tree_sha256"],
        "raw_tree_sha256": verification["raw_tree_sha256"],
        "completed_cells": verification["raw_files"],
        "invalid_cells": verification["invalid_checkpoint_entries"],
        "e1_authorized_after_this_freeze": True,
    }
    freeze_path = result_root / "audit" / FREEZE_NAME
    dump_json(freeze_path, freeze)
    print(json.dumps({"verification": verification, "freeze": freeze}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
