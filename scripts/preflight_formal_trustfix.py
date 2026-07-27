#!/usr/bin/env python3
"""Fail-closed preflight for the authorized 830-cell trust-fix formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oats_v2.experiments.run_matrix import build_run_matrix_v2  # noqa: E402


DEFAULT_AUTHORIZATION = (
    ROOT
    / "docs"
    / "p0_trust_feedback_repair_20260726"
    / "FORMAL_830_AUTHORIZATION.json"
)
DEFAULT_RECEIPT = (
    ROOT
    / "docs"
    / "p0_trust_feedback_repair_20260726"
    / "FORMAL_830_PREFLIGHT_RECEIPT.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_raw_tree(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = sorted(root.glob("*.json"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return len(paths), digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise RuntimeError(message)


def check_hash(relative: str, expected: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        fail(f"missing required file: {relative}")
    actual = sha256_file(path)
    if actual != expected:
        fail(f"hash mismatch for {relative}: expected={expected} actual={actual}")
    return {"path": relative, "sha256": actual, "bytes": path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    authorization_path = args.authorization.resolve()
    receipt_path = args.receipt.resolve()
    authorization = load_json(authorization_path)
    if authorization.get("status") != "AUTHORIZED":
        fail("formal authorization is not active")
    execution = authorization["formal_execution"]
    if int(execution["expected_cells"]) != 830:
        fail("authorization does not target exactly 830 cells")
    if int(execution["workers"]) != 25:
        fail("authorized worker count is not 25")
    if execution["output_root"] == "results/formal_realcal_pre_repair":
        fail("authorization attempts to overwrite the legacy formal result root")

    checks: list[dict[str, Any]] = []
    parent_paths = {
        "next_formal_freeze_template": "next_formal_freeze_template_sha256",
        "p0_document_manifest": "p0_document_manifest_sha256",
        "p0_gate_completion": "p0_gate_completion_sha256",
        "p0_result_completion": "p0_result_completion_sha256",
        "p0_audit_manifest": "p0_audit_manifest_sha256",
        "p0_test_receipt": "p0_test_receipt_sha256",
    }
    for path_key, hash_key in parent_paths.items():
        checks.append(
            check_hash(
                authorization["parent_gates"][path_key],
                authorization["parent_gates"][hash_key],
            )
        )
    for relative, expected in authorization["required_source_hashes"].items():
        checks.append(check_hash(relative, expected))
    checks.append(
        check_hash(
            execution["trace_hash_manifest"],
            execution["trace_hash_manifest_sha256"],
        )
    )

    p0_result = load_json(ROOT / authorization["parent_gates"]["p0_result_completion"])
    if (
        p0_result.get("status") != "P0_COMPLETE"
        or not p0_result.get("formal_830_rerun_required")
        or p0_result.get("formal_830_authorized")
    ):
        fail("P0 result receipt does not retain the required pre-authorization state")
    test_receipt = load_json(ROOT / authorization["parent_gates"]["p0_test_receipt"])
    post_tests = test_receipt.get("post_execution_canonical_suite_result", {})
    if int(post_tests.get("passed", 0)) != 167 or int(post_tests.get("failed", -1)) != 0:
        fail("authoritative repaired-code test receipt is not 167/167 PASS")

    matrix = build_run_matrix_v2()
    cell_ids = [cell.cell_id for cell in matrix]
    seeds = sorted({int(cell.seed) for cell in matrix})
    family_counts: dict[str, int] = {}
    for cell in matrix:
        family_counts[cell.family] = family_counts.get(cell.family, 0) + 1
    if len(matrix) != int(execution["expected_cells"]):
        fail(f"matrix count mismatch: {len(matrix)}")
    if len(set(cell_ids)) != int(execution["expected_unique_cell_ids"]):
        fail("matrix cell IDs are not globally unique")
    if seeds != [int(value) for value in execution["formal_seeds"]]:
        fail(f"formal seed mismatch: {seeds}")
    if family_counts != {
        key: int(value) for key, value in execution["expected_family_counts"].items()
    }:
        fail(f"matrix family counts mismatch: {family_counts}")

    manifest = load_json(ROOT / execution["trace_hash_manifest"])
    seed_hashes = manifest.get("seed_file_hashes", {})
    trace_checks: list[dict[str, Any]] = []
    for seed in seeds:
        entries = seed_hashes.get(str(seed))
        if not isinstance(entries, dict) or len(entries) != 10:
            fail(f"trace manifest does not contain 10 files for seed {seed}")
        seed_root = ROOT / execution["data_root"] / str(seed)
        for name, expected in sorted(entries.items()):
            relative = str(
                (Path(execution["data_root"]) / str(seed) / name).as_posix()
            )
            record = check_hash(relative, str(expected))
            record["seed"] = seed
            trace_checks.append(record)
    if len(trace_checks) != int(authorization["acceptance"]["trace_files_verified"]):
        fail(f"formal trace file count mismatch: {len(trace_checks)}")

    legacy_count, legacy_tree_hash = sha256_raw_tree(
        ROOT / authorization["isolation"]["preserve_legacy_result_root"] / "raw"
    )
    if legacy_count != 830:
        fail(f"legacy raw file count changed: {legacy_count}")
    if legacy_tree_hash != authorization["isolation"]["legacy_raw_tree_sha256"]:
        fail(
            "legacy raw tree hash mismatch: "
            f"expected={authorization['isolation']['legacy_raw_tree_sha256']} "
            f"actual={legacy_tree_hash}"
        )

    output_root = ROOT / execution["output_root"]
    if output_root.exists() and any(output_root.iterdir()):
        fail(f"new formal output root is not empty: {output_root}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 5 * (1 << 30):
        fail(f"insufficient local free space: {free_bytes}")

    receipt = {
        "schema_version": "oats-formal-trustfix-preflight-receipt-1.0.0",
        "status": "PASS",
        "authorization": str(authorization_path.relative_to(ROOT)).replace("\\", "/"),
        "authorization_sha256": sha256_file(authorization_path),
        "run_version": execution["run_version"],
        "output_root": execution["output_root"],
        "workers": int(execution["workers"]),
        "matrix": {
            "total_cells": len(matrix),
            "unique_cell_ids": len(set(cell_ids)),
            "seeds": seeds,
            "family_counts": family_counts,
        },
        "trace_verification": {
            "dataset_id": manifest.get("dataset_id"),
            "files_checked": len(trace_checks),
            "total_bytes": sum(int(item["bytes"]) for item in trace_checks),
            "all_hashes_match": True,
        },
        "parent_and_source_hashes": checks,
        "legacy_isolation": {
            "raw_files": legacy_count,
            "raw_tree_sha256": legacy_tree_hash,
            "unchanged": True,
        },
        "new_output_root_empty": True,
        "local_free_bytes": free_bytes,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
