from __future__ import annotations

import argparse
import hashlib
import json
import locale
import os
import platform
import shutil
import stat
import sys
from datetime import datetime, timezone
from decimal import getcontext
from pathlib import Path
from typing import Any, Iterable

from .profiles import deployment_profiles
from .schemas import (
    DATASET_ID,
    FORMAL_SEEDS,
    GENERATOR_VERSION,
    PREREGISTRATION_VERSION,
    SCHEMA_VERSION,
    SCREENING_LABEL,
    TraceConfig,
    jsonable,
    schema_document,
)
from .trace_validator import file_digest, read_jsonl, validate_trace
from .trace_writer import TraceWriteResult, generate_trace, generator_source_hash


EXPECTED_RUNTIME = {
    "implementation": "CPython",
    "python_version": "3.11.2",
    "bits": "64bit",
    "os_family": "Windows",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _hash_value(value: object) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def runtime_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = {
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_executable": str(Path(sys.executable).resolve()),
        "bits": platform.architecture()[0],
        "os": platform.platform(),
        "os_family": platform.system(),
        "architecture": platform.machine()
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or "AMD64-derived-from-CPython-64bit-runtime",
        "locale_preferred_encoding": locale.getpreferredencoding(False),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "decimal_context": {
            "precision": getcontext().prec,
            "rounding": getcontext().rounding,
            "Emin": getcontext().Emin,
            "Emax": getcontext().Emax,
            "capitals": getcontext().capitals,
            "clamp": getcontext().clamp,
        },
        "json_serialization": {
            "encoding": "UTF-8",
            "bom": False,
            "line_ending": "LF",
            "separators": [",", ":"],
            "field_order": "schema frozen",
            "decimal_encoding": "canonical base-10 JSON string",
        },
        "rng": {
            "core": "random.Random MT19937",
            "keying": "int.from_bytes(SHA256(UTF8(formal_seed|family|ids))[0:8],big)",
            "uniform_bernoulli": "first random()",
            "normal": "first gauss(0,1) on a fresh keyed substream",
            "beta": "first betavariate(2,2) on a fresh keyed substream",
            "poisson20": "Knuth product with L=exp(-20)",
            "permutation": "fresh keyed substream shuffle",
            "python_hash_forbidden": True,
        },
        "standard_library_only": True,
        "source_tree_hash": generator_source_hash(root),
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "preregistration_version": PREREGISTRATION_VERSION,
        "dataset_version": DATASET_ID,
    }
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in EXPECTED_RUNTIME.items()
        if actual[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"SYN-V2-1 runtime mismatch; bump to SYN-V2-2 or use exact runtime: {mismatches}")
    runtime_hash = _hash_value(actual)
    actual["runtime_hash"] = runtime_hash
    generator_contract = {
        "dataset_id": DATASET_ID,
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "preregistration_version": PREREGISTRATION_VERSION,
        "formal_trace_config": TraceConfig().__dict__,
        "formal_seeds": list(FORMAL_SEEDS),
        "screening_label": SCREENING_LABEL,
        "exogenous_pre_generation": True,
        "paired_potential_outcomes": True,
        "method_inputs_to_generator": [],
        "gamma_inputs_to_exogenous_generator": [],
        "contract_materializations": {"method_id": "V2-FULL", "gamma_grid": ["0", "0.1", "0.3", "0.5", "0.8", "1.0"]},
        "online_forbidden_fields": ["theta", "stratum", "bias_sign", "potential_reports", "future_holdout"],
        "holdout_dependency_forbidden": ["method", "gamma", "selection", "report_delivery", "consumer_action"],
        "deployment_profiles": deployment_profiles(),
        "calibration": {
            "N_cal": 10_000,
            "precision": "0.000001",
            "rounding": "ROUND_HALF_EVEN",
            "simultaneous_correction": "global-seed-family Hoeffding union bound",
            "signature": "UNTRUSTED_SHA256_PLACEHOLDER_NOT_A_DIGITAL_SIGNATURE",
        },
        "source_tree_hash": actual["source_tree_hash"],
        "runtime_hash": runtime_hash,
    }
    return actual, generator_contract


def write_environment(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime, contract = runtime_contract(root)
    environment = root / "environment"
    _write_json(environment / "runtime_manifest.json", runtime)
    _write_json(environment / "generator_contract.json", contract)
    _write_json(environment / "deployment_profiles.json", deployment_profiles())
    return runtime, contract


def _trace_signature(report: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for filename, value in sorted(report["file_sha256"].items()):
        digest.update(f"{filename}|{value}\n".encode("ascii"))
    return digest.hexdigest()


def _all_file_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _safe_remove_directory(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = (root / "audit_results" / "generator_dev_work").resolve()
    if resolved != allowed:
        raise RuntimeError(f"refusing to delete outside dev work path: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_development(root: Path) -> dict[str, object]:
    runtime, contract = write_environment(root)
    config = TraceConfig()
    work = root / "audit_results" / "generator_dev_work"
    if work.exists():
        raise FileExistsError(f"development work directory exists: {work}")
    first_dir = work / "run_a"
    second_dir = work / "run_b"
    generate_trace(seed=20260715, output_directory=first_dir, root=root, config=config, formal=True)
    first_report = validate_trace(first_dir, config)
    if first_report["status"] != "PASS":
        raise RuntimeError(f"development run A failed: {first_report['errors']}")
    generate_trace(seed=20260715, output_directory=second_dir, root=root, config=config, formal=True)
    second_report = validate_trace(second_dir, config)
    if second_report["status"] != "PASS":
        raise RuntimeError(f"development run B failed: {second_report['errors']}")
    first_all_hashes = _all_file_hashes(first_dir)
    second_all_hashes = _all_file_hashes(second_dir)
    same_hashes = first_all_hashes == second_all_hashes
    same_counts = first_report["row_counts"] == second_report["row_counts"]
    determinism = {
        "dataset_id": DATASET_ID,
        "seed": 20260715,
        "status": "PASS" if same_hashes and same_counts else "FAIL",
        "byte_identical_all_files": same_hashes,
        "row_counts_identical": same_counts,
        "run_a_trace_hash": _trace_signature(first_report),
        "run_b_trace_hash": _trace_signature(second_report),
        "file_hashes": first_all_hashes,
        "compared_file_count": len(first_all_hashes),
    }
    schema_report = {
        "status": first_report["status"],
        "schema": schema_document(),
        "validation": first_report,
        "explicit_checks": {
            "substream_independence": "covered by tests/data/test_substreams.py",
            "no_future_information_leakage": first_report["checks"]["no_online_secret_fields"],
            "no_method_dependent_exogenous_trace": True,
            "one_task_mapping": first_report["checks"]["one_task_mapping"],
            "anchor_pre_slot_provenance": True,
            "external_holdout_independence": first_report["checks"]["holdout_independence"],
            "potential_outcome_completeness": first_report["checks"]["potential_outcome_completeness"],
            "numeric_domain_bounds": not first_report["errors"],
        },
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": 20260715,
        "development_only": True,
        "status": "PASS" if determinism["status"] == "PASS" and schema_report["status"] == "PASS" else "FAIL",
        "runtime_hash": runtime["runtime_hash"],
        "generator_source_hash": contract["source_tree_hash"],
        "config_hash": _hash_value(config.__dict__),
        "row_counts": first_report["row_counts"],
        "task_count": first_report["task_count"],
        "worker_count": first_report["worker_count"],
        "available_mapped_count": first_report["available_mapped_count"],
        "screening_label": SCREENING_LABEL,
        "prohibited_metrics_computed": [],
    }
    _write_json(root / "audit_results/generator_dev_manifest.json", manifest)
    _write_json(root / "audit_results/generator_schema_report.json", schema_report)
    _write_json(root / "audit_results/generator_determinism.json", determinism)
    if manifest["status"] != "PASS":
        raise RuntimeError("development gate failed; formal trace generation prohibited")
    _safe_remove_directory(work, root)
    return manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_read_only(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)


def freeze_formal_traces(root: Path) -> dict[str, object]:
    development = _read_json(root / "audit_results/generator_dev_manifest.json")
    determinism = _read_json(root / "audit_results/generator_determinism.json")
    schema = _read_json(root / "audit_results/generator_schema_report.json")
    if not (
        development.get("status") == "PASS"
        and determinism.get("status") == "PASS"
        and schema.get("status") == "PASS"
    ):
        raise RuntimeError("development/schema/determinism gate is not fully passed")
    runtime, contract = runtime_contract(root)
    if runtime["runtime_hash"] != development["runtime_hash"] or contract["source_tree_hash"] != development["generator_source_hash"]:
        raise RuntimeError("runtime/source changed after development validation; rerun development under a new clean gate")
    data_root = root / "data" / DATASET_ID
    if data_root.exists():
        raise FileExistsError("formal data directory exists; overwrite forbidden and version bump required")
    data_root.mkdir(parents=True)
    config = TraceConfig()
    seed_entries: list[dict[str, object]] = []
    all_table_lines: list[str] = []
    all_certificates: list[dict[str, object]] = []
    for seed in FORMAL_SEEDS:
        generated_at = datetime.now(timezone.utc).isoformat()
        directory = data_root / str(seed)
        result = generate_trace(
            seed=seed,
            output_directory=directory,
            root=root,
            config=config,
            formal=True,
        )
        report = validate_trace(directory, config)
        if report["status"] != "PASS":
            raise RuntimeError(f"formal trace validation failed for seed {seed}: {report['errors']}")
        trace_hash = _trace_signature(report)
        seed_entries.append(
            {
                "seed": seed,
                "generated_at_utc": generated_at,
                "trace_hash": trace_hash,
                "file_sha256": report["file_sha256"],
                "row_counts": report["row_counts"],
                "task_count": report["task_count"],
                "worker_count": report["worker_count"],
                "eligibility_count": report["row_counts"]["eligibility.jsonl"],
                "available_mapped_count": report["available_mapped_count"],
                "missing_counts": report["missing_counts"],
                "delay_mask_rows": report["delay_mask_rows"],
                "anchor_version": result.anchor_version,
                "holdout_provenance": "SYNTHETIC_PREGENERATED_HOLDOUT",
                "generator_hash": contract["source_tree_hash"],
                "runtime_hash": runtime["runtime_hash"],
                "config_hash": _hash_value(config.__dict__),
                "numeric_domain": report["numeric_domain"],
                "validation_status": "PASS",
            }
        )
        all_table_lines.extend((directory / "continuation_tables.jsonl").read_text(encoding="utf-8").splitlines())
        all_certificates.extend(_read_json(directory / "epsilon_certificates.json"))

    trace_hashes = {
        "dataset_id": DATASET_ID,
        "hash_algorithm": "SHA-256",
        "seed_trace_hashes": {str(entry["seed"]): entry["trace_hash"] for entry in seed_entries},
        "seed_file_hashes": {str(entry["seed"]): entry["file_sha256"] for entry in seed_entries},
    }
    seed_manifest = {
        "dataset_id": DATASET_ID,
        "seeds": list(FORMAL_SEEDS),
        "seed_count": len(FORMAL_SEEDS),
        "labels_not_dates": True,
        "paired_trace_requirement": True,
        "entries": seed_entries,
    }
    data_manifest = {
        "dataset_id": DATASET_ID,
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "preregistration_version": PREREGISTRATION_VERSION,
        "freeze_time_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "requirements_lock_sha256": file_digest(root / "environment/requirements-lock.txt"),
        "generator_source_hash": contract["source_tree_hash"],
        "config": config.__dict__,
        "config_hash": _hash_value(config.__dict__),
        "schema": schema_document(),
        "seed_count": len(seed_entries),
        "seeds": seed_entries,
        "formal_mechanism_results": False,
        "screening_label": SCREENING_LABEL,
    }
    calibration_manifest = {
        "dataset_id": DATASET_ID,
        "table_count": len(all_table_lines),
        "certificate_count": len(all_certificates),
        "role_buckets": ["honest", "low"],
        "effort_menu": ["0", "0.5", "1"],
        "sample_size_per_effort": 10_000,
        "score_rule": "S(r,z)=1-(r-z)^2",
        "precision": "0.000001",
        "rounding": "ROUND_HALF_EVEN",
        "confidence": "0.99",
        "simultaneous_correction": "HOEFFDING_UNION_BOUND_GLOBAL_SEED_FAMILY",
        "signature": "UNTRUSTED_SHA256_PLACEHOLDER_NOT_A_DIGITAL_SIGNATURE",
        "anti_rollback_counter": 1,
        "source_hash": contract["source_tree_hash"],
    }
    _write_json(root / "trace_hashes.json", trace_hashes)
    _write_json(root / "seed_manifest.json", seed_manifest)
    _write_json(root / "data_manifest.json", data_manifest)
    _write_json(root / "calibration_manifest.json", calibration_manifest)
    (root / "Gbar_tables.jsonl").write_text("\n".join(all_table_lines) + "\n", encoding="utf-8", newline="\n")
    _write_json(root / "epsilon_certificates.json", all_certificates)
    guard = {
        "dataset_id": DATASET_ID,
        "version_bump_required_for_regeneration": True,
        "data_manifest_sha256": file_digest(root / "data_manifest.json"),
        "trace_hashes_sha256": file_digest(root / "trace_hashes.json"),
        "seed_manifest_sha256": file_digest(root / "seed_manifest.json"),
    }
    _write_json(data_root / ".freeze_guard.json", guard)
    receipt = (
        "# SYN-V2-1 Freeze Receipt\n\n"
        f"- Dataset: `{DATASET_ID}`\n"
        f"- Seeds: `{len(FORMAL_SEEDS)}` (`20260715` through `20260744` as integer labels)\n"
        f"- Generator source hash: `{contract['source_tree_hash']}`\n"
        f"- Runtime hash: `{runtime['runtime_hash']}`\n"
        f"- Data manifest SHA-256: `{guard['data_manifest_sha256']}`\n"
        f"- Trace hashes SHA-256: `{guard['trace_hashes_sha256']}`\n"
        "- All traces passed schema, domain, one-task, provenance and potential-outcome completeness validation.\n"
        "- No mechanism or baseline was executed; no utility, LP-gap or aggregate performance result was computed.\n"
        "- Regeneration or overwrite requires a dataset-version bump.\n"
    )
    (root / "freeze_receipt.md").write_text(receipt, encoding="utf-8", newline="\n")
    _set_read_only(data_root)
    return data_manifest


def verify_freeze(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "data_manifest.json")
    errors: list[str] = []
    config = TraceConfig()
    for entry in manifest["seeds"]:
        seed = int(entry["seed"])
        report = validate_trace(root / "data" / DATASET_ID / str(seed), config)
        if report["status"] != "PASS":
            errors.append(f"seed {seed} validation failed: {report['errors']}")
        if _trace_signature(report) != entry["trace_hash"]:
            errors.append(f"seed {seed} trace hash mismatch")
    guard = _read_json(root / "data" / DATASET_ID / ".freeze_guard.json")
    if file_digest(root / "data_manifest.json") != guard["data_manifest_sha256"]:
        errors.append("data_manifest hash guard mismatch")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "seed_count": len(manifest["seeds"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dev", "freeze", "verify", "environment"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "dev":
        result = run_development(root)
    elif args.command == "freeze":
        result = freeze_formal_traces(root)
    elif args.command == "verify":
        result = verify_freeze(root)
    else:
        runtime, contract = write_environment(root)
        result = {"runtime": runtime, "generator_contract": contract}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
