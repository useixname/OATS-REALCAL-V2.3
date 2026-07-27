from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from .provenance import validate_holdout_provenance
from .schemas import EFFORTS, GAMMAS, SCHEMAS, TraceConfig, validate_row


FILE_SCHEMAS = {
    "workers.jsonl": "workers",
    "tasks.jsonl": "tasks",
    "eligibility.jsonl": "eligibility",
    "anchors.jsonl": "anchors",
    "potential_reports.jsonl": "potential_reports",
    "continuation_tables.jsonl": "continuation_tables",
    "contracts.jsonl": "contracts",
    "holdout_provenance.jsonl": "holdout_provenance",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"{path.name}:{line_number} lacks LF terminator")
            yield json.loads(line)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace(directory: Path, config: TraceConfig) -> dict[str, object]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    available_mapped = 0
    mapped_pairs: set[tuple[int, str]] = set()
    task_ids: set[str] = set()
    worker_ids: set[str] = set()
    stratum_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    delay_counts: Counter[str] = Counter()
    numeric = {
        "theta_min": Decimal("1"),
        "theta_max": Decimal("0"),
        "z_min": Decimal("1"),
        "z_max": Decimal("0"),
        "report_min": Decimal("1"),
        "report_max": Decimal("0"),
        "score_min": Decimal("1"),
        "score_max": Decimal("0"),
    }

    for filename, schema_name in FILE_SCHEMAS.items():
        path = directory / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
            continue
        hashes[filename] = file_digest(path)
        count = 0
        for row in read_jsonl(path):
            count += 1
            if tuple(row) != SCHEMAS[schema_name]:
                errors.append(f"{filename}:{count} field order/schema mismatch")
                break
            row_errors = validate_row(schema_name, row)
            if row_errors:
                errors.extend(f"{filename}:{count}: {error}" for error in row_errors)
                break
            if schema_name == "workers":
                worker_ids.add(str(row["worker_id"]))
                stratum_counts[str(row["stratum"])] += 1
            elif schema_name == "tasks":
                task_ids.add(str(row["task_id"]))
                for setting, missing in row["missing_mask"].items():
                    if missing:
                        missing_counts[str(setting)] += 1
                for setting, delay in row["delay_mask"].items():
                    if int(delay) != int(Decimal(str(setting))):
                        errors.append(f"{filename}:{count}: delay mask mismatch for {setting}")
                    delay_counts[str(setting)] += 1
                for field in ("theta", "z"):
                    value = Decimal(row[field])
                    numeric[f"{field}_min"] = min(numeric[f"{field}_min"], value)
                    numeric[f"{field}_max"] = max(numeric[f"{field}_max"], value)
            elif schema_name == "eligibility":
                pair = (int(row["slot"]), str(row["worker_id"]))
                if pair in mapped_pairs:
                    errors.append(f"duplicate eligibility pair {pair}")
                    break
                mapped_pairs.add(pair)
                if row["available"] and row["mapped_task_id"] is not None:
                    available_mapped += 1
            elif schema_name == "potential_reports":
                for field in ("report", "score"):
                    value = Decimal(row[field])
                    numeric[f"{field}_min"] = min(numeric[f"{field}_min"], value)
                    numeric[f"{field}_max"] = max(numeric[f"{field}_max"], value)
            elif schema_name == "holdout_provenance":
                errors.extend(
                    f"{filename}:{count}: {error}" for error in validate_holdout_provenance(row)
                )
        counts[filename] = count

    expected = {
        "workers.jsonl": config.worker_count,
        "eligibility.jsonl": config.horizon * config.worker_count,
        "anchors.jsonl": config.cell_count * config.anchor_count_per_cell,
        "potential_reports.jsonl": available_mapped * len(EFFORTS),
        "continuation_tables.jsonl": config.cell_count * 2 * len(EFFORTS),
        "contracts.jsonl": available_mapped * len(GAMMAS),
        "holdout_provenance.jsonl": counts.get("tasks.jsonl", 0),
    }
    for filename, expected_count in expected.items():
        if counts.get(filename) != expected_count:
            errors.append(f"{filename} count {counts.get(filename)} != {expected_count}")
    if len(worker_ids) != config.worker_count:
        errors.append("worker ID completeness failed")
    if len(task_ids) != counts.get("tasks.jsonl", 0):
        errors.append("task ID uniqueness failed")
    if config.worker_count == 500 and stratum_counts != Counter(
        {"honest": 300, "low-quality": 100, "malicious": 50, "camouflage": 50}
    ):
        errors.append(f"stratum counts mismatch: {dict(stratum_counts)}")

    # Contract rows are the only method/gamma-bearing file.  Secrets and future
    # labels must not appear there or in eligibility.
    for filename in ("contracts.jsonl", "eligibility.jsonl"):
        fields = set(SCHEMAS[FILE_SCHEMAS[filename]])
        leaked = fields.intersection({"theta", "stratum", "bias_sign", "potential_reports", "z"})
        if leaked:
            errors.append(f"online input schema leaks {sorted(leaked)} in {filename}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "directory": str(directory),
        "errors": errors,
        "row_counts": counts,
        "file_sha256": hashes,
        "available_mapped_count": available_mapped,
        "task_count": counts.get("tasks.jsonl", 0),
        "worker_count": counts.get("workers.jsonl", 0),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "missing_counts": dict(sorted(missing_counts.items())),
        "delay_mask_rows": dict(sorted(delay_counts.items())),
        "numeric_domain": {key: str(value) for key, value in numeric.items()},
        "checks": {
            "schema_and_field_order": not any("schema" in error for error in errors),
            "one_task_mapping": not any("eligibility" in error for error in errors),
            "holdout_independence": not any("holdout" in error for error in errors),
            "potential_outcome_completeness": counts.get("potential_reports.jsonl")
            == available_mapped * len(EFFORTS),
            "no_online_secret_fields": not any("leaks" in error for error in errors),
        },
    }
