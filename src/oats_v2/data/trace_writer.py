from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from src.oats_v2.calibration.continuation_table import calibrate_seed

from .anchor_history_generator import generate_anchor_history
from .eligibility_generator import generate_eligibility
from .holdout_generator import generate_holdout_provenance
from .potential_report_generator import generate_potential_reports
from .schemas import (
    DATASET_ID,
    EFFORTS,
    GAMMAS,
    GBAR_PRECISION,
    GENERATOR_VERSION,
    MONEY_GRID,
    SCHEMA_VERSION,
    SCREENING_LABEL,
    TraceConfig,
    canonical_json,
    decimal_text,
    validate_row,
)
from .substreams import stable_digest
from .task_generator import generate_tasks
from .worker_generator import generate_workers


@dataclass(frozen=True)
class WrittenFile:
    name: str
    sha256: str
    rows: int
    bytes: int


@dataclass(frozen=True)
class TraceWriteResult:
    seed: int
    directory: Path
    files: tuple[WrittenFile, ...]
    task_count: int
    worker_count: int
    eligibility_count: int
    available_mapped_count: int
    anchor_version: str
    calibration_certificate_count: int


def generator_source_hash(root: Path) -> str:
    source_roots = (root / "src/oats_v2/data", root / "src/oats_v2/calibration")
    paths = sorted(path for source_root in source_roots for path in source_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _write_jsonl(
    path: Path,
    schema_name: str,
    rows: Iterable[Mapping[str, object]],
    *,
    task_value_band: tuple[Decimal, Decimal] = (Decimal("0.5"), Decimal("1.5")),
) -> WrittenFile:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    with path.open("wb") as handle:
        for row in rows:
            errors = validate_row(schema_name, row, task_value_band=task_value_band)
            if errors:
                raise ValueError(f"{path.name} row {count + 1}: {errors}")
            data = (canonical_json(row, schema_name=schema_name) + "\n").encode("utf-8")
            handle.write(data)
            digest.update(data)
            count += 1
            byte_count += len(data)
    return WrittenFile(path.name, digest.hexdigest(), count, byte_count)


def _write_json(path: Path, value: object) -> WrittenFile:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(data)
    return WrittenFile(path.name, hashlib.sha256(data).hexdigest(), 1, len(data))


def _calibration_input_hash(seed: int, anchor_rows: Iterable[Mapping[str, object]], config: TraceConfig) -> str:
    digest = hashlib.sha256()
    digest.update(f"{DATASET_ID}|{seed}|N_cal={config.calibration_n}|anchor_count={config.anchor_count_per_cell}".encode("ascii"))
    for row in anchor_rows:
        digest.update(canonical_json(row, schema_name="anchors").encode("utf-8"))
    return digest.hexdigest()


def _contract_rows(
    seed: int,
    workers: Iterable[Mapping[str, object]],
    tasks: Iterable[Mapping[str, object]],
    available_mappings: Iterable[Mapping[str, object]],
    continuation_rows: Iterable[Mapping[str, object]],
    config: TraceConfig,
) -> Iterator[dict[str, object]]:
    workers_by_id = {str(row["worker_id"]): row for row in workers}
    tasks_by_id = {str(row["task_id"]): row for row in tasks}
    gbar = {
        (int(row["cell"]), str(row["public_signal_role"]), Decimal(row["effort"])): Decimal(row["gbar"])
        for row in continuation_rows
    }
    for mapping in available_mappings:
        worker = workers_by_id[str(mapping["worker_id"])]
        task = tasks_by_id[str(mapping["mapped_task_id"])]
        role = str(worker["public_signal_role"])
        cell = int(task["cell"])
        coefficient = (
            Decimal(task["V"])
            / Decimal(int(task["K"]))
            * gbar[(cell, role, Decimal("0.5"))]
        ).quantize(GBAR_PRECISION, rounding=ROUND_HALF_EVEN)
        for gamma in GAMMAS:
            sbar = (gamma * Decimal(task["V"]) / Decimal(int(task["K"]))).quantize(
                GBAR_PRECISION, rounding=ROUND_HALF_EVEN
            )
            gbar_by_effort = {
                decimal_text(effort): (sbar * gbar[(cell, role, effort)]).quantize(
                    GBAR_PRECISION, rounding=ROUND_HALF_EVEN
                )
                for effort in EFFORTS
            }
            base = {
                "seed": seed,
                "method_id": "V2-FULL",
                "gamma": gamma,
                "slot": task["slot"],
                "task_id": task["task_id"],
                "worker_id": worker["worker_id"],
                "public_signal_role": role,
                "sbar": sbar,
                "Gbar_by_effort": gbar_by_effort,
                "vhat": {
                    "rule": "rho_i(t-)*coefficient",
                    "coefficient": coefficient,
                    "rho0_example": (config.rho0 * coefficient).quantize(GBAR_PRECISION, rounding=ROUND_HALF_EVEN),
                },
                "Delta_money": MONEY_GRID,
                "epsilon_rank": Decimal("0.001"),
            }
            contract_hash = hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()
            yield {**base, "contract_hash": contract_hash}


def generate_trace(
    *,
    seed: int,
    output_directory: Path,
    root: Path,
    config: TraceConfig,
    formal: bool,
) -> TraceWriteResult:
    if formal:
        config.assert_formal()
    if output_directory.exists():
        raise FileExistsError(f"trace directory already exists; version bump required: {output_directory}")
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary trace directory: {temporary}")
    temporary.mkdir(parents=True)
    try:
        workers = generate_workers(seed, config)
        tasks = generate_tasks(seed, config)
        anchors, anchor_bounds = generate_anchor_history(seed, config)
        files: list[WrittenFile] = []
        files.append(_write_jsonl(temporary / "workers.jsonl", "workers", workers))
        files.append(_write_jsonl(temporary / "tasks.jsonl", "tasks", tasks))
        files.append(_write_jsonl(temporary / "anchors.jsonl", "anchors", anchors))

        available_mappings: list[dict[str, object]] = []

        def eligibility_stream() -> Iterator[dict[str, object]]:
            for row in generate_eligibility(seed, config, workers, tasks):
                if row["available"] and row["mapped_task_id"] is not None:
                    available_mappings.append(row)
                yield row

        files.append(_write_jsonl(temporary / "eligibility.jsonl", "eligibility", eligibility_stream()))

        source_hash = generator_source_hash(root)
        calibration_input_hash = _calibration_input_hash(seed, anchors, config)
        continuation_rows, certificates = calibrate_seed(
            seed,
            anchor_bounds,
            config,
            data_hash=calibration_input_hash,
            code_hash=source_hash,
        )
        files.append(
            _write_jsonl(
                temporary / "continuation_tables.jsonl", "continuation_tables", continuation_rows
            )
        )
        files.append(
            _write_jsonl(
                temporary / "potential_reports.jsonl",
                "potential_reports",
                generate_potential_reports(seed, workers, tasks, available_mappings, anchor_bounds),
            )
        )
        files.append(
            _write_jsonl(
                temporary / "contracts.jsonl",
                "contracts",
                _contract_rows(seed, workers, tasks, available_mappings, continuation_rows, config),
            )
        )
        files.append(
            _write_jsonl(
                temporary / "holdout_provenance.jsonl",
                "holdout_provenance",
                generate_holdout_provenance(seed, tasks),
            )
        )
        files.append(_write_json(temporary / "epsilon_certificates.json", certificates))
        anchor_version = next(iter(anchor_bounds.values())).version
        metadata = {
            "dataset_id": DATASET_ID,
            "generator_version": GENERATOR_VERSION,
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "formal": formal,
            "screening_label": SCREENING_LABEL,
            "generator_source_hash": source_hash,
            "calibration_input_hash": calibration_input_hash,
            "anchor_version": anchor_version,
            "counts": {
                "workers": len(workers),
                "tasks": len(tasks),
                "eligibility": config.horizon * config.worker_count,
                "available_mapped": len(available_mappings),
                "anchors": len(anchors),
                "potential_reports": len(available_mappings) * len(EFFORTS),
                "continuation_tables": len(continuation_rows),
                "contracts": len(available_mappings) * len(GAMMAS),
                "holdout_provenance": len(tasks),
                "epsilon_certificates": len(certificates),
            },
        }
        files.append(_write_json(temporary / "trace_metadata.json", metadata))
        os.replace(temporary, output_directory)
        return TraceWriteResult(
            seed=seed,
            directory=output_directory,
            files=tuple(files),
            task_count=len(tasks),
            worker_count=len(workers),
            eligibility_count=config.horizon * config.worker_count,
            available_mapped_count=len(available_mappings),
            anchor_version=anchor_version,
            calibration_certificate_count=len(certificates),
        )
    except Exception:
        # Keep the temporary directory for forensic inspection.  A subsequent
        # run must remove it explicitly after diagnosing the failure.
        raise
