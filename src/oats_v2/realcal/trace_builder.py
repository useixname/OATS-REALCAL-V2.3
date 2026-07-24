"""Write a REAL-CAL trace directory (same schema/layout as SYN-V2-1).

Reuses the distribution-neutral machinery from the SYN pipeline (Gbar calibration,
contract rows, holdout provenance) and swaps in the profile-driven generators for
the distribution-bearing stages. Output goes to ``data/<dataset>/<seed>/`` and
is consumable by the existing ``trace_loader`` / ``formal_runner`` without change.

Dataset versions:
  * REAL-CAL-V1 — frozen; V in [0.5, 1.5] (inherited from SYN-V2-1).
  * REAL-CAL — identical distributions with the task-value unit rescaled by
    ``REALCAL_V2_VALUE_SCALE`` (preregistered market-viability recalibration).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ..data.schemas import DATASET_ID as SYN_DATASET_ID  # noqa: F401  (reference)
from ..data.schemas import EFFORTS, FORMAL_SEEDS, GAMMAS, TraceConfig, canonical_json, validate_row
from ..data.holdout_generator import generate_holdout_provenance
from ..data.trace_writer import _contract_rows, _write_json, _write_jsonl, generator_source_hash
from . import (
    REALCAL_DATASET_ID,
    REALCAL_PROFILE_VERSION,
    REALCAL_V2_DATASET_ID,
    REALCAL_V2_PROFILE_VERSION,
    REALCAL_V2_VALUE_SCALE,
)
from .calibration import calibrate_realcal_seed
from .config import RealCalConfig, load_realcal_config
from .generator import (
    generate_anchor_history,
    generate_eligibility,
    generate_potential_reports,
    generate_tasks,
    generate_workers,
)


def _calibration_input_hash(
    dataset_id: str, seed: int, anchor_rows: Iterable[Mapping[str, object]], config: TraceConfig
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{dataset_id}|{seed}|N_cal={config.calibration_n}|anchor_count={config.anchor_count_per_cell}".encode("ascii")
    )
    for row in anchor_rows:
        digest.update(canonical_json(row, schema_name="anchors").encode("utf-8"))
    return digest.hexdigest()


def generate_realcal_trace(
    *,
    seed: int,
    output_directory: Path,
    root: Path,
    profile_path: Path,
    dataset_version: int = 1,
) -> dict:
    cfg: RealCalConfig = load_realcal_config(profile_path)
    if dataset_version == 1:
        dataset_id = REALCAL_DATASET_ID
        profile_version = REALCAL_PROFILE_VERSION
    elif dataset_version == 2:
        dataset_id = REALCAL_V2_DATASET_ID
        profile_version = REALCAL_V2_PROFILE_VERSION
        scale = Decimal(REALCAL_V2_VALUE_SCALE)
        cfg = replace(
            cfg,
            task_value_low=cfg.task_value_low * scale,
            task_value_high=cfg.task_value_high * scale,
        )
    else:
        raise ValueError(f"unknown REAL-CAL dataset version: {dataset_version}")
    value_band = (cfg.task_value_low, cfg.task_value_high)
    calib_config = TraceConfig()  # frozen constants for Gbar calibration only

    if output_directory.exists():
        raise FileExistsError(f"trace directory already exists; version bump required: {output_directory}")
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary trace directory: {temporary}")
    temporary.mkdir(parents=True)

    workers = generate_workers(seed, cfg)
    tasks = generate_tasks(seed, cfg)
    anchors, anchor_bounds = generate_anchor_history(seed, cfg)

    files = []
    files.append(_write_jsonl(temporary / "workers.jsonl", "workers", workers))
    files.append(_write_jsonl(temporary / "tasks.jsonl", "tasks", tasks, task_value_band=value_band))
    files.append(_write_jsonl(temporary / "anchors.jsonl", "anchors", anchors))

    available_mappings: list[dict[str, object]] = []

    def eligibility_stream() -> Iterator[dict[str, object]]:
        for row in generate_eligibility(seed, cfg, workers, tasks):
            if row["available"] and row["mapped_task_id"] is not None:
                available_mappings.append(row)
            yield row

    files.append(_write_jsonl(temporary / "eligibility.jsonl", "eligibility", eligibility_stream()))

    source_hash = generator_source_hash(root)
    calibration_input_hash = _calibration_input_hash(dataset_id, seed, anchors, calib_config)
    seed_index = FORMAL_SEEDS.index(seed) if seed in FORMAL_SEEDS else 0
    continuation_rows, certificates = calibrate_realcal_seed(
        seed,
        seed_index,
        anchor_bounds,
        cfg,
        calib_config,
        data_hash=calibration_input_hash,
        code_hash=source_hash,
    )
    files.append(_write_jsonl(temporary / "continuation_tables.jsonl", "continuation_tables", continuation_rows))
    files.append(
        _write_jsonl(
            temporary / "potential_reports.jsonl",
            "potential_reports",
            generate_potential_reports(seed, cfg, workers, tasks, available_mappings, anchor_bounds),
        )
    )
    files.append(
        _write_jsonl(
            temporary / "contracts.jsonl",
            "contracts",
            _contract_rows(seed, workers, tasks, available_mappings, continuation_rows, calib_config),
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
        "dataset_id": dataset_id,
        "profile_version": profile_version,
        "profile_hash": cfg.profile_hash,
        "seed": seed,
        "semi_synthetic": True,
        "claim_ceiling": "real-data-calibrated semi-synthetic evidence only",
        "task_value_band": [str(value_band[0]), str(value_band[1])],
        "value_scale": REALCAL_V2_VALUE_SCALE if dataset_version == 2 else 1,
        "generator_source_hash": source_hash,
        "calibration_input_hash": calibration_input_hash,
        "anchor_version": anchor_version,
        "counts": {
            "workers": len(workers),
            "tasks": len(tasks),
            "eligibility": cfg.horizon * cfg.worker_count,
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
    return {
        "seed": seed,
        "directory": str(output_directory),
        "files": [{"name": f.name, "sha256": f.sha256, "rows": f.rows, "bytes": f.bytes} for f in files],
        "task_count": len(tasks),
        "worker_count": len(workers),
        "available_mapped": len(available_mappings),
        "anchor_version": anchor_version,
        "profile_hash": cfg.profile_hash,
    }

