from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping, Sequence

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.oats_external.adapters.oasis_tsc import (
    OasisCandidate,
    OasisSpecificationViolation,
    OasisTSCPolicy,
)
from src.oats_external.realcal_bridge import (
    canonical_hash,
    load_trace_hash_manifest,
    sha256_file,
    verify_seed_files,
)
from src.oats_external.types import CandidateKey


DEFAULT_DATA = ROOT / "data" / "REAL-CAL-V2"
DEFAULT_FORECAST = (
    ROOT
    / "reports"
    / "published_baseline_oasis_20260727"
    / "arrival_count_forecast.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "published_baseline_oasis_20260727_r1"
DEFAULT_BETAS = ("0.03", "0.05", "0.10", "0.25", "0.50")
EVALUATION_SEEDS = tuple(range(20260715, 20260725))
VALIDATION_SEEDS = tuple(range(20260735, 20260740))
ZERO = Decimal("0")
THREE = Decimal("3")


@dataclass(frozen=True, slots=True)
class Task:
    slot: int
    task_id: str
    cell: int
    value: Decimal
    capacity: int
    deadline: int


@dataclass(frozen=True, slots=True)
class Potential:
    report: Decimal
    score: Decimal


@dataclass(slots=True)
class Trace:
    tasks_by_slot: Mapping[int, tuple[Task, ...]]
    bids: Mapping[str, Decimal]
    candidates_by_task: Mapping[str, tuple[str, ...]]
    potential: Mapping[CandidateKey, Potential]
    task_count: int
    candidate_count: int
    reference_budget: Decimal


class ArrivalForecast:
    def __init__(self, payload: Mapping[str, object]) -> None:
        unsigned = dict(payload)
        supplied_hash = str(unsigned.pop("model_hash"))
        if canonical_hash(unsigned) != supplied_hash:
            raise RuntimeError("arrival forecast hash mismatch")
        self.model_hash = supplied_hash
        self.slot_cell = {
            str(key): Decimal(str(value))
            for key, value in dict(payload["slot_cell_mean"]).items()
        }
        self.cell = {
            str(key): Decimal(str(value))
            for key, value in dict(payload["cell_mean"]).items()
        }
        self.global_mean = Decimal(str(payload["global_mean"]))

    def predict(self, slot: int, cell: int) -> int:
        mean = self.slot_cell.get(
            f"{slot}|{cell}",
            self.cell.get(str(cell), self.global_mean),
        )
        return int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _public_order(seed: int, task_id: str, worker_id: str) -> bytes:
    return hashlib.sha256(
        f"{seed}|{task_id}|{worker_id}".encode("utf-8")
    ).digest()


def _load_trace(
    data_root: Path,
    seed: int,
    trace_manifest: Mapping[str, Mapping[str, str]],
) -> tuple[Trace, dict[str, str]]:
    verified = verify_seed_files(
        data_root,
        seed,
        (
            "tasks.jsonl",
            "workers.jsonl",
            "eligibility.jsonl",
            "potential_reports.jsonl",
        ),
        trace_manifest,
    )
    tasks_by_slot_mutable: dict[int, list[Task]] = defaultdict(list)
    task_ids: set[str] = set()
    reference_budget = ZERO
    for row in _rows(data_root / str(seed) / "tasks.jsonl"):
        task = Task(
            slot=int(row["slot"]),
            task_id=str(row["task_id"]),
            cell=int(row["cell"]),
            value=Decimal(str(row["V"])),
            capacity=int(row["K"]),
            deadline=int(row["deadline"]),
        )
        tasks_by_slot_mutable[task.slot].append(task)
        task_ids.add(task.task_id)
        reference_budget += THREE * Decimal(task.capacity) + task.value

    bids = {
        str(row["worker_id"]): Decimal(str(row["c_i"]))
        for row in _rows(data_root / str(seed) / "workers.jsonl")
    }
    candidates_mutable: dict[str, list[str]] = defaultdict(list)
    candidate_keys: set[CandidateKey] = set()
    for row in _rows(data_root / str(seed) / "eligibility.jsonl"):
        if bool(row["available"]) and row["mapped_task_id"] is not None:
            task_id = str(row["mapped_task_id"])
            worker_id = str(row["worker_id"])
            if task_id not in task_ids:
                raise RuntimeError(
                    f"seed {seed}: eligibility maps to unknown task {task_id}"
                )
            if worker_id not in bids:
                raise RuntimeError(
                    f"seed {seed}: eligibility maps to unknown worker {worker_id}"
                )
            key = CandidateKey(task_id=task_id, worker_id=worker_id)
            if key in candidate_keys:
                raise RuntimeError(f"seed {seed}: duplicate eligible pair {key}")
            candidate_keys.add(key)
            candidates_mutable[task_id].append(worker_id)

    potential: dict[CandidateKey, Potential] = {}
    for row in _rows(data_root / str(seed) / "potential_reports.jsonl"):
        if str(row["effort"]) != "0":
            continue
        key = CandidateKey(
            task_id=str(row["task_id"]),
            worker_id=str(row["worker_id"]),
        )
        if key not in candidate_keys:
            raise RuntimeError(f"seed {seed}: potential for ineligible pair {key}")
        if key in potential:
            raise RuntimeError(f"seed {seed}: duplicate effort-0 potential {key}")
        potential[key] = Potential(
            report=Decimal(str(row["report"])),
            score=Decimal(str(row["score"])),
        )
    missing = candidate_keys.difference(potential)
    if missing:
        raise RuntimeError(
            f"seed {seed}: {len(missing)} eligible pairs lack effort-0 potential"
        )

    tasks_by_slot = {
        slot: tuple(sorted(tasks, key=lambda task: task.task_id))
        for slot, tasks in tasks_by_slot_mutable.items()
    }
    candidates_by_task = {
        task_id: tuple(
            sorted(
                workers,
                key=lambda worker_id: (
                    _public_order(seed, task_id, worker_id),
                    worker_id,
                ),
            )
        )
        for task_id, workers in candidates_mutable.items()
    }
    return (
        Trace(
            tasks_by_slot=tasks_by_slot,
            bids=bids,
            candidates_by_task=candidates_by_task,
            potential=potential,
            task_count=sum(len(tasks) for tasks in tasks_by_slot.values()),
            candidate_count=len(candidate_keys),
            reference_budget=reference_budget,
        ),
        verified,
    )


def _run_cell(
    *,
    seed: int,
    beta: Decimal,
    trace: Trace,
    forecast: ArrivalForecast,
    trace_hashes: Mapping[str, str],
) -> dict[str, object]:
    started = time.perf_counter()
    policy = OasisTSCPolicy()
    policy.reset(seed)

    selected_count = 0
    gross_value = ZERO
    score_sum = ZERO
    native_quality_sum = ZERO
    basic_payment = ZERO
    actual_payment = ZERO
    observation_exclusions = 0
    sample_replacements = 0
    zero_range_tasks = 0
    truth_iterations = 0
    truth_nonconvergence = 0
    basic_budget_violations = 0
    actual_budget_violations = 0
    actual_budget_excess = ZERO
    ir_violations = 0
    deadline_satisfied = 0
    decision_hashes: list[str] = []
    settlement_hashes: list[str] = []

    for slot in sorted(trace.tasks_by_slot):
        pending: list[
            tuple[
                Task,
                object,
                Mapping[CandidateKey, Decimal],
                Mapping[CandidateKey, Decimal],
                Mapping[CandidateKey, Decimal],
            ]
        ] = []
        for task in trace.tasks_by_slot[slot]:
            worker_ids = trace.candidates_by_task.get(task.task_id, ())
            candidates = tuple(
                OasisCandidate(
                    key=CandidateKey(task.task_id, worker_id),
                    bid=trace.bids[worker_id],
                    long_term_quality=policy.quality_of(worker_id),
                )
                for worker_id in worker_ids
            )
            task_budget = beta * (
                THREE * Decimal(task.capacity) + task.value
            )
            selection = policy.select_task(
                task_id=task.task_id,
                task_budget=task_budget,
                capacity=task.capacity,
                ordered_candidates=candidates,
                forecast_candidate_count=forecast.predict(task.slot, task.cell),
            )
            decision_hashes.append(selection.decision_hash)
            observation_exclusions += selection.observation_exclusions
            sample_replacements += selection.sample_replacements
            basic_budget_violations += int(selection.basic_budget_violation)
            reports = {
                key: trace.potential[key].report for key in selection.winners
            }
            scores = {
                key: trace.potential[key].score for key in selection.winners
            }
            winner_bids = {
                key: trace.bids[key.worker_id] for key in selection.winners
            }
            pending.append((task, selection, reports, scores, winner_bids))

        updates: list[tuple[CandidateKey, Decimal]] = []
        for task, selection, reports, scores, winner_bids in pending:
            settlement = policy.settle_task(
                selection,
                reports=reports,
                bids=winner_bids,
            )
            settlement_hashes.append(settlement.settlement_hash)
            truth_iterations += settlement.truth_iterations
            truth_nonconvergence += int(not settlement.truth_converged)
            zero_range_tasks += int(settlement.zero_range_normalization)
            basic_payment += settlement.total_basic_payment
            actual_payment += settlement.total_actual_payment
            actual_budget_violations += int(settlement.actual_budget_violation)
            if settlement.actual_budget_violation:
                actual_budget_excess += (
                    settlement.total_actual_payment - selection.task_budget
                )
            ir_violations += settlement.individual_rationality_violations
            for key in selection.winners:
                selected_count += 1
                score = scores[key]
                score_sum += score
                native_quality_sum += settlement.current_quality[key]
                gross_value += task.value / Decimal(task.capacity) * score
                deadline_satisfied += int(task.slot <= task.deadline)
            updates.extend(settlement.current_quality.items())
        policy.apply_quality_updates(updates)

    runtime = time.perf_counter() - started
    run_budget = beta * trace.reference_budget
    net_value = gross_value - actual_payment
    comparison_status = (
        "ELIGIBLE"
        if actual_budget_violations == 0 and truth_nonconvergence == 0
        else (
            "NA_TRUTH_NONCONVERGENCE"
            if truth_nonconvergence
            else "NA_PAYMENT_BUDGET_GATE_FAILED"
        )
    )
    payload: dict[str, object] = {
        "schema": "oasis-realcal-cell-v1",
        "method": policy.method_id,
        "paper_doi": policy.paper_doi,
        "trace_seed": seed,
        "beta": str(beta),
        "delay": 0,
        "missing": "0",
        "horizon": 1000,
        "effort_bridge": "frozen_potential_effort_0",
        "task_budget_formula": "beta * (3*K_T + V_T)",
        "reference_budget": str(trace.reference_budget),
        "run_budget": str(run_budget),
        "task_budget_sum_equals_run_budget": True,
        "task_count": trace.task_count,
        "candidate_count": trace.candidate_count,
        "recruited_count": selected_count,
        "gross_external_value": str(gross_value),
        "total_basic_payment": str(basic_payment),
        "total_actual_payment": str(actual_payment),
        "platform_net_value": str(net_value),
        "budget_efficiency": (
            None if actual_payment == ZERO else str(net_value / actual_payment)
        ),
        "mean_selected_external_quality": (
            None if selected_count == 0 else str(score_sum / Decimal(selected_count))
        ),
        "mean_selected_oasis_current_quality": (
            None
            if selected_count == 0
            else str(native_quality_sum / Decimal(selected_count))
        ),
        "deadline_satisfaction_rate": (
            None
            if selected_count == 0
            else str(Decimal(deadline_satisfied) / Decimal(selected_count))
        ),
        "observation_exclusion_count": observation_exclusions,
        "sample_replacement_count": sample_replacements,
        "zero_range_normalization_task_count": zero_range_tasks,
        "truth_iteration_count": truth_iterations,
        "truth_nonconvergence_count": truth_nonconvergence,
        "basic_payment_budget_violation_count": basic_budget_violations,
        "actual_payment_budget_violation_count": actual_budget_violations,
        "actual_payment_budget_excess": str(actual_budget_excess),
        "individual_rationality_violation_count": ir_violations,
        "comparison_status": comparison_status,
        "arrival_forecast_model_hash": forecast.model_hash,
        "trace_file_hashes": dict(sorted(trace_hashes.items())),
        "decision_stream_hash": canonical_hash(decision_hashes),
        "settlement_stream_hash": canonical_hash(settlement_hashes),
        "policy_audit": policy.audit_state(),
    }
    payload["deterministic_result_hash"] = canonical_hash(payload)
    payload["runtime_seconds"] = runtime
    payload["process_rss_bytes_at_completion"] = psutil.Process().memory_info().rss
    unsigned = dict(payload)
    payload["cell_hash"] = canonical_hash(unsigned)
    return payload


def _run_seed_job(job: Mapping[str, object]) -> dict[str, object]:
    data_root = Path(str(job["data_root"]))
    seed = int(job["seed"])
    betas = tuple(Decimal(str(value)) for value in job["betas"])
    forecast_payload = json.loads(Path(str(job["forecast_path"])).read_text("utf-8"))
    forecast = ArrivalForecast(forecast_payload)
    trace_manifest = load_trace_hash_manifest(data_root)
    trace, trace_hashes = _load_trace(data_root, seed, trace_manifest)
    cells = [
        _run_cell(
            seed=seed,
            beta=beta,
            trace=trace,
            forecast=forecast,
            trace_hashes=trace_hashes,
        )
        for beta in betas
    ]
    if bool(job.get("determinism_replay")):
        replay_cells = [
            _run_cell(
                seed=seed,
                beta=beta,
                trace=trace,
                forecast=forecast,
                trace_hashes=trace_hashes,
            )
            for beta in betas
        ]
        for cell, replay in zip(cells, replay_cells, strict=True):
            cell["determinism_replay_match"] = (
                cell["deterministic_result_hash"]
                == replay["deterministic_result_hash"]
            )
            unsigned = dict(cell)
            unsigned.pop("cell_hash", None)
            cell["cell_hash"] = canonical_hash(unsigned)
    else:
        for cell in cells:
            cell["determinism_replay_match"] = None
            unsigned = dict(cell)
            unsigned.pop("cell_hash", None)
            cell["cell_hash"] = canonical_hash(unsigned)
    return {
        "seed": seed,
        "cells": cells,
        "trace_task_count": trace.task_count,
        "trace_candidate_count": trace.candidate_count,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "formal"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--betas", nargs="+", default=DEFAULT_BETAS)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    forecast_path = args.forecast.resolve()
    output_root = args.output_root.resolve()
    seeds = tuple(
        args.seeds
        if args.seeds
        else (VALIDATION_SEEDS if args.stage == "validation" else EVALUATION_SEEDS)
    )
    betas = tuple(str(Decimal(value)) for value in args.betas)
    if not forecast_path.is_file():
        raise SystemExit(f"missing frozen arrival forecast: {forecast_path}")
    if args.stage == "formal":
        validation_receipt = output_root / "validation" / "VALIDATION_GATE.json"
        if not validation_receipt.is_file():
            raise SystemExit("formal run requires a passed validation gate")
        gate = json.loads(validation_receipt.read_text("utf-8"))
        if gate.get("status") != "PASS":
            raise SystemExit("validation gate is not PASS")

    stage_root = output_root / args.stage
    if stage_root.exists():
        raise SystemExit(f"refusing to overwrite existing stage root: {stage_root}")
    stage_root.mkdir(parents=True)
    started = time.perf_counter()
    jobs = [
        {
            "data_root": str(data_root),
            "forecast_path": str(forecast_path),
            "seed": seed,
            "betas": betas,
            "determinism_replay": args.stage == "validation",
        }
        for seed in seeds
    ]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_seed_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            seed = int(result["seed"])
            for cell in result["cells"]:
                beta_slug = str(cell["beta"]).replace(".", "p")
                _write_json(
                    stage_root / "cells" / f"seed_{seed}_beta_{beta_slug}.json",
                    cell,
                )
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "seed": seed,
                        "cell_count": len(result["cells"]),
                    }
                ),
                flush=True,
            )

    cells = [
        cell
        for result in sorted(results, key=lambda row: int(row["seed"]))
        for cell in sorted(
            result["cells"], key=lambda row: Decimal(str(row["beta"]))
        )
    ]
    manifest: dict[str, object] = {
        "schema": "oasis-realcal-stage-manifest-v1",
        "stage": args.stage,
        "parent_freeze_id": "OATS-TRUSTFIX-PRE-PUBLISHED-BASELINE-20260727-R1",
        "method": "OASIS-TSC-2024",
        "paper_doi": "10.1109/TSC.2024.3354240",
        "seeds": list(seeds),
        "betas": list(betas),
        "cell_count": len(cells),
        "comparison_status_counts": {
            status: sum(1 for cell in cells if cell["comparison_status"] == status)
            for status in sorted({str(cell["comparison_status"]) for cell in cells})
        },
        "truth_nonconvergence_count": sum(
            int(cell["truth_nonconvergence_count"]) for cell in cells
        ),
        "basic_payment_budget_violation_count": sum(
            int(cell["basic_payment_budget_violation_count"]) for cell in cells
        ),
        "actual_payment_budget_violation_count": sum(
            int(cell["actual_payment_budget_violation_count"]) for cell in cells
        ),
        "individual_rationality_violation_count": sum(
            int(cell["individual_rationality_violation_count"]) for cell in cells
        ),
        "determinism_replay_mismatch_count": sum(
            int(cell["determinism_replay_match"] is False) for cell in cells
        ),
        "forecast_sha256": sha256_file(forecast_path),
        "source_hashes": {
            "adapter": sha256_file(
                ROOT / "src" / "oats_external" / "adapters" / "oasis_tsc.py"
            ),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "cell_hashes": [str(cell["cell_hash"]) for cell in cells],
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(stage_root / "STAGE_MANIFEST.json", manifest)

    if args.stage == "validation":
        # Budget incompatibility is an empirical boundary, not a software-gate
        # failure.  The implementation gates are determinism, convergence, and
        # exact compliance with the basic-payment budget.
        status = (
            "PASS"
            if manifest["truth_nonconvergence_count"] == 0
            and manifest["basic_payment_budget_violation_count"] == 0
            and manifest["determinism_replay_mismatch_count"] == 0
            and len(cells) == len(seeds) * len(betas)
            else "FAIL"
        )
        gate = {
            "status": status,
            "stage_manifest_hash": manifest["manifest_hash"],
            "required_cell_count": len(seeds) * len(betas),
            "observed_cell_count": len(cells),
            "truth_nonconvergence_count": manifest[
                "truth_nonconvergence_count"
            ],
            "basic_payment_budget_violation_count": manifest[
                "basic_payment_budget_violation_count"
            ],
            "determinism_replay_mismatch_count": manifest[
                "determinism_replay_mismatch_count"
            ],
            "actual_payment_budget_violations_are_reported_not_clipped": True,
        }
        gate["gate_hash"] = canonical_hash(gate)
        _write_json(stage_root / "VALIDATION_GATE.json", gate)
        if status != "PASS":
            raise SystemExit("validation gate failed")

    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
