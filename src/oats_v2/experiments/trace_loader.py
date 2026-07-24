from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from ..data.schemas import FORMAL_SEEDS, GAMMAS, MONEY_GRID, TraceConfig
from ..types import D


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    worker_id: str
    stratum: str
    public_signal_role: str
    c_i: Decimal
    bias_sign: int


@dataclass(frozen=True, slots=True)
class TaskRecord:
    slot: int
    task_id: str
    cell: int
    theta: Decimal
    value: Decimal
    capacity: int
    deadline: int
    z: Decimal
    delay_mask: dict[str, int]
    missing_mask: dict[str, bool]


@dataclass(frozen=True, slots=True)
class PotentialRecord:
    slot: int
    task_id: str
    worker_id: str
    effort: Decimal
    report: Decimal
    score: Decimal
    screen_status: str
    v_ijt: Decimal


@dataclass(frozen=True, slots=True)
class ContractRecord:
    slot: int
    task_id: str
    worker_id: str
    gamma: Decimal
    public_signal_role: str
    sbar: Decimal
    gbar_by_effort: dict[str, Decimal]
    vhat_coefficient: Decimal
    delta_money: Decimal
    epsilon_rank: Decimal


@dataclass
class TraceBundle:
    seed: int
    trace_dir: Path
    workers: dict[str, WorkerRecord] = field(default_factory=dict)
    tasks_by_slot: dict[int, list[TaskRecord]] = field(default_factory=dict)
    tasks_by_id: dict[str, TaskRecord] = field(default_factory=dict)
    eligibility_by_slot: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    available_by_slot: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    contracts: dict[tuple[int, str, str, Decimal], ContractRecord] = field(default_factory=dict)
    potential: dict[tuple[int, str, str, Decimal], PotentialRecord] = field(default_factory=dict)
    anchors_by_cell: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    continuation: dict[tuple[int, str, Decimal], dict[str, Any]] = field(default_factory=dict)
    anchor_version: str = ""
    trace_hash: str = ""
    file_hashes: dict[str, str] = field(default_factory=dict)

    def verify_hashes(self, expected: dict[str, str]) -> None:
        for name, digest in expected.items():
            if self.file_hashes.get(name) != digest:
                raise ValueError(f"trace hash mismatch for {name}: seed={self.seed}")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contracts(
    bundle: TraceBundle,
    *,
    load_gammas: frozenset[Decimal] | None = None,
) -> None:
    """Replace ``bundle.contracts`` with rows from the trace contracts file.

    Used by formal workers to keep the heavy potential/task payload resident
    while swapping only the active gamma's contracts (~6x smaller).
    """
    bundle.contracts.clear()
    gamma_needles: tuple[str, ...] | None = None
    if load_gammas is not None:
        # Contracts store gamma as a JSON string (e.g. "0.5", "1.0"). Cheap
        # substring filter avoids json.loads on the ~5/6 non-matching rows.
        gamma_needles = tuple(f'"gamma":"{g}"' for g in (str(g) for g in load_gammas))
    path = bundle.trace_dir / "contracts.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if gamma_needles is not None and not any(n in line for n in gamma_needles):
                continue
            row = json.loads(line)
            if row.get("method_id") != "V2-FULL":
                continue
            gamma = D(row["gamma"])
            if load_gammas is not None and gamma not in load_gammas:
                continue
            key = (int(row["slot"]), row["task_id"], row["worker_id"], gamma)
            gbar = {str(k): D(v) for k, v in row["Gbar_by_effort"].items()}
            bundle.contracts[key] = ContractRecord(
                slot=int(row["slot"]),
                task_id=row["task_id"],
                worker_id=row["worker_id"],
                gamma=gamma,
                public_signal_role=row["public_signal_role"],
                sbar=D(row["sbar"]),
                gbar_by_effort=gbar,
                vhat_coefficient=D(row["vhat"]["coefficient"]),
                delta_money=D(row["Delta_money"]),
                epsilon_rank=D(row["epsilon_rank"]),
            )


def load_trace(
    seed: int,
    data_root: Path,
    trace_hashes: dict[str, Any] | None = None,
    *,
    verify_hashes: bool = False,
    load_eligibility_index: bool = True,
    load_gammas: frozenset[Decimal] | None = None,
    load_contracts_flag: bool = True,
    allowed_seeds: frozenset[int] | None = None,
) -> TraceBundle:
    """Load a trace bundle.

    ``load_eligibility_index`` controls whether the full per-slot eligibility list
    (all workers, including unavailable ones) is retained. It is consumed ONLY by
    the offline LP comparator; the per-cell simulation uses ``available_by_slot``.
    Worker processes that only run ``simulate_cell`` can set this False to roughly
    halve resident memory without changing any computed output.

    ``load_gammas`` restricts which contract rows are retained. Formal simulation
    only ever looks up ``contracts[(slot, task, worker, cell.gamma)]``, so a
    worker can load one gamma at a time and cut contract RAM ~6x with identical
    cell outputs. Set ``load_contracts_flag=False`` to skip contracts entirely
    (then call ``load_contracts`` to fill them later).
    """
    seed_allowlist = frozenset(FORMAL_SEEDS) if allowed_seeds is None else allowed_seeds
    if seed not in seed_allowlist:
        raise ValueError(f"seed {seed} not in formal seed list")
    trace_dir = data_root / str(seed)
    bundle = TraceBundle(seed=seed, trace_dir=trace_dir)
    jsonl_files = (
        "workers.jsonl",
        "tasks.jsonl",
        "eligibility.jsonl",
        "anchors.jsonl",
        "potential_reports.jsonl",
        "continuation_tables.jsonl",
        "contracts.jsonl",
        "holdout_provenance.jsonl",
    )
    for name in jsonl_files:
        if verify_hashes:
            bundle.file_hashes[name] = _file_sha256(trace_dir / name)

    if trace_hashes is not None and verify_hashes:
        expected = trace_hashes.get("seed_file_hashes", {}).get(str(seed), {})
        bundle.verify_hashes(expected)

    for row in _read_jsonl(trace_dir / "workers.jsonl"):
        bundle.workers[row["worker_id"]] = WorkerRecord(
            worker_id=row["worker_id"],
            stratum=row["stratum"],
            public_signal_role=row["public_signal_role"],
            c_i=D(row["c_i"]),
            bias_sign=int(row["bias_sign"]),
        )

    for row in _read_jsonl(trace_dir / "tasks.jsonl"):
        task = TaskRecord(
            slot=int(row["slot"]),
            task_id=row["task_id"],
            cell=int(row["cell"]),
            theta=D(row["theta"]),
            value=D(row["V"]),
            capacity=int(row["K"]),
            deadline=int(row["deadline"]),
            z=D(row["z"]),
            delay_mask={str(k): int(v) for k, v in row["delay_mask"].items()},
            missing_mask={str(k): bool(v) for k, v in row["missing_mask"].items()},
        )
        bundle.tasks_by_slot.setdefault(task.slot, []).append(task)
        bundle.tasks_by_id[task.task_id] = task

    for row in _read_jsonl(trace_dir / "eligibility.jsonl"):
        slot = int(row["slot"])
        if load_eligibility_index:
            bundle.eligibility_by_slot.setdefault(slot, []).append(row)
        if row.get("available") and row.get("mapped_task_id"):
            # Simulation only needs these two fields; dropping the rest of the
            # eligibility row cuts resident memory without changing lookups.
            bundle.available_by_slot.setdefault(slot, []).append(
                {
                    "worker_id": row["worker_id"],
                    "mapped_task_id": row["mapped_task_id"],
                }
            )

    for row in _read_jsonl(trace_dir / "anchors.jsonl"):
        cell = int(row["cell"])
        bundle.anchors_by_cell.setdefault(cell, []).append(row)
        if not bundle.anchor_version:
            bundle.anchor_version = row["anchor_version"]

    for row in _read_jsonl(trace_dir / "continuation_tables.jsonl"):
        key = (int(row["cell"]), row["public_signal_role"], D(row["effort"]))
        bundle.continuation[key] = row

    for row in _read_jsonl(trace_dir / "potential_reports.jsonl"):
        effort = D(row["effort"])
        key = (int(row["slot"]), row["task_id"], row["worker_id"], effort)
        bundle.potential[key] = PotentialRecord(
            slot=int(row["slot"]),
            task_id=row["task_id"],
            worker_id=row["worker_id"],
            effort=effort,
            report=D(row["report"]),
            score=D(row["score"]),
            screen_status=row["screen_status"],
            v_ijt=D(row["v_ijt"]),
        )

    if load_contracts_flag:
        load_contracts(bundle, load_gammas=load_gammas)

    if trace_hashes is not None:
        bundle.trace_hash = trace_hashes.get("seed_trace_hashes", {}).get(str(seed), "")
    return bundle


def reference_budget(trace: TraceBundle, gamma_max: Decimal = Decimal("1")) -> Decimal:
    """Reference budget = the maximum spend envelope of the online mechanism.

    Per task: K winners at the public base cap plus the gamma_max score escrow.
    (The old formula also added the raw task value V, which inflated the budget
    so far beyond any feasible spend that the budget constraint and the dual
    controller never bound; budget ratios were meaningless.)
    """
    total = Decimal("0")
    public_cap = TraceConfig().public_base_cap
    for task in trace.tasks_by_id.values():
        escrow = gamma_max * task.value
        total += Decimal(task.capacity) * public_cap + escrow
    return total
