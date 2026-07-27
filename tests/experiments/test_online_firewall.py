from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.oats_v2.experiments.audit import run_baseline_parity_audit, run_online_firewall_audit
from src.oats_v2.experiments.formal_runner import (
    FormalRunner,
    _build_seed_gamma_jobs,
    _build_seed_gamma_lp_jobs,
    simulate_cell,
)
from src.oats_v2.experiments.lp_comparator import LPComparatorCache
from src.oats_v2.experiments.online_projection import (
    OnlineFirewallViolation,
    validate_online_payload,
)
from src.oats_v2.experiments.run_matrix import (
    build_run_matrix,
    build_run_matrix_v2,
    count_run_cells,
)
from src.oats_v2.experiments.trace_loader import load_trace


ROOT = Path(__file__).resolve().parents[2]


def test_online_firewall_fail_closed_on_forbidden_fields() -> None:
    with pytest.raises(OnlineFirewallViolation):
        validate_online_payload({"theta": "0.5"})
    with pytest.raises(OnlineFirewallViolation):
        validate_online_payload({"potential_reports": []})


def test_online_firewall_audit_passes() -> None:
    report = run_online_firewall_audit()
    assert report["status"] == "PASS"
    assert report["ORACLE_ROUTE"] == "R4"
    assert report["NO_REGRET_GUARANTEE"] is True


def test_baseline_parity_audit_passes() -> None:
    report = run_baseline_parity_audit()
    assert report["overall_status"] == "PASS"
    for method in ("V2-FULL", "B-P1", "B-NOSCREEN", "B-NOTRUST", "B-NODUAL", "B-MYOPIC"):
        assert report["methods"][method]["status"] == "PASS"


def test_run_matrix_frozen_counts() -> None:
    counts = count_run_cells()
    assert counts["total"] == 6840
    assert counts["F1_OVERALL"] == 3240
    assert counts["F2_EFFORT_OFF"] == 180


def test_run_matrix_cell_ids_are_globally_unique_and_family_scoped() -> None:
    matrix = build_run_matrix()
    cell_ids = [cell.cell_id for cell in matrix]
    assert len(cell_ids) == len(set(cell_ids)) == 6840
    assert all(cell.cell_id.startswith(f"f{cell.family}_") for cell in matrix)


def test_seed_gamma_partition_is_complete_disjoint_and_parallelizable() -> None:
    matrix = build_run_matrix()
    realcal = [cell for cell in matrix if 20260715 <= cell.seed <= 20260724]
    jobs = _build_seed_gamma_jobs(realcal, ROOT / "results" / "partition-test")
    assert len(jobs) == 60
    assert all(len({(cell["seed"], cell["gamma"]) for cell in job["cells"]}) == 1 for job in jobs)
    partitioned_ids = [cell["cell_id"] for job in jobs for cell in job["cells"]]
    assert len(partitioned_ids) == len(set(partitioned_ids)) == 2280
    assert set(partitioned_ids) == {cell.cell_id for cell in realcal}


def test_v2_seed_gamma_chunks_are_balanced_complete_and_disjoint() -> None:
    matrix = build_run_matrix_v2()
    assert len(matrix) == 830  # V2.1 frozen matrix (750 + 80 E1B_TIGHT)
    jobs = _build_seed_gamma_jobs(
        matrix,
        ROOT / "results" / "partition-v2-test",
        max_cells_per_job=5,
    )
    assert max(len(job["cells"]) for job in jobs) <= 5
    assert all(len({(cell["seed"], cell["gamma"]) for cell in job["cells"]}) == 1 for job in jobs)
    partitioned_ids = [cell["cell_id"] for job in jobs for cell in job["cells"]]
    assert len(partitioned_ids) == len(set(partitioned_ids)) == len(matrix)
    assert set(partitioned_ids) == {cell.cell_id for cell in matrix}


def test_v2_lp_chunks_keep_cache_keys_whole_and_parallelizable() -> None:
    matrix = build_run_matrix_v2()
    jobs = _build_seed_gamma_lp_jobs(
        matrix,
        ROOT / "results" / "partition-v2-lp-test",
        max_cache_keys_per_job=8,
    )
    partitioned_ids = [cell["cell_id"] for job in jobs for cell in job["cells"]]
    assert len(partitioned_ids) == len(set(partitioned_ids)) == len(matrix)
    assert set(partitioned_ids) == {cell.cell_id for cell in matrix}

    key_to_jobs: dict[tuple, set[int]] = {}
    for job_index, job in enumerate(jobs):
        for payload in job["cells"]:
            key = (
                payload["seed"],
                payload["gamma"],
                payload["budget_ratio"],
                payload["arrival_multiplier"],
                payload["delay"],
                payload["missing_prob"],
            )
            key_to_jobs.setdefault(key, set()).add(job_index)
    assert all(len(job_indexes) == 1 for job_indexes in key_to_jobs.values())


def test_dev_seed_single_cell_simulation() -> None:
    if not (ROOT / "trace_hashes.json").exists() or not (
        ROOT / "data" / "SYN-V2-1" / "20260715"
    ).exists():
        pytest.skip(
            "large development trace is not included in the public code release"
        )
    trace_hashes = json.loads((ROOT / "trace_hashes.json").read_text(encoding="utf-8"))
    trace = load_trace(20260715, ROOT / "data" / "SYN-V2-1", trace_hashes)
    cell = next(
        c
        for c in build_run_matrix()
        if c.seed == 20260715
        and c.method_id == "V2-FULL"
        and c.gamma == Decimal("0.5")
        and c.budget_ratio == Decimal("0.25")
    )
    result = simulate_cell(cell, trace, LPComparatorCache(), compute_lp=True)
    assert result.invariant_status == "PASS"
    assert "IDEAL_S2" in "".join(result.labels)
    assert result.lp.status in ("optimal", "FAILED_")
