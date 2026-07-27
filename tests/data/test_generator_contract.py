from __future__ import annotations

import json
from decimal import Decimal

from src.oats_v2.data.anchor_history_generator import generate_anchor_history
from src.oats_v2.data.eligibility_generator import generate_eligibility
from src.oats_v2.data.holdout_generator import generate_holdout_provenance
from src.oats_v2.data.potential_report_generator import generate_potential_reports
from src.oats_v2.data.profiles import deployment_profiles
from src.oats_v2.data.provenance import validate_holdout_provenance
from src.oats_v2.data.schemas import EFFORTS, TraceConfig, canonical_json
from src.oats_v2.data.task_generator import generate_tasks
from src.oats_v2.data.worker_generator import generate_workers


def _tiny() -> TraceConfig:
    return TraceConfig(
        horizon=8,
        worker_count=20,
        cell_count=4,
        anchor_count_per_cell=10,
        calibration_n=50,
    )


def test_potential_outcomes_are_complete_and_method_independent() -> None:
    seed = 20260715
    config = _tiny()
    workers = generate_workers(seed, config)
    tasks = generate_tasks(seed, config)
    anchors, bounds = generate_anchor_history(seed, config)
    eligibility = list(generate_eligibility(seed, config, workers, tasks))
    available = [row for row in eligibility if row["available"] and row["mapped_task_id"]]
    potential = list(generate_potential_reports(seed, workers, tasks, available, bounds))
    assert len(potential) == len(available) * len(EFFORTS)
    assert all("method" not in row and "gamma" not in row for row in potential)
    assert len({(row["slot"], row["worker_id"]) for row in available}) == len(available)
    assert len(anchors) == config.cell_count * config.anchor_count_per_cell


def test_external_holdout_is_pregenerated_and_forbidden_inputs_absent() -> None:
    config = _tiny()
    tasks = generate_tasks(20260715, config)
    provenance = list(generate_holdout_provenance(20260715, tasks))
    assert len(provenance) == len(tasks)
    assert all(not validate_holdout_provenance(row) for row in provenance)
    serialized = "\n".join(canonical_json(row, schema_name="holdout_provenance") for row in provenance)
    assert "SYNTHETIC_PREGENERATED_HOLDOUT" in serialized
    declaration = provenance[0]["input_dependency_declaration"]
    assert set(declaration["does_not_depend_on"]) >= {
        "selected_report", "gamma", "method", "selection", "report_delivery", "consumer_action"
    }


def test_online_contract_projection_excludes_generator_secrets() -> None:
    forbidden = {"theta", "stratum", "bias_sign", "potential_reports", "z"}
    online_contract_fields = {
        "seed", "method_id", "gamma", "slot", "task_id", "worker_id",
        "public_signal_role", "sbar", "Gbar_by_effort", "vhat",
        "Delta_money", "epsilon_rank", "contract_hash",
    }
    assert not forbidden.intersection(online_contract_fields)


def test_ic_and_robustness_populations_are_not_conflated() -> None:
    profiles = deployment_profiles()
    managed = profiles["MANAGED_FLEET_SYNTHETIC_PROFILE"]
    unsupported = profiles["OPEN_BYOD_UNSUPPORTED_PROFILE"]
    assert managed["ic_population"] == ["honest", "low-quality"]
    assert "malicious" in managed["excluded_byzantine_behavior"]
    assert unsupported["supported"] is False
