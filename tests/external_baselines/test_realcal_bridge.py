from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.oats_external.firewall import validate_slot_view
from src.oats_external.realcal_bridge import (
    CompletionForecast,
    RealCalExternalTrace,
    RealCalBridgeViolation,
    canonical_hash,
    sha256_file,
)
from src.oats_external.types import ExternalSlotView


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_trace(tmp_path: Path) -> tuple[Path, CompletionForecast]:
    data_root = tmp_path / "REAL-CAL-V2"
    seed_root = data_root / "20260740"
    seed_root.mkdir(parents=True)
    workers = [
        {
            "seed": 20260740,
            "worker_id": "w1",
            "stratum": "malicious",
            "public_signal_role": "honest",
            "c_i": "0.5",
            "bias_sign": -1,
        },
        {
            "seed": 20260740,
            "worker_id": "w2",
            "stratum": "honest",
            "public_signal_role": "honest",
            "c_i": "1.0",
            "bias_sign": 0,
        },
    ]
    tasks = [
        {
            "seed": 20260740,
            "slot": 1,
            "task_id": "t1",
            "cell": 7,
            "theta": "0.2",
            "V": "30",
            "K": 2,
            "deadline": 2,
            "z": "0.2",
            "delay_mask": {"0": 0, "5": 5},
            "missing_mask": {"0": False, "0.1": False},
        }
    ]
    eligibility = [
        {
            "seed": 20260740,
            "slot": 1,
            "worker_id": worker,
            "available": True,
            "mapped_task_id": "t1",
            "map_hash": worker,
        }
        for worker in ("w1", "w2")
    ]
    potentials = []
    for worker, score in (("w1", "0.7"), ("w2", "0.9")):
        for effort in ("0", "0.5", "1"):
            potentials.append(
                {
                    "seed": 20260740,
                    "slot": 1,
                    "task_id": "t1",
                    "worker_id": worker,
                    "effort": effort,
                    "report": "0.2",
                    "score": score,
                    "screen_status": "FAIL",
                    "v_ijt": "0",
                }
            )
    _write_jsonl(seed_root / "workers.jsonl", workers)
    _write_jsonl(seed_root / "tasks.jsonl", tasks)
    _write_jsonl(seed_root / "eligibility.jsonl", eligibility)
    _write_jsonl(seed_root / "potential_reports.jsonl", potentials)
    (seed_root / "trace_metadata.json").write_text(
        json.dumps({"dataset_id": "REAL-CAL-V2", "seed": 20260740}),
        encoding="utf-8",
    )
    file_hashes = {
        name: sha256_file(seed_root / name)
        for name in (
            "workers.jsonl",
            "tasks.jsonl",
            "eligibility.jsonl",
            "potential_reports.jsonl",
            "trace_metadata.json",
        )
    }
    (data_root / "trace_hashes_realcal.json").write_text(
        json.dumps(
            {
                "dataset_id": "REAL-CAL-V2",
                "seed_file_hashes": {"20260740": file_hashes},
            }
        ),
        encoding="utf-8",
    )
    model = CompletionForecast(
        model_id="fixture",
        training_seeds=(1,),
        alpha=Decimal("1"),
        scenario_probabilities={
            "delay=0|missing=0": {"1": Decimal("0.8")}
        },
        scenario_global_probabilities={
            "delay=0|missing=0": Decimal("0.8")
        },
        training_trace_hashes={"1": canonical_hash({"tasks.jsonl": "fixture"})},
    )
    return data_root, model


def test_bridge_exposes_only_public_presale_fields(tmp_path: Path) -> None:
    data_root, model = _fixture_trace(tmp_path)
    trace = RealCalExternalTrace(
        data_root=data_root,
        seed=20260740,
        completion_forecast=model,
        delay=0,
        missing_prob="0",
        max_arrivals=2,
    )
    assert tuple(candidate.current_bid for candidate in trace.candidates) == (
        Decimal("0.5"),
        Decimal("1.0"),
    )
    assert all(candidate.completion_probability == Decimal("0.8") for candidate in trace.candidates)
    validate_slot_view(
        ExternalSlotView(
            slot=1,
            trace_seed=20260740,
            remaining_budget=Decimal("10"),
            candidates=trace.candidates,
        )
    )


def test_bridge_outcome_ignores_oats_screen_and_v_ijt(tmp_path: Path) -> None:
    data_root, model = _fixture_trace(tmp_path)
    trace = RealCalExternalTrace(
        data_root=data_root,
        seed=20260740,
        completion_forecast=model,
        delay=0,
        missing_prob="0",
        max_arrivals=2,
    )
    key = trace.candidates[0].key
    feedback = trace.feedback_for(key, revealed_slot=2)
    assert feedback.selected_quality == Decimal("0.7")
    assert feedback.realized_external_value == Decimal("10.5")
    assert trace.completion_status(key) == (True, 1)


def test_selection_only_bridge_does_not_require_outcome_file(
    tmp_path: Path,
) -> None:
    data_root, model = _fixture_trace(tmp_path)
    (data_root / "20260740" / "potential_reports.jsonl").unlink()
    trace = RealCalExternalTrace(
        data_root=data_root,
        seed=20260740,
        completion_forecast=model,
        delay=0,
        missing_prob="0",
        max_arrivals=2,
        load_outcomes=False,
    )
    assert trace.outcomes == {}
    assert "potential_reports.jsonl" not in trace.verified_hashes
    with pytest.raises(RealCalBridgeViolation, match="feedback is disabled"):
        trace.feedback_for(trace.candidates[0].key, revealed_slot=2)
