from __future__ import annotations

from pathlib import Path

from src.oats_v2.data.schemas import TraceConfig
from src.oats_v2.data.trace_validator import validate_trace
from src.oats_v2.data.trace_writer import generate_trace


def _tiny() -> TraceConfig:
    return TraceConfig(
        horizon=10,
        worker_count=20,
        cell_count=3,
        anchor_count_per_cell=10,
        calibration_n=80,
    )


def test_tiny_trace_is_schema_valid_and_byte_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = _tiny()
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_trace(seed=20260715, output_directory=first, root=root, config=config, formal=False)
    generate_trace(seed=20260715, output_directory=second, root=root, config=config, formal=False)
    report_a = validate_trace(first, config)
    report_b = validate_trace(second, config)
    assert report_a["status"] == report_b["status"] == "PASS"
    assert report_a["file_sha256"] == report_b["file_sha256"]
    assert report_a["checks"] == {
        "schema_and_field_order": True,
        "one_task_mapping": True,
        "holdout_independence": True,
        "potential_outcome_completeness": True,
        "no_online_secret_fields": True,
    }
