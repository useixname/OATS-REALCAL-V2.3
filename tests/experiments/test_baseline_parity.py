from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.oats_v2.experiments.method_registry import METHOD_REGISTRY
from src.oats_v2.experiments.run_matrix import COMPARISON_METHODS


def test_all_comparison_methods_registered() -> None:
    for method_id in COMPARISON_METHODS:
        assert method_id in METHOD_REGISTRY


def test_baseline_parity_artifact_exists_after_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "audit_results" / "baseline_parity.json"
    if not path.exists():
        pytest.skip("baseline parity audit not yet written")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["overall_status"] == "PASS"
