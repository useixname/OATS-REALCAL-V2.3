from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_formal_runtime_archive_is_machine_readable_and_scoped() -> None:
    record = json.loads(
        (ROOT / "environment" / "formal_runtime_environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["schema_version"] == "oats-formal-runtime-archive-1.0.0"
    assert record["formal_run"]["completed_cells"] == 830
    assert record["formal_run"]["invalid_cells"] == 0
    assert record["formal_run"]["workers"] == 25
    assert record["hardware"]["memory"]["cgroup_limit_gib"] == 90
    assert record["hardware"]["gpu"]["used_by_formal_path"] is False
    assert record["software"]["formal_path_direct_third_party_packages"] == {
        "numpy": "1.26.4",
        "scipy": "1.13.1",
    }
    assert record["not_recoverable_after_host_deallocation"]


def test_formal_runtime_lock_matches_archived_versions() -> None:
    pins = {
        line.strip()
        for line in (
            ROOT / "environment" / "formal-requirements-lock.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert pins == {"numpy==1.26.4", "scipy==1.13.1"}
