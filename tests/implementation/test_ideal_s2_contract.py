from __future__ import annotations

import json
from decimal import Decimal

from src.oats_v2.anchor_registry import AnchorSnapshot
from src.oats_v2.contracts import ContinuationTable
from src.oats_v2.ideal_range_screen import IdealRangeScreen, SCREENING_BACKEND_LABEL
from src.oats_v2.settlement import failure_disposition
from src.oats_v2.types import MechanismStatus
from src.oats_v2.worker_action import WorkerSubmission, validate_submission


def _snapshot(version: str) -> AnchorSnapshot:
    return AnchorSnapshot(
        version,
        Decimal("0"),
        Decimal("1"),
        Decimal("-1"),
        Decimal("1"),
        3,
    )


def test_one_query_is_cached_without_distance_leakage() -> None:
    screen = IdealRangeScreen()
    first = screen.compare("tx", Decimal("0"), _snapshot("v1"), cold_start_authorized=False)
    second = screen.compare("tx", Decimal("100"), _snapshot("v1"), cold_start_authorized=False)
    assert first is second is MechanismStatus.SCREEN_PASS
    assert len(screen.event_log.events) == 1
    event = json.loads(screen.event_log.jsonl())
    assert set(event["payload"]) == {"transcript_digest"}
    assert SCREENING_BACKEND_LABEL == "SCREENING_BACKEND=IDEAL_S2; NO_CRYPTOGRAPHIC_SECURITY_CLAIM"


def test_cross_version_replay_fails_closed() -> None:
    screen = IdealRangeScreen()
    assert screen.compare("tx", Decimal("0"), _snapshot("v1"), cold_start_authorized=False) is MechanismStatus.SCREEN_PASS
    assert screen.compare("tx", Decimal("0"), _snapshot("v2"), cold_start_authorized=False) is MechanismStatus.PROTOCOL_ERROR


def test_gbar_is_fixed_public_auxiliary_not_adaptive_precision() -> None:
    table = ContinuationTable(
        {"low": Decimal("0.1234"), "high": Decimal("0.9876")},
        Decimal("0.01"),
        "gbar-v1",
        signed=True,
    )
    assert table.values == {"low": Decimal("0.12"), "high": Decimal("0.99")}
    assert table.precision == Decimal("0.01") and table.version == "gbar-v1"


def test_share_inconsistency_and_platform_fault_are_distinct() -> None:
    invalid_share = WorkerSubmission(
        "c", Decimal("0"), "n", "commit", True, False, True, "v1"
    )
    assert validate_submission(invalid_share) is MechanismStatus.WORKER_NONCOMPLIANT
    assert failure_disposition(MechanismStatus.PROTOCOL_FAULT, effort_started=True).base_action == "RELEASE"
