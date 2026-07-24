from __future__ import annotations

from decimal import Decimal
import json

import pytest

from src.oats_v2.events import EventLog
from src.oats_v2.ledger import LedgerState
from src.oats_v2.settlement import failure_disposition
from src.oats_v2.shadow_envelope import ShadowEnvelopeState
from src.oats_v2.types import MechanismStatus


def _ledger_trace() -> str:
    ledger = LedgerState(Decimal("10"))
    ledger.activate_task("1|t", Decimal("2"))
    ledger.lock_bases_atomic({"w|t": Decimal("3")})
    ledger.settle_base("w|t", pay=True)
    ledger.close_task("1|t", {"w|t": Decimal("1")})
    return ledger.event_log.jsonl()


def test_every_money_transition_emits_deterministic_jsonl() -> None:
    first = _ledger_trace()
    second = _ledger_trace()
    assert first == second
    assert len(first.splitlines()) == 4
    assert "raw_report" not in first and "secret_bounds" not in first
    required = {
        "event_id", "slot", "task", "worker", "contract_version", "anchor_version",
        "pre_hash", "post_hash", "delta_free", "delta_locked_base",
        "delta_locked_score", "delta_paid", "delta_shadow_free",
        "delta_shadow_consumed", "status",
    }
    for line in first.splitlines():
        assert required <= set(json.loads(line))


def test_public_log_rejects_secret_fields() -> None:
    log = EventLog()
    with pytest.raises(ValueError):
        log.emit("BAD", {"raw_report": "secret"})
    with pytest.raises(ValueError):
        log.emit("BAD", {"lower_bound": "secret"})


def test_ledger_and_shadow_are_distinct_conserved_accounts() -> None:
    ledger = LedgerState(Decimal("10"))
    shadow = ShadowEnvelopeState(Decimal("10"))
    ledger.activate_task("1|t", Decimal("2"))
    shadow.hold_task("1|t", Decimal("2"))
    ledger.lock_bases_atomic({"w|t": Decimal("1")})
    shadow.commit_admission(("1|t",), {"w|t": Decimal("4")})
    assert ledger.identity() == Decimal("10")
    assert shadow.identity() == Decimal("10")
    assert ledger.locked_base == Decimal("1")
    assert shadow.committed_bases["w|t"] == Decimal("4")


def test_section16_failure_states_have_explicit_dispositions() -> None:
    required = (
        MechanismStatus.TASK_NOT_ACTIVATED,
        MechanismStatus.MULTI_TASK_TYPE_UNSUPPORTED,
        MechanismStatus.CONTINUATION_TABLE_INVALID,
        MechanismStatus.TYPE_MODEL_UNSUPPORTED,
        MechanismStatus.MONOTONICITY_VIOLATION,
        MechanismStatus.CAP_INVARIANT_BROKEN,
        MechanismStatus.WORKER_NONCOMPLIANT,
        MechanismStatus.SCREEN_FAIL_COMPLIANT,
        MechanismStatus.COLD_START,
        MechanismStatus.PROTOCOL_FAULT,
        MechanismStatus.MISSING_OUTCOME,
        MechanismStatus.ENDOGENOUS_OUTCOME,
        MechanismStatus.LEDGER_FATAL,
    )
    for status in required:
        disposition = failure_disposition(status, effort_started=True)
        assert disposition.status is status
        assert disposition.base_action and disposition.task_action and disposition.data_trust_action
    assert failure_disposition(MechanismStatus.PROTOCOL_FAULT, effort_started=True).base_action == "RELEASE"
    assert failure_disposition(MechanismStatus.PROTOCOL_FAULT, effort_started=False).base_action == "RETURN"
