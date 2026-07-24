from __future__ import annotations

from decimal import Decimal

from src.oats_v2.admission import admit_atomic
from src.oats_v2.anchor_registry import (
    AnchorObservation,
    AnchorPolicy,
    HistoricalAnchorRegistry,
)
from src.oats_v2.critical_payment import build_selection
from src.oats_v2.ideal_range_screen import IdealRangeScreen
from src.oats_v2.invariants import assert_all
from src.oats_v2.ledger import LedgerState
from src.oats_v2.settlement import settle_base_for_status, settle_task_scores
from src.oats_v2.shadow_envelope import ShadowEnvelopeState
from src.oats_v2.task_activation import activate_task_atomic
from src.oats_v2.trust import TrustState
from src.oats_v2.types import AllocationSnapshot, Candidate, MechanismStatus, Task
from src.oats_v2.worker_action import WorkerSubmission, validate_submission


def test_one_slot_pass_path_is_executable_and_conserved() -> None:
    ledger = LedgerState(Decimal("10"))
    shadow = ShadowEnvelopeState(Decimal("10"))
    task = Task("t", capacity=1, score_escrow=Decimal("2"))
    activation = activate_task_atomic(task, 1, ledger, shadow)
    assert activation.status is MechanismStatus.ACTIVE

    candidate = Candidate("w", "t", Decimal("1"), Decimal("3"), Decimal("5"))
    snapshot = AllocationSnapshot(
        active_tasks=frozenset({"t"}),
        task_capacities={"t": 1},
        actual_base_capacity=ledger.free,
        shadow_base_capacity=shadow.free,
        epsilon_rank=Decimal("0.1"),
    )
    selection = build_selection((candidate,), snapshot, Decimal("1"))
    assert selection.status is MechanismStatus.ACTIVE
    assert selection.critical_bases == {"w|t": Decimal("3")}
    assert admit_atomic(
        selection,
        (candidate,),
        {"t": activation.task_key},
        ledger,
        shadow,
    ) is MechanismStatus.ACTIVE

    submission = WorkerSubmission(
        "w|t",
        Decimal("10.01"),
        "nonce",
        "commitment",
        True,
        True,
        True,
        "a-v1",
    )
    assert validate_submission(submission) is MechanismStatus.ACTIVE
    anchors = HistoricalAnchorRegistry(AnchorPolicy(3, Decimal("0.1"), Decimal("2")))
    anchors.register(
        "a-v1",
        tuple(
            AnchorObservation(f"a{index}", Decimal("10"), Decimal("1"), True, True, True)
            for index in range(3)
        ),
    )
    screen_status = IdealRangeScreen().compare(
        "transcript", submission.report, anchors.snapshot("a-v1"), cold_start_authorized=False
    )
    assert screen_status is MechanismStatus.SCREEN_PASS
    settle_base_for_status(ledger, "w|t", screen_status)
    settle_task_scores(ledger, activation.task_key, {"w|t": Decimal("0.5")}, Decimal("2"))

    trust = TrustState()
    trust.initialize("w", Decimal("0.5"))
    assert trust.update(
        "w", "outcome-1", Decimal("0.5"), Decimal("0.2"), available=True, independent=True
    ) == Decimal("0.50")
    assert_all(ledger, shadow, (candidate,))
    assert ledger.snapshot() == {
        "budget": "10",
        "free": "6.0",
        "locked_base": "0",
        "locked_score": "0",
        "paid": "4.0",
    }


def test_compliant_fail_still_releases_base_and_never_updates_trust() -> None:
    ledger = LedgerState(Decimal("5"))
    ledger.lock_bases_atomic({"w|t": Decimal("2")})
    settle_base_for_status(ledger, "w|t", MechanismStatus.SCREEN_FAIL_COMPLIANT)
    assert ledger.paid == Decimal("2")
    trust = TrustState()
    trust.initialize("w", Decimal("0.5"))
    assert trust.values["w"] == Decimal("0.5")
