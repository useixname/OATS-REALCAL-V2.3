from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from src.oats_v2.admission import admit_atomic
from src.oats_v2.allocation import allocate
from src.oats_v2.anchor_registry import (
    AnchorObservation,
    AnchorPolicy,
    HistoricalAnchorRegistry,
)
from src.oats_v2.contracts import ContinuationTable, TaskContract
from src.oats_v2.ideal_range_screen import IdealRangeScreen
from src.oats_v2.ledger import LedgerState
from src.oats_v2.outcome import OutcomeProvenance, gross_value
from src.oats_v2.shadow_envelope import ShadowEnvelopeState
from src.oats_v2.trust import TrustState, clipped_adaptive_alpha
from src.oats_v2.types import (
    AllocationSnapshot,
    Candidate,
    MechanismStatus,
    SelectionOutcome,
)


def _snapshot(cap: str = "4") -> AllocationSnapshot:
    return AllocationSnapshot(
        active_tasks=frozenset({"t"}),
        task_capacities={"t": 2},
        actual_base_capacity=Decimal(cap),
        shadow_base_capacity=Decimal(cap),
        epsilon_rank=Decimal("0.1"),
    )


def test_duplicate_bonus_reserve_is_one_task_escrow() -> None:
    ledger = LedgerState(Decimal("10"))
    assert ledger.activate_task("1|t", Decimal("3"))
    assert not ledger.activate_task("1|t", Decimal("3"))
    assert ledger.locked_score == Decimal("3")
    first = ledger.close_task("1|t", {"w": Decimal("1")})
    second = ledger.close_task("1|t", {"w": Decimal("1")})
    assert first == second
    assert ledger.identity() == ledger.budget


def test_critical_payment_cannot_break_public_or_actual_cap() -> None:
    candidates = (Candidate("w", "t", Decimal("0"), Decimal("2"), Decimal("1")),)
    allocation = allocate(candidates, _snapshot("2"))
    forged = SelectionOutcome(allocation, {"w|t": Decimal("3")}, MechanismStatus.ACTIVE)
    ledger = LedgerState(Decimal("5"))
    ledger.activate_task("1|t", Decimal("1"))
    shadow = ShadowEnvelopeState(Decimal("5"))
    shadow.hold_task("1|t", Decimal("1"))
    before = ledger.snapshot()
    status = admit_atomic(forged, candidates, {"t": "1|t"}, ledger, shadow)
    assert status is MechanismStatus.CAP_INVARIANT_BROKEN
    assert ledger.snapshot() == before


def test_lower_bid_does_not_flip_final_allocation_to_loss() -> None:
    candidates = (
        Candidate("w0", "t", Decimal("2"), Decimal("2"), Decimal("2")),
        Candidate("w1", "t", Decimal("1"), Decimal("2"), Decimal("1")),
    )
    original = allocate(candidates, _snapshot("2"))
    assert original.won("w0|t")
    lowered = (replace(candidates[0], bid=Decimal("0")), candidates[1])
    assert allocate(lowered, _snapshot("2")).won("w0|t")


def test_total_utility_underbid_is_unprofitable() -> None:
    contract = TaskContract(
        "t",
        ("e",),
        {"e": Decimal("1")},
        ContinuationTable({"e": Decimal("2")}, Decimal("1"), "v1"),
        Decimal("10"),
        Decimal("1"),
        Decimal("2"),
    )
    d, _ = contract.effective_cost(Decimal("6"))
    assert d == Decimal("4")
    threshold = Decimal("3")
    assert contract.truthful_bid(Decimal("6")) > threshold
    assert threshold - d < 0


def test_alpha_out_of_range_is_rejected_and_diagnostic_variant_clips() -> None:
    trust = TrustState()
    trust.initialize("w", Decimal("0.8"))
    with pytest.raises(ValueError):
        trust.update("w", "o", Decimal("0"), Decimal("1.2"), available=True, independent=True)
    assert clipped_adaptive_alpha(Decimal("1.2"), Decimal("0.1"), Decimal("1")) == Decimal("1")


def _anchor(value: str, weight: str = "1") -> AnchorObservation:
    return AnchorObservation("a", Decimal(value), Decimal(weight), True, True, True)


def test_mad_near_zero_uses_registered_floor() -> None:
    registry = HistoricalAnchorRegistry(AnchorPolicy(3, Decimal("0.01"), Decimal("3")))
    registry.register("v1", (_anchor("10"), _anchor("10"), _anchor("10")))
    snapshot = registry.snapshot("v1")
    assert snapshot is not None
    assert snapshot.sigma == Decimal("0.01")
    assert IdealRangeScreen().compare(
        "x", Decimal("10.0001"), snapshot, cold_start_authorized=False
    ) is MechanismStatus.SCREEN_PASS


def test_cold_start_is_explicit_and_quota_controlled() -> None:
    screen = IdealRangeScreen()
    assert screen.compare("allowed", Decimal("1"), None, cold_start_authorized=True) is MechanismStatus.COLD_START
    assert screen.compare("denied", Decimal("1"), None, cold_start_authorized=False) is MechanismStatus.PROTOCOL_ERROR


def test_colluding_anchor_majority_remains_unresolved() -> None:
    registry = HistoricalAnchorRegistry(AnchorPolicy(3, Decimal("0.1"), Decimal("2")))
    registry.register(
        "collusion",
        (_anchor("100", "4"), _anchor("100", "4"), _anchor("0", "1")),
    )
    snapshot = registry.snapshot("collusion")
    assert snapshot is not None and snapshot.center == Decimal("100")
    screen = IdealRangeScreen()
    assert screen.compare("malicious", Decimal("100"), snapshot, cold_start_authorized=False) is MechanismStatus.SCREEN_PASS
    assert screen.compare("honest", Decimal("0"), snapshot, cold_start_authorized=False) is MechanismStatus.SCREEN_FAIL_COMPLIANT


def test_endogenous_outcome_cannot_create_value_or_trust() -> None:
    provenance = OutcomeProvenance("p", "t", "c", "now", "consumer-action", "hash", False)
    assert provenance.validate() is MechanismStatus.ENDOGENOUS_OUTCOME
    assert gross_value(
        purchased=True,
        independent_outcome=False,
        task_value=Decimal("10"),
        task_capacity=1,
        quality=Decimal("1"),
    ) == 0
    trust = TrustState()
    trust.initialize("w", Decimal("0.5"))
    assert trust.update("w", "o", Decimal("1"), Decimal("0.2"), available=True, independent=False) == Decimal("0.5")


def test_time_varying_lambda_weighted_term_witness_is_preserved() -> None:
    gradients = (Decimal("1"), Decimal("-1"))
    lambdas = (Decimal("10"), Decimal("1"))
    assert sum(gradients) == 0
    assert sum((lam * gradient for lam, gradient in zip(lambdas, gradients)), Decimal("0")) == Decimal("9")
