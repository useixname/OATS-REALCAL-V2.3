from __future__ import annotations

from decimal import Decimal

from src.oats_v2.contracts import ContinuationTable, TaskContract
from src.oats_v2.ledger import LedgerState
from src.oats_v2.settlement import settle_base_for_status
from src.oats_v2.types import MechanismStatus


def _contract() -> TaskContract:
    return TaskContract(
        task_id="t",
        effort_levels=("low", "high"),
        effort_basis={"low": Decimal("1"), "high": Decimal("2")},
        continuation=ContinuationTable(
            {"low": Decimal("0.2"), "high": Decimal("0.9")},
            precision=Decimal("0.01"),
            version="gbar-v1",
        ),
        base_cap=Decimal("5"),
        money_grid=Decimal("1"),
        score_cap=Decimal("2"),
        missing_score=Decimal("0.2"),
    )


def test_truthful_compliant_winners_have_nonnegative_interim_utility() -> None:
    contract = _contract()
    checked = 0
    for scalar_cost in (Decimal("0.5"), Decimal("1"), Decimal("1.5"), Decimal("2"), Decimal("3")):
        d, _ = contract.effective_cost(scalar_cost)
        truthful_bid = contract.truthful_bid(scalar_cost)
        for critical_base in contract.grid_values():
            if truthful_bid <= critical_base:
                checked += 1
                assert critical_base - d >= 0
    assert checked > 0


def test_failure_branch_base_settlement_is_worker_protective_and_idempotent() -> None:
    pay_statuses = (
        MechanismStatus.SCREEN_PASS,
        MechanismStatus.SCREEN_FAIL_COMPLIANT,
        MechanismStatus.COLD_START,
        MechanismStatus.PROTOCOL_FAULT,
    )
    for index, status in enumerate(pay_statuses):
        ledger = LedgerState(Decimal("10"))
        contract_id = f"c{index}"
        ledger.lock_bases_atomic({contract_id: Decimal("3")})
        first = settle_base_for_status(ledger, contract_id, status)
        second = settle_base_for_status(ledger, contract_id, status)
        assert first == second == {"amount": "3", "kind": "release"}
        assert ledger.paid == Decimal("3")

    ledger = LedgerState(Decimal("10"))
    ledger.lock_bases_atomic({"invalid": Decimal("3")})
    receipt = settle_base_for_status(ledger, "invalid", MechanismStatus.WORKER_NONCOMPLIANT)
    assert receipt == {"amount": "3", "kind": "return"}
    assert ledger.paid == 0


def test_platform_cancellation_timing_table() -> None:
    before = LedgerState(Decimal("10"))
    before.lock_bases_atomic({"before": Decimal("3")})
    before_receipt = settle_base_for_status(
        before,
        "before",
        MechanismStatus.PROTOCOL_FAULT,
        cancelled_before_effort=True,
    )
    assert before_receipt["kind"] == "return"

    after = LedgerState(Decimal("10"))
    after.lock_bases_atomic({"after": Decimal("3")})
    after_receipt = settle_base_for_status(
        after,
        "after",
        MechanismStatus.PROTOCOL_FAULT,
        cancelled_before_effort=False,
    )
    assert after_receipt["kind"] == "release"


def test_rounding_and_calibration_bounds_are_not_ex_post_ir_claims() -> None:
    # Ceiling makes the contract-relative interim bound exact.
    d_published = Decimal("1.0004")
    critical_grid_base = Decimal("2")
    assert critical_grid_base - d_published >= 0

    # A certified continuation calibration error epsilon weakens the lower
    # bound to -epsilon; it does not imply realized/ex-post nonnegativity.
    epsilon = Decimal("0.05")
    d_true = d_published + epsilon
    assert critical_grid_base - d_true >= -epsilon

    realized_effort_cost = Decimal("3")
    realized_score = Decimal("0")
    assert critical_grid_base + realized_score - realized_effort_cost < 0
