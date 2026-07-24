from __future__ import annotations

import itertools
from decimal import Decimal

from src.oats_v2.contracts import ContinuationTable, TaskContract, best_response


GRID = Decimal("1")
EPS = Decimal("0.05")


def _contract(
    effort_basis: dict[str, Decimal], continuation: dict[str, Decimal], cap: Decimal = Decimal("8")
) -> TaskContract:
    return TaskContract(
        task_id="t",
        effort_levels=tuple(effort_basis),
        effort_basis=effort_basis,
        continuation=ContinuationTable(continuation, Decimal("0.01"), "gbar-v1"),
        base_cap=cap,
        money_grid=GRID,
        score_cap=Decimal("2"),
    )


def _utility(report: Decimal, threshold: Decimal, true_effective_cost: Decimal) -> Decimal:
    return threshold - true_effective_cost if report <= threshold else Decimal("0")


def test_exact_grid_truthfulness_for_all_finite_instances() -> None:
    contracts = (
        _contract(
            {"low": Decimal("1"), "high": Decimal("2")},
            {"low": Decimal("0"), "high": Decimal("1")},
        ),
        # At c=1 both efforts are best responses, exercising the tie branch.
        _contract(
            {"low": Decimal("1"), "high": Decimal("2")},
            {"low": Decimal("0"), "high": Decimal("1")},
        ),
        _contract(
            {"e0": Decimal("1"), "e1": Decimal("1.5"), "e2": Decimal("2")},
            {"e0": Decimal("0"), "e1": Decimal("0.4"), "e2": Decimal("0.8")},
        ),
    )
    scalar_costs = tuple(Decimal(index) / Decimal("2") for index in range(1, 9))
    checked = 0
    saw_boundary = False
    saw_multiple_best_responses = False
    for contract, scalar_cost in itertools.product(contracts, scalar_costs):
        continuous_d, best_efforts = contract.effective_cost(scalar_cost)
        truthful = contract.truthful_bid(scalar_cost)
        saw_boundary |= continuous_d % GRID == 0
        saw_multiple_best_responses |= len(best_efforts) > 1
        for threshold, report in itertools.product(contract.grid_values(), repeat=2):
            truthful_utility = _utility(truthful, threshold, continuous_d)
            deviation_utility = _utility(report, threshold, continuous_d)
            checked += 1
            assert truthful_utility >= deviation_utility, {
                "true_d": str(continuous_d),
                "truthful_grid_bid": str(truthful),
                "reported_grid_bid": str(report),
                "allocation": report <= threshold,
                "expected_score": str(contract.continuation.values[best_efforts[0]]),
                "base": str(threshold),
                "effort_best_responses": best_efforts,
                "expected_utility": str(deviation_utility),
            }
    assert checked > 1_000
    assert saw_boundary
    assert saw_multiple_best_responses


def test_best_response_realizes_effective_cost_even_with_ties() -> None:
    effort = ("low", "high")
    basis = {"low": Decimal("1"), "high": Decimal("2")}
    continuation = {"low": Decimal("0"), "high": Decimal("1")}
    d, best = best_response(Decimal("1"), effort, basis, continuation)
    assert d == Decimal("1")
    assert best == ("low", "high")
    for choice in best:
        assert Decimal("1") * basis[choice] - continuation[choice] == d


def test_calibration_error_gives_epsilon_ic_not_exact_ic() -> None:
    contract = _contract(
        {"low": Decimal("1"), "high": Decimal("2")},
        {"low": Decimal("0.2"), "high": Decimal("1.0")},
    )
    true_continuation_variants = (
        {"low": Decimal("0.15"), "high": Decimal("0.95")},
        {"low": Decimal("0.25"), "high": Decimal("1.05")},
        {"low": Decimal("0.15"), "high": Decimal("1.05")},
        {"low": Decimal("0.25"), "high": Decimal("0.95")},
    )
    witnessed_strict_break = False
    for scalar_cost in (Decimal("0.5"), Decimal("1"), Decimal("1.5"), Decimal("2")):
        published_d, _ = contract.effective_cost(scalar_cost)
        truthful = contract.truthful_bid(scalar_cost)
        for true_continuation in true_continuation_variants:
            true_d, _ = best_response(
                scalar_cost,
                contract.effort_levels,
                contract.effort_basis,
                true_continuation,
            )
            assert abs(true_d - published_d) <= EPS
            for threshold in contract.grid_values():
                truthful_u = _utility(truthful, threshold, true_d)
                best_deviation = max(
                    _utility(report, threshold, true_d) for report in contract.grid_values()
                )
                regret = best_deviation - truthful_u
                assert regret <= EPS
                witnessed_strict_break |= regret > 0
    assert witnessed_strict_break, "calibration error should not be mislabeled exact IC"


def test_screen_fail_and_missing_outcome_do_not_change_bid_threshold() -> None:
    # Gbar already integrates PASS/FAIL/missing branches.  Once signed, neither
    # realized branch may alter the critical base or revoke compliant base pay.
    contract = _contract(
        {"low": Decimal("1"), "high": Decimal("2")},
        {"low": Decimal("0.2"), "high": Decimal("0.9")},
    )
    d, _ = contract.effective_cost(Decimal("1"))
    bid = contract.truthful_bid(Decimal("1"))
    threshold = Decimal("2")
    assert bid <= threshold
    assert threshold - d >= 0
    assert _utility(bid, threshold, d) == threshold - d
