from __future__ import annotations

from decimal import Decimal

from src.oats_v2.oracle.exact_dp import solve_exact_dp
from src.oats_v2.oracle.exact_milp import solve_exact_milp_reference
from src.oats_v2.oracle.fixtures import activation_complementarity, hand_computed_basic, ratio_gap
from src.oats_v2.oracle.greedy_legacy import solve_legacy_density
from src.oats_v2.oracle.problem import LocalContract, LocalOracleProblem, LocalTask


def test_hand_computed_fixture_matches_both_exact_solvers() -> None:
    problem, expected_solution, expected_objective = hand_computed_basic()
    dp = solve_exact_dp(problem)
    milp = solve_exact_milp_reference(problem)
    assert dp.solution == milp.solution == expected_solution
    assert dp.objective == milp.objective == expected_objective
    assert dp.is_exact and milp.is_exact
    assert dp.certified_gap == milp.certified_gap == 0


def test_legacy_density_is_refuted_on_ratio_and_fixed_cost_fixtures() -> None:
    ratio = ratio_gap(100)
    assert solve_legacy_density(ratio).solution == ("small",)
    assert solve_exact_dp(ratio).solution == ("large",)
    activation = activation_complementarity()
    assert solve_legacy_density(activation).objective == Decimal("4")
    assert solve_exact_dp(activation).objective == Decimal("15")


def test_link_worker_task_deadline_domain_and_shadow_constraints() -> None:
    problem = LocalOracleProblem(
        tasks=(LocalTask("t", Decimal("2"), 1), LocalTask("closed", Decimal("0"), 1, active=False)),
        contracts=(
            LocalContract("a", "w", "t", Decimal("5"), Decimal("2"), Decimal("1")),
            LocalContract("b", "w2", "t", Decimal("4"), Decimal("2"), Decimal("1")),
            LocalContract("late", "w3", "t", Decimal("100"), Decimal("0"), Decimal("1"), deadline_ok=False),
            LocalContract("domain", "w4", "t", Decimal("100"), Decimal("0"), Decimal("1"), domain_ok=False),
            LocalContract("inactive", "w5", "closed", Decimal("100"), Decimal("0"), Decimal("1")),
        ),
        remaining_shadow_capacity=Decimal("4"),
        dual_lambda=Decimal("0"),
    )
    result = solve_exact_dp(problem)
    assert result.solution == ("a",)
    assert result.active_tasks == ("t",)
    assert result.envelope_used == Decimal("4")


def test_exact_dp_and_enumerated_milp_agree_over_small_domain() -> None:
    checked = 0
    for budget in (Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")):
        for activation in (Decimal("0"), Decimal("1")):
            for dual in (Decimal("0"), Decimal("0.5")):
                problem = LocalOracleProblem(
                    tasks=(LocalTask("t", activation, 2),),
                    contracts=(
                        LocalContract("a", "wa", "t", Decimal("3"), Decimal("1"), Decimal("1")),
                        LocalContract("b", "wb", "t", Decimal("2"), Decimal("2"), Decimal("1")),
                        LocalContract("c", "wc", "t", Decimal("1"), Decimal("1"), Decimal("0.5")),
                    ),
                    remaining_shadow_capacity=budget,
                    dual_lambda=dual,
                )
                dp = solve_exact_dp(problem)
                milp = solve_exact_milp_reference(problem)
                assert dp.objective == milp.objective
                assert dp.solution == milp.solution
                checked += 1
    assert checked == 16
