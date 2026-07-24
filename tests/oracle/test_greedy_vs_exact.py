from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from src.oats_v2.oracle import (
    OracleContract,
    density_greedy_oracle,
    exact_integral_oracle,
)


OUTPUT = Path("audit_results/r3_gap_instances.json")
SEED = 20260715


def _ratio_instance(scale: int) -> tuple[tuple[OracleContract, ...], dict[str, Decimal], dict[str, int], Decimal]:
    amount = Decimal(scale)
    contracts = (
        OracleContract("large", "wl", "t", amount, amount, amount),
        OracleContract("small", "ws", "t", Decimal("2"), Decimal("1"), Decimal("1")),
    )
    return contracts, {"t": Decimal("0")}, {"t": 2}, amount


def collect_gap_instances() -> dict[str, object]:
    ratio_family: list[dict[str, str]] = []
    for scale in (10, 20, 50, 100, 1000):
        contracts, costs, capacities, budget = _ratio_instance(scale)
        greedy = density_greedy_oracle(contracts, costs, capacities, budget)
        exact = exact_integral_oracle(contracts, costs, capacities, budget)
        ratio_family.append(
            {
                "scale": str(scale),
                "greedy_value": str(greedy.value),
                "exact_value": str(exact.value),
                "additive_gap": str(exact.value - greedy.value),
                "greedy_over_exact": str(greedy.value / exact.value),
                "greedy_contracts": ",".join(greedy.contracts),
                "exact_contracts": ",".join(exact.contracts),
            }
        )

    facility_contracts = (
        OracleContract("fixed-heavy", "wa", "a", Decimal("4"), Decimal("1"), Decimal("1")),
        *tuple(
            OracleContract(f"b{index}", f"wb{index}", "b", Decimal("3"), Decimal("2"), Decimal("2"))
            for index in range(5)
        ),
    )
    facility_costs = {"a": Decimal("9"), "b": Decimal("0")}
    facility_caps = {"a": 1, "b": 5}
    facility_greedy = density_greedy_oracle(
        facility_contracts, facility_costs, facility_caps, Decimal("10")
    )
    facility_exact = exact_integral_oracle(
        facility_contracts, facility_costs, facility_caps, Decimal("10")
    )

    optimum_lagrangian = Decimal("-10")
    selected_lagrangian = Decimal("-10")
    beta = Decimal("0.5")
    additive_gap = Decimal("0")
    return {
        "seed": SEED,
        "scope": "small exact integral audit; not paper results and not a T7 proof",
        "decision": "REDESIGN_ONLINE_ORACLE",
        "ratio_knapsack_family": ratio_family,
        "task_activation_fixed_cost": {
            "budget": "10",
            "task_costs": {key: str(value) for key, value in facility_costs.items()},
            "greedy_contracts": list(facility_greedy.contracts),
            "greedy_value": str(facility_greedy.value),
            "exact_contracts": list(facility_exact.contracts),
            "exact_value": str(facility_exact.value),
            "additive_gap": str(facility_exact.value - facility_greedy.value),
        },
        "time_varying_lambda": {
            "gradients": ["1", "-1"],
            "lambdas": ["10", "1"],
            "unweighted_sum": "0",
            "weighted_sum": "9",
        },
        "additive_vs_multiplicative": {
            "optimum_lagrangian": str(optimum_lagrangian),
            "selected_lagrangian": str(selected_lagrangian),
            "additive_gap": str(additive_gap),
            "beta": str(beta),
            "additive_condition": selected_lagrangian >= optimum_lagrangian - additive_gap,
            "multiplicative_condition": selected_lagrangian >= beta * optimum_lagrangian - additive_gap,
        },
        "delayed_estimator_bias": {
            "true_values": ["0", "0", "1", "1"],
            "two_slot_delayed_estimates": ["0", "0", "0", "0"],
            "cumulative_bias": "-2",
            "finding": "unbiasedness requires an explicit stochastic/filtration assumption",
        },
        "shadow_conservatism": {
            "budget": "10",
            "public_cap_per_contract": "10",
            "critical_base_per_contract": "1",
            "shadow_feasible_count": "1",
            "actual-money_feasible_count": "10",
            "finding": "valid for pathwise comparator alignment but potentially arbitrarily conservative in coverage",
        },
    }


def test_current_density_is_not_exact_argmax() -> None:
    contracts, costs, capacities, budget = _ratio_instance(100)
    greedy = density_greedy_oracle(contracts, costs, capacities, budget)
    exact = exact_integral_oracle(contracts, costs, capacities, budget)
    assert greedy.contracts == ("small",)
    assert exact.contracts == ("large",)
    assert exact.value - greedy.value == Decimal("98")


def test_ratio_greedy_gap_grows_with_scale() -> None:
    results = collect_gap_instances()["ratio_knapsack_family"]
    additive = [Decimal(item["additive_gap"]) for item in results]
    ratios = [Decimal(item["greedy_over_exact"]) for item in results]
    assert additive == sorted(additive)
    assert ratios == sorted(ratios, reverse=True)
    assert additive[-1] == Decimal("998")
    assert ratios[-1] == Decimal("0.002")


def test_fixed_task_activation_cost_creates_complementarity_gap() -> None:
    result = collect_gap_instances()["task_activation_fixed_cost"]
    assert result["greedy_value"] == "4"
    assert result["exact_value"] == "15"
    assert result["additive_gap"] == "11"


def test_additive_oracle_condition_does_not_imply_beta_condition() -> None:
    result = collect_gap_instances()["additive_vs_multiplicative"]
    assert result["additive_condition"] is True
    assert result["multiplicative_condition"] is False


if __name__ == "__main__":
    output = collect_gap_instances()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
