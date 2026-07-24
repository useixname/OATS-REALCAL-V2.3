from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Mapping

from .certified_result import CertifiedOracleResult
from .exact_dp import solve_exact_dp
from .exact_milp import solve_exact_milp_reference
from .greedy_legacy import solve_legacy_density
from .problem import LocalContract, LocalOracleProblem, LocalTask


# Compatibility facade retained for the Phase 3A audit tests.
@dataclass(frozen=True)
class OracleContract:
    contract_id: str
    worker_id: str
    task_id: str
    value: Decimal
    envelope: Decimal
    estimated_transfer: Decimal


@dataclass(frozen=True)
class OracleSolution:
    contracts: tuple[str, ...]
    value: Decimal
    envelope: Decimal


def _compat_problem(
    contracts: Iterable[OracleContract],
    task_costs: Mapping[str, Decimal],
    task_capacities: Mapping[str, int],
    budget: Decimal,
) -> LocalOracleProblem:
    contracts = tuple(contracts)
    task_ids = sorted({contract.task_id for contract in contracts})
    return LocalOracleProblem(
        tasks=tuple(LocalTask(task, task_costs.get(task, Decimal("0")), task_capacities[task]) for task in task_ids),
        contracts=tuple(
            LocalContract(
                contract.contract_id,
                contract.worker_id,
                contract.task_id,
                contract.value,
                contract.envelope,
                contract.estimated_transfer,
            )
            for contract in contracts
        ),
        remaining_shadow_capacity=budget,
        dual_lambda=Decimal("0"),
    )


def exact_integral_oracle(
    contracts: Iterable[OracleContract],
    task_costs: Mapping[str, Decimal],
    task_capacities: Mapping[str, int],
    budget: Decimal,
) -> OracleSolution:
    problem = _compat_problem(contracts, task_costs, task_capacities, budget)
    result = solve_exact_dp(problem)
    return OracleSolution(result.solution, result.objective, result.envelope_used)


def density_greedy_oracle(
    contracts: Iterable[OracleContract],
    task_costs: Mapping[str, Decimal],
    task_capacities: Mapping[str, int],
    budget: Decimal,
    *,
    dual_lambda: Decimal = Decimal("0"),
    epsilon: Decimal = Decimal("0.001"),
) -> OracleSolution:
    problem = _compat_problem(contracts, task_costs, task_capacities, budget)
    problem = LocalOracleProblem(problem.tasks, problem.contracts, problem.remaining_shadow_capacity, dual_lambda)
    result = solve_legacy_density(problem, epsilon)
    # Phase 3A compatibility expected gross value at lambda=0; under nonzero
    # lambda report the frozen Lagrangian objective by design.
    return OracleSolution(result.solution, result.objective, result.envelope_used)


__all__ = [
    "CertifiedOracleResult",
    "LocalContract",
    "LocalOracleProblem",
    "LocalTask",
    "OracleContract",
    "OracleSolution",
    "density_greedy_oracle",
    "exact_integral_oracle",
    "solve_exact_dp",
    "solve_exact_milp_reference",
    "solve_legacy_density",
]
