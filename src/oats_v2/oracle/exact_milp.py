from __future__ import annotations

import time
from decimal import Decimal

from .certified_result import CertifiedOracleResult
from .problem import LocalOracleProblem


def solve_exact_milp_reference(problem: LocalOracleProblem, *, max_contracts: int = 24) -> CertifiedOracleResult:
    """Solve the frozen MILP model by deterministic exhaustive enumeration.

    No external MILP backend is installed.  The returned status explicitly
    says ENUMERATED_MILP_MODEL so it cannot be presented as solver-backend
    performance.  This is an exact small-instance audit oracle only.
    """

    start = time.perf_counter()
    contracts = problem.eligible_contracts()
    if len(contracts) > max_contracts:
        return CertifiedOracleResult(
            objective=Decimal("0"),
            solution=(),
            active_tasks=(),
            envelope_used=Decimal("0"),
            optimality_status="UNSUPPORTED_SIZE_NO_MILP_BACKEND",
            certified_gap=None,
            runtime_seconds=time.perf_counter() - start,
            explored_states=0,
            explored_nodes=0,
        )
    best_objective = Decimal("0")
    best_solution: tuple[str, ...] = ()
    best_envelope = Decimal("0")
    best_tasks: tuple[str, ...] = ()
    nodes = 0
    for mask in range(1 << len(contracts)):
        nodes += 1
        solution = tuple(
            contracts[index].contract_id for index in range(len(contracts)) if mask & (1 << index)
        )
        try:
            objective, envelope, active_tasks = problem.evaluate(solution)
        except ValueError:
            continue
        ordered = tuple(sorted(solution))
        if objective > best_objective or (objective == best_objective and ordered < best_solution):
            best_objective, best_solution, best_envelope, best_tasks = (
                objective,
                ordered,
                envelope,
                active_tasks,
            )
    return CertifiedOracleResult(
        objective=best_objective,
        solution=best_solution,
        active_tasks=best_tasks,
        envelope_used=best_envelope,
        optimality_status="OPTIMAL_ENUMERATED_MILP_MODEL",
        certified_gap=Decimal("0"),
        runtime_seconds=time.perf_counter() - start,
        explored_states=nodes,
        explored_nodes=nodes,
    )
