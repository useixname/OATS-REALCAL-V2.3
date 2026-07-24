from __future__ import annotations

import time
from dataclasses import replace
from decimal import Decimal
from typing import Callable

from .certified_result import CertifiedOracleResult
from .problem import LocalOracleProblem


Solver = Callable[[LocalOracleProblem], CertifiedOracleResult]


def rerun_grid(
    problem: LocalOracleProblem,
    target_contract_id: str,
    bids: tuple[Decimal, ...],
    solver: Solver,
) -> dict[str, object]:
    runtimes: list[float] = []
    states: list[int] = []
    outcomes: list[tuple[str, bool]] = []
    statuses: list[str] = []
    certified_gaps: list[str | None] = []
    total_start = time.perf_counter()
    for bid in bids:
        contracts = tuple(
            replace(
                contract,
                reported_bid=bid,
                estimated_transfer=bid,
            )
            if contract.contract_id == target_contract_id
            else contract
            for contract in problem.contracts
        )
        result = solver(replace(problem, contracts=contracts))
        runtimes.append(result.runtime_seconds)
        states.append(result.explored_states)
        statuses.append(result.optimality_status)
        certified_gaps.append(None if result.certified_gap is None else str(result.certified_gap))
        outcomes.append((str(bid), target_contract_id in result.solution))
    ordered = sorted(runtimes)
    return {
        "target_contract_id": target_contract_id,
        "oracle_calls": len(bids),
        "critical_search_iterations": len(bids),
        "total_solver_seconds": sum(runtimes),
        "wall_seconds": time.perf_counter() - total_start,
        "p50_solver_seconds": ordered[len(ordered) // 2],
        "p95_solver_seconds": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "worst_solver_seconds": max(runtimes),
        "total_explored_states": sum(states),
        "max_explored_states": max(states),
        "optimality_statuses": sorted(set(statuses)),
        "certified_gaps": sorted(set(certified_gaps), key=lambda item: "" if item is None else item),
        "timeouts": sum("TIMEOUT" in status for status in statuses),
        "outcomes": outcomes,
    }
