from __future__ import annotations

import time
from decimal import Decimal

from .certified_result import CertifiedOracleResult
from .problem import LocalOracleProblem


def solve_legacy_density(problem: LocalOracleProblem, epsilon: Decimal = Decimal("0.001")) -> CertifiedOracleResult:
    start = time.perf_counter()
    tasks = problem.tasks_by_id

    def priority(contract: object) -> tuple[Decimal, str]:
        numerator = max(Decimal("0"), contract.vhat - problem.dual_lambda * contract.envelope)
        return (-(numerator / (contract.estimated_transfer + epsilon)), contract.contract_id)

    selected: list[str] = []
    workers: set[str] = set()
    active_tasks: set[str] = set()
    task_counts: dict[str, int] = {}
    used = Decimal("0")
    nodes = 0
    for contract in sorted(problem.eligible_contracts(), key=priority):
        nodes += 1
        if contract.worker_id in workers or task_counts.get(contract.task_id, 0) >= tasks[contract.task_id].capacity:
            continue
        incremental = contract.envelope
        if contract.task_id not in active_tasks:
            incremental += tasks[contract.task_id].activation_cost
        if used + incremental > problem.remaining_shadow_capacity:
            continue
        selected.append(contract.contract_id)
        workers.add(contract.worker_id)
        active_tasks.add(contract.task_id)
        task_counts[contract.task_id] = task_counts.get(contract.task_id, 0) + 1
        used += incremental
    solution = tuple(sorted(selected))
    objective, envelope, active = problem.evaluate(solution)
    return CertifiedOracleResult(
        objective=objective,
        solution=solution,
        active_tasks=active,
        envelope_used=envelope,
        optimality_status="HEURISTIC_REFUTED_NO_CERTIFICATE",
        certified_gap=None,
        runtime_seconds=time.perf_counter() - start,
        explored_states=nodes,
        explored_nodes=nodes,
    )
