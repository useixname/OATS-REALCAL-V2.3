from __future__ import annotations

import time
from decimal import Decimal
from functools import lru_cache

from .certified_result import CertifiedOracleResult
from .problem import LocalOracleProblem


def solve_exact_dp(problem: LocalOracleProblem) -> CertifiedOracleResult:
    start = time.perf_counter()
    contracts = problem.eligible_contracts()
    tasks = problem.tasks_by_id
    task_ids = tuple(sorted(tasks))
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    workers = tuple(sorted({contract.worker_id for contract in contracts}))
    worker_index = {worker: index for index, worker in enumerate(workers)}

    @lru_cache(maxsize=None)
    def search(
        index: int,
        used: Decimal,
        active_mask: int,
        worker_mask: int,
        task_counts: tuple[int, ...],
    ) -> tuple[Decimal, tuple[str, ...]]:
        if index == len(contracts):
            return Decimal("0"), ()
        contract = contracts[index]
        best_objective, best_solution = search(index + 1, used, active_mask, worker_mask, task_counts)
        task_position = task_index[contract.task_id]
        worker_bit = 1 << worker_index[contract.worker_id]
        task_bit = 1 << task_position
        if worker_mask & worker_bit or task_counts[task_position] >= tasks[contract.task_id].capacity:
            return best_objective, best_solution
        incremental_envelope = contract.envelope
        incremental_objective = contract.vhat - problem.dual_lambda * contract.envelope
        if not active_mask & task_bit:
            incremental_envelope += tasks[contract.task_id].activation_cost
            incremental_objective -= problem.dual_lambda * tasks[contract.task_id].activation_cost
        if used + incremental_envelope > problem.remaining_shadow_capacity:
            return best_objective, best_solution
        next_counts = list(task_counts)
        next_counts[task_position] += 1
        tail_objective, tail_solution = search(
            index + 1,
            used + incremental_envelope,
            active_mask | task_bit,
            worker_mask | worker_bit,
            tuple(next_counts),
        )
        take_objective = incremental_objective + tail_objective
        take_solution = tuple(sorted((contract.contract_id, *tail_solution)))
        if take_objective > best_objective or (
            take_objective == best_objective and take_solution < best_solution
        ):
            return take_objective, take_solution
        return best_objective, best_solution

    initial_counts = tuple(0 for _ in task_ids)
    objective, solution = search(0, Decimal("0"), 0, 0, initial_counts)
    checked_objective, envelope, active_tasks = problem.evaluate(solution)
    if checked_objective != objective:
        raise AssertionError("DP objective/evaluator mismatch")
    info = search.cache_info()
    return CertifiedOracleResult(
        objective=objective,
        solution=solution,
        active_tasks=active_tasks,
        envelope_used=envelope,
        optimality_status="OPTIMAL_EXACT_DP",
        certified_gap=Decimal("0"),
        runtime_seconds=time.perf_counter() - start,
        explored_states=info.misses,
        explored_nodes=info.hits + info.misses,
    )
