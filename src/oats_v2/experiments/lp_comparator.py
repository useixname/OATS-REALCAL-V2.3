from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from decimal import ROUND_CEILING, ROUND_DOWN

from ..data.schemas import TraceConfig
from ..types import D
from .method_registry import choose_effort
from .result_schema import LPComparatorResult
from .trace_loader import TraceBundle


LP_SOLVER = "highs"
LP_TOL = 1e-9
VIOLATION_TOL = Decimal("1e-8")
MONEY_GRID_LP = Decimal("0.001")


@dataclass
class LPOpportunity:
    index: int
    slot: int
    task_id: str
    worker_id: str
    v_ijt: Decimal
    # Realized hindsight cost of executing this trade: truthful bid (minimal
    # IC base payment) + the score bonus sbar*score actually paid on purchase.
    # OATS recycles worst-case reserves through shadow settlement, so the
    # old reserve-cost model (Abar per pair + full gamma*V escrow per task,
    # never returned) priced trades ~2-3x above what the online mechanism
    # actually consumes and stopped being an upper bound.
    cost: Decimal
    capacity: int


class LPComparatorCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, LPComparatorResult] = {}

    def get(self, key: tuple) -> LPComparatorResult | None:
        return self._cache.get(key)

    def put(self, key: tuple, value: LPComparatorResult) -> None:
        self._cache[key] = value


def _build_opportunities(
    trace: TraceBundle,
    gamma: Decimal,
    *,
    arrival_multiplier: Decimal,
    delay: int = 0,
    missing_prob: Decimal = Decimal("0"),
    horizon: int = 1000,
) -> list[LPOpportunity]:
    cfg = TraceConfig()
    missing_key = str(missing_prob)
    delay_key = str(delay)
    opportunities: list[LPOpportunity] = []
    idx = 0
    for slot in sorted(trace.tasks_by_slot):
        if slot > horizon:
            continue
        for task in sorted(trace.tasks_by_slot[slot], key=lambda t: t.task_id):
            if arrival_multiplier == Decimal("0.5"):
                token = f"{trace.seed}|{slot}|{task.task_id}"
                if int.from_bytes(__import__("hashlib").sha256(token.encode()).digest()[:4], "big") % 2 == 1:
                    continue
            # Gate outcome availability identically to the online path (Spec §12
            # I_out): a missing or too-late outcome yields no realizable value, so
            # the hindsight LP must not count it either.
            outcome_available = not task.missing_mask.get(missing_key, False)
            if outcome_available and delay > 0:
                outcome_available = (slot + task.delay_mask.get(delay_key, 0)) <= horizon
            if not outcome_available:
                continue
            # arrival x2 duplicates the task stream (same latent attributes,
            # cloned id); mirror of the online _tasks_for_slot expansion.
            replicas = ("", "#d2") if arrival_multiplier == Decimal("2") else ("",)
            for replica_suffix in replicas:
                replica_task_id = task.task_id + replica_suffix
                for row in trace.eligibility_by_slot.get(slot, []):
                    if not row.get("available") or row.get("mapped_task_id") != task.task_id:
                        continue
                    worker_id = row["worker_id"]
                    worker = trace.workers[worker_id]
                    contract_key = (slot, task.task_id, worker_id, gamma)
                    contract = trace.contracts.get(contract_key)
                    if contract is None:
                        continue
                    effort = choose_effort(worker.c_i, contract.gbar_by_effort, fixed=None)
                    pot = trace.potential.get((slot, task.task_id, worker_id, effort))
                    if pot is None:
                        continue
                    # Truthful bid (grid-ceiling of the effective cost, clamped
                    # at 0) + realized score bonus (grid-floor of sbar*score),
                    # matching the online settlement path.
                    utilities = [
                        worker.c_i * (Decimal("1") + Decimal("0.1") * e * e) - contract.gbar_by_effort[str(e)]
                        for e in (Decimal("0"), Decimal("0.5"), Decimal("1"))
                    ]
                    d = min(utilities)
                    if d <= 0:
                        bid = Decimal("0")
                    else:
                        bid = (d / MONEY_GRID_LP).to_integral_value(rounding=ROUND_CEILING) * MONEY_GRID_LP
                    bonus = (contract.sbar * pot.score).quantize(MONEY_GRID_LP, rounding=ROUND_DOWN)
                    opportunities.append(
                        LPOpportunity(
                            index=idx,
                            slot=slot,
                            task_id=replica_task_id,
                            worker_id=worker_id,
                            v_ijt=pot.v_ijt,
                            cost=bid + bonus,
                            capacity=task.capacity,
                        )
                    )
                    idx += 1
    return opportunities


def solve_global_lp(
    trace: TraceBundle,
    budget: Decimal,
    gamma: Decimal,
    *,
    arrival_multiplier: Decimal,
    delay: int = 0,
    missing_prob: Decimal = Decimal("0"),
    horizon: int = 1000,
) -> LPComparatorResult:
    opportunities = _build_opportunities(
        trace,
        gamma,
        arrival_multiplier=arrival_multiplier,
        delay=delay,
        missing_prob=missing_prob,
        horizon=horizon,
    )
    if not opportunities:
        return LPComparatorResult(status="INFEASIBLE_EMPTY")

    n = len(opportunities)
    task_ids = sorted({opp.task_id for opp in opportunities})
    task_index = {task_id: i for i, task_id in enumerate(task_ids)}
    n_tasks = len(task_ids)
    n_vars = n + n_tasks

    c = np.zeros(n_vars, dtype=np.float64)
    for opp in opportunities:
        c[opp.index] = -float(opp.v_ijt)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []
    row = 0

    for opp in opportunities:
        rows.extend([row, row])
        cols.extend([opp.index, n + task_index[opp.task_id]])
        data.extend([1.0, -1.0])
        b_ub.append(0.0)
        row += 1

    by_task: dict[str, list[LPOpportunity]] = {}
    for opp in opportunities:
        by_task.setdefault(opp.task_id, []).append(opp)
    for task_id, opps in by_task.items():
        cap = opps[0].capacity
        for opp in opps:
            rows.append(row)
            cols.append(opp.index)
            data.append(1.0)
        rows.append(row)
        cols.append(n + task_index[task_id])
        data.append(-float(cap))
        b_ub.append(0.0)
        row += 1

    # Per-(worker, slot) participation constraint. The online mechanism lets a
    # worker win at most one contract per slot but participate in every slot,
    # so the hindsight LP must use the same opportunity set. (The old
    # horizon-wide sum x <= 1 constraint made the LP strictly weaker than the
    # online mechanism and produced negative "gaps".)
    by_worker_slot: dict[tuple[str, int], list[LPOpportunity]] = {}
    for opp in opportunities:
        by_worker_slot.setdefault((opp.worker_id, opp.slot), []).append(opp)
    for opps in by_worker_slot.values():
        if len(opps) == 1:
            # Bound constraint x <= 1 already enforces it.
            continue
        for opp in opps:
            rows.append(row)
            cols.append(opp.index)
            data.append(1.0)
        b_ub.append(1.0)
        row += 1

    budget_row = [0.0] * n_vars
    for opp in opportunities:
        # Realized-cost accounting: each trade consumes its truthful bid
        # plus the realized score bonus. Worst-case reserves (Abar, task escrow)
        # are recycled by the online mechanism (shadow settlement, Eq. 91/93)
        # and therefore do not consume hindsight budget.
        budget_row[opp.index] = float(opp.cost)
    rows.extend([row] * n_vars)
    cols.extend(list(range(n_vars)))
    data.extend(budget_row)
    b_ub.append(float(budget))
    row += 1

    a_ub = coo_matrix((data, (rows, cols)), shape=(row, n_vars)).tocsr()
    bounds = [(0.0, 1.0)] * n_vars
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.array(b_ub, dtype=np.float64),
        bounds=bounds,
        method=LP_SOLVER,
        options={
            "primal_feasibility_tolerance": LP_TOL,
            "dual_feasibility_tolerance": LP_TOL,
            "ipm_optimality_tolerance": LP_TOL,
        },
    )
    if not result.success:
        return LPComparatorResult(status=f"FAILED_{result.message}")

    x = result.x
    opt_lp = Decimal(str(round(-float(result.fun), 12)))
    # Independent constraint-violation recomputation (Spec §14 / §5 LP rules):
    # vectorised lhs - b_ub, clamped at 0. Faster and free of the ndim>0 float() warning.
    lhs = np.asarray(a_ub.dot(x)).ravel()
    residual = lhs - np.asarray(b_ub, dtype=np.float64)
    worst = float(residual.max()) if residual.size else 0.0
    max_violation = D(str(max(0.0, worst)))

    if max_violation > VIOLATION_TOL:
        return LPComparatorResult(status="VIOLATION_EXCEEDS_TOL", opt_lp=opt_lp, max_violation=max_violation)

    return LPComparatorResult(
        status="optimal",
        opt_lp=opt_lp,
        max_violation=max_violation,
    )


def compute_lp_gap(
    cache: LPComparatorCache,
    cache_key: tuple,
    trace: TraceBundle,
    budget: Decimal,
    gamma: Decimal,
    u_online: Decimal,
    *,
    arrival_multiplier: Decimal,
    delay: int = 0,
    missing_prob: Decimal = Decimal("0"),
    horizon: int = 1000,
) -> LPComparatorResult:
    cached = cache.get(cache_key)
    if cached is None:
        cached = solve_global_lp(
            trace,
            budget,
            gamma,
            arrival_multiplier=arrival_multiplier,
            delay=delay,
            missing_prob=missing_prob,
            horizon=horizon,
        )
        cache.put(cache_key, cached)
    if cached.status != "optimal" or cached.opt_lp is None:
        return cached
    lp_gap = cached.opt_lp - u_online
    denom = max(cached.opt_lp, Decimal("1e-12"))
    return LPComparatorResult(
        status=cached.status,
        opt_lp=cached.opt_lp,
        u_online=u_online,
        lp_gap=lp_gap,
        normalized_lp_gap=lp_gap / denom,
        max_violation=cached.max_violation,
    )
