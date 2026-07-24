from __future__ import annotations

from decimal import Decimal

from src.oats_v2.oracle.critical_rerun import rerun_grid
from src.oats_v2.oracle.exact_dp import solve_exact_dp
from src.oats_v2.oracle.fixtures import hand_computed_basic
from src.oats_v2.oracle.greedy_legacy import solve_legacy_density


def test_every_grid_bid_invokes_complete_solver() -> None:
    problem, _, _ = hand_computed_basic()
    bids = tuple(Decimal(index) / Decimal("10") for index in range(11))
    dp = rerun_grid(problem, "a", bids, solve_exact_dp)
    greedy = rerun_grid(problem, "a", bids, solve_legacy_density)
    assert dp["oracle_calls"] == greedy["oracle_calls"] == len(bids)
    assert dp["critical_search_iterations"] == len(bids)
    assert len(dp["outcomes"]) == len(bids)
    assert dp["total_explored_states"] > 0
