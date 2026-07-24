from __future__ import annotations

from decimal import Decimal

import pytest

from src.oats_v2.contracts import ContinuationTable, TaskContract
from src.oats_v2.experiments.method_registry import choose_effort, fast_bid, METHOD_REGISTRY


def _gbar_increasing() -> dict[str, Decimal]:
    # Gbar increasing in effort, within the admissible domain (d = c_i*k - Gbar >= 0
    # for c_i in {0.5,1,2}); mirrors real sbar<=0.5 continuation tables.
    return {"0": Decimal("0.30"), "0.5": Decimal("0.40"), "1": Decimal("0.48")}


def _k(effort: Decimal) -> Decimal:
    return Decimal("1") + Decimal("0.1") * effort * effort


@pytest.mark.parametrize("c_i", [Decimal("0.5"), Decimal("1.0"), Decimal("2.0")])
def test_choose_effort_is_best_response(c_i: Decimal) -> None:
    gbar = _gbar_increasing()
    efforts = (Decimal("0"), Decimal("0.5"), Decimal("1"))
    # ground-truth best response: minimise effective cost c_i*k(e) - Gbar(e)
    eff_cost = {e: c_i * _k(e) - gbar[str(e)] for e in efforts}
    best = min(eff_cost.values())
    expected = min(e for e in efforts if eff_cost[e] == best)
    assert choose_effort(c_i, gbar, fixed=None) == expected


@pytest.mark.parametrize("c_i", [Decimal("0.5"), Decimal("1.0"), Decimal("2.0")])
def test_bid_equals_effective_cost_at_chosen_effort(c_i: Decimal) -> None:
    gbar = _gbar_increasing()
    method = METHOD_REGISTRY["V2-FULL"]
    e = choose_effort(c_i, gbar, fixed=None)
    bid = fast_bid(method, c_i, gbar, money_grid=Decimal("0.001"))
    # effective cost at the chosen effort must equal the (pre-quantisation) bid basis
    d = c_i * _k(e) - gbar[str(e)]
    # bid is ceil(d) on the grid, so bid-grid >= d and within one grid step
    assert bid >= d
    assert bid - d < Decimal("0.001")


def test_choose_effort_matches_contract_effective_cost() -> None:
    # Cross-check against the authoritative TaskContract.effective_cost argmin.
    gbar = _gbar_increasing()
    table = ContinuationTable(gbar, Decimal("0.000001"), "v-test", signed=True)
    contract = TaskContract(
        task_id="t",
        effort_levels=("0", "0.5", "1"),
        effort_basis={e: Decimal("1") + Decimal("0.1") * Decimal(e) * Decimal(e) for e in gbar},
        continuation=table,
        base_cap=Decimal("3.0"),
        money_grid=Decimal("0.001"),
        score_cap=Decimal("1.0"),
    )
    for c_i in (Decimal("0.5"), Decimal("1.0"), Decimal("2.0")):
        _, best_efforts = contract.effective_cost(c_i)
        assert choose_effort(c_i, gbar, fixed=None) == min(Decimal(e) for e in best_efforts)
