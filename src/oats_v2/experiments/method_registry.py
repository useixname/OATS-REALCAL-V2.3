from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from ..contracts import ContinuationTable, TaskContract, quantize_ceiling
from ..types import Candidate, D


METHOD_IDS = (
    "V2-FULL",
    "B-P1",
    "B-NOSCREEN",
    "B-NOTRUST",
    "B-NODUAL",
    "B-MYOPIC",
    "B-RANDOM",
    "B-COST",
    "B-TRUST",
    "B-QUALITY",
    "C-EFFORT-OFF",
)

FAIR_BASELINES = (
    "V2-FULL",
    "B-P1",
    "B-NOSCREEN",
    "B-NOTRUST",
    "B-NODUAL",
    "B-MYOPIC",
)


@dataclass(frozen=True)
class MethodConfig:
    """Method switches.

    ``rank_mode`` selects the slot-level ranking rule:
      density  — OATS adjusted-value density (paper Eq. 76)
      surplus  — per-slot surplus greedy (Per-slot BRA baseline)
      cost     — ascending bid (Cost-only baseline)
      value    — descending predicted value (Trust-only baseline via vhat=rho)
      quality  — descending predicted quality (Quality-only baseline)
      random   — deterministic salted hash order (Random baseline)

    ``use_reserve`` enables the platform reserve price (bid > vhat rejected;
    critical payment bounded by vhat). ``use_av_gate`` enables the paper
    Eq. (78) adjusted-value gate (dual-adjusted value must be positive).
    Both gates are monotone in the candidate's own bid.
    """

    method_id: str
    use_trust: bool = True
    use_dual: bool = True
    use_screening: bool = True
    use_density: bool = True
    rank_mode: str = "density"
    use_reserve: bool = True
    use_av_gate: bool = True
    fixed_effort: Decimal | None = None
    p1_payment: bool = False
    noscreen_gbar: bool = False
    trust_only_value: bool = False

    def validate(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ValueError(f"unknown method: {self.method_id}")


METHOD_REGISTRY: dict[str, MethodConfig] = {
    # OATS and controlled ablations (share reserve + AV gates).
    "V2-FULL": MethodConfig("V2-FULL"),
    "B-P1": MethodConfig("B-P1", p1_payment=True, fixed_effort=Decimal("0")),
    "B-NOSCREEN": MethodConfig("B-NOSCREEN", noscreen_gbar=True, use_screening=False),
    "B-NOTRUST": MethodConfig("B-NOTRUST", use_trust=False),
    "B-NODUAL": MethodConfig("B-NODUAL", use_dual=False),
    "B-MYOPIC": MethodConfig("B-MYOPIC", use_dual=False, use_density=False, rank_mode="surplus"),
    "C-EFFORT-OFF": MethodConfig("C-EFFORT-OFF", fixed_effort=Decimal("0")),
    # Naive selection baselines (no value model => no reserve/AV gates).
    "B-RANDOM": MethodConfig(
        "B-RANDOM", use_dual=False, use_density=False, rank_mode="random", use_reserve=False, use_av_gate=False
    ),
    "B-COST": MethodConfig(
        "B-COST", use_dual=False, use_density=False, rank_mode="cost", use_reserve=False, use_av_gate=False
    ),
    "B-TRUST": MethodConfig(
        "B-TRUST",
        use_dual=False,
        use_density=False,
        rank_mode="value",
        use_reserve=False,
        use_av_gate=False,
        trust_only_value=True,
    ),
    "B-QUALITY": MethodConfig(
        "B-QUALITY", use_dual=False, use_density=False, rank_mode="quality", use_reserve=False, use_av_gate=False
    ),
}


def effort_cost(c_i: Decimal, effort: Decimal) -> Decimal:
    return c_i * (Decimal("1") + Decimal("0.1") * effort * effort)


def choose_effort(
    c_i: Decimal,
    gbar: dict[str, Decimal],
    *,
    fixed: Decimal | None,
) -> Decimal:
    """Compliant rational best response (Spec §5).

    Worker minimises effective cost ``c_i*k(e) - Gbar(e)`` with
    ``k(e) = 1 + 0.1*e^2``; equivalently maximises net utility
    ``Gbar(e) - c_i*k(e)``. This MUST be the same effort that yields the
    truthful bid ``d = min_e (c_i*k(e) - Gbar(e))`` (see ``fast_bid`` /
    ``TaskContract.effective_cost``), otherwise bid and realised effort diverge.
    """
    if fixed is not None:
        return fixed
    efforts = (Decimal("0"), Decimal("0.5"), Decimal("1"))
    effective_cost = {
        e: c_i * (Decimal("1") + Decimal("0.1") * e * e) - Decimal(gbar[str(e)])
        for e in efforts
    }
    best = min(effective_cost.values())
    chosen = [e for e in efforts if effective_cost[e] == best]
    return min(chosen)


def fast_bid(method: MethodConfig, c_i: Decimal, gbar: dict[str, Decimal], *, money_grid: Decimal) -> Decimal:
    if method.p1_payment:
        return quantize_ceiling(c_i, money_grid)
    utilities = {
        e: c_i * (Decimal("1") + Decimal("0.1") * Decimal(e) * Decimal(e)) - Decimal(gbar[e])
        for e in gbar
    }
    d = min(utilities.values())
    # With a large score bonus (REAL-CAL-V2 value scale) the effective cost can
    # be negative: participation is strictly profitable even at zero base pay.
    # The truthful bid is then clamped at the grid floor (workers cannot pay
    # the platform); IC/IR are unaffected because winning at bid 0 weakly
    # dominates for any type whose d <= 0.
    if d < 0:
        return Decimal("0")
    return quantize_ceiling(d, money_grid)


def compute_bid(
    method: MethodConfig,
    c_i: Decimal,
    contract: TaskContract,
    *,
    gbar: dict[str, Decimal],
) -> Decimal:
    if method.p1_payment:
        return quantize_ceiling(effort_cost(c_i, Decimal("0")), contract.money_grid)
    return contract.truthful_bid(c_i)


def predicted_value(
    *,
    coefficient: Decimal,
    trust: Decimal,
    use_trust: bool,
) -> Decimal:
    weight = trust if use_trust else Decimal("1")
    return weight * coefficient


def rank_key(
    candidate: Candidate,
    *,
    dual_lambda: Decimal,
    epsilon_rank: Decimal,
    use_density: bool,
) -> tuple:
    if use_density:
        numerator = max(
            Decimal("0"), candidate.predicted_value - dual_lambda * candidate.estimated_reserve
        )
        density = numerator / (candidate.estimated_reserve + epsilon_rank)
        return (-density, candidate.worker_id, candidate.task_id)
    surplus = candidate.predicted_value - candidate.bid
    return (-surplus, candidate.worker_id, candidate.task_id)


def build_contract_from_row(
    row_gbar: dict[str, Decimal],
    *,
    sbar: Decimal,
    anchor_version: str,
    abar: Decimal,
) -> TaskContract:
    table = ContinuationTable(row_gbar, Decimal("0.000001"), anchor_version, signed=True)
    effort_levels = tuple(sorted(row_gbar.keys(), key=lambda x: Decimal(x)))
    basis = {e: Decimal("1") + Decimal("0.1") * Decimal(e) * Decimal(e) for e in effort_levels}
    return TaskContract(
        task_id="",
        effort_levels=effort_levels,
        effort_basis=basis,
        continuation=table,
        base_cap=abar,
        money_grid=Decimal("0.001"),
        score_cap=sbar,
    )
