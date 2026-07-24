from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import product

from ..data.schemas import FORMAL_SEEDS, GAMMAS
from ..realcal import REALCAL_SEEDS


BUDGET_RATIOS = (Decimal("0.10"), Decimal("0.25"), Decimal("0.50"))
CONTAMINATION_LEVELS = (Decimal("0"), Decimal("0.1"), Decimal("0.3"), Decimal("0.5"))
DELAY_LEVELS = (0, 5, 20)
MISSING_LEVELS = (Decimal("0"), Decimal("0.1"), Decimal("0.3"))
ARRIVAL_MULTIPLIERS = (Decimal("0.5"), Decimal("1"), Decimal("2"))

COMPARISON_METHODS = (
    "V2-FULL",
    "B-P1",
    "B-NOSCREEN",
    "B-NOTRUST",
    "B-NODUAL",
    "B-MYOPIC",
)

# Defaults for the V2 sensitivity dimensions (E7 varies one at a time).
DEFAULT_ALPHA = Decimal("0.2")
# V2.1: coverage-only anchor confidence; honest cells sit near 1.0, hijacked
# cells near 0.55, so 0.75 separates the regimes (V2 used 0.45 with the
# dual-signal confidence, which was untunable).
DEFAULT_THETA_A = Decimal("0.75")
DEFAULT_LAMBDA_MAX = Decimal("10")
DEFAULT_GAMMA_V2 = Decimal("0.3")
DEFAULT_BUDGET_V2 = Decimal("0.25")

# E1 main comparison methods (paper Experiment 1 mapping):
#   OATS            -> V2-FULL
#   Random          -> B-RANDOM
#   Cost-only       -> B-COST
#   Trust-only      -> B-TRUST
#   Quality-only    -> B-QUALITY
#   VD-greedy       -> B-NODUAL   (value-density greedy, no dual control)
#   Per-slot BRA    -> B-MYOPIC   (per-slot surplus greedy)
#   Offline Oracle  -> LP comparator (computed per cell, not a method)
E1_METHODS = (
    "V2-FULL",
    "B-RANDOM",
    "B-COST",
    "B-TRUST",
    "B-QUALITY",
    "B-NODUAL",
    "B-MYOPIC",
)

E3_CONTAMINATION = (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"), Decimal("0.4"), Decimal("0.5"))
E6_EXTRA_METHODS = ("B-NOSCREEN", "B-NOTRUST", "B-P1", "C-EFFORT-OFF")
E7_GAMMAS = (Decimal("0"), Decimal("0.1"), Decimal("0.5"), Decimal("0.8"), Decimal("1.0"))
E7_ALPHAS = (Decimal("0.05"), Decimal("0.1"), Decimal("0.4"))
E7_THETA_AS = (Decimal("0"), Decimal("0.6"), Decimal("0.9"))
E7_LAMBDA_MAXES = (Decimal("1"), Decimal("5"), Decimal("50"))

# V2.1 E1B: tight budgets where the budget constraint actually binds
# (reference-budget spend appetite is ~0.13 of the envelope, so 0.10-0.50
# never bound in V2; the dual controller had costs but no possible benefit).
E1B_METHODS = ("V2-FULL", "B-NODUAL", "B-COST", "B-MYOPIC")
E1B_BUDGETS = (Decimal("0.03"), Decimal("0.05"))


@dataclass(frozen=True)
class RunCell:
    cell_id: str
    family: str
    seed: int
    method_id: str
    gamma: Decimal
    budget_ratio: Decimal
    contamination: Decimal
    delay: int
    missing_prob: Decimal
    arrival_multiplier: Decimal
    order_index: int
    alpha: Decimal = DEFAULT_ALPHA
    theta_a: Decimal = DEFAULT_THETA_A
    lambda_max: Decimal = DEFAULT_LAMBDA_MAX

    def lp_cache_key(self) -> tuple:
        return (
            self.seed,
            str(self.gamma),
            str(self.budget_ratio),
            str(self.arrival_multiplier),
            self.delay,
            str(self.missing_prob),
        )

    def checkpoint_key(self) -> str:
        return self.cell_id


def _cell_id(
    seed: int,
    method: str,
    gamma: Decimal,
    budget: Decimal,
    contamination: Decimal,
    delay: int,
    missing: Decimal,
    arrival: Decimal,
) -> str:
    return (
        f"s{seed}_m{method}_g{gamma}_b{budget}_c{contamination}"
        f"_d{delay}_m{missing}_a{arrival}"
    )


def build_run_matrix() -> tuple[RunCell, ...]:
    """Legacy Phase-4A / REAL-CAL-V1 matrix (F1-F5). Frozen for reproducibility."""
    cells: list[RunCell] = []
    order = 0

    # F1: core overall/ablation — 6 methods × 6 gamma × 3 budget, canonical robustness settings
    for seed, method, gamma, budget in product(FORMAL_SEEDS, COMPARISON_METHODS, GAMMAS, BUDGET_RATIOS):
        cells.append(
            RunCell(
                cell_id="fF1_OVERALL_" + _cell_id(seed, method, gamma, budget, Decimal("0"), 0, Decimal("0"), Decimal("1")),
                family="F1_OVERALL",
                seed=seed,
                method_id=method,
                gamma=gamma,
                budget_ratio=budget,
                contamination=Decimal("0"),
                delay=0,
                missing_prob=Decimal("0"),
                arrival_multiplier=Decimal("1"),
                order_index=order,
            )
        )
        order += 1

    # F2: effort-off channel control
    for seed, gamma in product(FORMAL_SEEDS, GAMMAS):
        cells.append(
            RunCell(
                cell_id="fF2_EFFORT_OFF_" + _cell_id(seed, "C-EFFORT-OFF", gamma, Decimal("0.25"), Decimal("0"), 0, Decimal("0"), Decimal("1")),
                family="F2_EFFORT_OFF",
                seed=seed,
                method_id="C-EFFORT-OFF",
                gamma=gamma,
                budget_ratio=Decimal("0.25"),
                contamination=Decimal("0"),
                delay=0,
                missing_prob=Decimal("0"),
                arrival_multiplier=Decimal("1"),
                order_index=order,
            )
        )
        order += 1

    # F3: screening robustness — V2-FULL, contamination > 0
    for seed, gamma, contamination in product(FORMAL_SEEDS, GAMMAS, (Decimal("0.1"), Decimal("0.3"), Decimal("0.5"))):
        cells.append(
            RunCell(
                cell_id="fF3_SCREENING_" + _cell_id(seed, "V2-FULL", gamma, Decimal("0.25"), contamination, 0, Decimal("0"), Decimal("1")),
                family="F3_SCREENING",
                seed=seed,
                method_id="V2-FULL",
                gamma=gamma,
                budget_ratio=Decimal("0.25"),
                contamination=contamination,
                delay=0,
                missing_prob=Decimal("0"),
                arrival_multiplier=Decimal("1"),
                order_index=order,
            )
        )
        order += 1

    # F4: trust delay/missing — non-canonical only
    for seed, gamma, delay, missing in product(
        FORMAL_SEEDS,
        GAMMAS,
        DELAY_LEVELS,
        MISSING_LEVELS,
    ):
        if delay == 0 and missing == Decimal("0"):
            continue
        cells.append(
            RunCell(
                cell_id="fF4_TRUST_" + _cell_id(seed, "V2-FULL", gamma, Decimal("0.25"), Decimal("0"), delay, missing, Decimal("1")),
                family="F4_TRUST",
                seed=seed,
                method_id="V2-FULL",
                gamma=gamma,
                budget_ratio=Decimal("0.25"),
                contamination=Decimal("0"),
                delay=delay,
                missing_prob=missing,
                arrival_multiplier=Decimal("1"),
                order_index=order,
            )
        )
        order += 1

    # F5: adaptability — non-canonical arrival/delay
    for seed, gamma, arrival, delay in product(
        FORMAL_SEEDS,
        GAMMAS,
        ARRIVAL_MULTIPLIERS,
        DELAY_LEVELS,
    ):
        if arrival == Decimal("1") and delay == 0:
            continue
        cells.append(
            RunCell(
                cell_id="fF5_ADAPTABILITY_" + _cell_id(seed, "V2-FULL", gamma, Decimal("0.25"), Decimal("0"), delay, Decimal("0"), arrival),
                family="F5_ADAPTABILITY",
                seed=seed,
                method_id="V2-FULL",
                gamma=gamma,
                budget_ratio=Decimal("0.25"),
                contamination=Decimal("0"),
                delay=delay,
                missing_prob=Decimal("0"),
                arrival_multiplier=arrival,
                order_index=order,
            )
        )
        order += 1

    return tuple(cells)


def _v2_cell(
    family: str,
    order: int,
    seed: int,
    method: str,
    *,
    gamma: Decimal = DEFAULT_GAMMA_V2,
    budget: Decimal = DEFAULT_BUDGET_V2,
    contamination: Decimal = Decimal("0"),
    delay: int = 0,
    missing: Decimal = Decimal("0"),
    arrival: Decimal = Decimal("1"),
    alpha: Decimal = DEFAULT_ALPHA,
    theta_a: Decimal = DEFAULT_THETA_A,
    lambda_max: Decimal = DEFAULT_LAMBDA_MAX,
) -> RunCell:
    base = _cell_id(seed, method, gamma, budget, contamination, delay, missing, arrival)
    suffix = ""
    if alpha != DEFAULT_ALPHA or theta_a != DEFAULT_THETA_A or lambda_max != DEFAULT_LAMBDA_MAX:
        suffix = f"_al{alpha}_ta{theta_a}_lm{lambda_max}"
    return RunCell(
        cell_id=f"{family}_{base}{suffix}",
        family=family,
        seed=seed,
        method_id=method,
        gamma=gamma,
        budget_ratio=budget,
        contamination=contamination,
        delay=delay,
        missing_prob=missing,
        arrival_multiplier=arrival,
        order_index=order,
        alpha=alpha,
        theta_a=theta_a,
        lambda_max=lambda_max,
    )


def build_run_matrix_v2() -> tuple[RunCell, ...]:
    """REAL-CAL matrix implementing 实验.md experiments E1-E7 on 10 seeds.

    E2 (value identification), E4 (trust evolution) and E8 (runtime breakdown)
    are measured inside these cells rather than adding cells of their own.
    """
    cells: list[RunCell] = []
    order = 0

    # E1: overall comparison — 7 methods × 3 budgets, canonical settings.
    for seed, method, budget in product(REALCAL_SEEDS, E1_METHODS, BUDGET_RATIOS):
        cells.append(_v2_cell("E1_OVERALL", order, seed, method, budget=budget))
        order += 1

    # E1B (V2.1): tight budgets — the dual controller's intended regime.
    for seed, method, budget in product(REALCAL_SEEDS, E1B_METHODS, E1B_BUDGETS):
        cells.append(_v2_cell("E1B_TIGHT", order, seed, method, budget=budget))
        order += 1

    # E3: pre-sale screening under anchor contamination — with/without screening.
    for seed, method, contamination in product(REALCAL_SEEDS, ("V2-FULL", "B-NOSCREEN"), E3_CONTAMINATION):
        cells.append(_v2_cell("E3_SCREENING", order, seed, method, contamination=contamination))
        order += 1

    # E5: online adaptability — arrival × delay × missing (non-canonical points).
    for seed, arrival, delay, missing in product(REALCAL_SEEDS, ARRIVAL_MULTIPLIERS, DELAY_LEVELS, MISSING_LEVELS):
        if arrival == Decimal("1") and delay == 0 and missing == Decimal("0"):
            continue
        cells.append(_v2_cell("E5_ADAPT", order, seed, "V2-FULL", arrival=arrival, delay=delay, missing=missing))
        order += 1

    # E6: ablations not already covered by E1 (V2-FULL / B-NODUAL / B-MYOPIC).
    for seed, method in product(REALCAL_SEEDS, E6_EXTRA_METHODS):
        cells.append(_v2_cell("E6_ABLATION", order, seed, method))
        order += 1

    # E7: parameter sensitivity — one factor at a time around the defaults.
    for seed, gamma in product(REALCAL_SEEDS, E7_GAMMAS):
        cells.append(_v2_cell("E7_SENS_GAMMA", order, seed, "V2-FULL", gamma=gamma))
        order += 1
    for seed, alpha in product(REALCAL_SEEDS, E7_ALPHAS):
        cells.append(_v2_cell("E7_SENS_ALPHA", order, seed, "V2-FULL", alpha=alpha))
        order += 1
    for seed, theta_a in product(REALCAL_SEEDS, E7_THETA_AS):
        cells.append(_v2_cell("E7_SENS_THETAA", order, seed, "V2-FULL", theta_a=theta_a))
        order += 1
    for seed, lambda_max in product(REALCAL_SEEDS, E7_LAMBDA_MAXES):
        cells.append(_v2_cell("E7_SENS_LMAX", order, seed, "V2-FULL", lambda_max=lambda_max))
        order += 1

    return tuple(cells)


def is_regret_cell(cell: RunCell) -> bool:
    """Canonical cells that get the prefix-LP empirical-regret curve (E5)."""
    return (
        cell.family == "E1_OVERALL"
        and cell.method_id == "V2-FULL"
        and cell.gamma == DEFAULT_GAMMA_V2
        and cell.budget_ratio == DEFAULT_BUDGET_V2
    )


def count_run_cells() -> dict[str, int]:
    matrix = build_run_matrix()
    by_family: dict[str, int] = {}
    for cell in matrix:
        by_family[cell.family] = by_family.get(cell.family, 0) + 1
    return {"total": len(matrix), **by_family}


def count_run_cells_v2() -> dict[str, int]:
    matrix = build_run_matrix_v2()
    by_family: dict[str, int] = {}
    for cell in matrix:
        by_family[cell.family] = by_family.get(cell.family, 0) + 1
    return {"total": len(matrix), **by_family}

