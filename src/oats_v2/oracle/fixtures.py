from __future__ import annotations

from decimal import Decimal

from .problem import LocalContract, LocalOracleProblem, LocalTask


def hand_computed_basic() -> tuple[LocalOracleProblem, tuple[str, ...], Decimal]:
    problem = LocalOracleProblem(
        tasks=(LocalTask("t", Decimal("1"), 2),),
        contracts=(
            LocalContract("a", "wa", "t", Decimal("5"), Decimal("2"), Decimal("2")),
            LocalContract("b", "wb", "t", Decimal("3"), Decimal("1"), Decimal("1")),
        ),
        remaining_shadow_capacity=Decimal("4"),
        dual_lambda=Decimal("0.5"),
    )
    # Both fit: objective=(5-1)+(3-.5)-.5=6.
    return problem, ("a", "b"), Decimal("6.0")


def ratio_gap(scale: int = 100) -> LocalOracleProblem:
    amount = Decimal(scale)
    return LocalOracleProblem(
        tasks=(LocalTask("t", Decimal("0"), 2),),
        contracts=(
            LocalContract("large", "wl", "t", amount, amount, amount),
            LocalContract("small", "ws", "t", Decimal("2"), Decimal("1"), Decimal("1")),
        ),
        remaining_shadow_capacity=amount,
        dual_lambda=Decimal("0"),
    )


def activation_complementarity() -> LocalOracleProblem:
    return LocalOracleProblem(
        tasks=(LocalTask("a", Decimal("9"), 1), LocalTask("b", Decimal("0"), 5)),
        contracts=(
            LocalContract("fixed-heavy", "wa", "a", Decimal("4"), Decimal("1"), Decimal("1")),
            *tuple(
                LocalContract(f"b{index}", f"wb{index}", "b", Decimal("3"), Decimal("2"), Decimal("2"))
                for index in range(5)
            ),
        ),
        remaining_shadow_capacity=Decimal("10"),
        dual_lambda=Decimal("0"),
    )
