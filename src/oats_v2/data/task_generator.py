from __future__ import annotations

from decimal import Decimal

from .schemas import TraceConfig, decimal_from_float
from .substreams import beta22, normal, poisson20, uniform, weighted_choice


def clip01(value: Decimal) -> Decimal:
    return min(Decimal("1"), max(Decimal("0"), value))


def generate_tasks(seed: int, config: TraceConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot in range(1, config.horizon + 1):
        count = poisson20(seed, "task_count", slot)
        for index in range(1, count + 1):
            task_id = f"t{slot:04d}_{index:03d}"
            cell = 1 + int(uniform(seed, "task_cell", slot, index) * config.cell_count)
            theta = decimal_from_float(beta22(seed, "task_theta", slot, index))
            value_draw = uniform(seed, "task_value", slot, index)
            value = decimal_from_float(float(config.task_value_low) + value_draw * float(config.task_value_high - config.task_value_low))
            requested = weighted_choice(
                (3, 5, 10),
                (Decimal("0.4"), Decimal("0.4"), Decimal("0.2")),
                seed,
                "task_capacity",
                slot,
                index,
            )
            slack = weighted_choice(
                (1, 3, 5),
                (Decimal("0.5"), Decimal("0.3"), Decimal("0.2")),
                seed,
                "deadline_slack",
                slot,
                index,
            )
            holdout_noise = decimal_from_float(0.01 * normal(seed, "holdout_normal", slot, index))
            holdout = clip01(theta + holdout_noise)
            missing_uniform = Decimal(repr(uniform(seed, "missing_mask", slot, index)))
            rows.append(
                {
                    "seed": seed,
                    "slot": slot,
                    "task_id": task_id,
                    "cell": cell,
                    "theta": theta,
                    "V": value,
                    "K": requested,
                    "deadline": slot + slack,
                    "z": holdout,
                    "delay_mask": {"0": 0, "5": 5, "20": 20},
                    "missing_mask": {
                        "0": False,
                        "0.1": missing_uniform < Decimal("0.1"),
                        "0.3": missing_uniform < Decimal("0.3"),
                    },
                }
            )
    return rows
