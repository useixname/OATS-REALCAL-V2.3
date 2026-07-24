from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING
from typing import Iterable, Mapping

from src.oats_v2.data.schemas import GBAR_PRECISION, TraceConfig, decimal_text


CERTIFICATE_VERSION = "syn-v2-1-hoeffding-union-v1"
GLOBAL_CONFIDENCE = Decimal("0.99")


def hoeffding_radius(config: TraceConfig) -> Decimal:
    # Simultaneous family: 100 cells * 2 roles * 3 efforts *
    # (pibar and bounded I(PASS)*score) = 1200 quantities.
    quantity_count = config.cell_count * 2 * 3 * 2
    alpha = Decimal("1") - GLOBAL_CONFIDENCE
    value = math.sqrt(math.log((2.0 * quantity_count) / float(alpha)) / (2.0 * config.calibration_n))
    return Decimal(repr(value)).quantize(Decimal("0.000000001"), rounding=ROUND_CEILING)


def uniform_epsilon(config: TraceConfig) -> Decimal:
    # Maximum sbar is gamma_max*V_max/K_min = 1*1.5/3 = 0.5.
    # Include half a published 1e-6 quantization unit.
    return (Decimal("0.5") * hoeffding_radius(config) + GBAR_PRECISION / 2).quantize(
        Decimal("0.000000001"), rounding=ROUND_CEILING
    )


def build_certificate(
    *,
    seed: int,
    cell: int,
    role: str,
    table_hash: str,
    data_hash: str,
    code_hash: str,
    sample_counts: Iterable[int],
    intervals: Mapping[str, Mapping[str, Decimal]],
    config: TraceConfig,
    anchor_version: str,
    signature: str,
) -> dict[str, object]:
    radius = hoeffding_radius(config)
    epsilon = uniform_epsilon(config)
    return {
        "certificate_version": CERTIFICATE_VERSION,
        "seed": seed,
        "cell": cell,
        "role": role,
        "effort_menu": ["0", "0.5", "1"],
        "sample_counts": list(sample_counts),
        "confidence": decimal_text(GLOBAL_CONFIDENCE),
        "simultaneous_family_size": config.cell_count * 2 * 3 * 2,
        "correction": "HOEFFDING_UNION_BOUND_GLOBAL_SEED_FAMILY",
        "bounded_quantity_domain": "[0,1]",
        "raw_simultaneous_radius": decimal_text(radius),
        "max_sbar": "0.5",
        "rounding_allowance": decimal_text(GBAR_PRECISION / 2),
        "uniform_epsilon_Gbar": decimal_text(epsilon),
        "claim": "With probability at least 0.99 over the preregistered calibration sample, sup over all registered cell/role/effort Gbar errors is at most uniform_epsilon_Gbar",
        "intervals": {
            effort: {name: decimal_text(value) for name, value in values.items()}
            for effort, values in intervals.items()
        },
        "data_hash": data_hash,
        "code_hash": code_hash,
        "table_hash": table_hash,
        "anchor_version": anchor_version,
        "precision": decimal_text(GBAR_PRECISION),
        "signature": signature,
        "contract_internal_ic": "EXACT_RELATIVE_TO_PUBLISHED_GBAR",
        "deployment_relative_ic": "EPSILON_IC_USING_UNIFORM_EPSILON_GBAR",
    }
