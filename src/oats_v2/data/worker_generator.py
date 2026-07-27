from __future__ import annotations

from decimal import Decimal

from .schemas import TraceConfig
from .substreams import shuffled, weighted_choice


STRATUM_FRACTIONS = (
    ("honest", Decimal("0.60")),
    ("low-quality", Decimal("0.20")),
    ("malicious", Decimal("0.10")),
    ("camouflage", Decimal("0.10")),
)


def _counts(worker_count: int) -> dict[str, int]:
    if worker_count == 500:
        return {"honest": 300, "low-quality": 100, "malicious": 50, "camouflage": 50}
    raw = {name: int(worker_count * fraction) for name, fraction in STRATUM_FRACTIONS}
    raw["honest"] += worker_count - sum(raw.values())
    return raw


def generate_workers(seed: int, config: TraceConfig) -> list[dict[str, object]]:
    worker_ids = [f"w{index:04d}" for index in range(1, config.worker_count + 1)]
    permutation = shuffled(worker_ids, seed, "worker_permutation", "all")
    counts = _counts(config.worker_count)
    assigned: dict[str, str] = {}
    cursor = 0
    for stratum in ("honest", "low-quality", "malicious", "camouflage"):
        for worker_id in permutation[cursor : cursor + counts[stratum]]:
            assigned[worker_id] = stratum
        cursor += counts[stratum]

    rows: list[dict[str, object]] = []
    for worker_id in worker_ids:
        stratum = assigned[worker_id]
        cost = weighted_choice(
            (Decimal("0.5"), Decimal("1.0"), Decimal("2.0")),
            (Decimal("0.3"), Decimal("0.4"), Decimal("0.3")),
            seed,
            "worker_cost",
            worker_id,
        )
        bias_sign = 0
        if stratum in {"malicious", "camouflage"}:
            bias_sign = -1 if weighted_choice((0, 1), (Decimal("0.5"), Decimal("0.5")), seed, "bias_sign", worker_id) == 0 else 1
        public_role = "low" if stratum == "low-quality" else "honest"
        rows.append(
            {
                "seed": seed,
                "worker_id": worker_id,
                "stratum": stratum,
                "public_signal_role": public_role,
                "c_i": cost,
                "bias_sign": bias_sign,
            }
        )
    return rows
