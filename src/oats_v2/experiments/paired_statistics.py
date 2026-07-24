from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


BOOTSTRAP_SEED = 990000001
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    baseline: str
    method: str
    seed_values: dict[int, Decimal]
    mean_diff: Decimal | None
    median_diff: Decimal | None
    ci_low: Decimal | None
    ci_high: Decimal | None
    conclusion: str


def _percentile(sorted_vals: list[Decimal], q: float) -> Decimal:
    if not sorted_vals:
        return Decimal("0")
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def paired_bootstrap_ci(
    diffs: Sequence[Decimal],
    *,
    rng: random.Random,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    if not diffs:
        return Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")
    mean_diff = sum(diffs, Decimal("0")) / Decimal(len(diffs))
    sorted_diffs = sorted(diffs)
    median_diff = sorted_diffs[len(sorted_diffs) // 2]
    boot_means: list[Decimal] = []
    n = len(diffs)
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample, Decimal("0")) / Decimal(n))
    boot_means.sort()
    ci_low = _percentile(boot_means, 0.025)
    ci_high = _percentile(boot_means, 0.975)
    return mean_diff, median_diff, ci_low, ci_high


def compare_paired(
    metric: str,
    baseline: str,
    method: str,
    baseline_by_seed: dict[int, Decimal],
    method_by_seed: dict[int, Decimal],
    *,
    rng: random.Random | None = None,
) -> PairedComparison:
    rng = rng or random.Random(BOOTSTRAP_SEED)
    seeds = sorted(set(baseline_by_seed) & set(method_by_seed))
    diffs = [method_by_seed[s] - baseline_by_seed[s] for s in seeds]
    mean_diff, median_diff, ci_low, ci_high = paired_bootstrap_ci(diffs, rng=rng)
    if ci_low <= Decimal("0") <= ci_high:
        conclusion = "inconclusive"
    elif mean_diff > 0:
        conclusion = "method_higher"
    else:
        conclusion = "baseline_higher"
    return PairedComparison(
        metric=metric,
        baseline=baseline,
        method=method,
        seed_values={s: method_by_seed[s] - baseline_by_seed[s] for s in seeds},
        mean_diff=mean_diff,
        median_diff=median_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        conclusion=conclusion,
    )


def holm_adjustment(p_values: list[tuple[str, Decimal]]) -> list[tuple[str, Decimal, bool]]:
    """Return (name, adjusted_threshold, reject) using Holm step-down on pseudo-p from CI."""
    m = len(p_values)
    sorted_vals = sorted(p_values, key=lambda x: x[1])
    results: list[tuple[str, Decimal, bool]] = []
    for rank, (name, p) in enumerate(sorted_vals, start=1):
        threshold = Decimal(rank) / Decimal(m)
        results.append((name, threshold, p <= threshold))
    return results
