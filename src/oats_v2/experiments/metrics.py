from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from ..types import D, MechanismStatus
from .result_schema import ChannelMetrics, RunResult, ScreeningMetrics, TrustMetrics


POPULATION_WEIGHTS = {
    "honest": Decimal("0.6"),
    "low-quality": Decimal("0.2"),
    "malicious": Decimal("0.1"),
    "camouflage": Decimal("0.1"),
}


def _mean(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values, Decimal("0")) / Decimal(len(values))


def compute_channels(records: list[dict[str, Any]]) -> ChannelMetrics:
    pop = [D(r["potential_score"]) for r in records if "potential_score" in r]
    sel = [D(r["realized_score"]) for r in records if r.get("selected")]
    comp = []
    type_means: dict[str, list[Decimal]] = defaultdict(list)
    for r in records:
        if r.get("selected"):
            type_means[r["stratum"]].append(D(r["realized_score"]))
    weights = POPULATION_WEIGHTS
    if type_means:
        total_w = Decimal("0")
        weighted = Decimal("0")
        for stratum, vals in type_means.items():
            w = weights.get(stratum)
            if w is None or not vals:
                continue
            total_w += w
            weighted += w * _mean(vals)
        if total_w > 0:
            comp.append(weighted / total_w)
    within = [
        D(r["effort_delta_quality"])
        for r in records
        if r.get("selected") and "effort_delta_quality" in r
    ]
    return ChannelMetrics(
        population_quality=_mean(pop),
        selected_quality=_mean(sel),
        composition_standardized_quality=_mean(comp),
        effort_effect=_mean(within),
        screening_effect=None,
        payment_effect=None,
    )


def compute_screening_metrics(events: list[dict[str, Any]]) -> ScreeningMetrics:
    def rate(num: int, den: int) -> Decimal | None:
        if den == 0:
            return None
        return Decimal(num) / Decimal(den)

    pass_statuses = {MechanismStatus.SCREEN_PASS.value, MechanismStatus.SCREEN_SOFT_PASS.value}
    m_num = m_den = h_num = h_den = 0
    rare_num = rare_den = coll_pass = coll_den = honest_fail = honest_den = 0
    cold = cold_den = soft = 0
    for event in events:
        stratum = event.get("stratum")
        if event.get("screened"):
            status = event.get("status")
            if stratum == "malicious":
                coll_den += 1
                coll_pass += int(status in pass_statuses)
            if stratum in ("honest", "low-quality"):
                h_den += 1
                h_num += int(status == MechanismStatus.SCREEN_FAIL_COMPLIANT.value)
            if stratum == "honest":
                honest_den += 1
                honest_fail += int(status == MechanismStatus.SCREEN_FAIL_COMPLIANT.value)
            if event.get("rare_event"):
                rare_den += 1
                rare_num += int(status == MechanismStatus.SCREEN_FAIL_COMPLIANT.value)
            if stratum == "malicious":
                m_den += 1
                m_num += int(status == MechanismStatus.SCREEN_FAIL_COMPLIANT.value)
            if status == MechanismStatus.COLD_START.value:
                cold += 1
            if status == MechanismStatus.SCREEN_SOFT_PASS.value:
                soft += 1
            cold_den += 1
    return ScreeningMetrics(
        mrr=rate(m_num, m_den),
        hfr=rate(h_num, h_den),
        rare_event_rejection=rate(rare_num, rare_den),
        colluder_pass=rate(coll_pass, coll_den),
        honest_fail=rate(honest_fail, honest_den),
        cold_start_rate=rate(cold, cold_den),
        soft_pass_rate=rate(soft, cold_den),
    )


def compute_trust_metrics(
    trust_events: list[dict[str, Any]],
    population_size: int,
    final_trust_by_stratum: dict[str, list[Decimal]] | None = None,
) -> TrustMetrics:
    selected = sum(1 for e in trust_events if e.get("selected"))
    feedback = sum(1 for e in trust_events if e.get("feedback"))
    transitions = sum(
        1 for e in trust_events if e.get("feedback") and e.get("trust_transition_applied")
    )
    duplicate_suppressions = sum(
        1 for e in trust_events if e.get("feedback") and e.get("duplicate_feedback_suppressed")
    )
    qualities = [D(e["quality"]) for e in trust_events if e.get("feedback") and "quality" in e]
    trust_vals = [D(e["rho"]) for e in trust_events if e.get("feedback") and "rho" in e]
    brier = None
    if qualities and trust_vals and len(qualities) == len(trust_vals):
        errs = [(q - t) ** 2 for q, t in zip(qualities, trust_vals)]
        brier = sum(errs, Decimal("0")) / Decimal(len(errs))
    auc = None
    if final_trust_by_stratum:
        # AUC = P(trust_honest > trust_bad) + 0.5 P(tie): can the final trust
        # ranking separate honest workers from malicious/camouflage workers?
        positives = final_trust_by_stratum.get("honest", [])
        negatives = final_trust_by_stratum.get("malicious", []) + final_trust_by_stratum.get("camouflage", [])
        if positives and negatives:
            wins = ties = 0
            for p in positives:
                for n in negatives:
                    if p > n:
                        wins += 1
                    elif p == n:
                        ties += 1
            total = len(positives) * len(negatives)
            auc = (Decimal(wins) + Decimal(ties) / 2) / Decimal(total)
    return TrustMetrics(
        population_available=population_size,
        selected_count=selected,
        feedback_count=feedback,
        trust_transition_count=transitions,
        duplicate_feedback_suppressed_count=duplicate_suppressions,
        brier=brier,
        auc=auc,
    )


def spearman_and_topk(
    estimated: list[Decimal], realized: list[Decimal], *, top_fraction: Decimal = Decimal("0.1")
) -> tuple[Decimal | None, Decimal | None]:
    """E2 value-identification metrics: Spearman rho between the mechanism's
    pre-purchase value estimates and the realized post-use marginal value, plus
    top-k overlap (k = top_fraction of purchases)."""
    n = len(estimated)
    if n < 3 or n != len(realized):
        return None, None

    def ranks(values: list[Decimal]) -> list[Decimal]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [Decimal("0")] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (Decimal(i) + Decimal(j)) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = avg_rank
            i = j + 1
        return result

    rx = ranks(estimated)
    ry = ranks(realized)
    mean = (Decimal(n) + 1) / 2
    num = sum(((rx[i] - mean) * (ry[i] - mean) for i in range(n)), Decimal("0"))
    den_x = sum(((rx[i] - mean) ** 2 for i in range(n)), Decimal("0"))
    den_y = sum(((ry[i] - mean) ** 2 for i in range(n)), Decimal("0"))
    if den_x == 0 or den_y == 0:
        return None, None
    rho = num / (den_x.sqrt() * den_y.sqrt())

    k = max(1, int(n * float(top_fraction)))
    top_est = set(sorted(range(n), key=lambda i: estimated[i], reverse=True)[:k])
    top_real = set(sorted(range(n), key=lambda i: realized[i], reverse=True)[:k])
    overlap = Decimal(len(top_est & top_real)) / Decimal(k)
    return rho, overlap


def finalize_run_result(
    result: RunResult,
    *,
    records: list[dict[str, Any]],
    screening_events: list[dict[str, Any]],
    trust_events: list[dict[str, Any]],
    population_size: int,
    final_trust_by_stratum: dict[str, list[Decimal]] | None = None,
) -> RunResult:
    result.channels = compute_channels(records)
    result.screening = compute_screening_metrics(screening_events)
    result.trust = compute_trust_metrics(trust_events, population_size, final_trust_by_stratum)
    composition = Counter(r["stratum"] for r in records if r.get("selected"))
    total = sum(composition.values())
    if total:
        result.worker_type_composition = {
            k: Decimal(v) / Decimal(total) for k, v in composition.items()
        }
    return result
