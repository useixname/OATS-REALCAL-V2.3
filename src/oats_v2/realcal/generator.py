"""Profile-driven trace generators for REAL-CAL-V1.

Mirrors the SYN-V2-1 generators field-for-field (identical output schema, so the
existing ``trace_loader`` / ``formal_runner`` consume it unchanged) but swaps the
INPUT DISTRIBUTIONS for real-calibrated ones from ``RealCalConfig``:

  * worker strata          <- profile.stratum_fractions   (PurpleAir tiers)
  * per-slot availability  <- profile.availability * arrival_intensity[slot]  (T-Drive)
  * task difficulty theta  <- Beta matched to profile theta mean/std          (Beijing air)
  * task missingness       <- profile.missing_rate                            (Beijing air)
  * report noise (sigma)   <- profile.report_cv(tier)                         (PurpleAir)
  * bias sign split        <- profile.bias_sign_positive_fraction            (PurpleAir)

Frozen-by-design constants (cost support, value band, capacities, effort menu,
gamma grid, money grid, base cap) are kept identical to SYN-V2-1 so proofs,
capacities, and Phase-4A comparability hold. Determinism comes from the shared
keyed substream RNG; the calibration family token is "realcal" to keep streams
disjoint from SYN-V2-1.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Iterable, Iterator, Mapping

from ..data.schemas import EFFORTS, canonical_json, decimal_from_float
from ..data.substreams import (
    bernoulli,
    keyed_seed,
    normal,
    rng,
    stable_digest,
    uniform,
    weighted_choice,
)
from ..data.task_generator import clip01
from .config import RealCalConfig
from .score import gaussian_score, report_sigma

FAMILY = "realcal"
COST_SUPPORT = (Decimal("0.5"), Decimal("1.0"), Decimal("2.0"))
COST_WEIGHTS = (Decimal("0.3"), Decimal("0.4"), Decimal("0.3"))
CAPACITIES = (3, 5, 10)
CAPACITY_WEIGHTS = (Decimal("0.4"), Decimal("0.4"), Decimal("0.2"))
DEADLINE_SLACK = (1, 3, 5)
DEADLINE_WEIGHTS = (Decimal("0.5"), Decimal("0.3"), Decimal("0.2"))


def _beta_params(mean: float, std: float) -> tuple[float, float]:
    """Method-of-moments Beta(a,b) for a target mean/std on (0,1)."""
    var = std * std
    max_var = mean * (1.0 - mean)
    if var <= 0 or var >= max_var:
        var = max_var * 0.5
    k = mean * (1.0 - mean) / var - 1.0
    a = max(1e-3, mean * k)
    b = max(1e-3, (1.0 - mean) * k)
    return a, b


def _stratum_counts(cfg: RealCalConfig) -> dict[str, int]:
    fractions = cfg.stratum_fractions
    order = ("honest", "low-quality", "malicious", "camouflage")
    counts = {name: int(cfg.worker_count * float(fractions[name])) for name in order}
    counts["honest"] += cfg.worker_count - sum(counts.values())
    return counts


def generate_workers(seed: int, cfg: RealCalConfig) -> list[dict[str, object]]:
    worker_ids = [f"w{index:04d}" for index in range(1, cfg.worker_count + 1)]
    source = rng(seed, FAMILY + "_worker_permutation", "all")
    permutation = list(worker_ids)
    source.shuffle(permutation)
    counts = _stratum_counts(cfg)
    assigned: dict[str, str] = {}
    cursor = 0
    for stratum in ("honest", "low-quality", "malicious", "camouflage"):
        for worker_id in permutation[cursor : cursor + counts[stratum]]:
            assigned[worker_id] = stratum
        cursor += counts[stratum]

    pos_fraction = float(cfg.bias_sign_positive_fraction)
    rows: list[dict[str, object]] = []
    for worker_id in worker_ids:
        stratum = assigned[worker_id]
        cost = weighted_choice(COST_SUPPORT, COST_WEIGHTS, seed, FAMILY + "_worker_cost", worker_id)
        bias_sign = 0
        if stratum in {"malicious", "camouflage"}:
            draw = uniform(seed, FAMILY + "_bias_sign", worker_id)
            bias_sign = 1 if draw < pos_fraction else -1
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


def generate_tasks(seed: int, cfg: RealCalConfig) -> list[dict[str, object]]:
    a, b = _beta_params(float(cfg.theta_mean), float(cfg.theta_std))
    # arrival intensity modulates task count around a base Poisson-like rate of 20.
    rows: list[dict[str, object]] = []
    for slot in range(1, cfg.horizon + 1):
        intensity = float(cfg.arrival_multiplier(slot))
        base_count = 20.0 * intensity
        # Poisson draw with per-slot mean = base_count.
        source = rng(seed, FAMILY + "_task_count", slot)
        count = _poisson(source, base_count)
        for index in range(1, count + 1):
            task_id = f"t{slot:04d}_{index:03d}"
            cell = 1 + int(uniform(seed, FAMILY + "_task_cell", slot, index) * cfg.cell_count)
            theta_source = rng(seed, FAMILY + "_task_theta", slot, index)
            theta = decimal_from_float(theta_source.betavariate(a, b))
            value_draw = uniform(seed, FAMILY + "_task_value", slot, index)
            value = decimal_from_float(
                float(cfg.task_value_low)
                + value_draw * float(cfg.task_value_high - cfg.task_value_low)
            )
            requested = weighted_choice(CAPACITIES, CAPACITY_WEIGHTS, seed, FAMILY + "_task_capacity", slot, index)
            slack = weighted_choice(DEADLINE_SLACK, DEADLINE_WEIGHTS, seed, FAMILY + "_deadline_slack", slot, index)
            holdout_noise = decimal_from_float(0.01 * normal(seed, FAMILY + "_holdout_normal", slot, index))
            holdout = clip01(theta + holdout_noise)
            missing_uniform = Decimal(repr(uniform(seed, FAMILY + "_missing_mask", slot, index)))
            base_missing = float(cfg.missing_rate)
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
                        "0.1": missing_uniform < (Decimal("0.1") + Decimal(repr(base_missing))),
                        "0.3": missing_uniform < (Decimal("0.3") + Decimal(repr(base_missing))),
                    },
                }
            )
    return rows


def _poisson(source, mean: float) -> int:
    import math

    if mean <= 0:
        return 0
    limit = math.exp(-mean)
    product = 1.0
    draws = 0
    while product > limit:
        draws += 1
        product *= source.random()
    return draws - 1


def _tasks_by_slot(tasks: Iterable[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    result: dict[int, list[Mapping[str, object]]] = {}
    for task in tasks:
        result.setdefault(int(task["slot"]), []).append(task)
    for slot in result:
        result[slot].sort(key=lambda item: str(item["task_id"]))
    return result


def generate_eligibility(
    seed: int,
    cfg: RealCalConfig,
    workers: Iterable[Mapping[str, object]],
    tasks: Iterable[Mapping[str, object]],
) -> Iterator[dict[str, object]]:
    worker_ids = tuple(str(w["worker_id"]) for w in workers)
    slot_tasks = _tasks_by_slot(tasks)
    base_avail = float(cfg.availability_probability)
    for slot in range(1, cfg.horizon + 1):
        available_tasks = slot_tasks.get(slot, [])
        slot_prob = min(1.0, base_avail * float(cfg.arrival_multiplier(slot)))
        for worker_id in worker_ids:
            available = bernoulli(Decimal(repr(slot_prob)), seed, FAMILY + "_worker_availability", slot, worker_id)
            mapped_task_id = None
            if available and available_tasks:
                index = keyed_seed(seed, FAMILY + "_eligibility_map", slot, worker_id) % len(available_tasks)
                mapped_task_id = str(available_tasks[index]["task_id"])
            digest = stable_digest(seed, slot, worker_id, mapped_task_id or "NONE")
            yield {
                "seed": seed,
                "slot": slot,
                "worker_id": worker_id,
                "available": available,
                "mapped_task_id": mapped_task_id,
                "map_hash": digest,
            }


# --- anchors & potential reports -------------------------------------------
from statistics import median  # noqa: E402

from ..data.anchor_history_generator import FrozenAnchorBounds  # noqa: E402


def _report_sigma_from_cv(cv: Decimal, theta: Decimal) -> Decimal:
    """Multiplicative report noise on the [0,1] signal scale, matching the
    PurpleAir ratio-error interpretation (sigma scales with the signal)."""
    return cv * theta


def generate_anchor_history(
    seed: int, cfg: RealCalConfig
) -> tuple[list[dict[str, object]], dict[int, FrozenAnchorBounds]]:
    rows: list[dict[str, object]] = []
    bounds: dict[int, FrozenAnchorBounds] = {}
    version = f"real-cal-v1-anchor-{seed}-v1"
    honest_cv = cfg.honest_report_cv
    a, b = _beta_params(float(cfg.theta_mean), float(cfg.theta_std))
    anchor_count = 50
    for cell in range(1, cfg.cell_count + 1):
        cell_reports: list[Decimal] = []
        for index in range(1, anchor_count + 1):
            theta_source = rng(seed, FAMILY + "_anchor_theta", cell, index)
            theta = decimal_from_float(theta_source.betavariate(a, b))
            sigma = _report_sigma_from_cv(honest_cv, theta)
            zeta = decimal_from_float(normal(seed, FAMILY + "_anchor_report_normal", cell, index))
            report = clip01(theta + sigma * zeta)
            holdout = clip01(theta + Decimal("0.01") * decimal_from_float(normal(seed, FAMILY + "_anchor_holdout_normal", cell, index)))
            commitment_input = {
                "seed": seed,
                "cell": cell,
                "anchor_index": index,
                "report": report,
                "z": holdout,
                "anchor_version": version,
            }
            commitment = hashlib.sha256(canonical_json(commitment_input).encode("utf-8")).hexdigest()
            rows.append(
                {
                    "seed": seed,
                    "cell": cell,
                    "anchor_index": index,
                    "report": report,
                    "z": holdout,
                    "eligible": True,
                    "anchor_version": version,
                    "commitment": commitment,
                }
            )
            cell_reports.append(report)
        center = median(cell_reports)
        mad = median(abs(value - center) for value in cell_reports)
        scale = max(mad, Decimal("0.01"))
        bounds[cell] = FrozenAnchorBounds(
            cell=cell,
            version=version,
            center=center,
            mad=mad,
            sigma=scale,
            lower=center - Decimal("3") * scale,
            upper=center + Decimal("3") * scale,
        )
    return rows, bounds


def _signal_quality_cv(cfg: RealCalConfig, stratum: str) -> Decimal:
    return cfg.report_cv("low-quality" if stratum == "low-quality" else "honest")


def _malicious_mode(stratum: str, slot: int) -> bool:
    return stratum == "malicious" or (stratum == "camouflage" and slot >= 501)


def generate_potential_reports(
    seed: int,
    cfg: RealCalConfig,
    workers: Iterable[Mapping[str, object]],
    tasks: Iterable[Mapping[str, object]],
    available_mappings: Iterable[Mapping[str, object]],
    anchor_bounds: Mapping[int, FrozenAnchorBounds],
) -> Iterator[dict[str, object]]:
    workers_by_id = {str(w["worker_id"]): w for w in workers}
    tasks_by_id = {str(t["task_id"]): t for t in tasks}
    for mapping in available_mappings:
        if not bool(mapping["available"]) or mapping["mapped_task_id"] is None:
            continue
        worker_id = str(mapping["worker_id"])
        task_id = str(mapping["mapped_task_id"])
        worker = workers_by_id[worker_id]
        task = tasks_by_id[task_id]
        slot = int(task["slot"])
        stratum = str(worker["stratum"])
        manipulation_draw = uniform(seed, FAMILY + "_manipulation_uniform", slot, task_id, worker_id)
        deviation = Decimal("0")
        if _malicious_mode(stratum, slot) and manipulation_draw < 0.8:
            deviation = Decimal("0.4") * Decimal(int(worker["bias_sign"]))
        bounds = anchor_bounds[int(task["cell"])]
        theta = Decimal(task["theta"])
        # effort still lifts effective quality (modeled effort curve), realised via
        # a per-effort noise reduction consistent with the frozen effort menu.
        cv = _signal_quality_cv(cfg, stratum)
        base_noise = _report_sigma_from_cv(cv, theta)
        for effort in EFFORTS:
            # Effort reduces report noise (public signal technology). Uses the same
            # reduction schedule the REAL-CAL calibration assumes, so Gbar and the
            # realized reports are mutually consistent.
            sigma = report_sigma(base_noise, effort)
            zeta = decimal_from_float(normal(seed, FAMILY + "_potential_report_normal", slot, task_id, worker_id, str(effort)))
            report = clip01(theta + sigma * zeta + deviation)
            score = gaussian_score(report, Decimal(task["z"]))
            status = "PASS" if bounds.lower <= report <= bounds.upper else "FAIL"
            value = Decimal("0")
            if status == "PASS":
                value = Decimal(task["V"]) / Decimal(int(task["K"])) * score
            yield {
                "seed": seed,
                "slot": slot,
                "task_id": task_id,
                "worker_id": worker_id,
                "effort": effort,
                "report": report,
                "score": score,
                "screen_status": status,
                "v_ijt": value,
            }
