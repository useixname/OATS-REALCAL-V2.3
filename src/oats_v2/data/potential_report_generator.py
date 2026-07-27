from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Iterator, Mapping

from .anchor_history_generator import FrozenAnchorBounds, honest_quality, report_sigma
from .schemas import EFFORTS, decimal_from_float
from .substreams import normal, uniform
from .task_generator import clip01


def signal_quality(stratum: str, effort: Decimal) -> Decimal:
    if stratum == "low-quality":
        return Decimal("0.4") + Decimal("0.2") * effort
    return honest_quality(effort)


def malicious_mode(stratum: str, slot: int) -> bool:
    return stratum == "malicious" or (stratum == "camouflage" and slot >= 501)


def point_score(report: Decimal, holdout: Decimal) -> Decimal:
    return Decimal("1") - (report - holdout) ** 2


def generate_potential_reports(
    seed: int,
    workers: Iterable[Mapping[str, object]],
    tasks: Iterable[Mapping[str, object]],
    eligibility: Iterable[Mapping[str, object]],
    anchor_bounds: Mapping[int, FrozenAnchorBounds],
) -> Iterator[dict[str, object]]:
    workers_by_id = {str(row["worker_id"]): row for row in workers}
    tasks_by_id = {str(row["task_id"]): row for row in tasks}
    for mapping in eligibility:
        if not bool(mapping["available"]) or mapping["mapped_task_id"] is None:
            continue
        worker_id = str(mapping["worker_id"])
        task_id = str(mapping["mapped_task_id"])
        worker = workers_by_id[worker_id]
        task = tasks_by_id[task_id]
        slot = int(task["slot"])
        manipulation_draw = uniform(seed, "manipulation_uniform", slot, task_id, worker_id)
        deviation = Decimal("0")
        if malicious_mode(str(worker["stratum"]), slot) and manipulation_draw < 0.8:
            deviation = Decimal("0.4") * Decimal(int(worker["bias_sign"]))
        bounds = anchor_bounds[int(task["cell"])]
        for effort in EFFORTS:
            quality = signal_quality(str(worker["stratum"]), effort)
            zeta = decimal_from_float(normal(seed, "potential_report_normal", slot, task_id, worker_id, str(effort)))
            report = clip01(Decimal(task["theta"]) + report_sigma(quality) * zeta + deviation)
            score = point_score(report, Decimal(task["z"]))
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
