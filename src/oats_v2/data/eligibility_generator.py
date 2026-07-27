from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator, Mapping

from .schemas import TraceConfig
from .substreams import bernoulli, keyed_seed, stable_digest


def tasks_by_slot(tasks: Iterable[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    result: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for task in tasks:
        result[int(task["slot"])].append(task)
    for slot in result:
        result[slot].sort(key=lambda item: str(item["task_id"]))
    return result


def generate_eligibility(
    seed: int,
    config: TraceConfig,
    workers: Iterable[Mapping[str, object]],
    tasks: Iterable[Mapping[str, object]],
) -> Iterator[dict[str, object]]:
    worker_ids = tuple(str(worker["worker_id"]) for worker in workers)
    slot_tasks = tasks_by_slot(tasks)
    for slot in range(1, config.horizon + 1):
        available_tasks = slot_tasks.get(slot, [])
        for worker_id in worker_ids:
            available = bernoulli(config.availability_probability, seed, "worker_availability", slot, worker_id)
            mapped_task_id: str | None = None
            if available and available_tasks:
                index = keyed_seed(seed, "eligibility_map", slot, worker_id) % len(available_tasks)
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
