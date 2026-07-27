from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from .provenance import build_holdout_provenance


def generate_holdout_provenance(
    seed: int, tasks: Iterable[Mapping[str, object]]
) -> Iterator[dict[str, object]]:
    for task in tasks:
        yield build_holdout_provenance(seed, task)
