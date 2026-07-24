from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from src.oats_v2.types import D


@dataclass(frozen=True)
class LocalTask:
    task_id: str
    activation_cost: Decimal
    capacity: int
    active: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation_cost", D(self.activation_cost))
        if self.activation_cost < 0 or self.capacity < 1:
            raise ValueError("invalid local task")


@dataclass(frozen=True)
class LocalContract:
    contract_id: str
    worker_id: str
    task_id: str
    vhat: Decimal
    envelope: Decimal
    estimated_transfer: Decimal
    reported_bid: Decimal = Decimal("0")
    deadline_ok: bool = True
    domain_ok: bool = True

    def __post_init__(self) -> None:
        for name in ("vhat", "envelope", "estimated_transfer", "reported_bid"):
            object.__setattr__(self, name, D(getattr(self, name)))
        if any(getattr(self, name) < 0 for name in ("vhat", "envelope", "estimated_transfer", "reported_bid")):
            raise ValueError("negative local contract field")


@dataclass(frozen=True)
class LocalOracleProblem:
    tasks: tuple[LocalTask, ...]
    contracts: tuple[LocalContract, ...]
    remaining_shadow_capacity: Decimal
    dual_lambda: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "remaining_shadow_capacity", D(self.remaining_shadow_capacity))
        object.__setattr__(self, "dual_lambda", D(self.dual_lambda))
        if self.remaining_shadow_capacity < 0 or self.dual_lambda < 0:
            raise ValueError("invalid local problem")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate task IDs")
        known = set(task_ids)
        if any(contract.task_id not in known for contract in self.contracts):
            raise ValueError("contract references unknown task")

    @property
    def tasks_by_id(self) -> dict[str, LocalTask]:
        return {task.task_id: task for task in self.tasks}

    def eligible_contracts(self) -> tuple[LocalContract, ...]:
        tasks = self.tasks_by_id
        return tuple(
            sorted(
                (
                    contract
                    for contract in self.contracts
                    if contract.deadline_ok
                    and contract.domain_ok
                    and tasks[contract.task_id].active
                    and tasks[contract.task_id].deadline_ok
                ),
                key=lambda item: item.contract_id,
            )
        )

    def evaluate(self, contract_ids: Iterable[str]) -> tuple[Decimal, Decimal, tuple[str, ...]]:
        selected_ids = tuple(sorted(contract_ids))
        by_id = {contract.contract_id: contract for contract in self.eligible_contracts()}
        if any(contract_id not in by_id for contract_id in selected_ids):
            raise ValueError("solution contains ineligible contract")
        selected = tuple(by_id[contract_id] for contract_id in selected_ids)
        if len({contract.worker_id for contract in selected}) != len(selected):
            raise ValueError("worker capacity violated")
        task_counts: dict[str, int] = {}
        for contract in selected:
            task_counts[contract.task_id] = task_counts.get(contract.task_id, 0) + 1
        tasks = self.tasks_by_id
        if any(count > tasks[task_id].capacity for task_id, count in task_counts.items()):
            raise ValueError("task capacity violated")
        active_tasks = tuple(sorted(task_counts))
        envelope = sum((contract.envelope for contract in selected), Decimal("0")) + sum(
            (tasks[task_id].activation_cost for task_id in active_tasks), Decimal("0")
        )
        if envelope > self.remaining_shadow_capacity:
            raise ValueError("remaining shadow capacity violated")
        objective = sum(
            (contract.vhat - self.dual_lambda * contract.envelope for contract in selected),
            Decimal("0"),
        ) - self.dual_lambda * sum(
            (tasks[task_id].activation_cost for task_id in active_tasks), Decimal("0")
        )
        return objective, envelope, active_tasks
