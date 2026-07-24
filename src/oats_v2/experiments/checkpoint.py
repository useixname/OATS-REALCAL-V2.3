from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointState:
    run_version: str
    completed_cells: list[str]
    invalid_cells: list[dict[str, Any]]
    last_order_index: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CheckpointState:
        if not path.exists():
            return cls(run_version="", completed_cells=[], invalid_cells=[], last_order_index=-1)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def is_complete(self, cell_id: str) -> bool:
        return cell_id in self.completed_cells

    def mark_complete(self, cell_id: str, order_index: int) -> None:
        if cell_id not in self.completed_cells:
            self.completed_cells.append(cell_id)
        self.last_order_index = max(self.last_order_index, order_index)

    def mark_invalid(self, cell_id: str, reason: str, order_index: int) -> None:
        self.invalid_cells.append({"cell_id": cell_id, "reason": reason, "order_index": order_index})
