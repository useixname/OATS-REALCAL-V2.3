from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class CertifiedOracleResult:
    objective: Decimal
    solution: tuple[str, ...]
    active_tasks: tuple[str, ...]
    envelope_used: Decimal
    optimality_status: str
    certified_gap: Decimal | None
    runtime_seconds: float
    explored_states: int
    explored_nodes: int
    deterministic_tie_break: str = "lexicographically-smallest sorted contract-id tuple"

    @property
    def is_exact(self) -> bool:
        return self.optimality_status.startswith("OPTIMAL") and self.certified_gap == 0
