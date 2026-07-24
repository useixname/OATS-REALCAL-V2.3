"""Phase 4A formal R4 / IDEAL_S2 mechanism experiment infrastructure."""

from .audit import run_baseline_parity_audit, run_online_firewall_audit
from .formal_runner import FormalRunner
from .run_matrix import build_run_matrix, count_run_cells

__all__ = [
    "FormalRunner",
    "build_run_matrix",
    "count_run_cells",
    "run_baseline_parity_audit",
    "run_online_firewall_audit",
]
