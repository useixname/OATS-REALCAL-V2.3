"""Executable OATS-V2 restricted mechanism core.

This package implements the Phase-3A reference implementation path.  The ideal
range screen is not a cryptographic backend and no regret theorem is exposed.
"""

from .allocation import allocate
from .contracts import ContinuationTable, TaskContract
from .critical_payment import all_critical_values, critical_value
from .ledger import LedgerState
from .shadow_envelope import ShadowEnvelopeState
from .types import Candidate, MechanismStatus, Task

__all__ = [
    "Candidate",
    "ContinuationTable",
    "LedgerState",
    "MechanismStatus",
    "ShadowEnvelopeState",
    "Task",
    "TaskContract",
    "allocate",
    "all_critical_values",
    "critical_value",
]
