"""Isolated published-baseline adapters released with OATS REAL-CAL.

Only the Oasis adapter used by the manuscript's frozen published-baseline
extension is part of this public scope.
"""

from .types import (
    CandidateKey,
    ExternalCandidateView,
    ExternalFeedback,
    ExternalSlotView,
)

__all__ = [
    "CandidateKey",
    "ExternalCandidateView",
    "ExternalFeedback",
    "ExternalSlotView",
]
