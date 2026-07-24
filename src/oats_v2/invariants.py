from __future__ import annotations

from typing import Iterable

from .ledger import LedgerState
from .shadow_envelope import ShadowEnvelopeState
from .types import Candidate


def assert_all(
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
    candidates: Iterable[Candidate] = (),
) -> None:
    ledger.assert_invariants()
    shadow.assert_invariants()
    for candidate in candidates:
        if candidate.bid < 0 or candidate.bid > candidate.base_cap:
            raise AssertionError("candidate bid/cap invariant broken")
