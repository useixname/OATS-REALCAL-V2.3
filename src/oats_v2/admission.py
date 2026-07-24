from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Mapping

from .ledger import LedgerState
from .shadow_envelope import ShadowEnvelopeState
from .types import Candidate, MechanismStatus, Money, SelectionOutcome


def admit_atomic(
    selection: SelectionOutcome,
    candidates: Iterable[Candidate],
    task_keys: Mapping[str, str],
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
) -> MechanismStatus:
    if selection.status is not MechanismStatus.ACTIVE:
        return selection.status
    by_key = {candidate.key: candidate for candidate in candidates}
    winners = selection.allocation.winners
    bases = {key: selection.critical_bases[key] for key in winners}
    caps = {key: by_key[key].base_cap for key in winners}
    if any(bases[key] < 0 or bases[key] > caps[key] for key in winners):
        return MechanismStatus.CAP_INVARIANT_BROKEN
    if sum(bases.values(), Decimal("0")) > ledger.free:
        return MechanismStatus.CAP_INVARIANT_BROKEN
    if sum(caps.values(), Decimal("0")) > shadow.free:
        return MechanismStatus.CAP_INVARIANT_BROKEN
    used_task_keys = tuple(sorted({task_keys[by_key[key].task_id] for key in winners}))
    ledger_snapshot = (
        ledger.free,
        ledger.locked_base,
        dict(ledger.base_locks),
    )
    try:
        ledger.lock_bases_atomic(bases)
        shadow.commit_admission(used_task_keys, caps)
    except Exception:
        ledger.free, ledger.locked_base, old_locks = ledger_snapshot
        ledger.base_locks = old_locks
        ledger.assert_invariants()
        return MechanismStatus.CAP_INVARIANT_BROKEN
    return MechanismStatus.ACTIVE
