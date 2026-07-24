"""REAL-CAL-V1 report model and discriminating bounded score.

Design rationale (mechanism fix, Spec §5 permits any bounded S(r,z) that does not
depend on bid / peers / current anchor membership):

The SYN-V2-1 score ``S=1-(r-z)^2`` saturates near 1 in the small-noise regime, so
the continuation table ``Gbar(e)`` barely rewards effort and the escrow->bonus->
effort->quality chain collapses (effort is never a best response). We replace it
with a Gaussian-kernel accuracy score whose sensitivity is set by a declared
sensing tolerance ``tau``. This keeps S bounded in (0,1], monotone in accuracy,
and independent of bid/peer/anchor, so T2 monotonicity, effective-type IC, and the
screening/privacy assumptions are preserved; only the numeric Gbar values change.

``tau`` is a fixed accuracy tolerance (a report within ~tau of truth on the [0,1]
scale scores well). It is NOT fitted to any comparison outcome.
"""

from __future__ import annotations

import math
from decimal import Decimal

from ..data.schemas import decimal_from_float

# Declared sensing accuracy tolerance on the [0,1] measurement scale.
TAU = Decimal("0.1")
# Effort reduces report noise by up to this fraction at e=1 (public signal tech).
EFFORT_NOISE_REDUCTION = Decimal("0.6")


def report_sigma(base_noise: Decimal, effort: Decimal) -> Decimal:
    """Report noise std at a given effort: more effort -> less noise."""
    factor = Decimal("1") - EFFORT_NOISE_REDUCTION * effort
    if factor < Decimal("0"):
        factor = Decimal("0")
    return base_noise * factor


def gaussian_score(report: Decimal, holdout: Decimal, tau: Decimal = TAU) -> Decimal:
    """Bounded accuracy score S(r,z)=exp(-(r-z)^2/(2*tau^2)) in (0,1].

    Depends only on (r, z); independent of bid, peers, and anchor membership.
    """
    diff = float(report) - float(holdout)
    value = math.exp(-(diff * diff) / (2.0 * float(tau) * float(tau)))
    return decimal_from_float(value)
