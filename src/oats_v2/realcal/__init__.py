"""REAL-CAL-V1: real-data-calibrated, semi-synthetic profile builders.

This package DOES NOT touch the frozen SYN-V2-1 generator. It reads three real
public datasets and distils them into a single frozen calibration profile that a
separate REAL-CAL generation path can consume:

  * T-Drive taxi GPS         -> worker availability / arrival intensity
  * Beijing multi-site air   -> task difficulty / value / anchor structure / missingness
  * PurpleAir vs EPA (FRM)   -> sensor report noise / bias / quality tiers / contamination

Private cost, effort-response and counterfactual reports remain modeled (synthetic),
consistent with the mechanism-design literature. Nothing here upgrades the claim
beyond "real-data-calibrated semi-synthetic evidence".
"""

REALCAL_DATASET_ID = "REAL-CAL-V1"
REALCAL_PROFILE_VERSION = "real-cal-v1-profile-1.0.0"

# The current REAL-CAL benchmark uses identical calibrated distributions, but
# the task-value unit is
# rescaled so the market admits profitable trades. The legacy profile froze
# V in [0.5, 1.5]
# (inherited from SYN-V2-1) which put the per-report value (median v_ijt 0.17)
# strictly below the cheapest worker cost (0.5) — no mechanism can be
# economically viable there. The current profile scales the value band by
# VALUE_SCALE so
# that the median predicted pair value ~ 2-4x the median truthful bid. The
# scale rule was fixed before observing any formal comparison result.
REALCAL_V2_DATASET_ID = "REAL-CAL"
REALCAL_V2_PROFILE_VERSION = "real-cal-profile-1.0.0"
REALCAL_V2_VALUE_SCALE = 30

# Preregistered REAL-CAL seed set: the first 10 formal seeds. Fixed BEFORE any
# result is observed (Phase 4C preregistration). Paired-bootstrap statistics use
# exactly these 10 paired seeds; seeds are never added after seeing results.
# The same frozen set is reused for REAL-CAL under the same pre-results
# discipline.
REALCAL_SEEDS = tuple(range(20260715, 20260725))

__all__ = [
    "REALCAL_DATASET_ID",
    "REALCAL_PROFILE_VERSION",
    "REALCAL_SEEDS",
    "REALCAL_V2_DATASET_ID",
    "REALCAL_V2_PROFILE_VERSION",
    "REALCAL_V2_VALUE_SCALE",
]

