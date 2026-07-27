from __future__ import annotations

from src.oats_v2.data.provenance import DEPENDENCY_DECLARATION, PROVIDER_MODEL_ID


def test_holdout_dependency_firewall_matches_preregistration() -> None:
    assert PROVIDER_MODEL_ID == "SYNTHETIC_PREGENERATED_HOLDOUT"
    forbidden = {"selected_report", "gamma", "method", "selection", "report_delivery", "consumer_action"}
    assert forbidden <= set(DEPENDENCY_DECLARATION["does_not_depend_on"])
    assert not forbidden.intersection(DEPENDENCY_DECLARATION["depends_on"])
