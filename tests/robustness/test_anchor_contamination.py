from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from src.oats_v2.anchor_registry import (
    AnchorObservation,
    AnchorPolicy,
    HistoricalAnchorRegistry,
)
from src.oats_v2.ideal_range_screen import IdealRangeScreen
from src.oats_v2.types import MechanismStatus


ROOT = Path(__file__).resolve().parents[2]
LEVELS = (Decimal("0"), Decimal("0.1"), Decimal("0.3"), Decimal("0.5"))


def _history(level: Decimal) -> tuple[AnchorObservation, ...]:
    count = 100
    contaminated = int(level * count)
    honest = count - contaminated
    rows: list[AnchorObservation] = []
    for index in range(honest):
        value = Decimal("0.48") + Decimal(index % 5) * Decimal("0.01")
        rows.append(AnchorObservation(f"h{index}", value, Decimal("1"), True, True, True))
    for index in range(contaminated):
        # Reputation-weight asymmetry is frozen in the boundary witness; no
        # weight-cap defense is silently adopted.
        rows.append(AnchorObservation(f"c{index}", Decimal("0.9"), Decimal("1.1"), True, True, True))
    return tuple(rows)


def collect_contamination_boundary() -> dict[str, object]:
    records: list[dict[str, object]] = []
    baseline_center: Decimal | None = None
    baseline_lower: Decimal | None = None
    baseline_upper: Decimal | None = None
    for level in LEVELS:
        registry = HistoricalAnchorRegistry(AnchorPolicy(20, Decimal("0.01"), Decimal("3")))
        registry.register(f"contamination-{level}", _history(level))
        snapshot = registry.snapshot(f"contamination-{level}")
        assert snapshot is not None
        if baseline_center is None:
            baseline_center, baseline_lower, baseline_upper = snapshot.center, snapshot.lower, snapshot.upper
        screen = IdealRangeScreen()
        malicious = screen.compare(f"m-{level}", Decimal("0.9"), snapshot, cold_start_authorized=False)
        honest = screen.compare(f"h-{level}", Decimal("0.5"), snapshot, cold_start_authorized=False)
        rare = screen.compare(f"r-{level}", Decimal("0.05"), snapshot, cold_start_authorized=False)
        records.append(
            {
                "requested_contamination": str(level),
                "anchor_count": 100,
                "contaminated_count": int(level * 100),
                "colluder_weight": "1.1",
                "reference": str(snapshot.center),
                "mad_floor_scale": str(snapshot.sigma),
                "lower": str(snapshot.lower),
                "upper": str(snapshot.upper),
                "confidence": str(snapshot.confidence),
                "reference_shift": str(snapshot.center - baseline_center),
                "lower_shift": str(snapshot.lower - baseline_lower),
                "upper_shift": str(snapshot.upper - baseline_upper),
                "soft_mode": malicious is MechanismStatus.SCREEN_SOFT_PASS,
                "malicious_PASS": malicious is MechanismStatus.SCREEN_PASS,
                "malicious_purchasable": malicious
                in (MechanismStatus.SCREEN_PASS, MechanismStatus.SCREEN_SOFT_PASS),
                "honest_FAIL": honest is MechanismStatus.SCREEN_FAIL_COMPLIANT,
                "rare_event_rejection": rare is MechanismStatus.SCREEN_FAIL_COMPLIANT,
            }
        )
    return {
        "scope": "robustness boundary witness for the V2.1 coverage-confidence gate",
        "screening_backend": "IDEAL_S2",
        "records": records,
        "adopted_defense": (
            "coverage-confidence soft-screening gate (theta_A=0.75); adopted via "
            "the preregistered REAL-CAL evaluation protocol"
        ),
        "candidate_defenses_not_adopted": [
            "honest-majority assumption",
            "externally certified anchors",
            "robust trimmed history",
            "multi-source anchor quorum",
            "reputation-weight cap",
            "poisoning detector",
        ],
        "version_rule": "adopting any defense requires a new mechanism and preregistration version",
    }


def test_contamination_grid_and_boundary_outputs_are_complete() -> None:
    output = collect_contamination_boundary()
    assert [row["requested_contamination"] for row in output["records"]] == ["0", "0.1", "0.3", "0.5"]
    required = {
        "malicious_PASS", "honest_FAIL", "rare_event_rejection", "reference_shift",
        "mad_floor_scale", "lower_shift", "upper_shift", "soft_mode", "confidence",
    }
    assert all(required <= set(row) for row in output["records"])
    # Clean and lightly contaminated anchors: hard screening, malicious rejected,
    # honest kept.
    for row in output["records"][:2]:
        assert row["soft_mode"] is False
        assert row["malicious_PASS"] is False
        assert row["honest_FAIL"] is False
    # Point-mass hijack (0.5): the coverage confidence collapses, the screen
    # degrades to soft mode — malicious data are purchasable (and later punished
    # via outcome trust), but the market NEVER hard-rejects honest data.
    hijack = output["records"][-1]
    assert hijack["soft_mode"] is True
    assert hijack["malicious_PASS"] is False
    assert hijack["malicious_purchasable"] is True
    assert hijack["honest_FAIL"] is False


if __name__ == "__main__":
    output = collect_contamination_boundary()
    path = ROOT / "audit_results/anchor_contamination.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))

