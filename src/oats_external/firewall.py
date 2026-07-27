from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from .types import ExternalCandidateView, ExternalFeedback, ExternalSlotView


FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "current_unpurchased_raw_report",
        "current_raw_report",
        "current_realized_score_or_value",
        "realized_score",
        "realized_value",
        "future_availability",
        "future_trajectory",
        "future_outcome",
        "latent_worker_stratum",
        "stratum",
        "theta",
        "bias_sign",
        "attack_label",
        "lp_optimum",
        "other_method_winners",
        "counterfactual_unselected_quality",
        "oats_trust",
        "oats_screen",
        "oats_dual",
        "oats_payment",
        "oats_settlement",
        "potential_reports",
        "report",
        "score",
        "v_ijt",
        "screen_status",
    }
)


class ExternalFirewallViolation(RuntimeError):
    pass


def _walk(value: Any, *, path: str) -> None:
    if is_dataclass(value):
        for item in fields(value):
            if item.name in FORBIDDEN_DECISION_FIELDS:
                raise ExternalFirewallViolation(f"forbidden decision field: {path}.{item.name}")
            _walk(getattr(value, item.name), path=f"{path}.{item.name}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name in FORBIDDEN_DECISION_FIELDS:
                raise ExternalFirewallViolation(f"forbidden decision key: {path}.{name}")
            _walk(child, path=f"{path}.{name}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _walk(child, path=f"{path}[{index}]")


def validate_slot_view(view: ExternalSlotView) -> None:
    if not isinstance(view, ExternalSlotView):
        raise ExternalFirewallViolation("policy decision input must be ExternalSlotView")
    _walk(view, path="slot_view")
    if not all(isinstance(candidate, ExternalCandidateView) for candidate in view.candidates):
        raise ExternalFirewallViolation("non-public candidate type in slot view")


def validate_feedback(feedback: ExternalFeedback) -> None:
    if not isinstance(feedback, ExternalFeedback):
        raise ExternalFirewallViolation("policy feedback input must be ExternalFeedback")
