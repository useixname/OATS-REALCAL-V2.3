from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Mapping

from ..types import D


class TaintLevel(Enum):
    CLEAN = auto()
    FORBIDDEN = auto()


FORBIDDEN_ONLINE_FIELDS = frozenset(
    {
        "theta",
        "stratum",
        "bias_sign",
        "future_potential_reports",
        "future_holdout",
        "method_independent_oracle",
        "other_method_selection",
        "current_raw_report",
        "future_outcome",
        "lp_optimum",
        "potential_reports",
        "z",
        "report",
        "score",
        "v_ijt",
        "screen_status",
    }
)


@dataclass(frozen=True)
class OnlineContractView:
    """Bid-before public contract fields allowed for online policy."""

    slot: int
    task_id: str
    worker_id: str
    public_signal_role: str
    sbar: Decimal
    gbar_by_effort: Mapping[str, Decimal]
    vhat_coefficient: Decimal
    delta_money: Decimal
    epsilon_rank: Decimal
    anchor_version: str
    abar: Decimal
    task_value: Decimal
    capacity: int
    deadline: int
    gamma: Decimal
    taint: TaintLevel = TaintLevel.CLEAN

    def __post_init__(self) -> None:
        for name in ("sbar", "vhat_coefficient", "delta_money", "epsilon_rank", "abar", "task_value", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                object.__setattr__(self, name, D(value))


@dataclass(frozen=True)
class OnlineEligibilityView:
    slot: int
    worker_id: str
    available: bool
    mapped_task_id: str | None
    map_hash: str
    taint: TaintLevel = TaintLevel.CLEAN


@dataclass(frozen=True)
class OnlineWorkerStateView:
    worker_id: str
    prior_trust: Decimal
    taint: TaintLevel = TaintLevel.CLEAN


class OnlineFirewallViolation(RuntimeError):
    """Fail-closed when forbidden information reaches online policy."""


def assert_clean(field_name: str, value: Any) -> None:
    if field_name in FORBIDDEN_ONLINE_FIELDS:
        raise OnlineFirewallViolation(f"forbidden online field: {field_name}")
    if isinstance(value, Mapping):
        for key in value:
            if str(key) in FORBIDDEN_ONLINE_FIELDS:
                raise OnlineFirewallViolation(f"forbidden online mapping key: {key}")
    if hasattr(value, "taint") and getattr(value, "taint") is TaintLevel.FORBIDDEN:
        raise OnlineFirewallViolation(f"tainted object passed to online policy: {field_name}")


def validate_online_payload(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        assert_clean(str(key), value)
        if hasattr(value, "__dataclass_fields__"):
            for sub in fields(value):
                assert_clean(sub.name, getattr(value, sub.name))


def mark_forbidden(**kwargs: Any) -> dict[str, Any]:
    return {**kwargs, "taint": TaintLevel.FORBIDDEN}
