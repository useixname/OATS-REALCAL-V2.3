from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "raw_report",
    "report_value",
    "lower_bound",
    "upper_bound",
    "secret_bounds",
    "anchor_reports",
    "median",
    "mad",
}

REQUIRED_EVENT_FIELDS = {
    "slot": None,
    "task": None,
    "worker": None,
    "contract_version": None,
    "anchor_version": None,
    "pre_hash": None,
    "post_hash": None,
    "delta_free": "0",
    "delta_locked_base": "0",
    "delta_locked_score": "0",
    "delta_paid": "0",
    "delta_shadow_free": "0",
    "delta_shadow_consumed": "0",
    "status": "ACTIVE",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def stable_hash(value: Any) -> str:
    canonical = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_paths(nested, f"{prefix}[{index}]"))
    return found


@dataclass
class EventLog:
    events: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True

    def emit(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        # The audit JSONL trail (Spec §17) is never consumed by the formal
        # experiment outputs (RunResult carries only numeric ledger/shadow/metric
        # state). Disabling emission is therefore logic-preserving: it changes no
        # computed value, only skips building the log payload.
        if not self.enabled:
            return {}
        forbidden = _forbidden_paths(payload)
        if forbidden:
            raise ValueError(f"secret fields in public log: {forbidden}")
        payload = dict(payload)
        base = {
            "seq": len(self.events),
            "event_type": event_type,
        }
        for key, default in REQUIRED_EVENT_FIELDS.items():
            base[key] = _jsonable(payload.pop(key, default))
        base["payload"] = _jsonable(payload)
        canonical = json.dumps(base, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        event = dict(base)
        event["event_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.events.append(event)
        return event

    def jsonl(self) -> str:
        return "\n".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for event in self.events
        )
