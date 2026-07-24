from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .anchor_registry import AnchorSnapshot
from .events import EventLog, stable_hash
from .types import D, MechanismStatus


SCREENING_BACKEND_LABEL = "SCREENING_BACKEND=IDEAL_S2; NO_CRYPTOGRAPHIC_SECURITY_CLAIM"


@dataclass
class IdealRangeScreen:
    """Ideal one-bit functionality; this is deliberately not a security protocol.

    ``confidence_threshold`` (theta_A, paper Eq. 55) gates hard screening: when
    the anchor snapshot's dual-signal confidence falls below it, the screen
    refuses to hard-reject and returns SCREEN_SOFT_PASS instead. The final
    quality evaluation is then deferred to post-use outcome feedback. This is
    what prevents the "reject everything under heavy anchor contamination"
    failure mode: a poisoned anchor set can degrade screening to a no-op, but
    it can never shut the market down.
    """

    record_events: bool = True
    confidence_threshold: Decimal = Decimal("0.75")
    cache: dict[str, MechanismStatus] = field(default_factory=dict)
    transcript_versions: dict[str, str | None] = field(default_factory=dict)
    event_log: EventLog = field(default_factory=EventLog, repr=False)

    def compare(
        self,
        transcript_key: str,
        secret_report: Decimal,
        snapshot: AnchorSnapshot | None,
        *,
        cold_start_authorized: bool,
    ) -> MechanismStatus:
        if transcript_key in self.cache:
            requested_version = snapshot.version if snapshot is not None else None
            if self.transcript_versions[transcript_key] != requested_version:
                if self.record_events:
                    state_hash = stable_hash({"keys": sorted(self.cache), "versions": self.transcript_versions})
                    self.event_log.emit(
                        "SCREEN_VERSION_REPLAY_REJECTED",
                        {
                            "anchor_version": requested_version,
                            "pre_hash": state_hash,
                            "post_hash": state_hash,
                            "status": MechanismStatus.PROTOCOL_ERROR,
                            "transcript_digest": stable_hash(transcript_key),
                        },
                    )
                return MechanismStatus.PROTOCOL_ERROR
            return self.cache[transcript_key]
        if snapshot is None:
            status = MechanismStatus.COLD_START if cold_start_authorized else MechanismStatus.PROTOCOL_ERROR
        elif snapshot.confidence < self.confidence_threshold:
            # Low-confidence anchors: no hard rejection (soft screening mode).
            status = MechanismStatus.SCREEN_SOFT_PASS
        else:
            report = D(secret_report)
            status = (
                MechanismStatus.SCREEN_PASS
                if snapshot.lower <= report <= snapshot.upper
                else MechanismStatus.SCREEN_FAIL_COMPLIANT
            )
        self.cache[transcript_key] = status
        self.transcript_versions[transcript_key] = snapshot.version if snapshot is not None else None
        if self.record_events:
            pre_hash = stable_hash({"keys": sorted(self.cache), "versions": self.transcript_versions})
            post_hash = stable_hash({"keys": sorted(self.cache), "versions": self.transcript_versions})
            self.event_log.emit(
                "SCREEN_RESULT",
                {
                    "anchor_version": snapshot.version if snapshot is not None else None,
                    "pre_hash": pre_hash,
                    "post_hash": post_hash,
                    "status": status,
                    "transcript_digest": stable_hash(transcript_key),
                },
            )
        return status
