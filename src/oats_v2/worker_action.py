from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .types import D, MechanismStatus, ReportMode


@dataclass(frozen=True)
class WorkerSubmission:
    contract_id: str
    report: Decimal
    nonce: str
    commitment: str
    attestation_valid: bool
    share_valid: bool
    before_deadline: bool
    anchor_version: str
    report_mode: ReportMode = ReportMode.ATTESTED_POINT_ESTIMATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", D(self.report))


def validate_submission(submission: WorkerSubmission) -> MechanismStatus:
    if (
        not submission.nonce
        or not submission.commitment
        or not submission.anchor_version
        or not submission.attestation_valid
        or not submission.share_valid
        or not submission.before_deadline
        or submission.report_mode is not ReportMode.ATTESTED_POINT_ESTIMATE
    ):
        return MechanismStatus.WORKER_NONCOMPLIANT
    return MechanismStatus.ACTIVE
