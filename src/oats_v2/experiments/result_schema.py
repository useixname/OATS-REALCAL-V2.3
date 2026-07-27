from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


REQUIRED_LABELS = (
    "SCREENING_BACKEND=IDEAL_S2",
    "NO_CRYPTOGRAPHIC_SECURITY_CLAIM",
    "ORACLE_ROUTE=R4",
    "NO_REGRET_GUARANTEE",
)


@dataclass
class ChannelMetrics:
    population_quality: Decimal | None = None
    selected_quality: Decimal | None = None
    composition_standardized_quality: Decimal | None = None
    effort_effect: Decimal | None = None
    screening_effect: Decimal | None = None
    payment_effect: Decimal | None = None


@dataclass
class ScreeningMetrics:
    mrr: Decimal | None = None
    hfr: Decimal | None = None
    rare_event_rejection: Decimal | None = None
    colluder_pass: Decimal | None = None
    honest_fail: Decimal | None = None
    cold_start_rate: Decimal | None = None
    soft_pass_rate: Decimal | None = None


@dataclass
class TrustMetrics:
    population_available: int = 0
    selected_count: int = 0
    feedback_count: int = 0
    trust_transition_count: int = 0
    duplicate_feedback_suppressed_count: int = 0
    feedback_count_definition: str = (
        "completed independent worker-task feedback records submitted to the trust interface"
    )
    trust_transition_count_definition: str = (
        "worker-specific trust-state transitions actually applied"
    )
    feedback_id_fields: tuple[str, ...] = (
        "cell_id",
        "feedback_slot",
        "task_id",
        "worker_id",
    )
    brier: Decimal | None = None
    auc: Decimal | None = None


@dataclass
class LPComparatorResult:
    status: str
    opt_lp: Decimal | None = None
    u_online: Decimal | None = None
    lp_gap: Decimal | None = None
    normalized_lp_gap: Decimal | None = None
    max_violation: Decimal | None = None


@dataclass
class RunResult:
    cell_id: str
    config_hash: str
    trace_hash: str
    method_id: str
    seed: int
    gamma: str
    budget_ratio: str
    invariant_status: str
    labels: list[str] = field(default_factory=lambda: list(REQUIRED_LABELS))
    task_count: int = 0
    worker_count: int = 0
    activated_count: int = 0
    contracted_count: int = 0
    purchased_count: int = 0
    base_paid: str = "0"
    score_paid: str = "0"
    total_paid: str = "0"
    returned_locks: str = "0"
    final_ledger: dict[str, str] = field(default_factory=dict)
    final_shadow: dict[str, str] = field(default_factory=dict)
    effort_histogram: dict[str, int] = field(default_factory=dict)
    channels: ChannelMetrics = field(default_factory=ChannelMetrics)
    worker_type_composition: dict[str, Decimal] = field(default_factory=dict)
    screening: ScreeningMetrics = field(default_factory=ScreeningMetrics)
    trust: TrustMetrics = field(default_factory=TrustMetrics)
    deadline_satisfaction: Decimal | None = None
    gross_external_value: str = "0"
    platform_net_value: str = "0"
    lp: LPComparatorResult = field(default_factory=lambda: LPComparatorResult(status="PENDING"))
    mc_correlation: Decimal | None = None
    mc_top_k_overlap: Decimal | None = None
    mc_undefined_rate: Decimal | None = None
    failure_counts: dict[str, int] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    # E4: mean trust per stratum sampled every 50 slots ({slot: {stratum: rho}}).
    trust_trajectory: dict[str, dict[str, str]] = field(default_factory=dict)
    # E5: cumulative purchased external value at prefix horizons ({T: value}).
    value_prefix: dict[str, str] = field(default_factory=dict)
    # E8: wall-clock decomposition of the online loop (seconds).
    runtime_breakdown: dict[str, float] = field(default_factory=dict)
    # Rejection-reason counts from selection (reserve/AV gates, capacity).
    rejection_counts: dict[str, int] = field(default_factory=dict)
    # Calendar-time feedback/settlement audit. End-of-slot occupancy is sampled
    # after delay-zero feedback has been processed.
    feedback_queue_mode: str = "calendar-time"
    mean_outstanding_score_escrow: str = "0"
    peak_outstanding_score_escrow: str = "0"
    terminal_outstanding_score_escrow: str = "0"
    mean_outstanding_task_escrows: Decimal = Decimal("0")
    peak_outstanding_task_escrows: int = 0
    terminal_pending_task_count: int = 0
    terminal_pending_feedback_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Decimal):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        return convert(asdict(self))
