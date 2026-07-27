from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from ..admission import admit_atomic
from ..allocation import allocate, candidate_order
from ..anchor_registry import AnchorObservation, AnchorPolicy, HistoricalAnchorRegistry
from .fast_selection import build_selection_fast, build_selection_myopic_fast
from ..data.schemas import TraceConfig
from ..ideal_range_screen import IdealRangeScreen
from ..invariants import assert_all
from ..ledger import LedgerState
from ..feedback_calendar import FeedbackCalendar, QueuedTaskOutcome, QueuedWorkerFeedback
from ..settlement import settle_base_for_status, settle_task_scores
from ..shadow_envelope import DualController, ShadowEnvelopeState
from ..task_activation import activate_task_atomic
from ..trust import TrustState, trust_feedback_id
from ..types import AllocationOutcome, AllocationSnapshot, Candidate, MechanismStatus, Task
from .checkpoint import CheckpointState
from .lp_comparator import LPComparatorCache, compute_lp_gap
from .method_registry import (
    METHOD_REGISTRY,
    choose_effort,
    fast_bid,
    predicted_value,
    rank_key,
)
from .metrics import finalize_run_result, spearman_and_topk
from .online_projection import OnlineFirewallViolation, validate_online_payload
from .result_schema import REQUIRED_LABELS, RunResult
from .run_matrix import RunCell, build_run_matrix, build_run_matrix_v2, is_regret_cell
from .trace_loader import TraceBundle, load_contracts, load_trace, reference_budget


HORIZON = 1000
RHO0 = Decimal("0.5")
ALPHA0 = Decimal("0.2")
LAMBDA_MAX = Decimal("10")


def _eta() -> Decimal:
    import math

    return Decimal(str(math.sqrt(1.0 / HORIZON)))


ETA = _eta()


@dataclass
class SlotOutcome:
    task_key: str
    contract_id: str
    worker_id: str
    task_id: str
    slot: int
    effort: Decimal
    screen_status: MechanismStatus
    gross_value: Decimal
    score: Decimal
    stratum: str
    selected: bool
    rare_event: bool


def _apply_due_task_outcomes(
    events: tuple[QueuedTaskOutcome, ...],
    *,
    cell: RunCell,
    trust: TrustState,
    ledger: LedgerState,
    shadow: ShadowEnvelopeState,
    trust_events: list[dict[str, Any]],
    alpha: Decimal,
    money_grid: Decimal,
) -> tuple[Decimal, list[Decimal], list[Decimal]]:
    """Apply due outcomes in the formal worker-specific feedback order.

    Task events are already sorted by ``(feedback_slot, task_id, task_key)`` by
    ``FeedbackCalendar``. Worker feedback inside each task is sorted by
    ``(worker_id, contract_id)``. Because the event fixes one task, the combined
    order is the manuscript contract ``(feedback_slot, task_id, worker_id)``.
    """

    gross_delta = Decimal("0")
    estimated: list[Decimal] = []
    realized: list[Decimal] = []
    for event in events:
        remaining = ledger.task_escrows[event.task_key]
        score_payments: dict[str, Decimal] = {}
        for record in sorted(
            event.feedback_records,
            key=lambda item: (item.worker_id, item.contract_id),
        ):
            audit = trust_events[record.trust_event_index]
            if not record.feedback_available:
                audit["feedback_resolution"] = "missing"
                continue
            feedback_id = trust_feedback_id(
                cell.cell_id,
                event.available_slot,
                record.task_id,
                record.worker_id,
            )
            rho_before = trust.values[record.worker_id]
            transitions_before = trust.transition_count
            duplicate_suppressions_before = trust.duplicate_feedback_suppressed_count
            trust.update(
                record.worker_id,
                feedback_id,
                record.quality,
                alpha,
                available=True,
                independent=True,
            )
            transition_applied = trust.transition_count == transitions_before + 1
            duplicate_suppressed = (
                trust.duplicate_feedback_suppressed_count
                == duplicate_suppressions_before + 1
            )
            audit.update(
                {
                    "feedback": True,
                    "feedback_id": feedback_id,
                    "feedback_slot": event.available_slot,
                    "quality": record.quality,
                    "rho": rho_before,
                    "trust_transition_applied": transition_applied,
                    "duplicate_feedback_suppressed": duplicate_suppressed,
                    "feedback_resolution": "completed",
                }
            )
            amount = (record.score_cap * record.quality).quantize(
                money_grid,
                rounding=ROUND_DOWN,
            )
            if amount > remaining:
                amount = remaining.quantize(money_grid, rounding=ROUND_DOWN)
            score_payments[record.contract_id] = amount
            remaining -= amount
            if record.recognize_value_at_feedback:
                gross_delta += record.realized_value
                estimated.append(record.estimated_value)
                realized.append(record.realized_value)

        receipt = ledger.close_task(event.task_key, score_payments)
        released_total = Decimal(receipt["released"])
        shadow.settle_task(event.task_key, released_total)
    return gross_delta, estimated, realized


def _build_anchor_registry(trace: TraceBundle, contamination: Decimal) -> tuple[HistoricalAnchorRegistry, dict[int, Any]]:
    registry = HistoricalAnchorRegistry(AnchorPolicy(20, Decimal("0.01"), Decimal("3")))
    snapshots: dict[int, Any] = {}
    for cell, rows in trace.anchors_by_cell.items():
        observations: list[AnchorObservation] = []
        n = len(rows)
        contaminated = int(contamination * n)
        for index, row in enumerate(rows):
            if index >= n - contaminated:
                observations.append(
                    AnchorObservation(
                        f"c{index}",
                        Decimal("0.9"),
                        Decimal("1.1"),
                        True,
                        True,
                        True,
                    )
                )
            else:
                observations.append(
                    AnchorObservation(
                        f"h{index}",
                        Decimal(row["report"]),
                        Decimal("1"),
                        True,
                        True,
                        True,
                    )
                )
        version = f"{trace.anchor_version}-c{contamination}-cell{cell}"
        if version not in registry._versions:
            registry.register(version, tuple(observations))
        snapshots[cell] = registry.snapshot(version)
    return registry, snapshots


def _source_task_id(task_id: str) -> str:
    """Strip the arrival-x2 clone suffix to recover the trace task id."""
    return task_id.partition("#")[0]


def _tasks_for_slot(trace: TraceBundle, slot: int, arrival_multiplier: Decimal) -> list:
    tasks = list(trace.tasks_by_slot.get(slot, []))
    if arrival_multiplier == Decimal("0.5"):
        filtered = []
        for task in tasks:
            token = f"{trace.seed}|{slot}|{task.task_id}"
            if int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 2 == 0:
                filtered.append(task)
        return filtered
    if arrival_multiplier == Decimal("2"):
        # Deterministic task-stream duplication: every task arrives twice with
        # identical latent attributes (cloned id). Contracts/reports resolve to
        # the source id; escrow, capacity and value are counted per clone. The
        # LP comparator mirrors this expansion (lp_comparator._build_opportunities).
        doubled = []
        for task in tasks:
            doubled.append(task)
            doubled.append(dataclasses.replace(task, task_id=f"{task.task_id}#d2"))
        return doubled
    return tasks


def _myopic_allocate(candidates: tuple[Candidate, ...], snapshot: AllocationSnapshot) -> AllocationOutcome:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: rank_key(
                item,
                dual_lambda=snapshot.dual_lambda,
                epsilon_rank=snapshot.epsilon_rank,
                use_density=False,
            ),
        )
    )
    from collections import Counter

    task_counts: Counter[str] = Counter()
    worker_counts: Counter[str] = Counter()
    used_cap = Decimal("0")
    winners: list[str] = []
    rejected: dict[str, str] = {}
    total_cap = min(snapshot.actual_base_capacity, snapshot.shadow_base_capacity)
    for candidate in ordered:
        worker_cap = snapshot.worker_capacities.get(candidate.worker_id, 1)
        task_cap = snapshot.task_capacities.get(candidate.task_id, 0)
        feasible = (
            worker_counts[candidate.worker_id] < worker_cap
            and task_counts[candidate.task_id] < task_cap
            and used_cap + candidate.base_cap <= total_cap
        )
        if not feasible:
            rejected[candidate.key] = "DOWNWARD_CLOSED_CAPACITY"
            continue
        winners.append(candidate.key)
        worker_counts[candidate.worker_id] += 1
        task_counts[candidate.task_id] += 1
        used_cap += candidate.base_cap
    return AllocationOutcome(winners=tuple(winners), order=tuple(c.key for c in ordered), rejected=rejected)


PREFIX_CHECKPOINTS = tuple(range(100, HORIZON + 1, 100))


def simulate_cell(cell: RunCell, trace: TraceBundle, lp_cache: LPComparatorCache, *, compute_lp: bool = False) -> RunResult:
    start = time.perf_counter()
    cfg = TraceConfig()
    method = METHOD_REGISTRY[cell.method_id]
    method.validate()

    alpha0 = cell.alpha
    lambda_max = cell.lambda_max

    budget = reference_budget(trace) * cell.budget_ratio
    budget = budget.quantize(Decimal("0.001"))
    ledger = LedgerState(budget)
    shadow = ShadowEnvelopeState(budget)
    # Audit event logs are not consumed by RunResult; disable emission on the hot
    # path (logic-preserving — no computed value depends on the JSONL trail).
    ledger.event_log.enabled = False
    shadow.event_log.enabled = False
    trust = TrustState()
    trust.event_log.enabled = False
    for worker_id in trace.workers:
        trust.initialize(worker_id, RHO0)
    feedback_calendar = FeedbackCalendar()
    dual = DualController(Decimal("0"), lambda_max)
    screen_backend = IdealRangeScreen(record_events=False, confidence_threshold=cell.theta_a)
    anchors, anchor_snapshots = _build_anchor_registry(trace, cell.contamination)
    eta = _eta()

    workers_by_stratum: dict[str, list[str]] = {}
    for worker_id, worker in trace.workers.items():
        workers_by_stratum.setdefault(worker.stratum, []).append(worker_id)

    pop_quality_sum = Decimal("0")
    pop_quality_n = 0
    sel_quality_sum = Decimal("0")
    sel_quality_n = 0
    type_selected: dict[str, int] = {}
    type_quality: dict[str, Decimal] = {}
    type_quality_n: dict[str, int] = {}
    screening_events: list[dict[str, Any]] = []
    trust_events: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    effort_hist: dict[str, int] = {}
    effort_delta_sum = Decimal("0")
    effort_delta_n = 0
    mc_estimated: list[Decimal] = []
    mc_realized: list[Decimal] = []
    trust_trajectory: dict[str, dict[str, str]] = {}
    value_prefix: dict[str, str] = {}
    t_selection = t_screening = t_settlement = 0.0
    gross_total = Decimal("0")
    activated_count = contracted_count = purchased_count = 0
    deadline_met = deadline_total = 0
    outstanding_score_sum = Decimal("0")
    peak_outstanding_score = Decimal("0")
    outstanding_task_sum = 0
    peak_outstanding_tasks = 0
    invariant_status = "PASS"

    try:
        for slot in range(1, HORIZON + 1):
            # Calendar phase 1: feedback scheduled by earlier purchases becomes
            # part of the history before the current slot's online decision.
            t0 = time.perf_counter()
            due = feedback_calendar.pop_due(slot)
            if due:
                gross_delta, estimated_delta, realized_delta = _apply_due_task_outcomes(
                    due,
                    cell=cell,
                    trust=trust,
                    ledger=ledger,
                    shadow=shadow,
                    trust_events=trust_events,
                    alpha=alpha0,
                    money_grid=Decimal("0.001"),
                )
                gross_total += gross_delta
                mc_estimated.extend(estimated_delta)
                mc_realized.extend(realized_delta)
            t_settlement += time.perf_counter() - t0

            lambda_t = dual.value if method.use_dual else Decimal("0")
            slot_tasks = _tasks_for_slot(trace, slot, cell.arrival_multiplier)
            slot_tasks.sort(key=lambda t: t.task_id)
            tasks_this_slot = {t.task_id: t for t in slot_tasks}
            task_keys: dict[str, str] = {}
            # OATS dual pacing: remaining-quota flow control on the SHADOW
            # envelope (the resource that actually binds). Signal history:
            #   * V2 paced the per-slot gross reserve envelope C_env — lambda
            #     rose with zero budget pressure (15% net loss vs no-dual).
            #   * V2.1 paced cumulative LEDGER consumption — but the shadow
            #     envelope hit 100% at slot ~500 while the ledger sat at 60%;
            #     the market shut down at half-time unseen by the dual.
            #   * V2.2 paced cumulative SHADOW consumption against B*t/T — the
            #     right resource but a lagging integral signal: after the early
            #     diurnal binge, lambda kept rising while current flow was
            #     already zero, then decayed too slowly; 23% of the envelope
            #     went unused and net value fell 24% below no-dual.
            # OATS controls the instantaneous flow against the remaining quota:
            #     quota_t    = shadow_free(t-) / (T - t + 1)
            #     gradient_t = flow_t / quota_t - 1
            # (paper Eq. (71) with P_res = this slot's reserved worst-case
            # payments and B-bar = the adaptive per-slot reserve quota).
            # Properties, each killing one observed failure mode:
            #   * flow below quota gives gradient >= -1, so lambda decays at
            #     ~eta/slot as soon as the market over-throttles (no V2.2
            #     mid-horizon dead zone);
            #   * capacity wasted early raises the remaining quota, driving
            #     lambda back down until the envelope is fully used by T (no
            #     stranded capacity);
            #   * slack envelopes keep lambda at 0 exactly, so OATS coincides
            #     with the no-dual greedy (V2.1's verified property).
            # Budget safety is unchanged: it is enforced by the ledger/escrow
            # reserve checks, not by pacing.
            shadow_free_before = shadow.free

            for task in slot_tasks:
                escrow = cell.gamma * task.value
                mech_task = Task(task.task_id, task.capacity, escrow)
                activation = activate_task_atomic(mech_task, slot, ledger, shadow)
                if activation.status is MechanismStatus.ACTIVE:
                    activated_count += 1
                    task_keys[task.task_id] = activation.task_key
                else:
                    failure_counts[activation.status.value] = failure_counts.get(activation.status.value, 0) + 1

            candidates: list[Candidate] = []
            candidate_meta: dict[str, tuple] = {}
            firewall_checked = False
            for row in trace.available_by_slot.get(slot, []):
                mapped_task_id = row.get("mapped_task_id")
                if not mapped_task_id:
                    continue
                worker_id = row["worker_id"]
                worker = trace.workers[worker_id]
                # arrival x2: the worker is also a candidate for the clone of its
                # mapped task (same contract), competing under capacity 1/slot.
                target_ids = [tid for tid in (mapped_task_id, f"{mapped_task_id}#d2") if tid in task_keys]
                if not target_ids:
                    continue
                contract_row = trace.contracts.get((slot, mapped_task_id, worker_id, cell.gamma))
                if contract_row is None:
                    continue
                gbar = dict(contract_row.gbar_by_effort)
                if method.noscreen_gbar:
                    gbar = {k: contract_row.sbar for k in gbar}
                effort = choose_effort(worker.c_i, gbar, fixed=method.fixed_effort)
                bid = fast_bid(method, worker.c_i, gbar, money_grid=Decimal("0.001"))
                trust_val = trust.values[worker_id]
                if method.trust_only_value:
                    vhat = trust_val
                else:
                    vhat = predicted_value(
                        coefficient=contract_row.vhat_coefficient,
                        trust=trust_val,
                        use_trust=method.use_trust,
                    )
                pot0 = trace.potential.get((slot, mapped_task_id, worker_id, effort))
                potential_score = pot0.score if pot0 else Decimal("0")
                reserve = vhat if method.use_reserve else None
                for task_id in target_ids:
                    task = tasks_this_slot[task_id]
                    key = f"{worker_id}|{task_id}"
                    pop_quality_sum += potential_score
                    pop_quality_n += 1
                    candidate_meta[key] = (worker.stratum, effort, pot0, contract_row.sbar, task, vhat)
                    if not firewall_checked:
                        validate_online_payload(
                            {
                                "slot": slot,
                                "task_id": task_id,
                                "worker_id": worker_id,
                                "bid": bid,
                                "vhat": vhat,
                            }
                        )
                        firewall_checked = True
                    candidates.append(
                        Candidate(
                            worker_id=worker_id,
                            task_id=task_id,
                            bid=bid,
                            base_cap=cfg.public_base_cap,
                            predicted_value=vhat,
                            expected_score_at_reference=gbar.get("0.5", Decimal("0")),
                            public_role=worker.public_signal_role,
                            deadline_ok=task.deadline >= slot,
                            reserve_price=reserve,
                        )
                    )

            if not candidates:
                # Spec §7: activated-but-uncontracted tasks must return their escrow
                # immediately in BOTH accounts; nothing may keep an empty escrow.
                for task_key in task_keys.values():
                    if task_key in ledger.task_escrows:
                        ledger.close_task(task_key, {})
                    if task_key in shadow.held_tasks:
                        shadow.release_empty_task(task_key)
                quota = shadow_free_before / Decimal(HORIZON - slot + 1)
                if quota > 0:
                    dual.update((shadow_free_before - shadow.free) / quota, Decimal("1"), eta)
                if slot in PREFIX_CHECKPOINTS:
                    value_prefix[str(slot)] = str(gross_total)
                if slot % 50 == 0:
                    trust_trajectory[str(slot)] = {
                        stratum: str(sum((trust.values[w] for w in ids), Decimal("0")) / Decimal(len(ids)))
                        for stratum, ids in workers_by_stratum.items()
                    }
                outstanding_score_sum += ledger.locked_score
                peak_outstanding_score = max(peak_outstanding_score, ledger.locked_score)
                outstanding_task_sum += len(ledger.task_escrows)
                peak_outstanding_tasks = max(peak_outstanding_tasks, len(ledger.task_escrows))
                continue

            snapshot = AllocationSnapshot(
                active_tasks=frozenset(task_keys),
                task_capacities={tid: tasks_this_slot[tid].capacity for tid in task_keys},
                actual_base_capacity=ledger.free,
                shadow_base_capacity=shadow.free,
                dual_lambda=lambda_t,
                epsilon_rank=Decimal("0.001"),
            )
            candidates_tuple = tuple(candidates)
            money_grid = Decimal("0.001")
            t0 = time.perf_counter()
            selection = build_selection_fast(
                candidates_tuple,
                snapshot,
                money_grid,
                rank_mode=method.rank_mode,
                av_gate=method.use_av_gate,
                rank_salt=f"{cell.seed}|{slot}",
            )
            t_selection += time.perf_counter() - t0

            if selection.status is not MechanismStatus.ACTIVE:
                failure_counts[selection.status.value] = failure_counts.get(selection.status.value, 0) + 1
                invariant_status = "INVALID"
                break

            for reason in selection.allocation.rejected.values():
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            base_caps = {key: cfg.public_base_cap for key in selection.allocation.winners}
            admit_status = admit_atomic(
                selection,
                candidates_tuple,
                task_keys,
                ledger,
                shadow,
            )
            if admit_status is not MechanismStatus.ACTIVE:
                failure_counts[admit_status.value] = failure_counts.get(admit_status.value, 0) + 1
                invariant_status = "INVALID"
                break

            queued_feedback_by_task: dict[str, list[QueuedWorkerFeedback]] = {}
            queued_slot_by_task: dict[str, int] = {}

            for key in selection.allocation.winners:
                worker_id, task_id = key.split("|", 1)
                stratum, effort, pot, sbar, task, vhat = candidate_meta[key]
                if pot is None:
                    continue
                worker = trace.workers[worker_id]
                contracted_count += 1
                sel_quality_sum += pot.score
                sel_quality_n += 1
                type_selected[stratum] = type_selected.get(stratum, 0) + 1
                type_quality[stratum] = type_quality.get(stratum, Decimal("0")) + pot.score
                type_quality_n[stratum] = type_quality_n.get(stratum, 0) + 1
                effort_hist[str(effort)] = effort_hist.get(str(effort), 0) + 1
                pot_zero = trace.potential.get((slot, _source_task_id(task_id), worker_id, Decimal("0")))
                if pot_zero is not None:
                    effort_delta_sum += pot.score - pot_zero.score
                    effort_delta_n += 1

                if method.use_screening:
                    snapshot_anchor = anchor_snapshots.get(task.cell)
                    rare = task.theta < Decimal("0.1") or task.theta > Decimal("0.9")
                    t0 = time.perf_counter()
                    screen_status = screen_backend.compare(
                        f"{slot}|{task_id}|{worker_id}|{pot.report}",
                        pot.report,
                        snapshot_anchor,
                        cold_start_authorized=False,
                    )
                    t_screening += time.perf_counter() - t0
                else:
                    screen_status = MechanismStatus.SCREEN_PASS

                screening_events.append(
                    {
                        "stratum": worker.stratum,
                        "screened": True,
                        "status": screen_status.value,
                        "rare_event": rare if method.use_screening else False,
                    }
                )
                t0 = time.perf_counter()
                receipt = settle_base_for_status(ledger, key, screen_status)
                # OATS: settle the shadow base cap at the realized payment and
                # return the worst-case slack (paper Eq. (91) P_return). The V1
                # shadow kept every cap committed forever, which silently turned
                # the envelope into a hard market-capacity cap (~43k purchases
                # at b=0.25) that no controller could pace around.
                realized_base = (
                    Decimal(receipt["amount"]) if receipt.get("kind") == "release" else Decimal("0")
                )
                shadow.settle_base(key, realized_base)
                t_settlement += time.perf_counter() - t0

                missing_key = str(cell.missing_prob)
                delay_key = str(cell.delay)
                outcome_available = not task.missing_mask.get(missing_key, False)

                purchased_statuses = (
                    MechanismStatus.SCREEN_PASS,
                    MechanismStatus.SCREEN_SOFT_PASS,
                    MechanismStatus.COLD_START,
                )
                if screen_status in purchased_statuses:
                    purchased_count += 1
                    outcome_slot = slot + task.delay_mask.get(delay_key, 0)
                    recognize_value_at_feedback = outcome_available and outcome_slot > slot
                    if outcome_available and not recognize_value_at_feedback:
                        # Preserve the frozen delay-zero accounting order while
                        # still deferring its trust transition until after the
                        # current selection phase.
                        gross_total += pot.v_ijt
                        mc_estimated.append(vhat)
                        mc_realized.append(pot.v_ijt)
                    # A missing outcome retains the task escrow until its
                    # contractual deadline, then returns it without a score
                    # payment or trust transition.
                    settlement_slot = outcome_slot if outcome_available else max(slot, task.deadline)
                    trust_event_index = len(trust_events)
                    trust_events.append(
                        {
                            "selected": True,
                            "feedback": False,
                            "purchase_slot": slot,
                            "scheduled_feedback_slot": outcome_slot if outcome_available else None,
                            "task_id": task_id,
                            "worker_id": worker_id,
                            "feedback_resolution": "pending",
                        }
                    )
                    task_key = task_keys[task_id]
                    queued_feedback_by_task.setdefault(task_key, []).append(
                        QueuedWorkerFeedback(
                            worker_id=worker_id,
                            task_id=task_id,
                            contract_id=key,
                            quality=pot.score,
                            score_cap=sbar,
                            estimated_value=vhat,
                            realized_value=pot.v_ijt,
                            trust_event_index=trust_event_index,
                            feedback_available=outcome_available,
                            recognize_value_at_feedback=recognize_value_at_feedback,
                        )
                    )
                    existing_slot = queued_slot_by_task.setdefault(task_key, settlement_slot)
                    if existing_slot != settlement_slot:
                        raise AssertionError("one task produced inconsistent feedback slots")
                    if outcome_available:
                        deadline_total += 1
                        if outcome_slot <= task.deadline:
                            deadline_met += 1

            for task_id, task_key in task_keys.items():
                feedback_records = queued_feedback_by_task.get(task_key)
                if feedback_records:
                    feedback_calendar.schedule(
                        QueuedTaskOutcome(
                            purchase_slot=slot,
                            available_slot=queued_slot_by_task[task_key],
                            task_id=task_id,
                            task_key=task_key,
                            deadline=tasks_this_slot[task_id].deadline,
                            feedback_records=tuple(feedback_records),
                        )
                    )
                    continue
                # No purchased report remains for this task. Empty activations
                # and screen-failed committed escrows are released immediately.
                t0 = time.perf_counter()
                if task_key in ledger.task_escrows:
                    ledger.close_task(task_key, {})
                if task_key in shadow.held_tasks:
                    shadow.release_empty_task(task_key)
                elif task_key in shadow.committed_tasks:
                    shadow.settle_task(task_key, Decimal("0"))
                t_settlement += time.perf_counter() - t0

            # Calendar phase 2: delay-zero outcomes from the current purchases
            # are processed only after selection, preserving the online
            # information firewall while making them available to slot t+1.
            t0 = time.perf_counter()
            due = feedback_calendar.pop_due(slot)
            if due:
                gross_delta, estimated_delta, realized_delta = _apply_due_task_outcomes(
                    due,
                    cell=cell,
                    trust=trust,
                    ledger=ledger,
                    shadow=shadow,
                    trust_events=trust_events,
                    alpha=alpha0,
                    money_grid=Decimal("0.001"),
                )
                gross_total += gross_delta
                mc_estimated.extend(estimated_delta)
                mc_realized.extend(realized_delta)
            t_settlement += time.perf_counter() - t0

            quota = shadow_free_before / Decimal(HORIZON - slot + 1)
            if quota > 0:
                dual.update((shadow_free_before - shadow.free) / quota, Decimal("1"), eta)

            if slot in PREFIX_CHECKPOINTS:
                value_prefix[str(slot)] = str(gross_total)
            if slot % 50 == 0:
                trust_trajectory[str(slot)] = {
                    stratum: str(sum((trust.values[w] for w in ids), Decimal("0")) / Decimal(len(ids)))
                    for stratum, ids in workers_by_stratum.items()
                }
            outstanding_score_sum += ledger.locked_score
            peak_outstanding_score = max(peak_outstanding_score, ledger.locked_score)
            outstanding_task_sum += len(ledger.task_escrows)
            peak_outstanding_tasks = max(peak_outstanding_tasks, len(ledger.task_escrows))

    except OnlineFirewallViolation as exc:
        invariant_status = "FIREWALL_VIOLATION"
        failure_counts[str(exc)] = 1

    # Record terminal exposure before releasing unresolved beyond-horizon or
    # missing-feedback escrows. The terminal close is conservative: it pays no
    # score bonus and applies no trust update.
    terminal_outstanding_score = ledger.locked_score
    terminal_pending_task_count = feedback_calendar.pending_task_count
    terminal_pending_feedback_count = feedback_calendar.pending_feedback_count
    for event in feedback_calendar.drain():
        for record in event.feedback_records:
            trust_events[record.trust_event_index]["feedback_resolution"] = "terminal_expiry"
        if event.task_key in ledger.task_escrows:
            ledger.close_task(event.task_key, {})
        if event.task_key in shadow.committed_tasks:
            shadow.settle_task(event.task_key, Decimal("0"))

    try:
        assert_all(ledger, shadow, ())
    except Exception as exc:
        failure_counts[str(exc)] = failure_counts.get(str(exc), 0) + 1
        invariant_status = "INVALID"

    elapsed = time.perf_counter() - start

    records: list[dict[str, Any]] = []
    if pop_quality_n:
        records.append({"potential_score": pop_quality_sum / pop_quality_n, "selected": False, "stratum": "aggregate"})
    if sel_quality_n:
        record: dict[str, Any] = {
            "realized_score": sel_quality_sum / sel_quality_n,
            "selected": True,
            "stratum": "aggregate",
        }
        if effort_delta_n:
            record["effort_delta_quality"] = effort_delta_sum / Decimal(effort_delta_n)
        records.append(record)
    for stratum, count in type_selected.items():
        if type_quality_n.get(stratum):
            records.append(
                {
                    "stratum": stratum,
                    "realized_score": type_quality[stratum] / Decimal(type_quality_n[stratum]),
                    "selected": True,
                }
            )

    config_hash = hashlib.sha256(json.dumps(dataclasses.asdict(cell), default=str, sort_keys=True).encode()).hexdigest()
    result = RunResult(
        cell_id=cell.cell_id,
        config_hash=config_hash,
        trace_hash=trace.trace_hash,
        method_id=cell.method_id,
        seed=cell.seed,
        gamma=str(cell.gamma),
        budget_ratio=str(cell.budget_ratio),
        invariant_status=invariant_status,
        task_count=len(trace.tasks_by_id),
        worker_count=len(trace.workers),
        activated_count=activated_count,
        contracted_count=contracted_count,
        purchased_count=purchased_count,
        base_paid=str(ledger.paid_base),
        score_paid=str(ledger.paid_score),
        total_paid=str(ledger.paid),
        final_ledger=ledger.snapshot(),
        final_shadow=shadow.snapshot(),
        effort_histogram=effort_hist,
        gross_external_value=str(gross_total),
        platform_net_value=str(gross_total - ledger.paid),
        failure_counts=failure_counts,
        runtime_seconds=elapsed,
        peak_memory_mb=0.0,
        feedback_queue_mode="calendar-time-predecision-plus-postdecision-delay0",
        mean_outstanding_score_escrow=str(outstanding_score_sum / Decimal(HORIZON)),
        peak_outstanding_score_escrow=str(peak_outstanding_score),
        terminal_outstanding_score_escrow=str(terminal_outstanding_score),
        mean_outstanding_task_escrows=Decimal(outstanding_task_sum) / Decimal(HORIZON),
        peak_outstanding_task_escrows=peak_outstanding_tasks,
        terminal_pending_task_count=terminal_pending_task_count,
        terminal_pending_feedback_count=terminal_pending_feedback_count,
    )
    result.rejection_counts = rejection_counts
    result.trust_trajectory = trust_trajectory
    result.value_prefix = value_prefix
    result.runtime_breakdown = {
        "selection_and_critical_payment": round(t_selection, 3),
        "screening": round(t_screening, 3),
        "settlement": round(t_settlement, 3),
        "other": round(max(0.0, elapsed - t_selection - t_screening - t_settlement), 3),
        "total": round(elapsed, 3),
    }
    rho, topk = spearman_and_topk(mc_estimated, mc_realized)
    result.mc_correlation = rho
    result.mc_top_k_overlap = topk
    if mc_estimated:
        undefined = sum(1 for v in mc_realized if v == 0)
        result.mc_undefined_rate = Decimal(undefined) / Decimal(len(mc_realized))
    if deadline_total:
        result.deadline_satisfaction = Decimal(deadline_met) / Decimal(deadline_total)
    if compute_lp:
        lp_result = compute_lp_gap(
            lp_cache,
            cell.lp_cache_key(),
            trace,
            budget,
            cell.gamma,
            gross_total,
            arrival_multiplier=cell.arrival_multiplier,
        )
        result.lp = lp_result
    final_trust_by_stratum = {
        stratum: [trust.values[w] for w in ids] for stratum, ids in workers_by_stratum.items()
    }
    if trust.feedback_submission_count != sum(
        1 for event in trust_events if event.get("feedback")
    ):
        raise AssertionError("trust feedback submission count does not match completed feedback events")
    if trust.transition_count != sum(
        1 for event in trust_events if event.get("trust_transition_applied")
    ):
        raise AssertionError("trust transition count does not match applied transition events")
    if trust.duplicate_feedback_suppressed_count != sum(
        1 for event in trust_events if event.get("duplicate_feedback_suppressed")
    ):
        raise AssertionError("duplicate trust suppression count does not match event audit")
    return finalize_run_result(
        result,
        records=records,
        screening_events=screening_events,
        trust_events=trust_events,
        population_size=len(trace.workers),
        final_trust_by_stratum=final_trust_by_stratum,
        selected_count_by_stratum=type_selected,
    )


_WORKER: dict[str, Any] = {}


def _init_formal_worker(data_root: str, trace_hashes: dict[str, Any]) -> None:
    _WORKER["data_root"] = Path(data_root)
    _WORKER["trace_hashes"] = trace_hashes
    _WORKER["traces"] = {}
    _WORKER["lp_cache"] = LPComparatorCache()


def _get_worker_trace(seed: int) -> TraceBundle:
    traces: dict[int, TraceBundle] = _WORKER["traces"]
    if seed not in traces:
        traces[seed] = load_trace(
            seed,
            _WORKER["data_root"],
            _WORKER["trace_hashes"],
            verify_hashes=False,
            # Simulation only needs available_by_slot; skip the full eligibility
            # index (LP-only) to roughly halve per-worker memory so more workers
            # fit in RAM. Logic-preserving for simulate_cell.
            load_eligibility_index=False,
        )
    return traces[seed]


def _cell_from_payload(payload: dict[str, Any]) -> RunCell:
    return RunCell(
        cell_id=payload["cell_id"],
        family=payload["family"],
        seed=int(payload["seed"]),
        method_id=payload["method_id"],
        gamma=Decimal(payload["gamma"]),
        budget_ratio=Decimal(payload["budget_ratio"]),
        contamination=Decimal(payload["contamination"]),
        delay=int(payload["delay"]),
        missing_prob=Decimal(payload["missing_prob"]),
        arrival_multiplier=Decimal(payload["arrival_multiplier"]),
        alpha=Decimal(payload.get("alpha", "0.2")),
        theta_a=Decimal(payload.get("theta_a", "0.75")),
        lambda_max=Decimal(payload.get("lambda_max", "10")),
        order_index=int(payload["order_index"]),
    )


def _cell_payload(cell: RunCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "seed": cell.seed,
        "method_id": cell.method_id,
        "gamma": str(cell.gamma),
        "budget_ratio": str(cell.budget_ratio),
        "contamination": str(cell.contamination),
        "delay": cell.delay,
        "missing_prob": str(cell.missing_prob),
        "arrival_multiplier": str(cell.arrival_multiplier),
        "alpha": str(cell.alpha),
        "theta_a": str(cell.theta_a),
        "lambda_max": str(cell.lambda_max),
        "order_index": cell.order_index,
    }


def _write_result_files(output_root: Path, cell: RunCell, payload: dict[str, Any]) -> None:
    raw_dir = output_root / "raw"
    per_seed_dir = output_root / "per_seed" / str(cell.seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    tmp = raw_dir / f".{cell.cell_id}.tmp"
    final = raw_dir / f"{cell.cell_id}.json"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(final)
    (per_seed_dir / f"{cell.cell_id}.json").write_text(text, encoding="utf-8")


def _simulate_seed_batch(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one seed/gamma partition of pending simulation cells.

    The scheduler creates one job per seed/gamma pair so a 10-seed REAL-CAL run
    exposes 60 independent jobs to the process pool. Each job loads the heavy
    base once and only the active gamma's contracts.
    """
    output_root = Path(job["output_root"])
    seed = int(job["seed"])
    trace = load_trace(
        seed,
        _WORKER["data_root"],
        _WORKER["trace_hashes"],
        verify_hashes=False,
        load_eligibility_index=False,
        load_contracts_flag=False,
    )
    lp_cache = LPComparatorCache()
    out: list[dict[str, Any]] = []
    # Preserve original pending order within each gamma group.
    by_gamma: dict[str, list[dict[str, Any]]] = {}
    gamma_order: list[str] = []
    for cell_payload in job["cells"]:
        g = cell_payload["gamma"]
        if g not in by_gamma:
            by_gamma[g] = []
            gamma_order.append(g)
        by_gamma[g].append(cell_payload)
    for gamma_str in gamma_order:
        gamma = Decimal(gamma_str)
        load_contracts(trace, load_gammas=frozenset({gamma}))
        for cell_payload in by_gamma[gamma_str]:
            cell = _cell_from_payload(cell_payload)
            result = simulate_cell(cell, trace, lp_cache, compute_lp=False)
            _write_result_files(output_root, cell, result.to_dict())
            out.append(
                {
                    "cell_id": cell.cell_id,
                    "order_index": cell.order_index,
                    "invariant_status": result.invariant_status,
                    "seed": cell.seed,
                    "method_id": cell.method_id,
                    "gamma": str(cell.gamma),
                    "runtime_seconds": result.runtime_seconds,
                }
            )
            if len(out) == 1 or len(out) % 10 == 0:
                print(
                    f"[formal-worker] seed={seed} gamma={gamma_str} "
                    f"cells_written={len(out)} last={cell.cell_id} "
                    f"runtime={result.runtime_seconds:.1f}s",
                    flush=True,
                )
    return out


V2_SIMULATION_CHUNK_CELLS = 5
V2_LP_CHUNK_CACHE_KEYS = 8


def _build_seed_gamma_jobs(
    cells: list[RunCell] | tuple[RunCell, ...],
    output_root: Path,
    *,
    max_cells_per_job: int | None = None,
) -> list[dict[str, Any]]:
    """Partition cells deterministically by seed/gamma and optional chunks.

    REAL-CAL-V2 places 700/750 cells at gamma=0.3.  A single job per
    seed/gamma therefore leaves only ten useful workers after the 50 one-cell
    sensitivity jobs finish.  Fixed-size chunks preserve the one-gamma trace
    loading contract while exposing enough independent work to keep the pool
    busy.  ``None`` retains the legacy V1 partition exactly.
    """
    if max_cells_per_job is not None and max_cells_per_job <= 0:
        raise ValueError("max_cells_per_job must be positive or None")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell.seed, str(cell.gamma)), []).append(_cell_payload(cell))
    jobs: list[dict[str, Any]] = []
    for (seed, gamma), payloads in sorted(
        grouped.items(), key=lambda item: (item[0][0], Decimal(item[0][1]))
    ):
        size = max_cells_per_job or len(payloads)
        chunks = [payloads[start : start + size] for start in range(0, len(payloads), size)]
        for chunk_index, chunk in enumerate(chunks):
            jobs.append(
                {
                    "seed": seed,
                    "gamma": gamma,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "cells": chunk,
                    "output_root": str(output_root),
                }
            )
    return jobs


def _build_seed_gamma_lp_jobs(
    cells: list[RunCell] | tuple[RunCell, ...],
    output_root: Path,
    *,
    max_cache_keys_per_job: int | None = None,
) -> list[dict[str, Any]]:
    """Partition LP work without splitting cells that share one LP cache key.

    Chunking raw cells directly would solve the same offline LP repeatedly for
    method ablations that share an environment.  This builder keeps every
    cache-key equivalence class in one job, then packs a bounded number of
    unique keys per chunk.  It improves parallelism without sacrificing cache
    reuse inside a chunk.
    """
    if max_cache_keys_per_job is not None and max_cache_keys_per_job <= 0:
        raise ValueError("max_cache_keys_per_job must be positive or None")
    grouped: dict[tuple[int, str], dict[tuple, list[dict[str, Any]]]] = {}
    for cell in cells:
        by_key = grouped.setdefault((cell.seed, str(cell.gamma)), {})
        by_key.setdefault(cell.lp_cache_key(), []).append(_cell_payload(cell))

    jobs: list[dict[str, Any]] = []
    for (seed, gamma), by_key in sorted(
        grouped.items(), key=lambda item: (item[0][0], Decimal(item[0][1]))
    ):
        key_groups = list(by_key.values())
        size = max_cache_keys_per_job or len(key_groups)
        chunks = [key_groups[start : start + size] for start in range(0, len(key_groups), size)]
        for chunk_index, chunk_groups in enumerate(chunks):
            jobs.append(
                {
                    "seed": seed,
                    "gamma": gamma,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "cache_key_count": len(chunk_groups),
                    "cells": [payload for group in chunk_groups for payload in group],
                    "output_root": str(output_root),
                }
            )
    return jobs


def _lp_payload(result: LPComparatorResult) -> dict[str, str | None]:
    return {
        "status": result.status,
        "opt_lp": str(result.opt_lp) if result.opt_lp is not None else None,
        "u_online": str(result.u_online) if result.u_online is not None else None,
        "lp_gap": str(result.lp_gap) if result.lp_gap is not None else None,
        "normalized_lp_gap": (
            str(result.normalized_lp_gap) if result.normalized_lp_gap is not None else None
        ),
        "max_violation": str(result.max_violation) if result.max_violation is not None else None,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _apply_lp_seed_gamma_batch(job: dict[str, Any]) -> dict[str, Any]:
    """Solve and attach LP comparators for one seed/gamma partition.

    ``refresh`` discards previously stored LP results (used when the comparator
    definition changes, e.g. the repaired realized-cost model) and recomputes both
    the full-horizon LP and the prefix regret curve.
    """
    output_root = Path(job["output_root"])
    seed = int(job["seed"])
    gamma = Decimal(job["gamma"])
    refresh = bool(job.get("refresh", False))
    trace = load_trace(
        seed,
        _WORKER["data_root"],
        _WORKER["trace_hashes"],
        verify_hashes=False,
        load_eligibility_index=True,
        load_contracts_flag=False,
    )
    load_contracts(trace, load_gammas=frozenset({gamma}))
    base_budget = reference_budget(trace)
    cache = LPComparatorCache()
    solved_keys = 0
    updated_cells = 0
    skipped_optimal = 0
    statuses: dict[str, int] = {}

    for cell_payload in job["cells"]:
        cell = _cell_from_payload(cell_payload)
        path = output_root / "raw" / f"{cell.cell_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = cell.lp_cache_key()
        existing = payload.get("lp", {})
        if refresh:
            payload.pop("lp_prefix", None)
        elif existing.get("status") == "optimal" and existing.get("opt_lp") is not None:
            if cache.get(key) is None:
                cache.put(
                    key,
                    LPComparatorResult(
                        status="optimal",
                        opt_lp=Decimal(existing["opt_lp"]),
                        max_violation=(
                            Decimal(existing["max_violation"])
                            if existing.get("max_violation") is not None
                            else None
                        ),
                    ),
                )
            skipped_optimal += 1
            continue

        was_cached = cache.get(key) is not None
        budget = (base_budget * cell.budget_ratio).quantize(Decimal("0.001"))
        result = compute_lp_gap(
            cache,
            key,
            trace,
            budget,
            cell.gamma,
            Decimal(payload.get("gross_external_value", "0")),
            arrival_multiplier=cell.arrival_multiplier,
            delay=cell.delay,
            missing_prob=cell.missing_prob,
            horizon=HORIZON,
        )
        if not was_cached:
            solved_keys += 1
        payload["lp"] = _lp_payload(result)

        # E5 empirical regret: prefix-horizon LP curve for the canonical cells.
        # Reg(T) = OPT_LP(prefix-T arrivals, FULL budget B) - U_online(prefix T).
        # The hindsight benchmark faces the same total budget constraint but is
        # free to spend all of it within the first T slots — exactly like the
        # online mechanism. (V2.1 scaled the prefix budget by T/H, which made
        # the benchmark strictly weaker than a front-loading online policy and
        # produced negative "gaps".)
        if is_regret_cell(cell) and payload.get("value_prefix") and "lp_prefix" not in payload:
            prefix_out: dict[str, dict[str, str | None]] = {}
            for t_str, u_str in sorted(payload["value_prefix"].items(), key=lambda kv: int(kv[0])):
                horizon_t = int(t_str)
                res_t = compute_lp_gap(
                    cache,
                    key + (horizon_t,),
                    trace,
                    budget,
                    cell.gamma,
                    Decimal(u_str),
                    arrival_multiplier=cell.arrival_multiplier,
                    delay=cell.delay,
                    missing_prob=cell.missing_prob,
                    horizon=horizon_t,
                )
                prefix_out[t_str] = _lp_payload(res_t)
            payload["lp_prefix"] = prefix_out

        _write_json_atomic(path, payload)
        updated_cells += 1
        statuses[result.status] = statuses.get(result.status, 0) + 1

    return {
        "seed": seed,
        "gamma": str(gamma),
        "updated_cells": updated_cells,
        "skipped_optimal": skipped_optimal,
        "solved_keys": solved_keys,
        "statuses": statuses,
    }


def _adopt_existing_raw(checkpoint: CheckpointState, raw_dir: Path) -> int:
    """Mark cells already written under raw/ as complete (crash-safe resume)."""
    if not raw_dir.exists():
        return 0
    adopted = 0
    for path in raw_dir.glob("*.json"):
        if path.name.startswith("."):
            continue
        cell_id = path.stem
        if checkpoint.is_complete(cell_id):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        order_index = int(payload.get("order_index", -1))
        status = payload.get("invariant_status", "PASS")
        if status != "PASS":
            checkpoint.mark_invalid(cell_id, str(status), order_index)
        else:
            checkpoint.mark_complete(cell_id, order_index)
        adopted += 1
    return adopted


@dataclass
class FormalRunner:
    data_root: Path
    output_root: Path
    trace_hashes: dict[str, Any]
    run_version: str = "formal-r4-familyid-v1.1.0"
    workers: int | None = None
    seeds: tuple[int, ...] | None = None
    matrix_version: str = "v1"
    lp_refresh: bool = False
    lp_cache: LPComparatorCache = field(default_factory=LPComparatorCache)
    trace_cache: dict[int, TraceBundle] = field(default_factory=dict)

    def get_trace(self, seed: int) -> TraceBundle:
        if seed not in self.trace_cache:
            self.trace_cache[seed] = load_trace(
                seed,
                self.data_root,
                self.trace_hashes,
                verify_hashes=False,
            )
        return self.trace_cache[seed]

    def _worker_count(self) -> int:
        if self.workers is not None and self.workers > 0:
            return int(self.workers)
        return max(1, os.cpu_count() or 1)

    def _apply_lp_gaps(self, matrix: tuple[RunCell, ...], workers: int) -> None:
        chunk_keys = V2_LP_CHUNK_CACHE_KEYS if self.matrix_version == "v2" else None
        jobs = _build_seed_gamma_lp_jobs(
            matrix,
            self.output_root,
            max_cache_keys_per_job=chunk_keys,
        )
        if self.lp_refresh:
            for job in jobs:
                job["refresh"] = True
        print(
            f"[formal-lp] jobs={len(jobs)} partition=seed_gamma_lpkey_chunk "
            f"chunk_keys={chunk_keys or 'all'} refresh={self.lp_refresh} workers={workers}",
            flush=True,
        )
        if not jobs:
            return
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            initializer=_init_formal_worker,
            initargs=(str(self.data_root), self.trace_hashes),
        ) as pool:
            futures = {pool.submit(_apply_lp_seed_gamma_batch, job): job for job in jobs}
            completed_jobs = 0
            for future in as_completed(futures):
                result = future.result()
                completed_jobs += 1
                job = futures[future]
                print(
                    f"[formal-lp] seed={result['seed']} gamma={result['gamma']} "
                    f"chunk={job['chunk_index'] + 1}/{job['chunk_count']} "
                    f"updated={result['updated_cells']} solved_keys={result['solved_keys']} "
                    f"progress={completed_jobs}/{len(jobs)} statuses={result['statuses']}",
                    flush=True,
                )

    def run_all(self, *, resume: bool = True) -> dict[str, Any]:
        matrix = build_run_matrix_v2() if self.matrix_version == "v2" else build_run_matrix()
        if self.seeds is not None:
            allowed = set(self.seeds)
            matrix = tuple(cell for cell in matrix if cell.seed in allowed)
        checkpoint_path = self.output_root / "audit" / "checkpoint.json"
        checkpoint = CheckpointState.load(checkpoint_path) if resume else CheckpointState("", [], [], -1)
        if checkpoint.run_version not in ("", self.run_version):
            raise ValueError(
                f"checkpoint run_version mismatch: {checkpoint.run_version!r} != {self.run_version!r}"
            )
        checkpoint.run_version = self.run_version
        (self.output_root / "raw").mkdir(parents=True, exist_ok=True)
        (self.output_root / "per_seed").mkdir(parents=True, exist_ok=True)
        (self.output_root / "audit").mkdir(parents=True, exist_ok=True)

        if resume:
            adopted = _adopt_existing_raw(checkpoint, self.output_root / "raw")
            if adopted:
                checkpoint.save(checkpoint_path)
                print(f"[formal] adopted {adopted} existing raw cells into checkpoint", flush=True)

        pending = [cell for cell in matrix if not checkpoint.is_complete(cell.cell_id)]
        completed = len(matrix) - len(pending)
        invalid = len(checkpoint.invalid_cells)
        total = len(matrix)
        workers = self._worker_count()

        print(
            f"[formal] resume={resume} pending={len(pending)}/{total} "
            f"workers={workers} cpu_count={os.cpu_count()}",
            flush=True,
        )

        if not pending:
            self._apply_lp_gaps(matrix, workers)
            return {
                "total_cells": total,
                "completed": completed,
                "invalid": invalid,
                "workers": workers,
                "run_version": self.run_version,
            }

        chunk_cells = V2_SIMULATION_CHUNK_CELLS if self.matrix_version == "v2" else None
        simulation_jobs = _build_seed_gamma_jobs(
            pending,
            self.output_root,
            max_cells_per_job=chunk_cells,
        )
        print(
            f"[formal] simulation_jobs={len(simulation_jobs)} "
            f"partition=seed_gamma_chunk chunk_cells={chunk_cells or 'all'}",
            flush=True,
        )
        done_since_start = 0
        with ProcessPoolExecutor(
            max_workers=min(workers, len(simulation_jobs)),
            initializer=_init_formal_worker,
            initargs=(str(self.data_root), self.trace_hashes),
        ) as pool:
            futures = {pool.submit(_simulate_seed_batch, job): job for job in simulation_jobs}
            for future in as_completed(futures):
                batch = future.result()
                for result in batch:
                    done_since_start += 1
                    if result["invariant_status"] != "PASS":
                        checkpoint.mark_invalid(
                            result["cell_id"],
                            result["invariant_status"],
                            result["order_index"],
                        )
                        invalid += 1
                    else:
                        checkpoint.mark_complete(result["cell_id"], result["order_index"])
                        completed += 1
                checkpoint.save(checkpoint_path)
                job = futures[future]
                print(
                    f"[formal] seed={job['seed']} gamma={job['gamma']} "
                    f"chunk={job['chunk_index'] + 1}/{job['chunk_count']} "
                    f"done batch_size={len(batch)} "
                    f"progress {completed}/{total} (+{done_since_start}/{len(pending)})",
                    flush=True,
                )
            checkpoint.save(checkpoint_path)

        self._apply_lp_gaps(matrix, workers)

        return {
            "total_cells": total,
            "completed": completed,
            "invalid": invalid,
            "workers": workers,
            "run_version": self.run_version,
        }
