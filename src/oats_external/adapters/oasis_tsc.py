from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Iterable, Mapping, Sequence

from ..types import CandidateKey


ZERO = Decimal("0")
ONE = Decimal("1")
DEFAULT_PHI = Decimal("0.4")
DEFAULT_INITIAL_QUALITY = Decimal("0.5")
DEFAULT_TOLERANCE = Decimal("1e-9")
DEFAULT_EPSILON = Decimal("1e-18")
DEFAULT_MAX_ITERATIONS = 500


class OasisSpecificationViolation(RuntimeError):
    pass


def _decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class OasisCandidate:
    key: CandidateKey
    bid: Decimal
    long_term_quality: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "bid", _decimal(self.bid))
        object.__setattr__(
            self, "long_term_quality", _decimal(self.long_term_quality)
        )
        if self.bid <= ZERO:
            raise OasisSpecificationViolation("Oasis requires a strictly positive bid")
        if not ZERO <= self.long_term_quality <= ONE:
            raise OasisSpecificationViolation("long-term quality must be in [0, 1]")

    @property
    def rho(self) -> Decimal:
        return self.long_term_quality / self.bid


@dataclass(frozen=True, slots=True)
class OasisTaskSelection:
    task_id: str
    task_budget: Decimal
    capacity: int
    actual_candidate_count: int
    forecast_candidate_count: int
    observation_count: int
    winners: tuple[CandidateKey, ...]
    basic_payments: Mapping[CandidateKey, Decimal]
    threshold_rho: Decimal | None
    threshold_bid: Decimal | None
    observation_exclusions: int
    sample_replacements: int
    basic_budget_violation: bool
    decision_hash: str


@dataclass(frozen=True, slots=True)
class OasisTruthResult:
    truth: Decimal
    current_quality: Mapping[CandidateKey, Decimal]
    normalized_quality: Mapping[CandidateKey, Decimal]
    iterations: int
    converged: bool
    zero_range_normalization: bool


@dataclass(frozen=True, slots=True)
class OasisTaskSettlement:
    task_id: str
    truth: Decimal | None
    current_quality: Mapping[CandidateKey, Decimal]
    normalized_quality: Mapping[CandidateKey, Decimal]
    basic_payments: Mapping[CandidateKey, Decimal]
    actual_payments: Mapping[CandidateKey, Decimal]
    total_basic_payment: Decimal
    total_actual_payment: Decimal
    truth_iterations: int
    truth_converged: bool
    zero_range_normalization: bool
    actual_budget_violation: bool
    individual_rationality_violations: int
    settlement_hash: str


class OasisTSCPolicy:
    """Paper-faithful core of Oasis, IEEE TSC 2024, Algorithms 1 and 2.

    This adapter deliberately exposes task-level methods because Oasis assigns a
    budget to each sensing task.  The REAL-CAL runner is responsible for the
    preregistered public arrival order and for applying same-slot quality updates
    only after all task selections in that slot have completed.
    """

    method_id = "OASIS-TSC-2024"
    paper_doi = "10.1109/TSC.2024.3354240"

    def __init__(
        self,
        *,
        phi: Decimal | str = DEFAULT_PHI,
        initial_quality: Decimal | str = DEFAULT_INITIAL_QUALITY,
        truth_tolerance: Decimal | str = DEFAULT_TOLERANCE,
        truth_epsilon: Decimal | str = DEFAULT_EPSILON,
        truth_max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.phi = _decimal(phi)
        self.initial_quality = _decimal(initial_quality)
        self.truth_tolerance = _decimal(truth_tolerance)
        self.truth_epsilon = _decimal(truth_epsilon)
        self.truth_max_iterations = int(truth_max_iterations)
        if not ZERO <= self.phi <= ONE:
            raise OasisSpecificationViolation("phi must be in [0, 1]")
        if not ZERO <= self.initial_quality <= ONE:
            raise OasisSpecificationViolation("initial quality must be in [0, 1]")
        if self.truth_tolerance <= ZERO or self.truth_epsilon <= ZERO:
            raise OasisSpecificationViolation("truth tolerances must be positive")
        if self.truth_max_iterations < 1:
            raise OasisSpecificationViolation("truth iteration cap must be positive")
        self._quality: dict[str, Decimal] = {}
        self._trace_seed: int | None = None
        self._selection_count = 0
        self._settlement_count = 0
        self._update_count = 0

    def reset(self, trace_seed: int) -> None:
        self._trace_seed = int(trace_seed)
        self._quality.clear()
        self._selection_count = 0
        self._settlement_count = 0
        self._update_count = 0

    def quality_of(self, worker_id: str) -> Decimal:
        return self._quality.get(worker_id, self.initial_quality)

    def snapshot_candidates(
        self, task_id: str, bids: Mapping[str, Decimal]
    ) -> tuple[OasisCandidate, ...]:
        return tuple(
            OasisCandidate(
                key=CandidateKey(task_id=task_id, worker_id=worker_id),
                bid=bid,
                long_term_quality=self.quality_of(worker_id),
            )
            for worker_id, bid in sorted(bids.items())
        )

    @staticmethod
    def _basic_payment(candidate: OasisCandidate, threshold_rho: Decimal) -> Decimal:
        if threshold_rho <= ZERO:
            raise OasisSpecificationViolation("threshold rho must be positive")
        return candidate.bid * candidate.rho / threshold_rho

    def _initial_sample(
        self,
        observation: Sequence[OasisCandidate],
        task_budget: Decimal,
    ) -> list[OasisCandidate]:
        ordered = sorted(
            observation,
            key=lambda candidate: (-candidate.rho, candidate.key.worker_id),
        )
        best: list[OasisCandidate] = []
        for k in range(1, len(ordered) + 1):
            prefix = ordered[:k]
            threshold_rho = prefix[-1].rho
            total = sum(
                (self._basic_payment(candidate, threshold_rho) for candidate in prefix),
                ZERO,
            )
            if total <= task_budget:
                best = list(prefix)
        return best

    def select_task(
        self,
        *,
        task_id: str,
        task_budget: Decimal | str,
        capacity: int,
        ordered_candidates: Sequence[OasisCandidate],
        forecast_candidate_count: int,
    ) -> OasisTaskSelection:
        budget = _decimal(task_budget)
        if budget < ZERO:
            raise OasisSpecificationViolation("negative task budget")
        if capacity < 1:
            raise OasisSpecificationViolation("task capacity must be positive")
        if len({candidate.key for candidate in ordered_candidates}) != len(
            ordered_candidates
        ):
            raise OasisSpecificationViolation("duplicate task candidate")
        if any(candidate.key.task_id != task_id for candidate in ordered_candidates):
            raise OasisSpecificationViolation("candidate belongs to another task")

        actual_count = len(ordered_candidates)
        forecast = max(0, int(forecast_candidate_count))
        if actual_count < 2 or forecast < 2:
            observation_count = actual_count
            payload = {
                "task_id": task_id,
                "budget": str(budget),
                "capacity": capacity,
                "actual_candidate_count": actual_count,
                "forecast_candidate_count": forecast,
                "observation_count": observation_count,
                "winners": [],
                "reason": "insufficient_arrivals",
            }
            self._selection_count += 1
            return OasisTaskSelection(
                task_id=task_id,
                task_budget=budget,
                capacity=capacity,
                actual_candidate_count=actual_count,
                forecast_candidate_count=forecast,
                observation_count=observation_count,
                winners=(),
                basic_payments={},
                threshold_rho=None,
                threshold_bid=None,
                observation_exclusions=observation_count,
                sample_replacements=0,
                basic_budget_violation=False,
                decision_hash=_canonical_hash(payload),
            )

        with localcontext() as context:
            context.prec = 50
            # Paper Algorithm 1: delta = N/e.  Decimal keeps the frozen
            # forecast conversion independent of the platform math library.
            e_value = context.exp(ONE)
            observation_count = int(Decimal(forecast) / e_value)
        observation_count = min(max(1, observation_count), actual_count - 1)
        observation = ordered_candidates[:observation_count]
        sample = self._initial_sample(observation, budget)

        if not sample:
            payload = {
                "task_id": task_id,
                "budget": str(budget),
                "capacity": capacity,
                "actual_candidate_count": actual_count,
                "forecast_candidate_count": forecast,
                "observation_count": observation_count,
                "winners": [],
                "reason": "no_budget_feasible_sample",
            }
            self._selection_count += 1
            return OasisTaskSelection(
                task_id=task_id,
                task_budget=budget,
                capacity=capacity,
                actual_candidate_count=actual_count,
                forecast_candidate_count=forecast,
                observation_count=observation_count,
                winners=(),
                basic_payments={},
                threshold_rho=None,
                threshold_bid=None,
                observation_exclusions=observation_count,
                sample_replacements=0,
                basic_budget_violation=False,
                decision_hash=_canonical_hash(payload),
            )

        winners: list[CandidateKey] = []
        payments: dict[CandidateKey, Decimal] = {}
        payment_total = ZERO
        replacements = 0

        for candidate in ordered_candidates[observation_count:]:
            worst = sorted(
                sample,
                key=lambda item: (item.rho, item.key.worker_id),
            )[0]
            threshold_rho = worst.rho
            threshold_bid = worst.bid
            if candidate.rho > threshold_rho:
                proposed = self._basic_payment(candidate, threshold_rho)
                if (
                    candidate.bid < threshold_bid
                    and len(winners) < capacity
                    and payment_total + proposed <= budget
                ):
                    winners.append(candidate.key)
                    payments[candidate.key] = proposed
                    payment_total += proposed

                sample.remove(worst)
                sample.append(candidate)
                replacements += 1

        final_worst = sorted(
            sample,
            key=lambda item: (item.rho, item.key.worker_id),
        )[0]
        violation = payment_total > budget
        payload = {
            "task_id": task_id,
            "budget": str(budget),
            "capacity": capacity,
            "actual_candidate_count": actual_count,
            "forecast_candidate_count": forecast,
            "observation_count": observation_count,
            "winners": [
                {
                    "task_id": key.task_id,
                    "worker_id": key.worker_id,
                    "basic_payment": str(payments[key]),
                }
                for key in winners
            ],
            "threshold_rho": str(final_worst.rho),
            "threshold_bid": str(final_worst.bid),
            "sample_replacements": replacements,
        }
        self._selection_count += 1
        return OasisTaskSelection(
            task_id=task_id,
            task_budget=budget,
            capacity=capacity,
            actual_candidate_count=actual_count,
            forecast_candidate_count=forecast,
            observation_count=observation_count,
            winners=tuple(winners),
            basic_payments=dict(payments),
            threshold_rho=final_worst.rho,
            threshold_bid=final_worst.bid,
            observation_exclusions=observation_count,
            sample_replacements=replacements,
            basic_budget_violation=violation,
            decision_hash=_canonical_hash(payload),
        )

    def discover_truth(
        self, reports: Mapping[CandidateKey, Decimal | str]
    ) -> OasisTruthResult:
        if not reports:
            raise OasisSpecificationViolation("truth discovery requires reports")
        values = {key: _decimal(value) for key, value in reports.items()}
        ordered_keys = sorted(values)
        with localcontext() as context:
            context.prec = 50
            truth = sum((values[key] for key in ordered_keys), ZERO) / Decimal(
                len(ordered_keys)
            )
            converged = False
            iterations = 0
            current_quality: dict[CandidateKey, Decimal] = {}
            for iteration in range(1, self.truth_max_iterations + 1):
                distances = {
                    key: abs(values[key] - truth) for key in ordered_keys
                }
                distance_sum = sum(distances.values(), ZERO)
                if distance_sum <= self.truth_epsilon:
                    current_quality = {key: ONE for key in ordered_keys}
                    converged = True
                    iterations = iteration
                    break
                weights: dict[CandidateKey, Decimal] = {}
                for key in ordered_keys:
                    ratio = distances[key] / distance_sum
                    ratio = min(ONE, max(self.truth_epsilon, ratio))
                    weights[key] = -context.ln(ratio)
                weight_sum = sum(weights.values(), ZERO)
                if weight_sum <= self.truth_epsilon:
                    raise OasisSpecificationViolation(
                        "truth-discovery weight sum is zero"
                    )
                new_truth = (
                    sum(
                        (weights[key] * values[key] for key in ordered_keys),
                        ZERO,
                    )
                    / weight_sum
                )
                iterations = iteration
                if abs(new_truth - truth) <= self.truth_tolerance:
                    truth = new_truth
                    converged = True
                    distances = {
                        key: abs(values[key] - truth) for key in ordered_keys
                    }
                    current_quality = {
                        key: context.exp(-distances[key]) for key in ordered_keys
                    }
                    break
                truth = new_truth

            if not current_quality:
                distances = {
                    key: abs(values[key] - truth) for key in ordered_keys
                }
                current_quality = {
                    key: context.exp(-distances[key]) for key in ordered_keys
                }

            minimum = min(current_quality.values())
            maximum = max(current_quality.values())
            zero_range = maximum - minimum <= self.truth_epsilon
            if zero_range:
                normalized = {key: ONE for key in ordered_keys}
            else:
                normalized = {
                    key: (current_quality[key] - minimum) / (maximum - minimum)
                    for key in ordered_keys
                }

        return OasisTruthResult(
            truth=truth,
            current_quality=current_quality,
            normalized_quality=normalized,
            iterations=iterations,
            converged=converged,
            zero_range_normalization=zero_range,
        )

    def settle_task(
        self,
        selection: OasisTaskSelection,
        reports: Mapping[CandidateKey, Decimal | str],
        bids: Mapping[CandidateKey, Decimal | str],
    ) -> OasisTaskSettlement:
        winner_set = set(selection.winners)
        if set(reports) != winner_set or set(bids) != winner_set:
            raise OasisSpecificationViolation(
                "settlement inputs must exactly match recruited winners"
            )
        if not selection.winners:
            payload = {"task_id": selection.task_id, "winners": []}
            self._settlement_count += 1
            return OasisTaskSettlement(
                task_id=selection.task_id,
                truth=None,
                current_quality={},
                normalized_quality={},
                basic_payments={},
                actual_payments={},
                total_basic_payment=ZERO,
                total_actual_payment=ZERO,
                truth_iterations=0,
                truth_converged=True,
                zero_range_normalization=False,
                actual_budget_violation=False,
                individual_rationality_violations=0,
                settlement_hash=_canonical_hash(payload),
            )

        truth_result = self.discover_truth(reports)
        mean_normalized = sum(
            truth_result.normalized_quality.values(), ZERO
        ) / Decimal(len(selection.winners))
        if mean_normalized <= self.truth_epsilon:
            raise OasisSpecificationViolation("mean normalized quality is zero")
        actual = {
            key: (
                truth_result.normalized_quality[key]
                / mean_normalized
                * selection.basic_payments[key]
            )
            for key in selection.winners
        }
        total_basic = sum(selection.basic_payments.values(), ZERO)
        total_actual = sum(actual.values(), ZERO)
        ir_violations = sum(
            1 for key in selection.winners if actual[key] < _decimal(bids[key])
        )
        payload = {
            "task_id": selection.task_id,
            "truth": str(truth_result.truth),
            "current_quality": {
                f"{key.task_id}|{key.worker_id}": str(value)
                for key, value in sorted(truth_result.current_quality.items())
            },
            "normalized_quality": {
                f"{key.task_id}|{key.worker_id}": str(value)
                for key, value in sorted(truth_result.normalized_quality.items())
            },
            "actual_payments": {
                f"{key.task_id}|{key.worker_id}": str(value)
                for key, value in sorted(actual.items())
            },
            "truth_iterations": truth_result.iterations,
            "truth_converged": truth_result.converged,
        }
        self._settlement_count += 1
        return OasisTaskSettlement(
            task_id=selection.task_id,
            truth=truth_result.truth,
            current_quality=truth_result.current_quality,
            normalized_quality=truth_result.normalized_quality,
            basic_payments=dict(selection.basic_payments),
            actual_payments=actual,
            total_basic_payment=total_basic,
            total_actual_payment=total_actual,
            truth_iterations=truth_result.iterations,
            truth_converged=truth_result.converged,
            zero_range_normalization=truth_result.zero_range_normalization,
            actual_budget_violation=total_actual > selection.task_budget,
            individual_rationality_violations=ir_violations,
            settlement_hash=_canonical_hash(payload),
        )

    def apply_quality_updates(
        self, updates: Iterable[tuple[CandidateKey, Decimal | str]]
    ) -> None:
        for key, current_quality in sorted(
            updates, key=lambda item: (item[0].task_id, item[0].worker_id)
        ):
            current = _decimal(current_quality)
            if not ZERO <= current <= ONE:
                raise OasisSpecificationViolation("current quality outside [0, 1]")
            previous = self.quality_of(key.worker_id)
            self._quality[key.worker_id] = (
                self.phi * previous + (ONE - self.phi) * current
            )
            self._update_count += 1

    def audit_state(self) -> dict[str, object]:
        quality_payload = {
            worker_id: str(quality)
            for worker_id, quality in sorted(self._quality.items())
        }
        return {
            "method_id": self.method_id,
            "paper_doi": self.paper_doi,
            "trace_seed": self._trace_seed,
            "phi": str(self.phi),
            "initial_quality": str(self.initial_quality),
            "truth_tolerance": str(self.truth_tolerance),
            "truth_epsilon": str(self.truth_epsilon),
            "truth_max_iterations": self.truth_max_iterations,
            "selection_count": self._selection_count,
            "settlement_count": self._settlement_count,
            "quality_update_count": self._update_count,
            "workers_with_quality_state": len(self._quality),
            "quality_state_hash": _canonical_hash(quality_payload),
        }
