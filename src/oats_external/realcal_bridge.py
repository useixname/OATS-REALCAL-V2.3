from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .types import CandidateKey, ExternalCandidateView, ExternalFeedback


ZERO = Decimal("0")
ONE = Decimal("1")
PUBLIC_BASE_CAP = Decimal("3.0")
REFERENCE_GAMMA_MAX = Decimal("1")
FIXED_OUTCOME_EFFORT = "0"
PUBLIC_CELL_COUNT = 100


class RealCalBridgeViolation(RuntimeError):
    pass


def _json_lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RealCalBridgeViolation(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RealCalBridgeViolation(
                    f"non-object JSON row at {path}:{line_number}"
                )
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicTask:
    slot: int
    task_id: str
    cell: int
    value: Decimal
    capacity: int
    deadline: int


@dataclass(frozen=True, slots=True)
class PrivateTaskOutcome:
    """Evaluation-only task state that is never passed to a policy."""

    delay_by_scenario: Mapping[str, int]
    missing_by_scenario: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class PurchasedOutcome:
    """Evaluation-only purchased potential at the fixed common effort."""

    quality: Decimal


@dataclass(frozen=True, slots=True)
class CompletionForecast:
    model_id: str
    training_seeds: tuple[int, ...]
    alpha: Decimal
    scenario_probabilities: Mapping[str, Mapping[str, Decimal]]
    scenario_global_probabilities: Mapping[str, Decimal]
    training_trace_hashes: Mapping[str, str]

    @staticmethod
    def scenario_key(delay: int, missing_prob: Decimal | str) -> str:
        return f"delay={int(delay)}|missing={Decimal(str(missing_prob))}"

    def predict(
        self,
        *,
        slot: int,
        deadline: int,
        delay: int,
        missing_prob: Decimal | str,
    ) -> Decimal:
        scenario = self.scenario_key(delay, missing_prob)
        if scenario not in self.scenario_global_probabilities:
            raise RealCalBridgeViolation(
                f"completion scenario is absent from frozen model: {scenario}"
            )
        slack = str(deadline - slot)
        return self.scenario_probabilities.get(scenario, {}).get(
            slack,
            self.scenario_global_probabilities[scenario],
        )

    def to_payload(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "training_seeds": list(self.training_seeds),
            "alpha": str(self.alpha),
            "target": (
                "not_missing_and_slot_plus_scenario_delay_not_after_deadline"
            ),
            "features": ["deadline_slack", "delay_scenario", "missing_scenario"],
            "smoothing": "beta_binomial_laplace",
            "scenario_probabilities": {
                scenario: {
                    slack: str(probability)
                    for slack, probability in sorted(probabilities.items())
                }
                for scenario, probabilities in sorted(
                    self.scenario_probabilities.items()
                )
            },
            "scenario_global_probabilities": {
                scenario: str(probability)
                for scenario, probability in sorted(
                    self.scenario_global_probabilities.items()
                )
            },
            "training_trace_hashes": dict(sorted(self.training_trace_hashes.items())),
        }
        payload["model_hash"] = canonical_hash(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CompletionForecast":
        supplied_hash = str(payload.get("model_hash", ""))
        unsigned = dict(payload)
        unsigned.pop("model_hash", None)
        if supplied_hash and canonical_hash(unsigned) != supplied_hash:
            raise RealCalBridgeViolation("completion forecast model hash mismatch")
        raw_probabilities = payload["scenario_probabilities"]
        raw_global = payload["scenario_global_probabilities"]
        if not isinstance(raw_probabilities, Mapping) or not isinstance(
            raw_global, Mapping
        ):
            raise RealCalBridgeViolation("invalid completion forecast payload")
        return cls(
            model_id=str(payload["model_id"]),
            training_seeds=tuple(int(seed) for seed in payload["training_seeds"]),
            alpha=Decimal(str(payload["alpha"])),
            scenario_probabilities={
                str(scenario): {
                    str(slack): Decimal(str(probability))
                    for slack, probability in probabilities.items()
                }
                for scenario, probabilities in raw_probabilities.items()
            },
            scenario_global_probabilities={
                str(scenario): Decimal(str(probability))
                for scenario, probability in raw_global.items()
            },
            training_trace_hashes={
                str(seed): str(value)
                for seed, value in dict(payload["training_trace_hashes"]).items()
            },
        )


def _seed_trace_hash(file_hashes: Mapping[str, str]) -> str:
    return canonical_hash(dict(sorted(file_hashes.items())))


def load_trace_hash_manifest(data_root: Path) -> dict[str, dict[str, str]]:
    path = data_root / "trace_hashes_realcal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != "REAL-CAL-V2":
        raise RealCalBridgeViolation("unexpected trace dataset id")
    rows = payload.get("seed_file_hashes")
    if not isinstance(rows, dict):
        raise RealCalBridgeViolation("trace hash manifest lacks seed_file_hashes")
    return {
        str(seed): {str(name): str(value) for name, value in hashes.items()}
        for seed, hashes in rows.items()
    }


def verify_seed_files(
    data_root: Path,
    seed: int,
    names: Sequence[str],
    trace_hash_manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    expected = trace_hash_manifest.get(str(seed))
    if expected is None:
        raise RealCalBridgeViolation(f"seed absent from trace hash manifest: {seed}")
    verified: dict[str, str] = {}
    for name in names:
        path = data_root / str(seed) / name
        if not path.is_file():
            raise RealCalBridgeViolation(f"missing trace file: {path}")
        actual = sha256_file(path)
        if actual != expected.get(name):
            raise RealCalBridgeViolation(f"trace hash mismatch: {seed}/{name}")
        verified[name] = actual
    return verified


def fit_completion_forecast(
    data_root: Path,
    training_seeds: Sequence[int],
    scenarios: Sequence[tuple[int, Decimal | str]],
    *,
    alpha: Decimal = ONE,
) -> CompletionForecast:
    if not training_seeds:
        raise RealCalBridgeViolation("completion forecast requires training seeds")
    if alpha <= ZERO:
        raise RealCalBridgeViolation("completion forecast alpha must be positive")
    trace_hash_manifest = load_trace_hash_manifest(data_root)
    counts: dict[str, dict[str, list[int]]] = {}
    global_counts: dict[str, list[int]] = {}
    training_trace_hashes: dict[str, str] = {}

    for delay, missing_prob in scenarios:
        scenario = CompletionForecast.scenario_key(delay, missing_prob)
        counts[scenario] = {}
        global_counts[scenario] = [0, 0]

    for seed in training_seeds:
        verified = verify_seed_files(
            data_root,
            int(seed),
            ("tasks.jsonl",),
            trace_hash_manifest,
        )
        training_trace_hashes[str(seed)] = _seed_trace_hash(verified)
        for row in _json_lines(data_root / str(seed) / "tasks.jsonl"):
            slot = int(row["slot"])
            deadline = int(row["deadline"])
            slack = str(deadline - slot)
            delay_mask = dict(row["delay_mask"])
            missing_mask = dict(row["missing_mask"])
            for delay, missing_prob in scenarios:
                scenario = CompletionForecast.scenario_key(delay, missing_prob)
                delay_key = str(int(delay))
                missing_key = str(Decimal(str(missing_prob)))
                if delay_key not in delay_mask or missing_key not in missing_mask:
                    raise RealCalBridgeViolation(
                        f"scenario absent from trace masks: {scenario}"
                    )
                completed = (
                    not bool(missing_mask[missing_key])
                    and slot + int(delay_mask[delay_key]) <= deadline
                )
                bucket = counts[scenario].setdefault(slack, [0, 0])
                bucket[0] += int(completed)
                bucket[1] += 1
                global_counts[scenario][0] += int(completed)
                global_counts[scenario][1] += 1

    probabilities: dict[str, dict[str, Decimal]] = {}
    global_probabilities: dict[str, Decimal] = {}
    for scenario, by_slack in counts.items():
        probabilities[scenario] = {}
        for slack, (successes, total) in by_slack.items():
            probabilities[scenario][slack] = (
                Decimal(successes) + alpha
            ) / (Decimal(total) + Decimal("2") * alpha)
        successes, total = global_counts[scenario]
        global_probabilities[scenario] = (
            Decimal(successes) + alpha
        ) / (Decimal(total) + Decimal("2") * alpha)

    return CompletionForecast(
        model_id="OTIM-REALCAL-COMPLETION-FORECAST-R1",
        training_seeds=tuple(int(seed) for seed in training_seeds),
        alpha=alpha,
        scenario_probabilities=probabilities,
        scenario_global_probabilities=global_probabilities,
        training_trace_hashes=training_trace_hashes,
    )


class RealCalExternalTrace:
    """Independent REAL-CAL reader with a narrow public-policy boundary."""

    def __init__(
        self,
        *,
        data_root: Path,
        seed: int,
        completion_forecast: CompletionForecast,
        delay: int,
        missing_prob: Decimal | str,
        max_arrivals: int,
        load_outcomes: bool = True,
    ) -> None:
        if max_arrivals < 2:
            raise RealCalBridgeViolation("smoke arrival cap must be at least two")
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.completion_forecast = completion_forecast
        self.delay = int(delay)
        self.missing_prob = Decimal(str(missing_prob))
        self.max_arrivals = int(max_arrivals)
        self.load_outcomes = bool(load_outcomes)
        self._trace_hash_manifest = load_trace_hash_manifest(self.data_root)
        verified_names = [
            "workers.jsonl",
            "tasks.jsonl",
            "eligibility.jsonl",
            "trace_metadata.json",
        ]
        if self.load_outcomes:
            verified_names.append("potential_reports.jsonl")
        self.verified_hashes = verify_seed_files(
            self.data_root,
            self.seed,
            tuple(verified_names),
            self._trace_hash_manifest,
        )
        self.trace_hash = _seed_trace_hash(self.verified_hashes)
        self.tasks, self._private_task_outcomes = self._load_tasks()
        self.reference_budget = self._reference_budget()
        self._bids = self._load_reported_bids()
        self.candidates = self._load_candidate_prefix()
        self.outcomes = (
            self._load_purchased_outcome_potentials()
            if self.load_outcomes
            else {}
        )
        self.candidate_keys = frozenset(candidate.key for candidate in self.candidates)
        if len(self.candidates) != self.max_arrivals:
            raise RealCalBridgeViolation(
                f"trace has only {len(self.candidates)} candidates before cap "
                f"{self.max_arrivals}"
            )

    def _load_tasks(
        self,
    ) -> tuple[dict[str, PublicTask], dict[str, PrivateTaskOutcome]]:
        public: dict[str, PublicTask] = {}
        private: dict[str, PrivateTaskOutcome] = {}
        path = self.data_root / str(self.seed) / "tasks.jsonl"
        for row in _json_lines(path):
            task_id = str(row["task_id"])
            if task_id in public:
                raise RealCalBridgeViolation(f"duplicate task id: {task_id}")
            public[task_id] = PublicTask(
                slot=int(row["slot"]),
                task_id=task_id,
                cell=int(row["cell"]),
                value=Decimal(str(row["V"])),
                capacity=int(row["K"]),
                deadline=int(row["deadline"]),
            )
            private[task_id] = PrivateTaskOutcome(
                delay_by_scenario={
                    str(key): int(value)
                    for key, value in dict(row["delay_mask"]).items()
                },
                missing_by_scenario={
                    str(key): bool(value)
                    for key, value in dict(row["missing_mask"]).items()
                },
            )
        return public, private

    def _reference_budget(self) -> Decimal:
        return sum(
            (
                Decimal(task.capacity) * PUBLIC_BASE_CAP
                + REFERENCE_GAMMA_MAX * task.value
                for task in self.tasks.values()
            ),
            ZERO,
        )

    def _load_reported_bids(self) -> dict[str, Decimal]:
        bids: dict[str, Decimal] = {}
        path = self.data_root / str(self.seed) / "workers.jsonl"
        for row in _json_lines(path):
            worker_id = str(row["worker_id"])
            bid = Decimal(str(row["c_i"]))
            if bid <= ZERO:
                raise RealCalBridgeViolation(
                    f"nonpositive truthful reported bid for {worker_id}"
                )
            bids[worker_id] = bid
        return bids

    def _candidate(
        self,
        *,
        slot: int,
        task_id: str,
        worker_id: str,
    ) -> ExternalCandidateView:
        task = self.tasks.get(task_id)
        if task is None:
            raise RealCalBridgeViolation(f"eligibility references unknown task: {task_id}")
        if task.slot != slot:
            raise RealCalBridgeViolation(
                f"eligibility/task slot mismatch for {task_id}: {slot} != {task.slot}"
            )
        probability = self.completion_forecast.predict(
            slot=slot,
            deadline=task.deadline,
            delay=self.delay,
            missing_prob=self.missing_prob,
        )
        return ExternalCandidateView(
            slot=slot,
            task_id=task_id,
            worker_id=worker_id,
            current_bid=self._bids[worker_id],
            public_task_value=task.value,
            method_independent_forecast=(
                task.value / Decimal(task.capacity) * probability
            ),
            capacity=task.capacity,
            deadline=task.deadline,
            public_cell=task.cell,
            completion_probability=probability,
        )

    def _load_candidate_prefix(self) -> tuple[ExternalCandidateView, ...]:
        candidates: list[ExternalCandidateView] = []
        crossing_slot: int | None = None
        path = self.data_root / str(self.seed) / "eligibility.jsonl"
        for row in _json_lines(path):
            slot = int(row["slot"])
            if crossing_slot is not None and slot > crossing_slot:
                break
            if not bool(row["available"]) or row["mapped_task_id"] is None:
                continue
            candidates.append(
                self._candidate(
                    slot=slot,
                    task_id=str(row["mapped_task_id"]),
                    worker_id=str(row["worker_id"]),
                )
            )
            if len(candidates) >= self.max_arrivals and crossing_slot is None:
                crossing_slot = slot
        candidates.sort(key=lambda item: (item.slot, item.task_id, item.worker_id))
        return tuple(candidates[: self.max_arrivals])

    def _load_purchased_outcome_potentials(
        self,
    ) -> dict[CandidateKey, PurchasedOutcome]:
        keys = frozenset(candidate.key for candidate in self.candidates)
        max_slot = max(candidate.slot for candidate in self.candidates)
        outcomes: dict[CandidateKey, PurchasedOutcome] = {}
        path = self.data_root / str(self.seed) / "potential_reports.jsonl"
        for row in _json_lines(path):
            slot = int(row["slot"])
            if slot > max_slot:
                break
            if str(row["effort"]) != FIXED_OUTCOME_EFFORT:
                continue
            key = CandidateKey(
                task_id=str(row["task_id"]),
                worker_id=str(row["worker_id"]),
            )
            if key not in keys:
                continue
            outcomes[key] = PurchasedOutcome(
                quality=Decimal(str(row["score"])),
            )
        missing = keys - set(outcomes)
        if missing:
            raise RealCalBridgeViolation(
                f"missing fixed-effort purchased outcomes: {len(missing)}"
            )
        return outcomes

    def by_slot(self) -> tuple[tuple[int, tuple[ExternalCandidateView, ...]], ...]:
        grouped: dict[int, list[ExternalCandidateView]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.slot, []).append(candidate)
        return tuple(
            (slot, tuple(rows)) for slot, rows in sorted(grouped.items())
        )

    def completion_status(self, key: CandidateKey) -> tuple[bool, int]:
        task = self.tasks[key.task_id]
        private = self._private_task_outcomes[key.task_id]
        delay_key = str(self.delay)
        missing_key = str(self.missing_prob)
        if (
            delay_key not in private.delay_by_scenario
            or missing_key not in private.missing_by_scenario
        ):
            raise RealCalBridgeViolation(
                "requested delay/missing scenario absent from trace"
            )
        outcome_slot = task.slot + private.delay_by_scenario[delay_key]
        completed = (
            not private.missing_by_scenario[missing_key]
            and outcome_slot <= task.deadline
        )
        return completed, outcome_slot

    def feedback_for(
        self,
        key: CandidateKey,
        *,
        revealed_slot: int,
    ) -> ExternalFeedback:
        if not self.load_outcomes:
            raise RealCalBridgeViolation(
                "feedback is disabled for this selection-only trace"
            )
        if key not in self.outcomes:
            raise RealCalBridgeViolation("feedback requested for non-prefix candidate")
        task = self.tasks[key.task_id]
        outcome = self.outcomes[key]
        gross_value = task.value / Decimal(task.capacity) * outcome.quality
        return ExternalFeedback(
            key=key,
            purchase_slot=task.slot,
            revealed_slot=revealed_slot,
            realized_external_value=gross_value,
            selected_quality=outcome.quality,
        )


def source_hash(paths: Iterable[Path]) -> str:
    return canonical_hash(
        {
            str(path.as_posix()): sha256_file(path)
            for path in sorted((Path(path) for path in paths), key=str)
        }
    )
