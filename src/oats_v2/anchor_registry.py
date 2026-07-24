from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from .events import EventLog
from .types import D


def weighted_median(values: Iterable[tuple[Decimal, Decimal]]) -> Decimal:
    points = sorted((D(value), D(weight)) for value, weight in values)
    if not points or any(weight < 0 for _, weight in points):
        raise ValueError("invalid weighted-median input")
    total = sum((weight for _, weight in points), Decimal("0"))
    if total <= 0:
        raise ValueError("zero anchor weight")
    cumulative = Decimal("0")
    for value, weight in points:
        cumulative += weight
        if cumulative * 2 >= total:
            return value
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class AnchorObservation:
    worker_id: str
    value: Decimal
    weight: Decimal
    verified: bool
    external_outcome: bool
    purchased_before_slot: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", D(self.value))
        object.__setattr__(self, "weight", D(self.weight))


@dataclass(frozen=True)
class AnchorSnapshot:
    version: str
    center: Decimal
    sigma: Decimal
    lower: Decimal
    upper: Decimal
    count: int
    # Coverage confidence in [0,1]: the anchor weight fraction inside
    # [lower, upper]. V2.1 design (replaces the V2 dual-signal min):
    #   * honest anchor sets: the 3-sigma band covers nearly all mass -> ~0.97+
    #   * point-mass hijack: the attack cluster drags the center and collapses
    #     the MAD, leaving the honest mass outside the band -> ~0.5-0.6
    #   * diffuse contamination inflates the band -> coverage stays high, but
    #     the wide band makes hard screening a near no-op anyway (equivalent
    #     to soft screening), so no separate dispersion signal is needed.
    # The V2 dispersion signal exp(-MAD/sigma0) sat at ~0.5 for HONEST cells,
    # overlapping the hijack regime and making theta_A untunable (V2 E7/E3
    # evidence); coverage alone separates the regimes cleanly.
    confidence: Decimal = Decimal("1")


@dataclass(frozen=True)
class AnchorPolicy:
    min_count: int
    sigma_floor: Decimal
    width_multiplier: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma_floor", D(self.sigma_floor))
        object.__setattr__(self, "width_multiplier", D(self.width_multiplier))
        if self.min_count < 1 or self.sigma_floor <= 0 or self.width_multiplier <= 0:
            raise ValueError("invalid anchor policy")


class HistoricalAnchorRegistry:
    def __init__(self, policy: AnchorPolicy, event_log: EventLog | None = None) -> None:
        self.policy = policy
        self._versions: dict[str, tuple[AnchorObservation, ...]] = {}
        self.event_log = event_log or EventLog()

    def register(self, version: str, observations: Iterable[AnchorObservation]) -> None:
        if not version or version in self._versions:
            raise ValueError("anchor version missing or reused")
        eligible = tuple(
            item
            for item in observations
            if item.verified and item.external_outcome and item.purchased_before_slot
        )
        self._versions[version] = eligible
        self.event_log.emit("ANCHOR_VERSION_REGISTERED", {"version": version, "eligible_count": len(eligible)})

    def snapshot(self, version: str) -> AnchorSnapshot | None:
        observations = self._versions.get(version)
        if observations is None:
            raise KeyError(version)
        if len(observations) < self.policy.min_count:
            return None
        center = weighted_median((item.value, item.weight) for item in observations)
        mad = weighted_median((abs(item.value - center), item.weight) for item in observations)
        sigma = max(mad, self.policy.sigma_floor)
        width = self.policy.width_multiplier * sigma
        lower, upper = center - width, center + width
        # Coverage confidence: anchor weight mass inside the band. A point-mass
        # hijack pulls the center to the attack cluster and leaves the honest
        # mass outside, so this signal drops even when MAD ~ 0.
        total_weight = sum((item.weight for item in observations), Decimal("0"))
        inside_weight = sum(
            (item.weight for item in observations if lower <= item.value <= upper),
            Decimal("0"),
        )
        confidence = inside_weight / total_weight if total_weight > 0 else Decimal("0")
        return AnchorSnapshot(version, center, sigma, lower, upper, len(observations), confidence)
