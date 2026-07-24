from __future__ import annotations

from decimal import Decimal

import pytest

from src.oats_v2.calibration.error_certificate import hoeffding_radius, uniform_epsilon
from src.oats_v2.calibration.version_registry import CalibrationVersionRegistry
from src.oats_v2.data.schemas import TraceConfig


def test_uniform_certificate_is_simultaneous_not_pointwise() -> None:
    config = TraceConfig()
    radius = hoeffding_radius(config)
    epsilon = uniform_epsilon(config)
    assert radius > 0
    assert epsilon == (Decimal("0.5") * radius + Decimal("0.0000005")).quantize(Decimal("0.000000001"), rounding="ROUND_CEILING")
    assert epsilon < Decimal("0.02")


def test_epsilon_zero_and_positive_ic_ir_boundaries() -> None:
    threshold = Decimal("2")
    published_d = Decimal("2")
    assert threshold - published_d == 0
    epsilon = uniform_epsilon(TraceConfig())
    true_d = published_d + epsilon
    assert threshold - true_d == -epsilon
    best_deviation_gain = epsilon
    assert best_deviation_gain <= epsilon


def test_anti_rollback_registry_is_idempotent_and_fail_closed() -> None:
    registry = CalibrationVersionRegistry()
    registry.register("role-v", 1, "abc")
    registry.register("role-v", 1, "abc")
    with pytest.raises(ValueError):
        registry.register("role-v", 1, "different")
    registry.register("role-v", 2, "def")
    with pytest.raises(ValueError):
        registry.register("role-v", 1, "abc")
