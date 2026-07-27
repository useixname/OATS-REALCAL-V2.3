from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Iterable, Mapping


DATASET_ID = "SYN-V2-1"
GENERATOR_VERSION = "syn-v2-1-gen-1.0.0"
SCHEMA_VERSION = "syn-v2-1-schema-1.0.0"
PREREGISTRATION_VERSION = "v2-prereg-20260715"
SCREENING_LABEL = "SCREENING_BACKEND=IDEAL_S2; NO_CRYPTOGRAPHIC_SECURITY_CLAIM"
TRACE_PRECISION = Decimal("0.000000001")
GBAR_PRECISION = Decimal("0.000001")
MONEY_GRID = Decimal("0.001")
EFFORTS = (Decimal("0"), Decimal("0.5"), Decimal("1"))
GAMMAS = (
    Decimal("0"),
    Decimal("0.1"),
    Decimal("0.3"),
    Decimal("0.5"),
    Decimal("0.8"),
    Decimal("1.0"),
)
FORMAL_SEEDS = tuple(range(20260715, 20260745))


SCHEMAS: dict[str, tuple[str, ...]] = {
    "workers": ("seed", "worker_id", "stratum", "public_signal_role", "c_i", "bias_sign"),
    "tasks": (
        "seed",
        "slot",
        "task_id",
        "cell",
        "theta",
        "V",
        "K",
        "deadline",
        "z",
        "delay_mask",
        "missing_mask",
    ),
    "eligibility": ("seed", "slot", "worker_id", "available", "mapped_task_id", "map_hash"),
    "anchors": (
        "seed",
        "cell",
        "anchor_index",
        "report",
        "z",
        "eligible",
        "anchor_version",
        "commitment",
    ),
    "potential_reports": (
        "seed",
        "slot",
        "task_id",
        "worker_id",
        "effort",
        "report",
        "score",
        "screen_status",
        "v_ijt",
    ),
    "continuation_tables": (
        "seed",
        "cell",
        "public_signal_role",
        "effort",
        "pibar",
        "gbar",
        "Gbar_precision",
        "anchor_version",
        "table_hash",
    ),
    "contracts": (
        "seed",
        "method_id",
        "gamma",
        "slot",
        "task_id",
        "worker_id",
        "public_signal_role",
        "sbar",
        "Gbar_by_effort",
        "vhat",
        "Delta_money",
        "epsilon_rank",
        "contract_hash",
    ),
    "holdout_provenance": (
        "seed",
        "provider_model_id",
        "task_id",
        "cell",
        "time",
        "generation_process",
        "input_dependency_declaration",
        "record_hash",
        "independence_attestation",
        "version",
        "fixed_before_selection",
    ),
}


@dataclass(frozen=True)
class TraceConfig:
    horizon: int = 1000
    worker_count: int = 500
    cell_count: int = 100
    poisson_rate: int = 20
    anchor_count_per_cell: int = 50
    calibration_n: int = 10_000
    availability_probability: Decimal = Decimal("0.20")
    task_value_low: Decimal = Decimal("0.5")
    task_value_high: Decimal = Decimal("1.5")
    public_base_cap: Decimal = Decimal("3.0")
    rho0: Decimal = Decimal("0.5")

    def assert_formal(self) -> None:
        expected = TraceConfig()
        if self != expected:
            raise ValueError("formal SYN-V2-1 generation requires the exact frozen TraceConfig")


def decimal_from_float(value: float, precision: Decimal = TRACE_PRECISION) -> Decimal:
    return Decimal(repr(value)).quantize(precision, rounding=ROUND_HALF_EVEN)


def decimal_text(value: Decimal) -> str:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if value == 0:
        return "0"
    return format(value, "f")


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return decimal_text(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def ordered_row(schema_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    fields = SCHEMAS[schema_name]
    missing = [field for field in fields if field not in row]
    extra = [field for field in row if field not in fields]
    if missing or extra:
        raise ValueError(f"{schema_name} schema mismatch missing={missing} extra={extra}")
    return {field: jsonable(row[field]) for field in fields}


def canonical_json(value: Any, *, schema_name: str | None = None) -> str:
    normalized = ordered_row(schema_name, value) if schema_name else jsonable(value)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def validate_row(
    schema_name: str,
    row: Mapping[str, Any],
    *,
    task_value_band: tuple[Decimal, Decimal] = (Decimal("0.5"), Decimal("1.5")),
) -> list[str]:
    errors: list[str] = []
    try:
        ordered = ordered_row(schema_name, row)
    except ValueError as exc:
        return [str(exc)]
    if int(ordered["seed"]) <= 0:
        errors.append("seed must be positive")
    if schema_name == "workers":
        if ordered["stratum"] not in {"honest", "low-quality", "malicious", "camouflage"}:
            errors.append("invalid stratum")
        if Decimal(ordered["c_i"]) not in {Decimal("0.5"), Decimal("1.0"), Decimal("2.0")}:
            errors.append("invalid c_i")
    elif schema_name == "tasks":
        for field in ("theta", "z"):
            if not Decimal("0") <= Decimal(ordered[field]) <= Decimal("1"):
                errors.append(f"{field} outside [0,1]")
        if not task_value_band[0] <= Decimal(ordered["V"]) <= task_value_band[1]:
            errors.append(f"V outside [{task_value_band[0]},{task_value_band[1]}]")
        if int(ordered["K"]) not in {3, 5, 10}:
            errors.append("invalid K")
    elif schema_name == "potential_reports":
        for field in ("report", "score"):
            if not Decimal("0") <= Decimal(ordered[field]) <= Decimal("1"):
                errors.append(f"{field} outside [0,1]")
        if ordered["screen_status"] not in {"PASS", "FAIL"}:
            errors.append("invalid screen status")
    return errors


def schema_document() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "schema_version": SCHEMA_VERSION,
        "numeric_encoding": "Decimal values are canonical base-10 JSON strings; integers remain JSON integers",
        "jsonl_encoding": "UTF-8 without BOM, LF line ending, compact separators, frozen field order",
        "schemas": {name: list(fields) for name, fields in SCHEMAS.items()},
    }


def rows_to_bytes(schema_name: str, rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row, schema_name=schema_name) + "\n").encode("utf-8") for row in rows)
