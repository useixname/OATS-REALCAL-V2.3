from __future__ import annotations

import hashlib
from typing import Mapping

from .schemas import canonical_json


PROVIDER_MODEL_ID = "SYNTHETIC_PREGENERATED_HOLDOUT"
PROVENANCE_VERSION = "syn-v2-1-holdout-v1"
DEPENDENCY_DECLARATION = {
    "depends_on": ["formal_seed", "slot", "task_index", "latent_theta", "holdout_normal_substream"],
    "does_not_depend_on": [
        "selected_report",
        "gamma",
        "method",
        "selection",
        "report_delivery",
        "consumer_action",
    ],
}


def build_holdout_provenance(seed: int, task: Mapping[str, object]) -> dict[str, object]:
    base = {
        "seed": seed,
        "provider_model_id": PROVIDER_MODEL_ID,
        "task_id": task["task_id"],
        "cell": task["cell"],
        "time": task["slot"],
        "generation_process": "z=clip(theta+0.01*zeta_H,0,1); SHA256-keyed pre-selection MT19937 substream",
        "input_dependency_declaration": DEPENDENCY_DECLARATION,
        "independence_attestation": True,
        "version": PROVENANCE_VERSION,
        "fixed_before_selection": True,
    }
    digest_input = {**base, "z": task["z"]}
    digest = hashlib.sha256(canonical_json(digest_input).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "provider_model_id": PROVIDER_MODEL_ID,
        "task_id": task["task_id"],
        "cell": task["cell"],
        "time": task["slot"],
        "generation_process": base["generation_process"],
        "input_dependency_declaration": DEPENDENCY_DECLARATION,
        "record_hash": digest,
        "independence_attestation": True,
        "version": PROVENANCE_VERSION,
        "fixed_before_selection": True,
    }


def validate_holdout_provenance(row: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    declaration = row.get("input_dependency_declaration", {})
    forbidden = {
        "selected_report",
        "gamma",
        "method",
        "selection",
        "report_delivery",
        "consumer_action",
    }
    excluded = set(declaration.get("does_not_depend_on", [])) if isinstance(declaration, Mapping) else set()
    if not forbidden <= excluded:
        errors.append("holdout dependency declaration omits forbidden inputs")
    if row.get("provider_model_id") != PROVIDER_MODEL_ID:
        errors.append("holdout is not labeled synthetic pregenerated")
    if row.get("independence_attestation") is not True or row.get("fixed_before_selection") is not True:
        errors.append("holdout independence/fixation attestation failed")
    return errors
