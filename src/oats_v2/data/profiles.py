from __future__ import annotations


MANAGED_FLEET_SYNTHETIC_PROFILE = {
    "profile_id": "MANAGED_FLEET_SYNTHETIC_PROFILE",
    "certified_device_roles": ["honest", "low"],
    "public_signal_technology": {
        "honest": "q(e)=0.6+0.3e",
        "low": "q(e)=0.4+0.2e",
    },
    "effort_menu": ["0", "0.5", "1"],
    "report_pipeline": "ATTESTED_POINT_ESTIMATE from frozen synthetic sensor pipeline",
    "attestation_assumption": "synthetic generator binds compliant role, effort technology, nonce and report pipeline",
    "one_task_mapping": "SHA256-keyed pre-method mapping h_t(i)",
    "external_holdout_source": "SYNTHETIC_PREGENERATED_HOLDOUT",
    "gbar_calibration_source": "independent preregistered N_cal=10000 sampler per seed/cell/role/effort",
    "epsilon_interpretation": "simultaneous high-probability deployment-relative bound; contract-internal model is exact",
    "ic_population": ["honest", "low-quality"],
    "excluded_byzantine_behavior": ["malicious", "camouflage", "attestation bypass", "bid attack", "collusion"],
}


OPEN_BYOD_UNSUPPORTED_PROFILE = {
    "profile_id": "OPEN_BYOD_UNSUPPORTED_PROFILE",
    "supported": False,
    "reasons": [
        "physical sensor input is not certified end-to-end",
        "device role and effort technology are heterogeneous/private",
        "workers may choose multiple tasks",
        "attestation may be bypassed",
        "outcome may be affected by consumer action",
    ],
    "theorem_consequence": "T3/T4 do not apply; fallback P1 and robustness-only screening labels are required",
}


def deployment_profiles() -> dict[str, object]:
    return {
        MANAGED_FLEET_SYNTHETIC_PROFILE["profile_id"]: MANAGED_FLEET_SYNTHETIC_PROFILE,
        OPEN_BYOD_UNSUPPORTED_PROFILE["profile_id"]: OPEN_BYOD_UNSUPPORTED_PROFILE,
    }
