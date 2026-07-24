from __future__ import annotations

import hashlib


SIGNATURE_SCHEME = "UNTRUSTED_SHA256_PLACEHOLDER_NOT_A_DIGITAL_SIGNATURE"


def placeholder_signature(payload_hash: str, version: str, anti_rollback_counter: int) -> str:
    digest = hashlib.sha256(
        f"{SIGNATURE_SCHEME}|{payload_hash}|{version}|{anti_rollback_counter}".encode("ascii")
    ).hexdigest()
    return f"{SIGNATURE_SCHEME}:{digest}"
