from __future__ import annotations

import hashlib
import random

from src.oats_v2.data.substreams import key_text, keyed_seed, normal, uniform


def test_keyed_seed_matches_preregistered_sha256_formula() -> None:
    text = "20260715|family|1|w0001"
    expected = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    assert key_text(20260715, "family", 1, "w0001") == text
    assert keyed_seed(20260715, "family", 1, "w0001") == expected
    assert random.Random(expected).random() == uniform(20260715, "family", 1, "w0001")


def test_substreams_are_order_independent_and_family_separated() -> None:
    first = uniform(20260715, "a", 1)
    _ = [uniform(20260715, "noise", index) for index in range(100)]
    assert uniform(20260715, "a", 1) == first
    assert keyed_seed(20260715, "a", 1) != keyed_seed(20260715, "b", 1)
    assert normal(20260715, "normal", 1) == normal(20260715, "normal", 1)


def test_python_hash_is_not_part_of_seed_contract() -> None:
    source = __import__("inspect").getsource(__import__("src.oats_v2.data.substreams", fromlist=["*"]))
    assert "hash(" not in source.replace("hashlib.sha256", "")
