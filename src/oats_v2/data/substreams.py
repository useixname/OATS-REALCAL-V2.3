from __future__ import annotations

import hashlib
import math
import random
from decimal import Decimal
from typing import Iterable, Sequence, TypeVar


T = TypeVar("T")


def _ascii_token(value: object) -> str:
    text = str(value)
    text.encode("ascii")
    if "|" in text:
        raise ValueError("substream tokens cannot contain '|'")
    return text


def key_text(formal_seed: int, family: str, *ids: object) -> str:
    tokens = (_ascii_token(formal_seed), _ascii_token(family), *(_ascii_token(value) for value in ids))
    return "|".join(tokens)


def keyed_seed(formal_seed: int, family: str, *ids: object) -> int:
    digest = hashlib.sha256(key_text(formal_seed, family, *ids).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng(formal_seed: int, family: str, *ids: object) -> random.Random:
    return random.Random(keyed_seed(formal_seed, family, *ids))


def uniform(formal_seed: int, family: str, *ids: object) -> float:
    return rng(formal_seed, family, *ids).random()


def normal(formal_seed: int, family: str, *ids: object) -> float:
    return rng(formal_seed, family, *ids).gauss(0.0, 1.0)


def beta22(formal_seed: int, family: str, *ids: object) -> float:
    return rng(formal_seed, family, *ids).betavariate(2.0, 2.0)


def bernoulli(probability: Decimal, formal_seed: int, family: str, *ids: object) -> bool:
    return uniform(formal_seed, family, *ids) < float(probability)


def poisson20(formal_seed: int, family: str, *ids: object) -> int:
    source = rng(formal_seed, family, *ids)
    limit = math.exp(-20.0)
    product = 1.0
    draws = 0
    while product > limit:
        draws += 1
        product *= source.random()
    return draws - 1


def weighted_choice(
    values: Sequence[T], probabilities: Sequence[Decimal], formal_seed: int, family: str, *ids: object
) -> T:
    if len(values) != len(probabilities) or sum(probabilities, Decimal("0")) != Decimal("1"):
        raise ValueError("invalid weighted choice")
    draw = Decimal(repr(uniform(formal_seed, family, *ids)))
    cumulative = Decimal("0")
    for value, probability in zip(values, probabilities):
        cumulative += probability
        if draw < cumulative:
            return value
    return values[-1]


def shuffled(values: Iterable[T], formal_seed: int, family: str, *ids: object) -> list[T]:
    result = list(values)
    rng(formal_seed, family, *ids).shuffle(result)
    return result


def stable_digest(*tokens: object) -> str:
    text = "|".join(_ascii_token(token) for token in tokens)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
