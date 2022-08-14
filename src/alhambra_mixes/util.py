from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def _maybesequence(object_or_sequence: Sequence[T] | T) -> list[T]:
    if isinstance(object_or_sequence, Sequence):
        return list(object_or_sequence)
    return [object_or_sequence]


def _none_as_empty_string(v: str | None) -> str:
    return "" if v is None else v
