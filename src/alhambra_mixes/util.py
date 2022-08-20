from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def _maybesequence(object_or_sequence: Sequence[T] | T) -> list[T]:
    if isinstance(object_or_sequence, Sequence):
        return list(object_or_sequence)
    return [object_or_sequence]


def _none_as_empty_string(v: str | None) -> str:
    return "" if v is None else v


def _unstructure(x):
    if isinstance(x, ureg.Quantity):
        return str(x)
    elif isinstance(x, list):
        return [_unstructure(y) for y in x]
    elif isinstance(x, WellPos):
        return str(x)
    elif hasattr(x, "__attrs_attrs__"):
        d = {}
        d["class"] = x.__class__.__name__
        for att in x.__attrs_attrs__:
            if att.name in ["reference"]:
                continue
            val = getattr(x, att.name)
            if val is att.default:
                continue
            d[att.name] = _unstructure(val)
        return d
    else:
        return x
