from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

def _none_as_empty_string(v: str | None) -> str:
    return "" if v is None else v
