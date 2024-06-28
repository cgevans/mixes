from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

# Largely replaced by concrete type code.
# def _maybesequence(object_or_sequence: Sequence[T] | T) -> list[T]:
#     if isinstance(object_or_sequence, Sequence):
#         return list(object_or_sequence)
#     return [object_or_sequence]


def _none_as_empty_string(v: str | None) -> str:
    return "" if v is None else v

def _get_picklist_class() -> type[PickList]:
    try:
        from kithairon.picklists import PickList  # type: ignore
        return PickList
    except ImportError as err:
        if err.name != "kithairon":
            raise err
        raise ImportError("kithairon is required for Echo support, but it is not installed.", name="kithairon")

__all__ = (
    "_none_as_empty_string",
    "_get_picklist_class",
)
