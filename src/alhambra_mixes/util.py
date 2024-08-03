from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import functools

if TYPE_CHECKING:
    from kithairon.picklists import PickList

T = TypeVar("T")

CACHE_HITS = 0
CACHE_MISSES = 0
CACHE_SKIPS = 0

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
        raise ImportError(
            "kithairon is required for Echo support, but it is not installed.",
            name="kithairon",
        )


__all__ = (
    "_none_as_empty_string",
    "_get_picklist_class",
)


def maybe_cache_once(fun):
    last_hash = None
    last_cache_data = None

    def inner(*args, _cache_key=None, **kwargs):
        nonlocal last_hash, last_cache_data
        # print((_cache_key, *args, tuple((k, v) for k, v in kwargs.items())))
        # print(fun)
        current_hash = hash(
            (
                _cache_key,
                tuple(a if not isinstance(a, list) else tuple(a) for a in args),
                tuple(
                    [
                        (k, v if not isinstance(v, list) else tuple(v))
                        for k, v in kwargs.items()
                    ]
                ),
            )
        )
        # print((current_hash, last_hash, last_cache_data))
        if (
            (_cache_key is not None)
            and (current_hash == last_hash)
            and (last_cache_data is not None)
        ):
            global CACHE_HITS
            CACHE_HITS += 1
            return last_cache_data

        data = fun(*args, **kwargs, _cache_key=_cache_key)
        if _cache_key is not None:
            global CACHE_MISSES
            CACHE_MISSES += 1
            last_hash = current_hash
            last_cache_data = data
        else:
            global CACHE_SKIPS
            CACHE_SKIPS += 1
            # raise ValueError("Cache key was not provided, so cache was not used.")
            # print(fun, args, kwargs)
        return data

    functools.update_wrapper(inner, fun)

    return inner


def gen_random_hash():
    import random
    import string

    return "".join(random.choices(string.ascii_lowercase + string.digits, k=15))
