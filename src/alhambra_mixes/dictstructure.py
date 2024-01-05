from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .locations import WellPos
from .units import ureg

if TYPE_CHECKING:  # pragma: no cover
    from .experiments import Experiment

__all__ = (
    "_STRUCTURE_CLASSES",
    "_structure",
    "_unstructure",
)

_STRUCTURE_CLASSES: dict[str, Any] = {}


def _structure(x: dict[str, Any], experiment: "Experiment" | None = None) -> Any:
    if isinstance(x, dict) and ("class" in x):
        c = _STRUCTURE_CLASSES[x["class"]]
        del x["class"]
        if hasattr(c, "_structure"):
            return c._structure(x, experiment)
        for k in x.keys():
            x[k] = _structure(x[k])
        return c(**x)
    elif isinstance(x, list):
        return [_structure(y) for y in x]
    else:
        return x


def _unstructure(x: Any) -> Any:
    if isinstance(x, ureg.Quantity):
        return str(x)
    elif isinstance(x, list):
        return [_unstructure(y) for y in x]
    elif isinstance(x, WellPos):
        return str(x)
    elif hasattr(x, "_unstructure"):
        return x._unstructure()
    elif hasattr(x, "__attrs_attrs__"):
        d = {}
        d["class"] = x.__class__.__name__
        for att in x.__attrs_attrs__:  # type: Attribute
            if att.name in ["reference"]:
                continue
            val = getattr(x, att.name)
            if val is att.default:
                continue
            d[att.name] = _unstructure(val)
        return d
    else:
        return x
