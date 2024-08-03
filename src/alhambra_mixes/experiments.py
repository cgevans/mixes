from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    Set,
    TextIO,
    Tuple,
)

import attrs

from .dictstructure import _structure
from .mixes import Mix, VolumeError
from .units import NAN_VOL, Q_, DecimalQuantity, uL
from .util import _get_picklist_class, gen_random_hash, maybe_cache_once

if TYPE_CHECKING:  # pragma: no cover
    from kithairon import PickList

    from alhambra_mixes.actions import AbstractAction

    from .components import AbstractComponent
    from .references import Reference


from abc import ABCMeta, abstractmethod


def _exp_attr_set_reference(
    self, attribute: Any, reference: Reference | None
) -> Reference | None:
    if reference is not None:
        self.use_reference(reference)
    return reference
    #     self.reference = reference
    # else:
    #     self.reference = None



class AbstractLocationType(metaclass=ABCMeta):
    __slots__ = ()
    @property
    @abstractmethod
    def name(self):
        ...

    @property
    @abstractmethod
    def is_echo_source_compatible(self) -> bool:
        return False

# class LocationType(AbstractLocationType):
#     __slots__ = ("name", "loc_type", "is_echo_source_compatible")
#     name: str
#     loc_type: Literal["plate96", "plate384", "tube"]
#     is_echo_source_compatible: bool

#     def __init__(self, name: str, loc_type: Literal["plate96", "plate384", "tube"], is_echo_source_compatible: bool = False):
#         self.name = name
#         self.loc_type = loc_type
#         self.is_echo_source_compatible = is_echo_source_compatible

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return f"LocationType({self.name}, {self.loc_type}, {self.is_echo_source_compatible})"

#     def __eq__(self, other):
#         return self.name == other.name and self.loc_type == other.loc_type and self.is_echo_source_compatible == other.is_echo_source_compatible

#     def __hash__(self):
#         return hash((self.name, self.loc_type, self.is_echo_source_compatible))

# LOCATION_TYPE_MAP = {
#     '384PP_AQ_BP': LocationType('384PP_AQ_BP', 'plate384', True),
# }

# def _location_type_converter(value: AbstractLocationType | str) -> AbstractLocationType:
#     if isinstance(value, AbstractLocationType):
#         return value
#     elif isinstance(value, str):
#         return LOCATION_TYPE_MAP[value]
#     else:
#         raise ValueError(f"Invalid location type: {value}")

@attrs.define()
class LocationInfo:
    echo_source_type: str | None = None
    echo_dest_type: str | None = None
    full_location: tuple[str, ...] = ()
    info: dict[str, Any] = attrs.field(factory=dict)

    @classmethod
    def from_obj(self, obj) -> LocationInfo:
        if isinstance(obj, LocationInfo):
            return obj
        elif isinstance(obj, dict):
            return LocationInfo(**obj)
        else:
            raise ValueError(f"Invalid location info: {obj}")

class LocationDict:
    _locs: dict[str, LocationInfo]

    def __init__(self, locs: dict[str, LocationInfo | Any]):
        self._locs = {k: LocationInfo.from_obj(v) for k, v in locs.items()}

    @classmethod
    def from_obj(cls, obj) -> LocationDict:
        if isinstance(obj, LocationDict):
            return obj
        elif isinstance(obj, dict):
            return cls(obj)
        else:
            raise ValueError(f"Invalid location dict: {obj}")

    def __getitem__(self, key: str) -> LocationInfo:
        return self._locs[key]

    def __setitem__(self, key: str, value: LocationInfo | Any) -> None:
        self._locs[key] = LocationInfo.from_obj(value)

    def __delitem__(self, key: str) -> None:
        del self._locs[key]

    def __contains__(self, key: str) -> bool:
        return key in self._locs

    def __iter__(self) -> Iterator[str]:
        return iter(self._locs)

    def __len__(self) -> int:
        return len(self._locs)

    def keys(self) -> Set[str]:
        return self._locs.keys()

    def values(self) -> Set[LocationInfo]:
        return self._locs.values()

    def items(self) -> Set[Tuple[str, LocationInfo]]:
        return self._locs.items()

    def __repr__(self) -> str:
        return f"LocationDict({self._locs})"

    def __str__(self) -> str:
        return f"LocationDict({self._locs})"

    def __eq__(self, other) -> bool:
        return self._locs == other._locs

    def get(self, key: str, default: Any | None = None) -> LocationInfo | None:
        if default is not None:
            default = LocationInfo.from_obj(default)
        return self._locs.get(key, default)

@attrs.define(eq=False)
class Experiment:
    """
    A class collecting many related mixes and components, allowing methods to be run that consider all of them
    together.

    Components can be referenced, and set, by name with [], and can be iterated through.
    """

    __hash__ = object.__hash__

    components: dict[str, AbstractComponent] = attrs.field(
        factory=dict
    )  # FIXME: CompRef
    volume_checks: bool = True
    reference: Reference | None = attrs.field(
        default=None, on_setattr=_exp_attr_set_reference
    )
    locations: LocationDict = attrs.field(factory=dict, converter=LocationDict.from_obj)

    def generate_picklist(self, _cache_key=None) -> PickList:
        _cache_key = gen_random_hash() if _cache_key is None else _cache_key
        PickList = _get_picklist_class()

        pls: list[PickList] = []
        for c in self.components.values():
            if hasattr(c, "generate_picklist"):
                p = c.generate_picklist(self, _cache_key=_cache_key)
                if p is not None:
                    pls.append(p)
        p = PickList.concat(pls)

        import networkx as nx
        import polars as pl

        g = p.well_transfer_multigraph()

        a = list(enumerate(nx.topological_generations(g)))

        topogen = sum(([x[0]] * len(x[1]) for x in a), [])
        plate = [y[0] for x in a for y in x[1]]
        well = [y[1] for x in a for y in x[1]]

        tgl = pl.DataFrame({
            'plate': plate,
            'well': well,
            'topogen': topogen
        }).lazy()

        return PickList(p.data.lazy().join(
            tgl,
            left_on=["Destination Plate Name", "Destination Well"],
            right_on=["plate", "well"],
            how="inner",
        ).sort(by=["topogen", "Destination Plate Name", "Source Plate Name"]).drop('topogen').collect())

    def add(
        self,
        component: AbstractComponent,
        *,
        check_volumes: bool | None = None,
        apply_reference: bool = True,
        check_existing: bool | Literal["equal"] = "equal",
    ) -> Experiment:
        if check_volumes is None:
            check_volumes = self.volume_checks

        if not component.name:
            raise ValueError("Component must have a name to be added to an experiment.")

        existing = self.get(component.name, None)
        if check_existing and (existing is not None):
            if check_existing == "equal" and existing != component:
                raise ValueError(
                    f"{component.name} already exists in experiment, and is different."
                )
            else:
                raise ValueError(f"{component.name} already exists in experiment.")
        self.components[component.name] = component

        if isinstance(component, Mix):
            component = component.with_experiment(self, inplace=True)
            if apply_reference and self.reference:
                component = component.with_reference(self.reference, inplace=True)

        if check_volumes:
            try:
                self.check_volumes(display=False, raise_error=True)
            except VolumeError as e:
                del self.components[component.name]
                raise e

        return self

    def add_mix(
        self,
        mix_or_actions: Mix | Sequence[AbstractAction] | AbstractAction,
        name: str = "",
        test_tube_name: str | None = None,
        *,
        fixed_total_volume: DecimalQuantity | str = NAN_VOL,
        fixed_concentration: str | DecimalQuantity | None = None,
        buffer_name: str = "Buffer",
        min_volume: DecimalQuantity | str = Q_("0.5", uL),
        check_volumes: bool | None = None,
        apply_reference: bool = True,
        check_existing: bool | Literal["equal"] = "equal",
    ) -> Experiment:
        """
        Add a mix to the experiment, either as a Mix object, or by creating a new Mix.

        Either the first argument should be a Mix, or arguments should be passed as for
        initializing a Mix.

        If check_volumes is True (by default), the mix will be added to the experiment, and
        volumes checked.  If the mix causes a volume usage problem, it will not be added to
        the Experiment, and a VolumeError will be raised.

        If check_existing is True (by default), then a exception is raised if the experiment
        already contains a mix with the name `name`. Otherwise, the existing mix is replaced
        with the new mix.
        """
        if isinstance(mix_or_actions, Mix):
            mix = mix_or_actions
            name = mix.name
        else:
            mix = Mix(
                mix_or_actions,
                name=name,
                test_tube_name=test_tube_name,
                fixed_total_volume=fixed_total_volume,
                fixed_concentration=fixed_concentration,
                buffer_name=buffer_name,
                min_volume=min_volume,
            )

        return self.add(
            mix,
            check_volumes=check_volumes,
            apply_reference=apply_reference,
            check_existing=check_existing,
        )

    def __setitem__(self, name: str, mix: AbstractComponent) -> None:
        if not mix.name:
            try:
                mix.name = name  # type: ignore
            except ValueError:  # pragma: no cover
                # This will only happen in a hypothetical component where
                # the name cannot be changed.
                raise ValueError(f"Component does not have a settable name: {mix}.")
        elif mix.name != name:
            raise ValueError(f"Component name {mix.name} does not match {name}.")
        mix = mix.with_experiment(self, inplace=True)
        if self.reference:
            mix = mix.with_reference(self.reference, inplace=True)
        self.components[name] = mix
        if self.volume_checks:
            try:
                self.check_volumes(display=False, raise_error=True)
            except VolumeError as e:
                del self.components[name]
                raise e

    def get(self, key: str, default=None):
        return self.components.get(key, default)

    def __getitem__(self, name: str) -> AbstractComponent:
        return self.components[name]

    def __delitem__(self, name: str) -> None:
        del self.components[name]

    def __contains__(self, name: str) -> bool:
        return name in self.components

    def remove_mix(self, name: str) -> None:
        """
        Remove a mix from the experiment, referenced by name,
        """
        self.remove(name)

    def remove(self, name: str) -> None:
        """
        Remove a mix from the experiment, referenced by name,
        """
        del self.components[name]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[AbstractComponent]:
        return iter(self.components.values())

    def consumed_and_produced_volumes(
        self,
    ) -> Mapping[str, Tuple[DecimalQuantity, DecimalQuantity]]:
        consumed_volume: dict[str, DecimalQuantity] = {}
        produced_volume: dict[str, DecimalQuantity] = {}
        for component in self.components.values():
            component._update_volumes(consumed_volume, produced_volume)
        return {
            k: (consumed_volume[k], produced_volume[k]) for k in consumed_volume
        }  # FIXME

    def check_volumes(
        self, showall: bool = False, display: bool = True, raise_error: bool = False
    ) -> str | None:
        """
        Check to ensure that consumed volumes are less than made volumes.
        """
        volumes = self.consumed_and_produced_volumes()
        conslines = []
        badlines = []
        for k, (consumed, made) in volumes.items():
            if made.m == 0:
                conslines.append(f"Consuming {consumed} of untracked {k}.")
            elif consumed > made:
                badlines.append(f"Making {made} of {k} but need at least {consumed}.")
            elif showall:
                conslines.append(f"Consuming {consumed} of {k}, making {made}.")

        if badlines and raise_error:
            raise VolumeError("\n".join(badlines))

        if display:
            print("\n".join(badlines))
            print("\n")
            print("\n".join(conslines))
            return None
        else:
            return "\n".join(badlines) + "\n" + "\n".join(conslines)

    def _unstructure(self) -> dict[str, Any]:
        """
        Create a dict representation of the Experiment.
        """
        return {
            "class": "Experiment",
            "components": {
                k: v._unstructure(experiment=self) for k, v in self.components.items()
            },
        }

    @classmethod
    def _structure(cls, d: dict[str, Any]) -> Experiment:
        """
        Create an Experiment from a dict representation.
        """
        if ("class" not in d) or (d["class"] != "Experiment"):
            raise ValueError("Not an Experiment dict.")
        del d["class"]
        for k, v in d["components"].items():
            d["components"][k] = _structure(v)
        return cls(**d)

    @classmethod
    def load(cls, filename_or_stream: str | PathLike | TextIO) -> Experiment:
        """
        Load an experiment from a JSON-formatted file created by Experiment.save.
        """
        if isinstance(filename_or_stream, (str, PathLike)):
            p = Path(filename_or_stream)
            if not p.suffix:
                p = p.with_suffix(".json")
            s: TextIO = open(p)
            close = True
        else:
            s = filename_or_stream
            close = False

        exp = cls._structure(json.load(s))
        if close:
            s.close()
        return exp

    def resolve_components(self) -> None:
        """
        Resolve string/blank-component components in mixes, searching through the mixes
        in the experiment.  FIXME Add used mixes to the experiment if they are not already there.
        """
        for mix in self:
            if not isinstance(mix, Mix):
                continue
            mix.with_experiment(self, inplace=True)

    def save(self, filename_or_stream: str | PathLike | TextIO) -> None:
        """
        Save an experiment to a JSON-formatted file.

        Tries to store each component/mix only once, with other mixes referencing those components.
        """
        if isinstance(filename_or_stream, (str, PathLike)):
            p = Path(filename_or_stream)
            if not p.suffix:
                p = p.with_suffix(".json")
            s: TextIO = open(p, "w")
            close = True
        else:
            s = filename_or_stream
            close = False

        json.dump(self._unstructure(), s, indent=2, ensure_ascii=True)
        if close:
            s.close()

    def use_reference(self, reference: Reference) -> Experiment:
        """
        Apply a Reference, in place, to all components in the Experiment.
        """
        for component in self:
            component.with_reference(reference, inplace=True)
        return self
