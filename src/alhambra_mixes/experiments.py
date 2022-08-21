from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    Mapping,
    Sequence,
    Set,
    TextIO,
    Tuple,
    cast,
)

import attrs

from .dictstructure import _structure, _unstructure
from .units import DNAN, Q_, ZERO_VOL, Decimal, Quantity, uL
from .mixes import Mix

if TYPE_CHECKING:  # pragma: no cover
    from alhambra_mixes.actions import AbstractAction
    from .components import AbstractComponent
    from .references import Reference


@attrs.define()
class Experiment:
    """
    A class collecting many related mixes and components, allowing methods to be run that consider all of them
    together.

    Components can be referenced, and set, by name with [], and can be iterated through.
    """

    components: Dict[str, AbstractComponent] = attrs.field(
        factory=dict
    )  # FIXME: CompRef

    def add_mix(
        self,
        mix_or_actions: Mix | Sequence[AbstractAction] | AbstractAction,
        name: str = "",
        test_tube_name: str | None = None,
        *,
        fixed_total_volume: Quantity[Decimal] | str = Q_(DNAN, uL),
        fixed_concentration: str | Quantity[Decimal] | None = None,
        buffer_name: str = "Buffer",
        reference: Reference | None = None,
        min_volume: Quantity[Decimal] | str = Q_(Decimal("0.5"), uL),
    ) -> None:
        """
        Add a mix to the experiment, either as a Mix object, or by creating a new Mix.

        Either the first argument should be a Mix, or arguments should be passed as for
        initializing a Mix.
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
                reference=reference,
                min_volume=min_volume,
            )
        if not name:
            raise ValueError("Mix must have a name to be added to an experiment.")
        elif mix.name in self.components:
            raise ValueError(f"Mix {mix.name} already exists in experiment.")
        mix = mix.with_experiment(self, True)
        self.components[mix.name] = mix

    def __setitem__(self, name: str, value: AbstractComponent) -> None:
        if not value.name:
            try:
                value.name = name  # type: ignore
            except ValueError:  # pragma: no cover
                # This will only happen in a hypothetical component where
                # the name cannot be changed.
                raise ValueError(f"Component does not have a settable name: {value}.")
        else:
            if value.name != name:
                raise ValueError(f"Component name {value.name} does not match {name}.")
        value = value.with_experiment(self, True)
        self.components[name] = value

    def __getitem__(self, name: str) -> AbstractComponent:
        return self.components[name]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[AbstractComponent]:
        return iter(self.components.values())

    def consumed_and_produced_volumes(self) -> Mapping[str, Tuple[Quantity, Quantity]]:
        consumed_volume: Dict[str, Quantity] = {}
        produced_volume: Dict[str, Quantity] = {}
        for component in self.components.values():
            component._update_volumes(consumed_volume, produced_volume)
        return {
            k: (consumed_volume[k], produced_volume[k]) for k in consumed_volume
        }  # FIXME

    def check_volumes(self, showall: bool = False) -> None:
        """
        Check to ensure that consumed volumes are less than made volumes.
        """
        volumes = self.consumed_and_produced_volumes()
        conslines = []
        badlines = []
        for k, (consumed, made) in volumes.items():
            if made.m == 0:
                conslines.append(f"Consuming {consumed} of untracked {k}.")
            elif consumed >= made:
                badlines.append(f"Making {made} of {k} but need at least {consumed}.")
            elif showall:
                conslines.append(f"Consuming {consumed} of {k}, making {made}.")
        print("\n".join(badlines))
        print("\n")
        print("\n".join(conslines))

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
    def _structure(cls, d: dict[str, Any]) -> "Experiment":
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
    def load(cls, filename_or_stream: str | PathLike | TextIO) -> "Experiment":
        """
        Load an experiment from a JSON-formatted file created by Experiment.save.
        """
        if isinstance(filename_or_stream, (str, PathLike)):
            p = Path(filename_or_stream)
            if not p.suffix:
                p = p.with_suffix(".json")
            s: TextIO = open(p, "r")
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
            mix.with_experiment(self, True)

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

        json.dump(self._unstructure(), s, indent=2, ensure_ascii=False)
        if close:
            s.close()
