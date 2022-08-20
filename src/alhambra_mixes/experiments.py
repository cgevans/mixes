from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Set, TextIO, Tuple, cast

import attrs

from alhambra_mixes.actions import AbstractAction

from .components import AbstractComponent
from .dictstructure import _structure, _unstructure
from .mixes import Mix
from .references import Reference
from .units import DNAN, Q_, ZERO_VOL, Decimal, Quantity, uL


@attrs.define()
class Experiment:
    """
    A class collecting many related mixes and components, allowing methods to be run that consider all of them
    together.
    """

    components: Dict[str, AbstractComponent] = attrs.field(
        factory=dict
    )  # FIXME: CompRef

    def add_mix(
        self,
        mix_or_actions: Mix | Sequence[AbstractAction] | AbstractAction,
        name: str | None = None,
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
        else:
            if name is None:
                raise ValueError("Mix must have a name.")
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
        if mix.name in self.components:
            raise ValueError(f"Mix {mix.name} already exists in experiment.")
        mix.with_experiment(self, True)
        self.components[mix.name] = cast(AbstractComponent, mix)

    def __setitem__(self, name: str, value: AbstractComponent) -> None:
        if value.name is None:
            value.name = name
        else:
            assert value.name == name
        self.components[name] = value

    def __getitem__(self, name: str) -> AbstractComponent:
        return self.components[name]

    def __iter__(self) -> Iterator[AbstractComponent]:
        return iter(self.components.values())

    def consumed_volumes(self) -> Mapping[str, Tuple[Quantity, Quantity]]:
        consumed_volume: Dict[str, Quantity] = {}
        made_volume: Dict[str, Quantity] = {}
        for component in self.components.values():
            component._update_volumes(consumed_volume, made_volume)
        return {
            k: (consumed_volume[k], made_volume[k]) for k in consumed_volume
        }  # FIXME

    def check_volumes(self, showall: bool = False) -> None:
        """
        Check to ensure that consumed volumes are less than made volumes.
        """
        volumes = self.consumed_volumes()
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
        if d["class"] != "Experiment":
            raise ValueError("Not an Experiment dict.")
        del d["class"]
        for k, v in d["components"].items():
            d["components"][k] = _structure(v)
        return cls(**d)

    @classmethod
    def load(cls, filename_or_stream: str | PathLike | TextIO) -> "Experiment":
        """
        Load an experiment from a JSON-formatted file created by Experiment.save
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

    def save(self, filename_or_stream: str | PathLike | TextIO) -> None:
        """
        Save an experiment to a JSON-formatted file.
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
