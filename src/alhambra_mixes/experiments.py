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
from .units import DNAN, Q_, ZERO_VOL, Decimal, DecimalQuantity, uL
from .mixes import Mix
from .mixes import VolumeError

if TYPE_CHECKING:  # pragma: no cover
    from alhambra_mixes.actions import AbstractAction
    from .components import AbstractComponent
    from .references import Reference


def _exp_attr_set_reference(
    self, attribute: Any, reference: Reference | None
) -> Reference | None:
    if reference is not None:
        self.use_reference(reference)
    return reference
    #     self.reference = reference
    # else:
    #     self.reference = None


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
    volume_checks: bool = True
    reference: Reference | None = attrs.field(
        default=None, on_setattr=_exp_attr_set_reference
    )

    def add_mix(
        self,
        mix_or_actions: Mix | Sequence[AbstractAction] | AbstractAction,
        name: str = "",
        test_tube_name: str | None = None,
        *,
        fixed_total_volume: DecimalQuantity | str = Q_(DNAN, uL),
        fixed_concentration: str | DecimalQuantity | None = None,
        buffer_name: str = "Buffer",
        min_volume: DecimalQuantity | str = Q_("0.5", uL),
        check_volumes: bool | None = None,
        apply_reference: bool = True,
    ) -> Experiment:
        """
        Add a mix to the experiment, either as a Mix object, or by creating a new Mix.

        Either the first argument should be a Mix, or arguments should be passed as for
        initializing a Mix.

        If check_volumes is True (by default), the mix will be added to the experiment, and
        volumes checked.  If the mix causes a volume usage problem, it will not be added to
        the Experiment, and a VolumeError will be raised.
        """
        if check_volumes is None:
            check_volumes = self.volume_checks
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
        if not name:
            raise ValueError("Mix must have a name to be added to an experiment.")
        elif mix.name in self.components:
            raise ValueError(f"Mix {mix.name} already exists in experiment.")

        mix = mix.with_experiment(self, True)
        if apply_reference and self.reference:
            mix = mix.with_reference(self.reference, inplace=True)

        self.components[mix.name] = mix

        if check_volumes:
            try:
                self.check_volumes(display=False, raise_error=True)
            except VolumeError as e:
                del self.components[mix.name]
                raise e

        return self

    def __setitem__(self, name: str, mix: AbstractComponent) -> None:
        if not mix.name:
            try:
                mix.name = name  # type: ignore
            except ValueError:  # pragma: no cover
                # This will only happen in a hypothetical component where
                # the name cannot be changed.
                raise ValueError(f"Component does not have a settable name: {mix}.")
        else:
            if mix.name != name:
                raise ValueError(f"Component name {mix.name} does not match {name}.")
        mix = mix.with_experiment(self, True)
        if self.reference:
            mix = mix.with_reference(self.reference, inplace=True)
        self.components[name] = mix
        if self.volume_checks:
            try:
                self.check_volumes(display=False, raise_error=True)
            except VolumeError as e:
                del self.components[name]
                raise e

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
        del self.components[name]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[AbstractComponent]:
        return iter(self.components.values())

    def consumed_and_produced_volumes(
        self,
    ) -> Mapping[str, Tuple[DecimalQuantity, DecimalQuantity]]:
        consumed_volume: Dict[str, DecimalQuantity] = {}
        produced_volume: Dict[str, DecimalQuantity] = {}
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

    def use_reference(self, reference: Reference) -> Experiment:
        """
        Apply a Reference, in place, to all components in the Experiment.
        """
        for component in self:
            component.with_reference(reference, inplace=True)
        return self
