from __future__ import annotations

from typing import Dict, Iterator, Mapping, Sequence, Set, Tuple, cast

from alhambra_mixes.actions import AbstractAction

from .references import Reference
from .mixes import Mix
from .components import Component
from .units import DNAN, Q_, Quantity, ZERO_VOL, Decimal, uL
import attrs


@attrs.define()
class Experiment:
    """
    A class collecting many related mixes and components, allowing methods to be run that consider all of them
    together.
    """

    components: Dict[str, Component]  # FIXME: CompRef

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
        self.components[mix.name] = cast(Component, mix)

    def __setitem__(self, name: str, value: Component) -> None:
        if value.name is None:
            value.name = name
        else:
            assert value.name == name
        self.components[name] = value

    def __getitem__(self, name: str) -> Component:
        return self.components[name]

    def __iter__(self) -> Iterator[Component]:
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
