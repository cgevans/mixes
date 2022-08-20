from __future__ import annotations

from typing import Dict, Mapping, Sequence, Set, Tuple
from .mixes import Mix
from .components import Component
from .units import Quantity, ZERO_VOL
import attrs


class Experiment:
    mixes: Sequence[Mix | Component]

    def consumed_volumes(self) -> Mapping[str, Tuple[Quantity, Quantity]]:
        consumed_volume: Dict[str, Quantity] = {}
        made_volume: Dict[str, Quantity] = {}
        for mix in self.mixes:
            if not isinstance(mix, Mix):
                continue
            mix._update_volumes(consumed_volume, made_volume)
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
