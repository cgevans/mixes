from __future__ import annotations

from abc import ABC, abstractmethod
from math import isnan
from typing import Any, Sequence, TypeVar

import attrs
import pandas as pd

from .locations import WellPos, _parse_wellpos_optional
from .logging import log
from .printing import TableFormat
from .references import Reference
from .units import Decimal, Quantity, _parse_conc_optional, nM, ureg
from .util import _none_as_empty_string

T = TypeVar("T")

__all__ = ["AbstractComponent", "Component", "Strand"]


class AbstractComponent(ABC):
    """Abstract class for a component in a mix.  Custom components that don't inherit from
    a concrete class should inherit from this class and implement the methods here.
    """

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        "Name of the component."
        ...

    @property
    def location(self) -> tuple[str, WellPos | None]:
        return ("", None)

    @property
    def plate(self) -> str:
        return ""

    @property
    def is_mix(self) -> bool:
        return False

    @property
    def well(self) -> WellPos | None:
        return None

    @property
    def _well_list(self) -> list[WellPos]:
        if self.well is not None:
            return [self.well]
        return []

    @property
    @abstractmethod
    def concentration(self) -> Quantity[Decimal]:  # pragma: no cover
        "(Source) concentration of the component as a pint Quantity.  NaN if undefined."
        ...

    @abstractmethod
    def all_components(self) -> pd.DataFrame:  # pragma: no cover
        "A dataframe of all components."
        ...

    @abstractmethod
    def with_reference(self: T, reference: Reference) -> T:  # pragma: no cover
        ...

    def printed_name(self, tablefmt: str | TableFormat) -> str:
        return self.name


@attrs.define()
class Component(AbstractComponent):
    """A single named component, potentially with a concentration and location.

    Location is stored as a `plate` and `well` property. `plate` is

    """

    name: str
    concentration: Quantity[Decimal] = attrs.field(
        converter=_parse_conc_optional, default=None, on_setattr=attrs.setters.convert
    )
    # FIXME: this is not a great way to do this: should make code not give None
    # Fortuitously, mypy doesn't support this converter, so problems should give type errors.
    plate: str = attrs.field(
        default="",
        kw_only=True,
        converter=_none_as_empty_string,
        on_setattr=attrs.setters.convert,
    )
    well: WellPos | None = attrs.field(
        converter=_parse_wellpos_optional,
        default=None,
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )

    def __eq__(self, other: Any) -> bool:
        if not other.__class__ == Component:
            return False
        if self.name != other.name:
            return False
        if isinstance(self.concentration, Quantity) and isinstance(
            other.concentration, Quantity
        ):
            if isnan(self.concentration.m) and isnan(other.concentration.m):
                return True
            return self.concentration == other.concentration
        elif hasattr(self, "concentration") and hasattr(other, "concentration"):
            return bool(self.concentration == other.concentration)
        return False

    @property
    def location(self) -> tuple[str, WellPos | None]:
        return (self.plate, self.well)

    def all_components(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "concentration_nM": [self.concentration.to(nM).magnitude],
                "component": [self],
            },
            index=pd.Index([self.name], name="name"),
        )
        return df

    def with_reference(self: Component, reference: Reference) -> Component:
        if reference.df.index.name == "Name":
            ref_by_name = reference.df
        else:
            ref_by_name = reference.df.set_index("Name")
        try:
            ref_comps = ref_by_name.loc[
                [self.name], :
            ]  # using this format to force a dataframe result
        except KeyError:
            return self

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not isnan(self.concentration.m) and not (ref_conc == self.concentration):
                mismatches.append(("Concentration (nM)", ref_comp))
                continue

            ref_plate = ref_comp["Plate"]
            if self.plate and ref_plate != self.plate:
                mismatches.append(("Plate", ref_comp))
                continue

            ref_well = _parse_wellpos_optional(ref_comp["Well"])
            if self.well and self.well != ref_well:
                mismatches.append(("Well", ref_well))
                continue

            matches.append(ref_comp)

        if len(matches) > 1:
            log.warning(
                "Component %s has more than one location: %s.  Choosing first.",
                self.name,
                [(x["Plate"], x["Well"]) for x in matches],
            )
        elif (len(matches) == 0) and len(mismatches) > 0:
            raise ValueError(
                "Component has only mismatched references: %s", self, mismatches
            )

        match = matches[0]
        ref_conc = ureg.Quantity(match["Concentration (nM)"], nM)
        ref_plate = match["Plate"]
        ref_well = _parse_wellpos_optional(match["Well"])

        return attrs.evolve(
            self, name=self.name, concentration=ref_conc, plate=ref_plate, well=ref_well
        )


@attrs.define()
class Strand(Component):
    """A single named strand, potentially with a concentration, location and sequence."""

    sequence: str | None = None

    def with_reference(self: Strand, reference: Reference) -> Strand:
        if reference.df.index.name == "Name":
            ref_by_name = reference.df
        else:
            ref_by_name = reference.df.set_index("Name")
        try:
            ref_comps = ref_by_name.loc[
                [self.name], :
            ]  # using this format to force a dataframe result
        except KeyError:
            return self

        mismatches = []
        matches = []
        for _, ref_comp in ref_comps.iterrows():
            ref_conc = ureg.Quantity(ref_comp["Concentration (nM)"], nM)
            if not isnan(self.concentration.m) and not (ref_conc == self.concentration):
                mismatches.append(("Concentration (nM)", ref_comp))
                continue

            ref_plate = ref_comp["Plate"]
            if self.plate and ref_plate != self.plate:
                mismatches.append(("Plate", ref_comp))
                continue

            ref_well = _parse_wellpos_optional(ref_comp["Well"])
            if self.well and self.well != ref_well:
                mismatches.append(("Well", ref_well))
                continue

            if isinstance(self.sequence, str) and isinstance(ref_comp["Sequence"], str):
                y = ref_comp["Sequence"]
                self.sequence = self.sequence.replace(" ", "").replace("-", "")
                y = y.replace(" ", "").replace("-", "")
                if self.sequence != y:
                    mismatches.append(("Sequence", ref_comp["Sequence"]))
                    continue

            matches.append(ref_comp)

        del ref_comp  # Ensure we never use this again

        if len(matches) > 1:
            log.warning(
                "Strand %s has more than one location: %s.  Choosing first.",
                self.name,
                [(x["Plate"], x["Well"]) for x in matches],
            )
        elif (len(matches) == 0) and len(mismatches) > 0:
            raise ValueError(
                "Strand has only mismatched references: %s", self, mismatches
            )

        m = matches[0]
        ref_conc = ureg.Quantity(m["Concentration (nM)"], nM)
        ref_plate = m["Plate"]
        ref_well = _parse_wellpos_optional(m["Well"])
        ss, ms = self.sequence, m["Sequence"]
        if (ss is None) and (ms is None):
            seq = None
        elif isinstance(ss, str) and ((ms is None) or (ms == "")):
            seq = ss
        elif isinstance(ms, str) and ((ss is None) or isinstance(ss, str)):
            seq = ms
        else:
            raise RuntimeError("should be unreachable")

        return attrs.evolve(
            self,
            name=self.name,
            concentration=ref_conc,
            plate=ref_plate,
            well=ref_well,
            sequence=seq,
        )


def _maybesequence_comps(
    object_or_sequence: Sequence[AbstractComponent | str] | AbstractComponent | str,
) -> list[AbstractComponent]:
    if isinstance(object_or_sequence, str):
        return [Component(object_or_sequence)]
    elif isinstance(object_or_sequence, Sequence):
        return [Component(x) if isinstance(x, str) else x for x in object_or_sequence]
    return [object_or_sequence]


def _empty_components() -> pd.DataFrame:
    cps = pd.DataFrame(
        index=pd.Index([], name="name"),
    )
    cps["concentration_nM"] = pd.Series([], dtype=object)
    cps["component"] = pd.Series([], dtype=object)
    return cps
