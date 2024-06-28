from __future__ import annotations

from abc import ABC, abstractmethod
from math import isnan
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, TypeVar, cast

import attrs
import pandas as pd

from .dictstructure import _STRUCTURE_CLASSES, _structure, _unstructure
from .locations import WellPos, _parse_wellpos_optional
from .logging import log
from .printing import TableFormat
from .units import (
    NAN_VOL,
    Q_,
    ZERO_VOL,
    DecimalQuantity,
    _parse_conc_optional,
    _parse_vol_optional,
    nM,
    ureg,
)

if TYPE_CHECKING:  # pragma: no cover
    from attrs import Attribute

    from .experiments import Experiment
    from .references import Reference


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
    def plate(self) -> str | None:
        return None

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
    def volume(self) -> DecimalQuantity:
        return NAN_VOL

    @property
    @abstractmethod
    def concentration(self) -> DecimalQuantity:  # pragma: no cover
        "(Source) concentration of the component as a pint Quantity.  NaN if undefined."
        ...

    @abstractmethod
    def all_components(self) -> pd.DataFrame:  # pragma: no cover
        "A dataframe of all components."
        ...

    @abstractmethod
    def with_reference(
        self: T, reference: Reference, inplace: bool = False
    ) -> T:  # pragma: no cover
        ...

    @abstractmethod
    def with_experiment(
        self, reference: Experiment, inplace: bool = True
    ) -> AbstractComponent:  # pragma: no cover
        ...

    @classmethod
    @abstractmethod
    def _structure(
        cls, d: dict[str, Any], experiment: Experiment | None = None
    ) -> AbstractComponent:  # pragma: no cover
        ...

    @abstractmethod
    def _unstructure(self, experiment: Experiment | None = None) -> dict[str, Any]:
        ...

    def printed_name(self, tablefmt: str | TableFormat) -> str:
        return self.name

    def _update_volumes(
        self,
        consumed_volumes: Dict[str, DecimalQuantity] = {},
        made_volumes: Dict[str, DecimalQuantity] = {},
    ) -> Tuple[Dict[str, DecimalQuantity], Dict[str, DecimalQuantity]]:
        """
        Given a
        """
        if self.name in made_volumes:
            # We've already been seen.  Ignore our components.
            return consumed_volumes, made_volumes

        made_volumes[self.name] = ZERO_VOL

        return consumed_volumes, made_volumes


def norm_nan_for_eq(v1: DecimalQuantity) -> DecimalQuantity:
    if isnan(v1.m):
        return Q_(-1, v1.u)
    return v1


@attrs.define()
class Component(AbstractComponent):
    """A single named component, potentially with a concentration and location.

    Location is stored as a `plate` and `well` property. `plate` is

    """

    name: str
    concentration: DecimalQuantity = attrs.field(
        converter=_parse_conc_optional,
        default=None,
        on_setattr=attrs.setters.convert,
        eq=norm_nan_for_eq,
    )
    plate: str | None = attrs.field(
        default=None,
        kw_only=True
    )
    well: str | WellPos | None = attrs.field(
        converter=_parse_wellpos_optional,
        default=None,
        kw_only=True,
        on_setattr=attrs.setters.convert,
    )
    volume: DecimalQuantity = attrs.field(
        converter=_parse_vol_optional,
        default=NAN_VOL,
        on_setattr=attrs.setters.convert,
        eq=norm_nan_for_eq,
    )

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

    def _unstructure(self, experiment: Experiment | None = None) -> dict[str, Any]:
        d = {}
        d["class"] = self.__class__.__name__
        for att in cast("Sequence[Attribute]", self.__attrs_attrs__):
            if att.name in ["reference"]:
                continue
            val = getattr(self, att.name)
            if val is att.default:
                continue
            if isinstance(val, ureg.Quantity) and isnan(val.m):
                continue
            d[att.name] = _unstructure(val)
        return d

    @classmethod
    def _structure(
        cls, d: dict[str, Any], experiment: Experiment | None = None
    ) -> Component:
        for k, v in d.items():
            d[k] = _structure(v, experiment)
        return cls(**d)

    def with_experiment(
        self: Component, experiment: Experiment, inplace: bool = True
    ) -> AbstractComponent:
        if self.name in experiment.components:
            return experiment.components[self.name]
            # FIXME: add checks
        else:
            return self

    def with_reference(
        self: Component, reference: Reference, inplace: bool = False
    ) -> Component:
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
            ref_conc = Q_(ref_comp["Concentration (nM)"], nM)
            if not isnan(self.concentration.m) and ref_conc != self.concentration:
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

        if inplace:
            self.concentration = ref_conc
            self.plate = ref_plate
            self.well = ref_well
            return self
        else:
            return attrs.evolve(
                self,
                name=self.name,
                concentration=ref_conc,
                plate=ref_plate,
                well=ref_well,
            )


@attrs.define()
class Strand(Component):
    """A single named strand, potentially with a concentration, location and sequence."""

    sequence: str | None = None

    def with_reference(
        self: Strand, reference: Reference, inplace: bool = False
    ) -> Strand:
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
            if not isnan(self.concentration.m) and ref_conc != self.concentration:
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
        ref_conc = Q_(m["Concentration (nM)"], nM)
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

        if inplace:
            self.concentration = ref_conc
            self.plate = ref_plate
            self.well = ref_well
            self.sequence = seq
            return self
        else:
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


for c in [Component, Strand]:
    _STRUCTURE_CLASSES[c.__name__] = c
