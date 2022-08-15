from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, Sequence, TypeVar

import attrs
import pandas as pd

from warnings import warn

from .components import AbstractComponent, _empty_components, _maybesequence_comps
from .locations import WellPos, mixgaps
from .printing import MixLine, TableFormat
from .references import Reference
from .units import *
from .units import (
    VolumeError,
    _parse_conc_required,
    _parse_vol_optional,
    _parse_vol_required,
    _ratio,
)

T = TypeVar("T")


class AbstractAction(ABC):
    """
    Abstract class defining an action in a mix recipe.
    """

    @property
    def name(self) -> str:  # pragma: no cover
        ...

    @abstractmethod
    def tx_volume(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> Quantity[Decimal]:  # pragma: no cover
        """The total volume transferred by the action to the sample.  May depend on the total mix volume.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
    ) -> Sequence[MixLine]:  # pragma: no cover
        ...

    @abstractmethod
    def all_components(
        self, mix_vol: Quantity[Decimal]
    ) -> pd.DataFrame:  # pragma: no cover
        """A dataframe containing all base components added by the action.

        Parameters
        ----------

        mix_vol
            The mix volume.  Does not accept strings.
        """
        ...

    @abstractmethod
    def with_reference(self: T, reference: Reference) -> T:  # pragma: no cover
        """Returns a copy of the action updated from a reference dataframe."""
        ...

    def dest_concentration(self, mix_vol: Quantity) -> Quantity:
        """The destination concentration added to the mix by the action.

        Raises
        ------

        ValueError
            There is no good definition for a single destination concentration
            (the action may add multiple components).
        """
        raise ValueError("Single destination concentration not defined.")

    def dest_concentrations(self, mix_vol: Quantity) -> Sequence[Quantity[Decimal]]:
        raise ValueError

    @property
    @abstractmethod
    def components(self) -> list[AbstractComponent]:
        ...

    @abstractmethod
    def each_volumes(self, total_volume: Quantity[Decimal]) -> list[Quantity[Decimal]]:
        ...


@attrs.define()
class FixedVolume(AbstractAction):
    """An action adding one or multiple components, with a set transfer volume.

    Parameters
    ----------

    components
        A list of :ref:`Components`.

    fixed_volume
        A fixed volume for the action.  Input can be a string (eg, "5 µL") or a pint Quantity.  The interpretation
        of this depends on equal_conc.

    set_name
        The name of the mix.  If not set, name is based on components.

    compact_display
        If True (default), the action tries to display compactly in mix recipes.  If False, it displays
        each component as a separate line.

    Examples
    --------

    >>> from alhambra.mixes import *
    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ... ]

    >>> print(Mix([FixedVolume(components, "5 uL")], name="example"))
    Table: Mix: example, Conc: 66.67 nM, Total Vol: 15.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   |   # | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|----:|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 66.67 nM  |   3 | 5.00 µl     | 15.00 µl     |       |        |
    """

    components: list[AbstractComponent] = attrs.field(
        converter=_maybesequence_comps, on_setattr=attrs.setters.convert
    )
    fixed_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    compact_display: bool = True

    # components: Sequence[AbstractComponent | str] | AbstractComponent | str, fixed_volume: str | Quantity, set_name: str | None = None, compact_display: bool = True, equal_conc: bool | str = False
    def __new__(cls, *args, **kwargs):
        if (cls is FixedVolume) and ("equal_conc" in kwargs):
            if kwargs["equal_conc"] is not False:
                c = super().__new__(EqualConcentration)
                # print(kwargs)
                # c.__init__(*args, **kwargs, method=equal_conc)
                # print("Hi")
                return c
        c = super().__new__(cls)
        # c.__init__(*args, **kwargs)
        return c

    def with_reference(self, reference: Reference) -> FixedVolume:
        return attrs.evolve(
            self, components=[c.with_reference(reference) for c in self.components]
        )

    @property
    def source_concentrations(self) -> Sequence[Quantity[Decimal]]:
        return [c.concentration.to(nM) for c in self.components]

    def all_components(self, mix_vol: Quantity[Decimal]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= _ratio(dc, sc)

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=Decimal("0.0"))
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        return [
            x * y
            for x, y in zip(
                self.source_concentrations, _ratio(self.each_volumes(mix_vol), mix_vol)
            )
        ]

    def each_volumes(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        return [self.fixed_volume.to(uL)] * len(self.components)

    def tx_volume(self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)) -> Quantity[Decimal]:
        return sum(self.each_volumes(mix_vol), ureg("0.0 uL"))

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
        locations: pd.DataFrame | None = None,
    ) -> list[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    [comp.printed_name(tablefmt=tablefmt)],
                    comp.concentration,
                    dc,
                    ev,
                    plate=comp.plate,
                    wells=comp._well_list,
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(tablefmt=tablefmt, mix_vol=mix_vol))

        return ml

    @property
    def number(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.components)
        else:
            return self.set_name

    def _compactstrs(
        self, tablefmt: str | TableFormat, mix_vol: Quantity
    ) -> list[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_concs": self.source_concentrations,
                "dest_concs": self.dest_concentrations(mix_vol),
                "ea_vols": self.each_volumes(mix_vol),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            }
        )

        locdf.fillna({"plate": ""}, inplace=True)

        locdf.sort_values(
            by=["plate", "ea_vols", "well"], ascending=[True, False, True]
        )

        names: list[list[str]] = []
        source_concs: list[Quantity[Decimal]] = []
        dest_concs: list[Quantity[Decimal]] = []
        numbers: list[int] = []
        ea_vols: list[Quantity[Decimal]] = []
        tot_vols: list[Quantity[Decimal]] = []
        plates: list[str] = []
        wells_list: list[list[WellPos]] = []

        for plate, plate_comps in locdf.groupby("plate"):  # type: str, pd.DataFrame
            for vol, plate_vol_comps in plate_comps.groupby(
                "ea_vols"
            ):  # type: Quantity[Decimal], pd.DataFrame
                if pd.isna(plate_vol_comps["well"].iloc[0]):
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names.append(list(plate_vol_comps["names"]))
                    ea_vols.append((vol))
                    tot_vols.append((vol * len(plate_vol_comps)))
                    numbers.append((len(plate_vol_comps)))
                    source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                    dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                    plates.append(plate)
                    wells_list.append([])
                    continue
                byrow = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_byrow),
                    by="row",
                )
                bycol = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_bycol),
                    by="col",
                )

                sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                names.append(list(plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbers.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                plates.append(plate)
                wells_list.append(list(plate_vol_comps["well"]))

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                plate=plate,
                wells=wells,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, plate, wells in zip(
                names,
                source_concs,
                dest_concs,
                numbers,
                ea_vols,
                tot_vols,
                plates,
                wells_list,
            )
        ]


@attrs.define(init=False)
class EqualConcentration(FixedVolume):
    """An action adding an equal concentration of each component, without setting that concentration.

    Depending on the setting of
    `equal_conc`, it may require that the concentrations all be equal to begin with, or may treat the fixed
    transfer volume as the volume as the minimum or maximum volume to transfer, adjusting volumes of each
    strand to make this work and have them at equal destination concentrations.

    Parameters
    ----------

    components
        A list of :ref:`Components`.

    fixed_volume
        A fixed volume for the action.  Input can be a string (eg, "5 µL") or a pint Quantity.  The interpretation
        of this depends on equal_conc.

    set_name
        The name of the mix.  If not set, name is based on components.

    compact_display
        If True (default), the action tries to display compactly in mix recipes.  If False, it displays
        each component as a separate line.

    method
        If `"check"`, the action still transfers the same volume of each component, but will
        raise a `ValueError` if this will not result in every component having the same concentration added
        (ie, if they have different source concentrations).  If `"min_volume"`, the action will transfer *at least*
        `fixed_volume` of each component, but will transfer more for components with lower source concentration,
        so that the destination concentrations are all equal (but not fixed to a specific value).  If `"max_volume"`,
        the action instead transfers *at most* `fixed_volume` of each component, tranferring less for higher
        source concentration components.  If ('max_fill', buffer_name), the fixed volume is the maximum, while for
        every component that is added at a lower volume, a corresponding volume of buffer is added to bring the total
        volume of the two up to the fixed volume.

    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ...     Component("c4", "100 nM")
    ... ]

    >>> print(Mix([EqualConcentration(components, "5 uL", method="min_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 5.00 µl     | 15.00 µl     |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 10.00 µl    | 10.00 µl     |       |        |

    >>> print(Mix([EqualConcentration(components, "5 uL", method="max_volume")], name="example"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 12.50 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 40.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 40.00 nM  | 1   | 5.00 µl     | 5.00 µl      |       |        |
    """

    def __init__(
        self,
        components: Sequence[AbstractComponent | str] | AbstractComponent | str,
        fixed_volume: str | Quantity,
        set_name: str | None = None,
        compact_display: bool = True,
        method: Literal["max_volume", "min_volume", "check"]
        | tuple[Literal["max_fill"], str] = "min_volume",
        equal_conc: bool | str | None = None,
    ):

        if equal_conc is not None:
            warn(
                "The equal_conc parameter for FixedVolume is no longer supported.  Use EqualConcentration and method instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if equal_conc is True:
                equal_conc = "check"

            method = equal_conc  # type: ignore

        self.__attrs_init__(components, fixed_volume, set_name, compact_display, method)  # type: ignore

    method: Literal["max_volume", "min_volume", "check"] | tuple[
        Literal["max_fill"], str
    ] = "min_volume"

    @property
    def source_concentrations(self) -> Sequence[Quantity[Decimal]]:
        concs = super().source_concentrations
        if any(x != concs[0] for x in concs) and (self.method == "check"):
            raise ValueError("Not all components have equal concentration.")
        return concs

    def each_volumes(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        # match self.equal_conc:
        if self.method == "min_volume":
            sc = self.source_concentrations
            scmax = max(sc)
            return [self.fixed_volume * x for x in _ratio(scmax, sc)]
        elif (self.method == "max_volume") | (
            isinstance(self.method, Sequence) and self.method[0] == "max_fill"
        ):
            sc = self.source_concentrations
            scmin = min(sc)
            return [self.fixed_volume * x for x in _ratio(scmin, sc)]
        elif self.method is "check":
            sc = self.source_concentrations
            if any(x != sc[0] for x in sc):
                raise ValueError("Concentrations")
            return [self.fixed_volume.to(uL)] * len(self.components)
        raise ValueError(f"equal_conc={repr(self.method)} not understood")

    def tx_volume(self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)) -> Quantity[Decimal]:
        if isinstance(self.method, Sequence) and (self.method[0] == "max_fill"):
            return self.fixed_volume * len(self.components)
        return sum(self.each_volumes(mix_vol), ureg("0.0 uL"))

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
        locations: pd.DataFrame | None = None,
    ) -> list[MixLine]:
        ml = super()._mixlines(tablefmt, mix_vol, locations)
        if isinstance(self.method, Sequence) and (self.method[0] == "max_fill"):
            fv = self.fixed_volume * len(self.components) - sum(self.each_volumes())
            if not fv == Q_(Decimal("0.0"), uL):
                ml.append(MixLine([self.method[1]], None, None, fv))
        return ml


@attrs.define()
class FixedConcentration(AbstractAction):
    """An action adding one or multiple components, with a set destination concentration per component (adjusting volumes).

    FixedConcentration adds a selection of components, with a specified destination concentration.

    Parameters
    ----------

    components
        A list of :ref:`Components`.

    fixed_concentration
        A fixed concentration for the action.  Input can be a string (eg, "50 nM") or a pint Quantity.

    set_name
        The name of the mix.  If not set, name is based on components.

    compact_display
        If True (default), the action tries to display compactly in mix recipes.  If False, it displays
        each component as a separate line.

    min_volume
        Specifies a minimum volume that must be transferred per component.  Currently, this is for
        validation only: it will cause a VolumeError to be raised if a volume is too low.

    Raises
    ------

    VolumeError
        One of the volumes to transfer is less than the specified min_volume.

    Examples
    --------

    >>> from alhambra.mixes import *
    >>> components = [
    ...     Component("c1", "200 nM"),
    ...     Component("c2", "200 nM"),
    ...     Component("c3", "200 nM"),
    ...     Component("c4", "100 nM")
    ... ]

    >>> print(Mix([FixedConcentration(components, "20 nM")], name="example", fixed_total_volume="25 uL"))
    Table: Mix: example, Conc: 40.00 nM, Total Vol: 25.00 µl
    <BLANKLINE>
    | Comp       | Src []    | Dest []   | #   | Ea Tx Vol   | Tot Tx Vol   | Loc   | Note   |
    |:-----------|:----------|:----------|:----|:------------|:-------------|:------|:-------|
    | c1, c2, c3 | 200.00 nM | 20.00 nM  | 3   | 2.50 µl     | 7.50 µl      |       |        |
    | c4         | 100.00 nM | 20.00 nM  |     | 5.00 µl     | 5.00 µl      |       |        |
    | Buffer     |           |           |     |             | 12.50 µl     |       |        |
    | *Total:*   |           | 40.00 nM  |     |             | 25.00 µl     |       |        |
    """

    components: list[AbstractComponent] = attrs.field(
        converter=_maybesequence_comps, on_setattr=attrs.setters.convert
    )
    fixed_concentration: Quantity[Decimal] = attrs.field(
        converter=_parse_conc_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    compact_display: bool = True
    min_volume: Quantity[Decimal] = attrs.field(
        converter=_parse_vol_optional,
        default=Q_(DNAN, uL),
        on_setattr=attrs.setters.convert,
    )

    def with_reference(self, reference: Reference) -> FixedConcentration:
        return attrs.evolve(
            self, components=[c.with_reference(reference) for c in self.components]
        )

    @property
    def source_concentrations(self) -> list[Quantity[Decimal]]:
        concs = [c.concentration for c in self.components]
        return concs

    def all_components(self, mix_vol: Quantity[Decimal]) -> pd.DataFrame:
        newdf = _empty_components()

        for comp, dc, sc in zip(
            self.components,
            self.dest_concentrations(mix_vol),
            self.source_concentrations,
        ):
            comps = comp.all_components()
            comps.concentration_nM *= _ratio(dc, sc)

            newdf, _ = newdf.align(comps)

            # FIXME: add checks
            newdf.loc[comps.index, "concentration_nM"] = newdf.loc[
                comps.index, "concentration_nM"
            ].add(comps.concentration_nM, fill_value=Decimal("0.0"))
            newdf.loc[comps.index, "component"] = comps.component

        return newdf

    def dest_concentrations(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        return [self.fixed_concentration] * len(self.components)

    def each_volumes(
        self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)
    ) -> list[Quantity[Decimal]]:
        ea_vols = [
            mix_vol * r
            for r in _ratio(self.fixed_concentration, self.source_concentrations)
        ]
        if not math.isnan(self.min_volume.m):
            below_min = []
            for comp, vol in zip(self.components, ea_vols):
                if vol < self.min_volume:
                    below_min.append((comp.name, vol))
            if below_min:
                raise VolumeError(
                    "Volume of some components is below minimum: "
                    + ", ".join(f"{n} at {v}" for n, v in below_min)
                    + ".",
                    below_min,
                )
        return ea_vols

    def tx_volume(self, mix_vol: Quantity[Decimal] = Q_(DNAN, uL)) -> Quantity[Decimal]:
        return sum(self.each_volumes(mix_vol), Q_(Decimal("0"), "uL"))

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: Quantity[Decimal],
        locations: pd.DataFrame | None = None,
    ) -> list[MixLine]:
        if not self.compact_display:
            ml = [
                MixLine(
                    [comp.printed_name(tablefmt=tablefmt)],
                    comp.concentration,
                    dc,
                    ev,
                    plate=comp.plate,
                    wells=comp._well_list,
                )
                for dc, ev, comp in zip(
                    self.dest_concentrations(mix_vol),
                    self.each_volumes(mix_vol),
                    self.components,
                )
            ]
        else:
            ml = list(self._compactstrs(tablefmt=tablefmt, mix_vol=mix_vol))

        return ml

    @property
    def number(self) -> int:
        return len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return ", ".join(c.name for c in self.components)
        else:
            return self.set_name

    def _compactstrs(
        self, tablefmt: str | TableFormat, mix_vol: Quantity
    ) -> Sequence[MixLine]:
        # locs = [(c.name,) + c.location for c in self.components]
        # names = [c.name for c in self.components]

        # if any(x is None for x in locs):
        #     raise ValueError(
        #         [name for name, loc in zip(names, locs) if loc is None]
        #     )

        locdf = pd.DataFrame(
            {
                "names": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_concs": self.source_concentrations,
                "dest_concs": self.dest_concentrations(mix_vol),
                "ea_vols": self.each_volumes(mix_vol),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            }
        )

        locdf.fillna({"plate": ""}, inplace=True)

        locdf.sort_values(
            by=["plate", "ea_vols", "well"], ascending=[True, False, True]
        )

        names: list[list[str]] = []
        source_concs: list[Quantity[Decimal]] = []
        dest_concs: list[Quantity[Decimal]] = []
        numbers: list[int] = []
        ea_vols: list[Quantity[Decimal]] = []
        tot_vols: list[Quantity[Decimal]] = []
        plates: list[str] = []
        wells_list: list[list[WellPos]] = []

        for plate, plate_comps in locdf.groupby("plate"):  # type: str, pd.DataFrame
            for vol, plate_vol_comps in plate_comps.groupby(
                "ea_vols"
            ):  # type: Quantity[Decimal], pd.DataFrame
                if pd.isna(plate_vol_comps["well"].iloc[0]):
                    if not pd.isna(plate_vol_comps["well"]).all():
                        raise ValueError
                    names.append(list(plate_vol_comps["names"]))
                    ea_vols.append((vol))
                    tot_vols.append((vol * len(plate_vol_comps)))
                    numbers.append((len(plate_vol_comps)))
                    source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                    dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                    plates.append(plate)
                    wells_list.append([])
                    continue

                byrow = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_byrow),
                    by="row",
                )
                bycol = mixgaps(
                    sorted(list(plate_vol_comps["well"]), key=WellPos.key_bycol),
                    by="col",
                )

                sortkey = WellPos.key_bycol if bycol <= byrow else WellPos.key_byrow

                plate_vol_comps["sortkey"] = [
                    sortkey(c) for c in plate_vol_comps["well"]
                ]

                plate_vol_comps.sort_values(by="sortkey", inplace=True)

                names.append(list(plate_vol_comps["names"]))
                ea_vols.append((vol))
                numbers.append((len(plate_vol_comps)))
                tot_vols.append((vol * len(plate_vol_comps)))
                source_concs.append((plate_vol_comps["source_concs"].iloc[0]))
                dest_concs.append((plate_vol_comps["dest_concs"].iloc[0]))
                plates.append(plate)
                wells_list.append(list(plate_vol_comps["well"]))

        return [
            MixLine(
                name,
                source_conc=source_conc,
                dest_conc=dest_conc,
                number=number,
                each_tx_vol=each_tx_vol,
                total_tx_vol=total_tx_vol,
                plate=p,
                wells=wells,
            )
            for name, source_conc, dest_conc, number, each_tx_vol, total_tx_vol, p, wells in zip(
                names,
                source_concs,
                dest_concs,
                numbers,
                ea_vols,
                tot_vols,
                plates,
                wells_list,
            )
        ]


MultiFixedConcentration = FixedConcentration
MultiFixedVolume = FixedVolume
