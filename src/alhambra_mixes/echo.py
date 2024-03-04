import math
import attrs
from tabulate import TableFormat
from .actions import ActionWithComponents, AbstractAction
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Sequence, cast
import polars as pl

from .printing import MixLine

from .experiments import Experiment
from .mixes import Mix


import attrs
from abc import ABCMeta

from kithairon.picklists import PickList


from .units import (
    DNAN,
    DecimalQuantity,
    _parse_conc_required,
    _parse_vol_optional,
    Q_,
    _parse_vol_required,
    _ratio,
    nM,
    uL,
)

DEFAULT_DROPLET_VOL = Q_(25, "nL")


class AbstractEchoAction(AbstractAction, metaclass=ABCMeta):
    """Abstract base class for Echo actions."""

    def to_picklist(self, mix: Mix, experiment: Experiment | None = None) -> "PickList":
        mix_vol = mix.total_volume
        dconcs = self.dest_concentrations(mix_vol, mix.actions)
        eavols = self.each_volumes(mix_vol, mix.actions)
        locdf = PickList(
            pl.DataFrame(
                {
                    "Sample Name": [
                        c.printed_name(tablefmt="plain") for c in self.components
                    ],
                    "Source Concentration": [
                        float(c.m_as("nM")) for c in self.source_concentrations
                    ],
                    "Destination Concentration": [float(c.m_as("nM")) for c in dconcs],
                    "Concentration Units": "nM",
                    "Transfer Volume": [float(v.m_as("nL")) for v in eavols],
                    "Source Plate Name": [c.plate for c in self.components],
                    "Source Plate Type": [
                        getattr(
                            experiment.locations.get(c.plate, None),
                            "echo_source_type",
                            None,
                        )
                        for c in self.components
                    ],
                    "Source Well": [str(c.well) for c in self.components],
                    "Destination Plate Name": mix.plate,
                    "Destination Plate Type": getattr(
                        experiment.locations.get(mix.plate, None),
                        "echo_dest_type",
                        None,
                    ),
                    "Destination Well": str(mix.well),
                    "Destination Sample Name": mix.name,
                },
                schema_overrides={
                    "Destination Plate Type": pl.String,
                    "Source Plate Type": pl.String,
                },
                # , schema_overrides={"Source Concentration": pl.Decimal(scale=6), "Destination Concentration": pl.Decimal(scale=6), "Transfer Volume": pl.Decimal(scale=6)} # FIXME: when new polars is released
            )
        )
        return locdf


@attrs.define(eq=True)
class EchoFixedVolume(ActionWithComponents, AbstractEchoAction):
    """Transfer a fixed volume of liquid to a target mix."""

    fixed_volume: DecimalQuantity = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = False

    def _check_volume(self) -> None:
        fv = self.fixed_volume.m_as("nL")
        dv = self.droplet_volume.m_as("nL")
        # ensure that fv is an integer multiple of dv
        if fv % dv != 0:
            raise ValueError(
                f"Fixed volume {fv} is not an integer multiple of droplet volume {dv}."
            )

    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self.source_concentrations, _ratio(self.each_volumes(mix_vol), mix_vol)
            )
        ]

    def each_volumes(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        return [cast(DecimalQuantity, self.fixed_volume.to(uL))] * len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions)
        eavols = self.each_volumes(mix_vol, actions)

        locdf = pl.DataFrame(
            {
                "name": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_conc": list(self.source_concentrations),
                "dest_conc": list(dconcs),
                "ea_vols": list(eavols),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            },
            schema_overrides={
                "source_conc": pl.Object,
                "dest_conc": pl.Object,
                "ea_vols": pl.Object,
            },
        )

        vs = locdf.group_by(("source_conc", "dest_conc"), maintain_order=True).agg(
            pl.col("name"), pl.col("plate")
        )

        if not self.compact_display:
            ml = [
                MixLine(
                    q["name"],
                    q["source_conc"],
                    q["dest_conc"],
                    self.fixed_volume,
                    plate=q["plate"],
                    wells=[],
                    note="ECHO",
                )
                for q in vs.iter_rows(named=True)
            ]
        else:
            ml = [
                MixLine(
                    f"{len(q['name'])} components",
                    q["source_conc"],
                    q["dest_conc"],
                    self.fixed_volume,
                    plate=q["plate"],
                    wells=[],
                    note="ECHO",
                )
                for q in vs.iter_rows(named=True)
            ]

        return ml


@attrs.define(eq=True)
class EchoTargetConcentration(ActionWithComponents):
    """Get as close as possible (using direct transfers) to a target concentration, possibly varying mix volume."""

    target_concentration: DecimalQuantity = attrs.field(
        converter=_parse_conc_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = True

    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self.source_concentrations,
                _ratio(self.each_volumes(mix_vol, actions), mix_vol),
            )
        ]

    def each_volumes(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        ea_vols = [
            round((mix_vol * r / self.droplet_volume).m_as("")) * self.droplet_volume
            for r in _ratio(self.target_concentration, self.source_concentrations)
        ]
        return ea_vols

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions)
        eavols = self.each_volumes(mix_vol, actions)
        if not self.compact_display:
            ml = [
                MixLine(
                    [comp.printed_name(tablefmt=tablefmt)],
                    comp.concentration,
                    dc,
                    ev,
                    plate=comp.plate,
                    wells=comp._well_list,
                    note=f"ECHO, target {self.target_concentration}",
                )
                for dc, ev, comp in zip(
                    dconcs,
                    eavols,
                    self.components,
                )
            ]
        else:
            ml = list(
                self._compactstrs(
                    tablefmt=tablefmt,
                    dconcs=dconcs,
                    eavols=eavols,
                ))
            for m in ml:
                m.note = f"ECHO, target {self.target_concentration}"
            

        return ml

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name


@attrs.define(eq=True)
class EchoFillToVolume(ActionWithComponents, AbstractEchoAction):
    target_total_volume: DecimalQuantity = attrs.field(
        converter=_parse_vol_optional, default=None
    )
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL

    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self.source_concentrations,
                _ratio(self.each_volumes(mix_vol, actions), mix_vol),
            )
        ]

    def each_volumes(
        self,
        mix_vol: DecimalQuantity = Q_(DNAN, uL),
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[DecimalQuantity]:
        othervol = sum(
            [a.tx_volume(mix_vol, actions) for a in actions if a is not self]
        )

        if len(self.components) > 1:
            raise NotImplementedError(
                "EchoTargetConcentration with multiple components is not implemented."
            )

        if math.isnan(self.target_total_volume.m):
            tvol = mix_vol
        else:
            tvol = self.target_total_volume

        ea_vols = [
            round(((tvol - othervol) / self.droplet_volume).m_as(""))
            * self.droplet_volume
        ]
        return ea_vols

    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = tuple(),
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions)
        eavols = self.each_volumes(mix_vol, actions)
        return [
            MixLine(
                [comp.printed_name(tablefmt=tablefmt)],
                comp.concentration,
                dc,
                ev,
                plate=comp.plate,
                wells=comp._well_list,
                note="ECHO",
            )
            for dc, ev, comp in zip(
                dconcs,
                eavols,
                self.components,
            )
        ]

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name


# class EchoTwoStepConcentration(ActionWithComponents):
#     """Use an intermediate mix to obtain a target concentration."""

#     ...
