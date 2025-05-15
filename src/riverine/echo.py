from __future__ import annotations

import math
from abc import ABCMeta
from typing import TYPE_CHECKING, Literal, Sequence, cast

import attrs
import polars as pl
from tabulate import TableFormat

from riverine.util import gen_random_hash, maybe_cache_once

from .actions import AbstractAction, ActionWithComponents
from .experiments import Experiment
from .mixes import Mix
from .printing import MixLine

if TYPE_CHECKING:
    from kithairon.picklists import PickList


from .units import (
    NAN_VOL,
    Q_,
    DecimalQuantity,
    _parse_conc_required,
    _parse_vol_optional,
    _parse_vol_required,
    _ratio,
    uL,
)

try:
    from kithairon.picklists import PickList  # type: ignore
except ImportError as err:
    if err.name != "kithairon":
        raise err
    raise ImportError(
        "kithairon is required for Echo support, but it is not installed.",
        name="kithairon",
    )


DEFAULT_DROPLET_VOL = Q_(25, "nL")


class AbstractEchoAction(ActionWithComponents, metaclass=ABCMeta):
    """Abstract base class for Echo actions."""

    @maybe_cache_once
    def to_picklist(self, mix: Mix, experiment: Experiment | None = None, _cache_key=None) -> PickList:
        def el_get(key):
            if experiment is None:
                return None
            return experiment.locations.get(key, None)

        mix_vol = mix._get_total_volume(_cache_key=_cache_key)
        dconcs = self.dest_concentrations(mix_vol, mix.actions, _cache_key=_cache_key)
        eavols = self.each_volumes(mix_vol, mix.actions, _cache_key=_cache_key)
        locdf = PickList(
            pl.DataFrame(
                {
                    "Sample Name": [
                        c.printed_name(tablefmt="plain") for c in self.components
                    ],
                    "Source Concentration": [
                        float(c.m_as("nM")) for c in self._get_source_concentrations(_cache_key=_cache_key)
                    ],
                    "Destination Concentration": [float(c.m_as("nM")) for c in dconcs],
                    "Concentration Units": "nM",
                    "Transfer Volume": [float(v.m_as("nL")) for v in eavols],
                    "Source Plate Name": [c.plate for c in self.components],
                    "Source Plate Type": [
                        getattr(
                            el_get(c.plate),
                            "echo_source_type",
                            None,
                        )
                        for c in self.components
                    ],
                    "Source Well": [str(c.well) for c in self.components],
                    "Destination Plate Name": mix.plate,
                    "Destination Plate Type": getattr(
                        el_get(mix.plate),
                        "echo_dest_type",
                        None,
                    ),
                    "Destination Well": str(mix.well),
                    "Destination Sample Name": mix.name,
                },
                schema_overrides={
                    "Sample Name": pl.String,
                    "Source Concentration": pl.Float64,
                    "Destination Concentration": pl.Float64,
                    "Concentration Units": pl.String,
                    "Transfer Volume": pl.Float64,
                    "Source Plate Name": pl.String,
                    "Source Well": pl.String,
                    "Destination Plate Name": pl.String,
                    "Destination Well": pl.String,
                    "Destination Sample Name": pl.String,
                    "Destination Plate Type": pl.String,
                    "Source Plate Type": pl.String,
                },
                # , schema_overrides={"Source Concentration": pl.Decimal(scale=6), "Destination Concentration": pl.Decimal(scale=6), "Transfer Volume": pl.Decimal(scale=6)} # FIXME: when new polars is released
            )
        )
        return locdf


@attrs.define(eq=False)
class EchoFixedVolume(AbstractEchoAction):
    """Transfer a fixed volume of liquid to a target mix."""

    fixed_volume: DecimalQuantity = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = True

    def _check_volume(self) -> None:
        fv = self.fixed_volume.m_as("nL")
        dv = self.droplet_volume.m_as("nL")
        # ensure that fv is an integer multiple of dv
        if fv % dv != 0:
            raise ValueError(
                f"Fixed volume {fv} is not an integer multiple of droplet volume {dv}."
            )

    @maybe_cache_once
    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        _cache_key = gen_random_hash() if _cache_key is None else _cache_key
        return [
            x * y
            for x, y in zip(
                self._get_source_concentrations(_cache_key=_cache_key),
                _ratio(self.each_volumes(mix_vol, _cache_key=_cache_key), mix_vol),
            )
        ]

    def each_volumes(
        self,
        mix_volume: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        return [cast(DecimalQuantity, self.fixed_volume.to(uL))] * len(self.components)

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name

    @maybe_cache_once
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[MixLine]:
        _cache_key = gen_random_hash() if _cache_key is None else _cache_key
        dconcs = self.dest_concentrations(mix_vol, actions, _cache_key=_cache_key)
        eavols = self.each_volumes(mix_vol, actions, _cache_key=_cache_key)

        locdf = pl.DataFrame(
            {
                "name": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_conc": list(
                    self._get_source_concentrations(_cache_key=_cache_key)
                ),
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

        vs = locdf.group_by(
            ("source_conc", "dest_conc", "ea_vols"), maintain_order=True
        ).agg(pl.col("name"), pl.col("plate").unique())

        ml = [
            MixLine(
                [f"{len(q['name'])} comps: {q['name'][0]}, ..."]
                if len(q["name"]) > 5
                else [", ".join(q["name"])],
                q["source_conc"],
                q["dest_conc"],
                len(q["name"]) * self.fixed_volume,
                number=self.number,
                each_tx_vol=self.fixed_volume,
                plate=(", ".join(x for x in q["plate"] if x) if q["plate"] else "?"),
                wells=[],
                note="ECHO",
            )
            for q in vs.iter_rows(named=True)
        ]

        return ml


@attrs.define(eq=False)
class EchoEqualTargetConcentration(AbstractEchoAction):
    """Transfer a fixed volume of liquid to a target mix."""

    fixed_volume: DecimalQuantity = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = False
    method: (
        Literal["max_volume", "min_volume", "check"] | tuple[Literal["max_fill"], str]
    ) = "min_volume"

    def _check_volume(self) -> None:
        fv = self.fixed_volume.m_as("nL")
        dv = self.droplet_volume.m_as("nL")
        # ensure that fv is an integer multiple of dv
        if fv % dv != 0:
            raise ValueError(
                f"Fixed volume {fv} is not an integer multiple of droplet volume {dv}."
            )

    @maybe_cache_once
    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self._get_source_concentrations(_cache_key=_cache_key),
                _ratio(self.each_volumes(mix_vol, _cache_key=_cache_key), mix_vol),
            )
        ]

    @maybe_cache_once
    def each_volumes(
        self,
        mix_volume: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        if self.method == "min_volume":
            sc = self._get_source_concentrations(_cache_key=_cache_key)
            scmax = max(sc)
            return [
                round((self.fixed_volume * x / self.droplet_volume).m_as(""))
                * self.droplet_volume
                for x in _ratio(scmax, sc)
            ]
        elif (self.method == "max_volume") | (
            isinstance(self.method, Sequence) and self.method[0] == "max_fill"
        ):
            sc = self._get_source_concentrations(_cache_key=_cache_key)
            scmin = min(sc)
            return [
                round((self.fixed_volume * x / self.droplet_volume).m_as(""))
                * self.droplet_volume
                for x in _ratio(scmin, sc)
            ]
        elif self.method == "check":
            sc = self._get_source_concentrations(_cache_key=_cache_key)
            if any(x != sc[0] for x in sc):
                raise ValueError("Concentrations")
            return [cast(DecimalQuantity, self.fixed_volume.to(uL))] * len(
                self.components
            )
        raise ValueError(f"equal_conc={self.method!r} not understood")

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name

    @maybe_cache_once
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions, _cache_key=_cache_key)
        eavols = self.each_volumes(mix_vol, actions, _cache_key=_cache_key)

        locdf = pl.DataFrame(
            {
                "name": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_conc": list(self._get_source_concentrations(_cache_key=_cache_key)),
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

        vs = locdf.group_by(
            ("source_conc", "dest_conc", "ea_vols"), maintain_order=True
        ).agg(pl.col("name"), pl.col("plate").unique())

        ml = [
            MixLine(
                [f"{len(q['name'])} comps: {q['name'][0]}, ..."]
                if len(q["name"]) > 5
                else [", ".join(q["name"])],
                q["source_conc"],
                q["dest_conc"],
                len(q["name"]) * q["ea_vols"],
                number=self.number,
                each_tx_vol=q["ea_vols"],
                plate=(", ".join(x for x in q["plate"] if x) if q["plate"] else "?"),
                wells=[],
                note="ECHO",
            )
            for q in vs.iter_rows(named=True)
        ]

        return ml


@attrs.define(eq=False)
class EchoTargetConcentration(AbstractEchoAction):
    """Get as close as possible (using direct transfers) to a target concentration, possibly varying mix volume."""

    target_concentration: DecimalQuantity = attrs.field(
        converter=_parse_conc_required, on_setattr=attrs.setters.convert
    )
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = True

    @maybe_cache_once
    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self._get_source_concentrations(_cache_key=_cache_key),
                _ratio(
                    self.each_volumes(mix_vol, actions, _cache_key=_cache_key), mix_vol
                ),
            )
        ]

    @maybe_cache_once
    def each_volumes(
        self,
        mix_volume: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        _cache_key = gen_random_hash() if _cache_key is None else _cache_key
        ea_vols = [
            (
                round((mix_volume * r / self.droplet_volume).m_as(""))
                * self.droplet_volume
            )
            if not math.isnan(mix_volume.m) and not math.isnan(r)
            else NAN_VOL
            for r in _ratio(
                self.target_concentration,
                self._get_source_concentrations(_cache_key=_cache_key),
            )
        ]
        return ea_vols

    @maybe_cache_once
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions, _cache_key=_cache_key)
        eavols = self.each_volumes(mix_vol, actions, _cache_key=_cache_key)

        locdf = pl.DataFrame(
            {
                "name": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_conc": list(self._get_source_concentrations(_cache_key=_cache_key)),
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

        vs = locdf.group_by(
            ("source_conc", "dest_conc", "ea_vols"), maintain_order=True
        ).agg(pl.col("name"), pl.col("plate").unique())

        ml = [
            MixLine(
                [f"{len(q['name'])} comps: {q['name'][0]}, ..."]
                if len(q["name"]) > 5
                else [", ".join(q["name"])],
                q["source_conc"],
                q["dest_conc"],
                len(q["name"]) * q["ea_vols"],
                number=self.number,
                each_tx_vol=q["ea_vols"],
                plate=(", ".join(x for x in q["plate"] if x) if q["plate"] else "?"),
                wells=[],
                note=f"ECHO, target {self.target_concentration}",
            )
            for q in vs.iter_rows(named=True)
        ]

        return ml

    @property
    def name(self) -> str:
        if self.set_name is None:
            return super().name
        else:
            return self.set_name


@attrs.define(eq=False)
class EchoFillToVolume(AbstractEchoAction):
    target_total_volume: DecimalQuantity = attrs.field(
        converter=_parse_vol_optional, default=None
    )
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL

    @maybe_cache_once
    def dest_concentrations(
        self,
        mix_vol: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        return [
            x * y
            for x, y in zip(
                self._get_source_concentrations(_cache_key=_cache_key),
                _ratio(
                    self.each_volumes(mix_vol, actions, _cache_key=_cache_key), mix_vol
                ),
            )
        ]

    @maybe_cache_once
    def each_volumes(
        self,
        mix_volume: DecimalQuantity = NAN_VOL,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[DecimalQuantity]:
        _cache_key = gen_random_hash() if _cache_key is None else _cache_key
        othervol = sum(
            [
                a.tx_volume(mix_volume, actions, _cache_key=_cache_key)
                for a in actions
                if a is not self
            ]
        )

        if len(self.components) > 1:
            raise NotImplementedError(
                "EchoTargetConcentration with multiple components is not implemented."
            )

        if math.isnan(self.target_total_volume.m):
            tvol = mix_volume
        else:
            tvol = self.target_total_volume

        maybe_vol = ((tvol - othervol) / self.droplet_volume).m_as("")
        if math.isnan(maybe_vol):
            return [NAN_VOL] * len(self.components)

        return [
            round(maybe_vol)
            * self.droplet_volume
        ]
        return ea_vols

    @maybe_cache_once
    def _mixlines(
        self,
        tablefmt: str | TableFormat,
        mix_vol: DecimalQuantity,
        actions: Sequence[AbstractAction] = (),
        _cache_key=None,
    ) -> list[MixLine]:
        dconcs = self.dest_concentrations(mix_vol, actions, _cache_key=_cache_key)
        eavols = self.each_volumes(mix_vol, actions, _cache_key=_cache_key)
        return [
            MixLine(
                [comp.printed_name(tablefmt=tablefmt)],
                comp.concentration
                if not math.isnan(comp.concentration.m)
                else None,  # FIXME: should be better handled
                dc if not math.isnan(dc.m) else None,
                ev,
                number=self.number,
                plate=comp.plate if comp.plate else "?",
                wells=comp._well_list,
                note="ECHO",
            )
            for dc, ev, comp in zip(
                dconcs,
                eavols,
                self.components,
            )
        ]


# class EchoTwoStepConcentration(ActionWithComponents):
#     """Use an intermediate mix to obtain a target concentration."""

#     ...
