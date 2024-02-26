import attrs
from tabulate import TableFormat
from .actions import ActionWithComponents, AbstractAction
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Sequence, cast
import polars as pl

from .printing import MixLine


from .units import DNAN, DecimalQuantity, _parse_vol_optional, Q_, _parse_vol_required, _ratio, nM, uL

DEFAULT_DROPLET_VOL = Q_(25, "nL")

from .experiments import Experiment
from .mixes import Mix


import attrs
from abc import ABCMeta, abstractmethod
from typing import Any, Literal

from kithairon.picklists import PickList


class AbstractLocationType(metaclass=ABCMeta):
    __slots__ = ()
    @property
    @abstractmethod
    def name(self):
        ...
    
    @property
    @abstractmethod
    def is_echo_source_compatible(self) -> bool:
        return False
    
class LocationType(AbstractLocationType):
    __slots__ = ("name", "loc_type", "is_echo_source_compatible")
    name: str
    loc_type: Literal["plate96", "plate384", "tube"]
    is_echo_source_compatible: bool
    
    def __init__(self, name: str, loc_type: Literal["plate96", "plate384", "tube"], is_echo_source_compatible: bool = False):
        self.name = name
        self.loc_type = loc_type
        self.is_echo_source_compatible = is_echo_source_compatible
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"LocationType({self.name}, {self.loc_type}, {self.is_echo_source_compatible})"
    
    def __eq__(self, other):
        return self.name == other.name and self.loc_type == other.loc_type and self.is_echo_source_compatible == other.is_echo_source_compatible
    
    def __hash__(self):
        return hash((self.name, self.loc_type, self.is_echo_source_compatible))
    
LOCATION_TYPE_MAP = {
    '384PP_AQ_BP': LocationType('384PP_AQ_BP', 'plate384', True),
}    

def _location_type_converter(value: AbstractLocationType | str) -> AbstractLocationType:
    if isinstance(value, AbstractLocationType):
        return value
    elif isinstance(value, str):
        return LOCATION_TYPE_MAP[value]
    else:
        raise ValueError(f"Invalid location type: {value}")
    
@attrs.define()
class LocationInfo:
    name: str
    location_type: AbstractLocationType = attrs.field(converter=_location_type_converter)
    full_location: tuple[str, ...] = ()
    info: dict[str, Any] = attrs.field(factory=dict)

class AbstractEchoAction(AbstractAction, metaclass=ABCMeta):
    """Abstract base class for Echo actions."""

    @abstractmethod
    def to_picklist(
        self, mix: 'Mix', experiment: 'Experiment | None' = None
    ) -> 'PickList':
        ...


@attrs.define(eq=True)
class EchoFixedVolume(ActionWithComponents, AbstractEchoAction):
    """Transfer a fixed volume of liquid to a target mix."""

    fixed_volume: DecimalQuantity = attrs.field(converter=_parse_vol_required)
    set_name: str | None = None
    droplet_volume: DecimalQuantity = DEFAULT_DROPLET_VOL
    compact_display: bool = False

    def to_picklist(
        self, mix: Mix, experiment: Experiment | None = None
    ) -> 'PickList':
        mix_vol = mix.total_volume
        dconcs = self.dest_concentrations(mix_vol)
        eavols = self.each_volumes(mix_vol)
        locdf = PickList(pl.DataFrame(
            {
                "Sample Name": [c.printed_name(tablefmt='plain') for c in self.components],
                "Source Concentration": [float(c.m_as("nM")) for c in self.source_concentrations],
                "Destination Concentration": [float(c.m_as("nM")) for c in dconcs],
                "Concentration Units": "nM",
                "Transfer Volume": [float(v.m_as("nL")) for v in eavols],
                "Source Plate Name": [c.plate for c in self.components],
                "Source Well": [str(c.well) for c in self.components],
                "Destination Plate Name": mix.plate,
                "Destination Well": str(mix.well),
                "Destination Sample Name": mix.name,
            } # , schema_overrides={"Source Concentration": pl.Decimal(scale=6), "Destination Concentration": pl.Decimal(scale=6), "Transfer Volume": pl.Decimal(scale=6)} # FIXME: when new polars is released
        ))
        return locdf
        

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
        dconcs = self.dest_concentrations(mix_vol)
        eavols = self.each_volumes(mix_vol)

        locdf = pl.DataFrame(
            {
                "name": [c.printed_name(tablefmt=tablefmt) for c in self.components],
                "source_conc": list(self.source_concentrations),
                "dest_conc": list(dconcs),
                "ea_vols": list(eavols),
                "plate": [c.plate for c in self.components],
                "well": [c.well for c in self.components],
            }
            , schema_overrides={"source_conc": pl.Object, "dest_conc": pl.Object, "ea_vols": pl.Object}
        )

        vs = locdf.group_by(('source_conc','dest_conc'), maintain_order=True).agg(pl.col('name'), pl.col('plate'))

        if not self.compact_display:
            ml = [
                MixLine(
                    q['name'],
                    q['source_conc'],
                    q['dest_conc'],
                    self.fixed_volume,
                    plate=q['plate'],
                    wells=[],
                    note="ECHO"
                )
                for q in vs.iter_rows(named=True)
            ]
        else:
            ml = [
                MixLine(
                    f"{len(q['name'])} components",
                    q['source_conc'],
                    q['dest_conc'],
                    self.fixed_volume,
                    plate=q['plate'],
                    wells=[],
                    note="ECHO"
                )
                for q in vs.iter_rows(named=True)
            ]

        return ml

# class EchoTargetConcentration(ActionWithComponents):
#     """Get as close as possible (using direct transfers) to a target concentration, possibly varying mix volume."""

#     ...


# class EchoTwoStepConcentration(ActionWithComponents):
#     """Use an intermediate mix to obtain a target concentration."""

#     ...
