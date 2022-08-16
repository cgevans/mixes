from __future__ import annotations

from math import isnan
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TextIO, cast

import attrs
from typing_extensions import TypeAlias

from .locations import PlateType, WellPos, _parse_wellpos_optional
from .units import (
    DNAN,
    Q_,
    Decimal,
    Quantity,
    _parse_conc_optional,
    _parse_conc_required,
    nM,
    ureg,
)

if TYPE_CHECKING:  # pragma: no cover
    from .mixes import PlateMap
    from pandas.core.indexing import _LocIndexer

import numpy as np
import pandas as pd

_REF_COLUMNS = ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"]
_REF_DTYPES = [object, object, object, np.float64, object]

RefFile: TypeAlias = "str | tuple[str, Quantity | str | dict[str, Quantity]]"


def _new_ref_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=_REF_COLUMNS)
    df["Concentration (nM)"] = df["Concentration (nM)"].astype("float")
    return df


@attrs.define()
class Reference:
    df: pd.DataFrame = attrs.field(factory=_new_ref_df)

    @property
    def loc(self) -> _LocIndexer:
        return self.df.loc

    def __getitem__(self, key: Any) -> Any:
        return self.df.__getitem__(key)

    def __eq__(self: Reference, other: object) -> bool:
        if isinstance(other, Reference):
            return (
                ((other.df == self.df) | (other.df.isna() & self.df.isna())).all().all()
            )
        elif isinstance(other, pd.DataFrame):
            return ((other == self.df) | (other.isna() & self.df.isna())).all().all()
        return False

    def __len__(self) -> int:
        return len(self.df)

    def plate_map(
        self,
        name: str,
        plate_type: PlateType = PlateType.wells96,
    ) -> PlateMap:
        """
        :param name:
            Name of plate to make a :class:`PlateMap` for.
        :param plate_type:
            Either :data:`PlateType.wells96` or :data:`PlateType.wells384`;
            default is :data:`PlateType.wells96`.
        :return:
            a :class:`PlateMap` consisting of all strands in this Reference object from plate named
            `name`. Currently always makes a 96-well plate.
        """
        well_to_strand_name = {}
        for row in self.df.itertuples():
            if row.Plate == name:  # type: ignore
                well = row.Well  # type: ignore
                sequence = row.Sequence  # type: ignore
                strand = Strand(name=row.Name, sequence=sequence)  # type: ignore
                well_to_strand_name[well] = strand.name

        plate_map = PlateMap(
            plate_name=name,
            plate_type=plate_type,
            well_to_strand_name=well_to_strand_name,
        )
        return plate_map

    def search(
        self,
        name: str | None = None,
        plate: str | None = None,
        well: str | WellPos | None = None,
        concentration: str | Quantity[Decimal] | None = None,
        sequence: str | None = None,
    ) -> Reference:
        well = _parse_wellpos_optional(well)
        concentration = _parse_conc_optional(concentration)
        cdf = self.df

        if name is not None:
            cdf = cdf.loc[cdf["Name"] == name, :]
        if plate is not None:
            cdf = cdf.loc[cdf["Plate"] == plate, :]
        if well is not None:
            cdf = cdf.loc[cdf["Well"] == str(well), :]
        if not isnan(concentration.m):
            conc = concentration.m_as("nM")
            cdf = cdf.loc[cdf["Concentration (nM)"] == conc, :]
        if sequence is not None:
            cdf = cdf.loc[cdf["Sequence"] == sequence, :]
        return Reference(cdf)

    def get_concentration(
        self,
        name: str | None = None,
        plate: str | None = None,
        well: str | WellPos | None = None,
        concentration: str | Quantity | None = None,
        sequence: str | None = None,
    ) -> Quantity[Decimal]:
        valref = self.search(name, plate, well, concentration, sequence)

        if len(valref) == 1:
            return Q_(valref.df["Concentration (nM)"].iloc[0], nM)
        elif len(valref) > 1:
            raise ValueError(
                f"Found multiple possible components: {str(valref)}", valref
            )

        raise ValueError("Did not find any matching components.")

    @classmethod
    def from_csv(cls, filename_or_file: str | TextIO | PathLike[str]) -> Reference:
        """
        Load reference information from a CSV file.

        The reference information loaded by this function should be compiled manually, fitting the :ref:`mix reference` format, or
        be loaded with :func:`compile_reference` or :func:`update_reference`.
        """
        df = pd.read_csv(filename_or_file, converters={"Concentration (nM)": Decimal})

        df = df.reindex(
            ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"], axis="columns"
        )

        return cls(df)

    def to_csv(self, filename: str | PathLike[str]) -> None:
        self.df.to_csv(filename, index=None, float_format="%.6f")

    def update(
        self: Reference, files: Sequence[RefFile] | RefFile, round: int = -1
    ) -> Reference:
        """
        Update reference information.

        This updates an existing reference dataframe with new files, with the same methods as :func:`compile_reference`.
        """
        if isinstance(files, str) or (
            len(files) == 2
            and isinstance(files[1], str)
            and not Path(files[1]).exists()
        ):
            files_list: Sequence[RefFile] = [cast(RefFile, files)]
        else:
            files_list = cast(Sequence[RefFile], files)

        # FIXME: how to deal with repeats?
        for filename in files_list:
            filetype = None
            all_conc = None
            conc_dict: dict[str, Quantity] = {}

            if isinstance(filename, tuple):
                conc_info = filename[1]
                filepath = Path(filename[0])

                if isinstance(conc_info, Mapping):
                    conc_dict = {
                        k: _parse_conc_required(v) for k, v in conc_info.values()
                    }
                    if "default" in conc_dict:
                        all_conc = _parse_conc_required(conc_dict["default"])
                        del conc_dict["default"]
                else:
                    all_conc = _parse_conc_required(conc_info)
            else:
                filepath = Path(filename)

            if filepath.suffix in (".xls", ".xlsx"):
                data: dict[str, pd.DataFrame] = pd.read_excel(filepath, sheet_name=None)
                if "Plate Specs" in data:
                    if len(data) > 1:
                        raise ValueError(
                            f"Plate specs file {filepath} should only have one sheet, but has {len(data)}."
                        )
                    sheet: pd.DataFrame = data["Plate Specs"]
                    filetype = "plate-specs"

                    sheet.rename(lambda x: x.lower(), inplace=True, axis="columns")

                    sheet.loc[:, "Concentration (nM)"] = 1000 * sheet.loc[
                        :, "measured concentration µm "
                    ].round(round)
                    sheet.loc[:, "Sequence"] = [
                        x.replace(" ", "") for x in sheet.loc[:, "sequence"]
                    ]
                    sheet.loc[:, "Well"] = [
                        str(WellPos(x)) for x in sheet.loc[:, "well position"]
                    ]
                    sheet.rename(
                        {
                            "plate name": "Plate",
                            "sequence name": "Name",
                        },
                        axis="columns",
                        inplace=True,
                    )

                    self.df = pd.concat(
                        (self.df, sheet.loc[:, _REF_COLUMNS]), ignore_index=True
                    )

                    continue

                else:
                    # FIXME: need better check here
                    # if not all(
                    #    next(iter(data.values())).columns
                    #    == ["Well Position", "Name", "Sequence"]
                    # ):
                    #    raise ValueError
                    filetype = "plates-order"
                    for k, v in data.items():
                        if "Plate" in v.columns:
                            # There's already a plate column.  That's problematic.  Let's check,
                            # then delete it.
                            if not all(v["Plate"] == k):
                                raise ValueError(
                                    "Not all rows in sheet {k} have same plate value (normal IDT order files do not have a plate column)."
                                )
                            del v["Plate"]
                        v["Concentration (nM)"] = conc_dict.get(
                            k, all_conc if all_conc is not None else Q_(DNAN, nM)
                        ).m_as(nM)
                    all_seqs = (
                        pd.concat(
                            data.values(), keys=data.keys(), names=["Plate"], copy=False
                        )
                        .reset_index()
                        .drop(columns=["level_1"])
                    )
                    all_seqs.rename(
                        {"Well Position": "Well", "Well position": "Well"},
                        axis="columns",
                        inplace=True,
                    )
                    all_seqs.loc[:, "Well"] = all_seqs.loc[:, "Well"].map(
                        lambda x: str(WellPos(x))
                    )

                    self.df = pd.concat((self.df, all_seqs), ignore_index=True)
                    continue

            if filepath.suffix == ".csv":
                # Are we a COA file?  If so, it isn't valid Unicode...
                # We'll check initially in binary mode.
                with filepath.open("rb") as f:
                    testbin = f.read(25)
                if testbin == b'"Sales Order","Reference"':
                    # We're a COA file... in case IDT fixes things, we'll try UTF-8
                    try:
                        df = pd.read_csv(filepath)
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding="iso8859-1")
                    self.df = pd.concat(
                        (self.df, _parse_idt_coa(df)), ignore_index=True
                    )
                    continue
                else:
                    tubedata = pd.read_csv(filepath)
                    filetype = "idt-bulk"

            if filepath.suffix == ".txt":
                tubedata = pd.read_table(filepath)
                filetype = "idt-bulk"

            if filetype == "idt-bulk":
                tubedata["Plate"] = "tube"
                tubedata["Well"] = None
                tubedata["Concentration (nM)"] = (
                    all_conc.m_as(nM) if all_conc is not None else DNAN
                )
                self.df = pd.concat(
                    (self.df, tubedata.loc[:, _REF_COLUMNS]), ignore_index=True
                )
                continue

            raise NotImplementedError

        # FIXME: validation

        return self

    @classmethod
    def compile(cls, files: Sequence[RefFile] | RefFile, round: int = -1) -> Reference:
        """
        Compile reference information.

        This loads information from the following sources:

        - An IDT plate order spreadsheet.  This does not include concentration.  To add concentration information, list it as a tuple of
        :code:`(file, concentration)`.
        - An IDT bulk order entry text file.
        - An IDT plate spec sheet.
        """
        return cls().update(files, round=round)


_REF_COLUMNS = ["Name", "Plate", "Well", "Concentration (nM)", "Sequence"]


def _parse_idt_coa(df: pd.DataFrame) -> pd.DataFrame:
    df.rename({"Sequence Name": "Name"}, axis="columns", inplace=True)
    df.loc[:, "Well"] = df.loc[:, "Well Position"].map(lambda x: str(WellPos(x)))
    df.loc[:, "Concentration (nM)"] = df.loc[:, "Conc"].map(lambda x: ureg(x).m_as(nM))
    df.loc[:, "Plate"] = None
    df.loc[:, "Sequence"] = df.loc[:, "Sequence"].str.replace(" ", "")
    return df.loc[:, _REF_COLUMNS]


def load_reference(filename_or_file: str | TextIO) -> Reference:
    return Reference.from_csv(filename_or_file)
