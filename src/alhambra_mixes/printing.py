from __future__ import annotations

from math import isnan
from typing import Any, Iterable, Literal, Sequence, Union, cast

import attrs
import numpy as np
import pandas as pd
from tabulate import Line, TableFormat, tabulate

from .locations import WellPos, mixgaps
from .units import *
from .units import VolumeError


def emphasize(text: str, tablefmt: str | TableFormat, strong: bool = False) -> str:
    """
    Emphasizes `text` according to `tablefmt`, e.g., for Markdown (e.g., `tablefmt` = `'pipe'`),
    surrounds with pair of *'s; if `strong` is True, with double *'s. For `tablefmt` = `'html'`,
    uses ``<emph>`` or ``<strong>``.

    :param text:
        text to emphasize
    :param tablefmt:
        format in which to add emphasis markup
    :return:
        emphasized version of `text`
    """
    # formats a title for a table produced using tabulate,
    # in the formats tabulate understands
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:  # type: ignore
        if strong:
            emph_text = f"<strong>{text}</strong>"
        else:
            emph_text = f"<em>{text}</em>"
    elif tablefmt in ["latex", "latex_raw", "latex_booktabs", "latex_longtable"]:
        if strong:
            emph_text = r"\textbf{" + text + r"}"
        else:
            emph_text = r"\emph{" + text + r"}"
    else:  # use the emphasis for tablefmt == "pipe" (Markdown)
        star = "**" if strong else "*"
        emph_text = f"{star}{text}{star}"
    return emph_text


def _format_error_span(out, tablefmt):
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        return f"<span style='color:red'>{out}</span>"
    else:
        return f"**{out}**"


from functools import partial

cell_with_border_css_class = "cell-with-border"


# https://bitbucket.org/astanin/python-tabulate/issues/57/html-class-options-for-tables
def _html_row_with_attrs(
    celltag: str,
    cell_values: Sequence[str],
    colwidths: Sequence[int],
    colaligns: Sequence[str],
) -> str:
    alignment = {
        "left": "",
        "right": ' style="text-align: right;"',
        "center": ' style="text-align: center;"',
        "decimal": ' style="text-align: right;"',
    }
    values_with_attrs = [
        f"<{celltag}{alignment.get(a, '')} class=\"{cell_with_border_css_class}\">{c}</{celltag}>"
        for c, a in zip(cell_values, colaligns)
    ]
    return "<tr>" + "".join(values_with_attrs).rstrip() + "</tr>"


html_with_borders_tablefmt = TableFormat(
    lineabove=Line(
        f"""\
<style>
th.{cell_with_border_css_class}, td.{cell_with_border_css_class} {{
    border: 1px solid black;
}}
</style>
<table>\
""",
        "",
        "",
        "",
    ),
    linebelowheader=None,
    linebetweenrows=None,
    linebelow=Line("</table>", "", "", ""),
    headerrow=partial(_html_row_with_attrs, "th"),  # type: ignore
    datarow=partial(_html_row_with_attrs, "td"),  # type: ignore
    padding=0,
    with_header_hide=None,
)
"""
Pass this as the parameter `tablefmt` in any method that accepts that parameter to have the table
be an HTML table with borders around each cell.
"""

_ALL_TABLEFMTS = [
    "plain",
    "simple",
    "github",
    "grid",
    "fancy_grid",
    "pipe",
    "orgtbl",
    "jira",
    "presto",
    "pretty",
    "psql",
    "rst",
    "mediawiki",
    "moinmoin",
    "youtrack",
    "html",
    "unsafehtml",
    "latex",
    "latex_raw",
    "latex_booktabs",
    "latex_longtable",
    "textile",
    "tsv",
    html_with_borders_tablefmt,
]

# cast is to shut mypy up; should always be a str if not == html_with_borders_tablefmt
_ALL_TABLEFMTS_NAMES: list[str] = [
    (
        cast(str, fmt)
        if fmt != html_with_borders_tablefmt
        else "html_with_borders_tablefmt"
    )
    for fmt in _ALL_TABLEFMTS
]

_SUPPORTED_TABLEFMTS_TITLE = [
    "github",
    "pipe",
    "simple",
    "grid",
    "html",
    "unsafehtml",
    "rst",
    "latex",
    "orgtbl",
    "latex_raw",
    "latex_booktabs",
    "latex_longtable",
    html_with_borders_tablefmt,
]


_NL: dict[Union[str, TableFormat], str] = {
    "pipe": "\n",
    "html": "<br/>",
    "unsafehtml": "<br/>",
    html_with_borders_tablefmt: "<br/>",
}


def _formatter(
    x: int | float | str | list[str] | Quantity[Decimal] | None,
    italic: bool = False,
    tablefmt: str | TableFormat = "pipe",
    splits: list = [],
) -> str:
    if isinstance(x, (int, str)):
        out = str(x)
    elif x is None:
        out = ""
    elif isinstance(x, float):
        out = f"{x:,.2f}"
        if isnan(x):
            out = _format_error_span(out, tablefmt)
    elif isinstance(x, Quantity):
        out = f"{x:,.2f~#P}"
        if isnan(x.m):
            out = _format_error_span(out, tablefmt)
        if x.m < 0:
            out = _format_error_span(out, tablefmt)
    elif isinstance(x, (list, np.ndarray, pd.Series)):
        out = ", ".join(
            (_NL[tablefmt] if i - 1 in splits else "") + _formatter(y)
            for i, y in enumerate(x)
        )
    else:
        raise TypeError
    if not out:
        return ""
    if italic:
        return emphasize(out, tablefmt=tablefmt, strong=False)
    return out


@attrs.define(eq=True)
class MixLine:
    """Class for handling a line of a (processed) mix recipe.

    Each line should represent a single step, or series of similar steps (same volume per substep)
    in the mixing process.

    Parameters
    ----------

    names
        A list of component names.  For a single step, use [name].

    source_conc
        The source concentration; may not be provided (will be left blank), or be a descriptive string.

    dest_conc
        The destination/target concentration; may not be provided (will be left blank), or be a descriptive string.

    total_tx_vol
        The total volume added to the mix by the step.  If zero, the amount will still be included in tables.
        If None, the amount will be blank.  If provided, and the line is not fake, the value must be correct
        and interpretable for calculations involving the mix.

    number
        The number of components added / subste

    each_tx_vol
        The volume per component / substep.  May be omitted, or a descriptive string.

    plate
        The plate name for the mix, a descriptive string for location / source type (eg, "tube") or None (omitted).
        A single MixLine, at present, should not involve multiple plates.

    wells
        A list of wells for the components in a plate.  If the components are not in a plate, this must be an
        empty list.  This *does not* parse strings; wells must be provided as WellPos instances.

    note
        A note to add for the line

    fake
        Denotes that the line is not a real step, eg, for a summary/total information line.  The line
        will be distinguished in some way in tables (eg, italics) and will not be included in calculations.
    """

    names: list[str] = attrs.field(factory=list)
    source_conc: Quantity[Decimal] | str | None = None
    dest_conc: Quantity[Decimal] | str | None = None
    total_tx_vol: Quantity[Decimal] = NAN_VOL
    number: int = 1
    each_tx_vol: Quantity[Decimal] = NAN_VOL  # | str | None = None
    plate: str = ""
    wells: list[WellPos] = attrs.field(factory=list)
    note: str | None = None
    fake: bool = False

    def __attrs_post_init__(self):
        if (
            isnan(self.each_tx_vol.m)
            and not isnan(self.total_tx_vol.m)
            and self.number == 1
        ):
            self.each_tx_vol = self.total_tx_vol

    @wells.validator
    def _check_wells(self, _: str, v: Any) -> None:
        if (not isinstance(v, list)) or any(not isinstance(x, WellPos) for x in v):
            raise TypeError(f"MixLine.wells of {v} is not a list of WellPos.")

    @names.validator
    def _check_names(self, _: str, v: Any) -> None:
        if (not isinstance(v, list)) or any(not isinstance(x, str) for x in v):
            raise TypeError(f"MixLine.names of {v} is not a list of strings.")

    def location(
        self, tablefmt: str | TableFormat = "pipe", split: bool = True
    ) -> tuple[str | list[str], list[int]]:
        "A formatted string (according to `tablefmt`) for the location of the component/components."
        if len(self.wells) == 0:
            return f"{self.plate}", []
        elif len(self.wells) == 1:
            return f"{self.plate}: {self.wells[0]}", []

        byrow = mixgaps(
            sorted(list(self.wells), key=WellPos.key_byrow),
            by="row",
        )
        bycol = mixgaps(
            sorted(list(self.wells), key=WellPos.key_bycol),
            by="col",
        )

        sortnext = WellPos.next_bycol if bycol <= byrow else WellPos.next_byrow

        splits = []
        wells_formatted = []
        next_well_iter = iter(self.wells)
        prevpos = next(next_well_iter)
        formatted_prevpos = emphasize(f"{self.plate}: {prevpos}", tablefmt, strong=True)
        wells_formatted.append(formatted_prevpos)
        for i, well in enumerate(next_well_iter):
            if (sortnext(prevpos) != well) or (
                (prevpos.col != well.col) and (prevpos.row != well.row)
            ):
                formatted_well = emphasize(f"{well}", tablefmt, strong=True)
                wells_formatted.append(formatted_well)
                if split:
                    splits.append(i)
            else:
                wells_formatted.append(f"{well}")
            prevpos = well

        return wells_formatted, splits

    def toline(
        self, incea: bool, tablefmt: str | TableFormat = "pipe"
    ) -> Sequence[str]:
        locations, splits = self.location(tablefmt=tablefmt)
        if incea:
            return [
                _formatter(
                    self.names, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.source_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.dest_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.number, italic=self.fake, tablefmt=tablefmt)
                if self.number != 1
                else "",
                _formatter(self.each_tx_vol, italic=self.fake, tablefmt=tablefmt)
                if not isnan(self.each_tx_vol.m)
                else "",
                _formatter(self.total_tx_vol, italic=self.fake, tablefmt=tablefmt),
                _formatter(
                    locations, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.note, italic=self.fake),
            ]
        else:
            return [
                _formatter(
                    self.names, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.source_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.dest_conc, italic=self.fake, tablefmt=tablefmt),
                _formatter(self.total_tx_vol, italic=self.fake, tablefmt=tablefmt),
                _formatter(
                    locations, italic=self.fake, tablefmt=tablefmt, splits=splits
                ),
                _formatter(self.note, italic=self.fake, tablefmt=tablefmt),
            ]


def _format_title(
    raw_title: str,
    level: int,
    tablefmt: str | TableFormat,
) -> str:
    # formats a title for a table produced using tabulate,
    # in the formats tabulate understands
    if tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        title = f"<h{level}>{raw_title}</h{level}>"
    elif tablefmt == "rst":
        # https://draft-edx-style-guide.readthedocs.io/en/latest/ExampleRSTFile.html#heading-levels
        # #############
        # Heading 1
        # #############
        #
        # *************
        # Heading 2
        # *************
        #
        # ===========
        # Heading 3
        # ===========
        #
        # Heading 4
        # ************
        #
        # Heading 5
        # ===========
        #
        # Heading 6
        # ~~~~~~~~~~~
        raw_title_width = len(raw_title)
        if level == 1:
            line = "#" * raw_title_width
        elif level in [2, 4]:
            line = "*" * raw_title_width
        elif level in [3, 5]:
            line = "=" * raw_title_width
        else:
            line = "~" * raw_title_width

        if level in [1, 2, 3]:
            title = f"{line}\n{raw_title}\n{line}"
        else:
            title = f"{raw_title}\n{line}"
    elif tablefmt in ["latex", "latex_raw", "latex_booktabs", "latex_longtable"]:
        if level == 1:
            size = r"\Huge"
        elif level == 2:
            size = r"\huge"
        elif level == 3:
            size = r"\LARGE"
        elif level == 4:
            size = r"\Large"
        elif level == 5:
            size = r"\large"
        elif level == 6:
            size = r"\normalsize"
        else:
            assert False
        newline = r"\\"
        noindent = r"\noindent"
        title = f"{noindent} {{ {size} {raw_title} }} {newline}"
    elif tablefmt == "orgtbl":  # use the title for tablefmt == "pipe"
        hashes = "*" * level
        title = f"{hashes} {raw_title}"
    else:  # use the title for tablefmt == "pipe"
        hashes = "#" * level
        title = f"{hashes} {raw_title}"
    return title


def _format_errors(errors: list[VolumeError], tablefmt: TableFormat | str) -> str:
    if tablefmt in ["latex"]:
        raise NotImplementedError
    elif tablefmt in ["html", "unsafehtml", html_with_borders_tablefmt]:
        if not errors:
            return ""
        s = "<div style='color: red'><ul>\n"
        s += "\n".join(f"<li>{e.args[0]}</li>" for e in errors)
        s += "</ul></div>\n"
        return s
    else:
        return "".join(f"- {e.args[0]}\n" for e in errors)


def _format_location(loc: tuple[str | None, WellPos | None]) -> str:
    p, w = loc
    if isinstance(p, str) and isinstance(w, WellPos):
        return f"{p}: {w}"
    elif isinstance(p, str) and (w is None):
        return p
    elif (p is None) and (w is None):
        return ""
    raise ValueError
