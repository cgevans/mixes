"""
A module for handling "quantitation": measuring concentration of strands,
and diluting and hydrating to reach a desired concentration.

The main "easy" functions to use are :func:`hydrate_from_specs` and
:func:`hydrate_and_measure_conc_and_dilute_from_specs`.

>>> from alhambra_mixes.quantitate import hydrate_from_specs, hydrate_and_measure_conc_and_dilute_from_specs
>>> specs_file = 'path/to/coa.csv'
>>> target_conc_high = '200 uM'
>>> target_conc_low = '100 uM'
>>> hydrate_from_specs(
...     filename=specs_file,
...     target_conc=target_conc_high,
...     strands=['5RF', '3RQ'],
... )
nmoles = 8.9 nmol
nmoles = 15.7 nmol
{'5RF': <Quantity(44.5, 'microliter')>,
 '3RQ': <Quantity(78.5, 'microliter')>}
>>> # now go to the lab and add the above quantities of water/buffer to the dry samples,
>>> # then measure absorbances, e.g., with a NanoDrop, to populate the dict `absorbances` below
>>> absorbances = {
...     '5RF': [48.46, 48.28],
...     '3RQ': [34.36, 34.82],
... }
>>> hydrate_and_measure_conc_and_dilute_from_specs(
...     filename=specs_file,
...     target_conc_high=target_conc_high,
...     target_conc_low=target_conc_low,
...     absorbances=absorbances,
... )
{'5RF': (<Quantity(213.931889, 'micromolar')>, <Quantity(48.4210528, 'microliter')>),
 '3RQ': (<Quantity(190.427429, 'micromolar')>, <Quantity(69.176983, 'microliter')>)}

For convenience in Jupyter notebooks, there are also versions of these functions beginning with ``display_``:
:func:`display_hydrate_from_specs` and :func:`display_hydrate_and_measure_conc_and_dilute_from_specs`.
Instead of returning a dictionary, these methods display the result in the Jupyter notebook,
as nicely-formatted Markdown.
"""

from __future__ import annotations

import decimal
import warnings
from decimal import Decimal as D
from typing import Any, Iterable, Sequence, Type, Union, cast

import pandas
import pint
from pint import Quantity

from .units import DNAN, Q_, _parse_vol_optional, nM, uL, uM, ureg


def parse_vol(vol: Union[float, int, str, Quantity[D]]) -> Quantity[D]:
    if isinstance(vol, (float, int)):
        vol = Quantity(D(vol), "µL")
    return _parse_vol_optional(vol)


__all__ = (
    "measure_conc_and_dilute",
    "hydrate_and_measure_conc_and_dilute",
    "hydrate_from_specs",
    "hydrate_and_measure_conc_and_dilute_from_specs",
)

# This needs to be here to make Decimal NaNs behave the way that NaNs
# *everywhere else in the standard library* behave.
decimal.setcontext(decimal.ExtendedContext)


warnings.filterwarnings(
    "ignore",
    "The unit of the quantity is " "stripped when downcasting to ndarray",
    pint.UnitStrippedWarning,
)

warnings.filterwarnings(
    "ignore",
    "pint-pandas does not support magnitudes of class <class 'int'>",
    RuntimeWarning,
)


def normalize(quantity: Quantity[D]) -> Quantity[D]:
    """
    Normalize `quantity` so that it is "compact" (uses units within the correct "3 orders of magnitude":
    https://pint.readthedocs.io/en/0.18/tutorial.html#simplifying-units)
    and eliminate trailing zeros.

    :param quantity:
        a pint Quantity[Decimal]
    :return:
        `quantity` normalized to be compact and without trailing zeros.
    """
    quantity = quantity.to_compact()
    mag_int = quantity.magnitude.to_integral()
    if mag_int == quantity.magnitude:
        # can be represented exactly as integer, so return that;
        # quantity.magnitude.normalize() would use scientific notation in this case, which we don't want
        quantity = Q_(mag_int, quantity.units)
    else:
        # is not exact integer, so normalize will return normal float literal such as 10.2
        # and not scientific notation like it would for an integer
        mag_norm = quantity.magnitude.normalize()
        quantity = Q_(mag_norm, quantity.units)
    return quantity


def parse_conc(conc: float | int | str | Quantity[D]) -> Quantity[D]:
    """
    Default units for conc being a float/int is µM (micromolar).
    """
    if isinstance(conc, (float, int)):
        conc = f"{conc} µM"

    if isinstance(conc, str):
        q = ureg(conc)
        if not q.check(uM):
            raise ValueError(
                f"{conc} is not a valid quantity here (should be concentration)."
            )
        return q
    elif isinstance(conc, Quantity):
        if not conc.check(uM):
            raise ValueError(
                f"{conc} is not a valid quantity here (should be concentration)."
            )
        conc = Q_(D(conc.m), conc.u)
        return normalize(conc)
    elif conc is None:
        return Q_(DNAN, uL)
    raise ValueError


def parse_nmol(nmoles: float | int | str | Quantity[D]) -> Quantity[D]:
    """
    Default units for molar amount being a float/int is nmol (nanomoles).
    """
    if isinstance(nmoles, (float, int)):
        nmoles = f"{nmoles} nmol"

    if isinstance(nmoles, str):
        q = ureg(nmoles)
        if not q.check(ureg.nmol):
            raise ValueError(f"{nmoles} is not a valid quantity here (should be nmol).")
        return q
    elif isinstance(nmoles, Quantity):
        if not nmoles.check(ureg.nmol):
            raise ValueError(f"{nmoles} is not a valid quantity here (should be nmol).")
        nmoles = Q_(D(nmoles.m), nmoles.u)
        return normalize(nmoles)
    elif nmoles is None:
        return Q_(DNAN, ureg.nmol)
    raise ValueError


# initial hydration of dry DNA
def hydrate(
    target_conc: float | int | str | Quantity[D], nmol: float | int | str | Quantity[D]
) -> Quantity[D]:
    """
    Indicates how much buffer/water volume to add to a dry DNA sample to reach a particular concentration.

    :param target_conc:
        target concentration. If float/int, units are µM (micromolar).
    :param nmol:
        number of nmol (nanomoles) of dry product.
    :return:
        number of µL (microliters) of water/buffer to pipette to reach `target_conc` concentration
    """
    target_conc = parse_conc(target_conc)
    nmol = parse_nmol(nmol)
    vol = nmol / target_conc
    vol = vol.to("uL")
    vol = normalize(vol)
    return vol


def dilute(
    target_conc: float | int | str | Quantity[D],
    start_conc: float | int | str | Quantity[D],
    vol: float | int | str | Quantity[D],
) -> Quantity[D]:
    """
    Indicates how much buffer/water volume to add to a wet DNA sample to reach a particular concentration.

    :param target_conc:
        target concentration. If float/int, units are µM (micromolar).
    :param start_conc:
        current concentration of sample. If float/int, units are µM (micromolar).
    :param vol:
        current volume of sample. If float/int, units are µL (microliters)
    :return:
        number of µL (microliters) of water/buffer to add to dilate to concentration `target_conc`
    """
    target_conc = parse_conc(target_conc)
    start_conc = parse_conc(start_conc)
    if start_conc < target_conc:
        raise ValueError(
            f"start_conc = {start_conc} is below target_conc = {target_conc}; must be above"
        )
    vol = parse_vol(vol)
    added_vol = (vol * start_conc / target_conc) - vol
    added_vol = normalize(added_vol)
    return added_vol


def _has_length(lst: Any) -> bool:
    # indicates if lst has __len__ method, i.e., we can call len(lst) on it
    try:
        _ = len(lst)
        return True
    except TypeError:
        return False


def measure_conc(
    absorbance: float | int | Sequence[float | int], ext_coef: float | int
) -> Quantity[D]:
    """
    Calculates concentration of DNA sample given an absorbance reading on a NanoDrop machine.

    :param absorbance:
        UV absorbance at 260 nm. Can either be a single float/int or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    :param ext_coef:
        Extinction coefficient in L/mol*cm.
    :return:
        concentration of DNA sample
    """
    if isinstance(absorbance, (float, int)):
        ave_absorbance = absorbance
    elif _has_length(absorbance):
        if len(absorbance) == 0:
            raise ValueError(f"absorbance cannot be an empty sequence")
        if not isinstance(absorbance[0], (int, float)):
            raise TypeError(
                f"absorbance sequence must contain ints or floats, "
                f"but the first element is {absorbance[0]}, "
                f"of type {type(absorbance[0])}"
            )
        ave_absorbance = sum(absorbance) / len(absorbance)
    else:
        raise TypeError(
            f"absorbance must either be float/int or iterable of floats/ints, but it is not:\n"
            f"type(absorbance) = {type(absorbance)}\n"
            f"absorbance = {absorbance}"
        )

    conc_float = (ave_absorbance / ext_coef) * 10**6
    conc = parse_conc(f"{conc_float} uM")
    conc = normalize(conc)
    return conc


def measure_conc_and_dilute(
    absorbance: float | int | Sequence[float | int],
    ext_coef: float | int,
    target_conc: float | int | str | Quantity[D],
    vol: float | int | str | Quantity[D],
    vol_removed: None | float | int | str | Quantity[D] = None,
) -> tuple[Quantity[D], Quantity[D]]:
    """
    Calculates concentration of DNA sample given an absorbance reading on a NanoDrop machine,
    then calculates the amount of buffer/water that must be added to dilute it to a target concentration.

    :param absorbance:
        UV absorbance at 260 nm. Can either be a single float/int or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    :param ext_coef:
        Extinction coefficient in L/mol*cm.
    :param target_conc:
        target concentration. If float/int, units are µM (micromolar).
    :param vol:
        current volume of sample. If float/int, units are µL (microliters)
        NOTE: This is the volume *before* samples are taken to measure absorbance.
        It is assumed that each sample taken to measure absorbance is 1 µL.
        If that is not the case, then set the parameter `vol_removed` to the total volume removed.
    :param vol_removed:
        Total volume removed from `vol` to measure absorbance.
        For example, if two samples were taken, one at 1 µL and one at 1.5 µL, then set
        `vol_removed` = 2.5 µL.
        If not specified, it is assumed that each sample is 1 µL, and that the total number of samples
        taken is the number of entries in `absorbance`.
        If `absorbance` is a single volume (e.g., ``float``, ``int``, ``str``, ``Quantity[Decimal]``),
        then it is assumed the number of samples is 1 (i.e., `vol_removed` = 1 µL),
        otherwise if `absorbance` is a list, then the length of the list is assumed to be the
        number of samples taken, each at 1 µL.
    :return:
        The pair (current concentration of DNA sample, volume to add to reach `target_conc`)
    """
    if vol_removed is None:
        if isinstance(absorbance, (tuple, list)):
            vol_removed = parse_vol(f"{len(absorbance)} uL")
        else:
            vol_removed = parse_vol("1 uL")
    else:
        vol_removed = parse_vol(vol_removed)

    start_conc = measure_conc(absorbance, ext_coef)
    target_conc = parse_conc(target_conc)
    vol = parse_vol(vol)

    vol_remaining = vol - vol_removed
    vol_to_add = dilute(target_conc, start_conc, vol_remaining)
    return start_conc, vol_to_add


def get_vols_of_strands_from_dataframe(dataframe: pandas.DataFrame) -> dict[str, str]:
    # takes care of some ugly special cases and variants IDT uses
    name_key = "Sequence Name"
    vol_key = find_volume_key(dataframe)
    vols = key_to_prop_from_dataframe(dataframe, name_key, vol_key)
    if "µL" in vol_key:
        # set units if units were listed in header rather than individual entries
        new_vols = {}
        for name, vol in vols.items():
            if isinstance(vol, (int, float)) or isinstance(vol, str) and "L" not in vol:
                new_vol = f"{vol} µL"
            else:
                new_vol = vol
            new_vols[name] = new_vol
        vols = new_vols
    return vols


def measure_conc_and_dilute_from_specs(
    filename: str,
    target_conc: float | int | str | Quantity[D],
    absorbances: dict[str, float | int | Sequence[float | int]],
    vols_removed: dict[str, None | float | int | str | Quantity[D]] | None = None,
    enforce_utf8: bool = True,
) -> dict[str, tuple[Quantity[D], Quantity[D]]]:
    """
    Measures concentrations of DNA samples given an IDT spec file to look up existing volumes and
    extinction coefficients, and given absorbances measured by a Nanodrop machine. Returns concentrations
    as well as additional volume to be added to diluate each strand to a particular target concentration.

    Example usage:

    >>> from alhambra_mixes.quantitate import measure_conc_and_dilute_from_specs
    >>> specs_file = 'coa-stub.csv'
    >>> target_conc = '150 uM'
    >>> absorbances = {'mon0': [38.88, 39.3], 'adp0': [77.96, 78.72]}
    >>> measure_conc_and_dilute_from_specs(specs_file, target_conc, absorbances)
    {'mon0': (<Quantity(186.765409, 'micromolar')>, <Quantity(28.186813, 'microliter')>),
     'adp0': (<Quantity(190.933463, 'micromolar')>, <Quantity(30.563653, 'microliter')>)}

    :param filename:
        IDT specs file (e.g., coa.csv)
    :param target_conc:
        target concentration to dilute to from measured concentration
    :param absorbances:
        measured absorbance of each strand. Should be a dict mapping each strand name (as it appears in
        the "Sequence name" column of `filename`) to an absorbance or nonempty list of absorbances, meaning
        UV absorbance at 260 nm. If a list then an average is taken.
    :param vols_removed:
        dict mapping each strand name to the volume that was removed to take absorbance measurements.
        For any strand name not appearing as a key in the dict, it is assumed that 1 microliter was taken
        for each absorbance measurement made.
    :param enforce_utf8:
        If `filename` is a text CSV file and this paramter is True, it enforces that `filename` is valid
        UTF-8, raising an exception if not. This helps to avoid accidentally dropping Unicode characters
        such as µ, which would silently convert a volume from µL to L.
        If do not want to convert the specs file to UTF-8 and you are certain that no important Unicode
        characters would be dropped, then you can set this parameter to false.
    :return:
        dict mapping each strand name to a pair `(conc, vol)`, where `conc` is its measured concentration
        and `vol` is the volume that should be subsequently added to reach concentration `target_conc`
    """
    if vols_removed is None:
        vols_removed = {}

    dataframe = _read_dataframe_from_excel_or_csv(filename, enforce_utf8)
    vols_of_strands = get_vols_of_strands_from_dataframe(dataframe)

    name_key = "Sequence Name"
    ext_coef_key = find_extinction_coefficient_key(dataframe)
    ext_coef_of_strand = key_to_prop_from_dataframe(dataframe, name_key, ext_coef_key)

    concs_and_vols_to_add = {}
    for name, absorbance in absorbances.items():
        vol_removed = vols_removed.get(name)  # None if name not a key in vol_removed
        ext_coef_str = ext_coef_of_strand[name]
        ext_coef = float(ext_coef_str)
        vol = vols_of_strands[name]
        conc_and_vol_to_add = measure_conc_and_dilute(
            absorbance=absorbance,
            ext_coef=ext_coef,
            target_conc=target_conc,
            vol=vol,
            vol_removed=vol_removed,
        )
        concs_and_vols_to_add[name] = conc_and_vol_to_add

    return concs_and_vols_to_add


def display_measure_conc_and_dilute_from_specs(
    filename: str,
    target_conc: float | int | str | Quantity[D],
    absorbances: dict[str, float | int | Sequence[float | int]],
    vols_removed: dict[str, None | float | int | str | Quantity[D]] | None = None,
    enforce_utf8: bool = True,
) -> None:
    """
    Like :meth:`measure_conc_and_dilute_from_specs`, but displays the value in a Jupyter
    notebook instead of returning it.

    :param filename:
        IDT specs file (e.g., coa.csv)
    :param target_conc:
        target concentration to dilute to from measured concentration
    :param absorbances:
        measured absorbance of each strand. Should be a dict mapping each strand name (as it appears in
        the "Sequence name" column of `filename`) to an absorbance or nonempty list of absorbances, meaning
        UV absorbance at 260 nm. If a list then an average is taken.
    :param vols_removed:
        dict mapping each strand name to the volume that was removed to take absorbance measurements.
        For any strand name not appearing as a key in the dict, it is assumed that 1 microliter was taken
        for each absorbance measurement made.
    :param enforce_utf8:
        If `filename` is a text CSV file and this paramter is True, it enforces that `filename` is valid
        UTF-8, raising an exception if not. This helps to avoid accidentally dropping Unicode characters
        such as µ, which would silently convert a volume from µL to L.
        If do not want to convert the specs file to UTF-8 and you are certain that no important Unicode
        characters would be dropped, then you can set this parameter to false.
    """
    from tabulate import tabulate
    from IPython.display import display, Markdown

    names_to_concs_and_vols_to_add = measure_conc_and_dilute_from_specs(
        filename=filename,
        target_conc=target_conc,
        absorbances=absorbances,
        vols_removed=vols_removed,
        enforce_utf8=enforce_utf8,
    )

    headers = ["name", "measured conc", "volume to add"]
    table_list = [
        (name, round(conc, 2), round(vol_to_add, 2))  # type: ignore
        for name, (conc, vol_to_add) in names_to_concs_and_vols_to_add.items()
    ]
    table = tabulate(table_list, headers=headers, tablefmt="pipe", floatfmt=".2f")
    from alhambra_mixes.mixes import _format_title

    raw_title = "Initial measured concentrations and subsequent dilution volumes"
    title = _format_title(raw_title, level=2, tablefmt="pipe")
    display(Markdown(title + "\n\n" + table))


def hydrate_and_measure_conc_and_dilute(
    nmol: float | int | str | Quantity[D],
    target_conc_high: float | int | str | Quantity[D],
    target_conc_low: float | int | str | Quantity[D],
    absorbance: float | int | Sequence[float | int],
    ext_coef: float | int,
    vol_removed: None | float | int | str | Quantity[D] = None,
) -> tuple[Quantity[D], Quantity[D]]:
    """
    Assuming :func:`hydrate` is called with parameters `nmol` and `target_conc_high` to give initial
    volumes to add to a dry sample to reach a "high" concentration `target_conc_high`,
    and assuming absorbances are then measured,
    calculates subsequent dilution volumes to reach "low" concentration `target_conc_low`,
    and also actual "start" concentration (i.e., actual concentration after adding initial hydration
    that targeted `target_conc_high`, according to `absorbance`).

    This is on the assumption that the first hydration step could result in a concentration below
    `target_conc_high`, so `target_conc_high` should be chosen sufficiently larger than
    `target_conc_low` so that the actual measured concentration after the first step is
    likely to be above `target_conc_low`, so that it is possible to reach concentration
    `target_conc_low` with a subsequent dilution step. (As opposed to requiring a vacufuge to
    concentrate the sample higher).

    :param nmol:
        number of nmol (nanomoles) of dry product.
    :param target_conc_high:
        target concentration for initial hydration. Should be higher than `target_conc_low`,
    :param target_conc_low:
        the "real" target concentration that we will try to hit after the second
        addition of water/buffer.
    :param absorbance:
        UV absorbance at 260 nm. Can either be a single float/int or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    :param ext_coef:
        Extinction coefficient in L/mol*cm.
    :param vol_removed:
        Total volume removed from `vol` to measure absorbance.
        For example, if two samples were taken, one at 1 µL and one at 1.5 µL, then set
        `vol_removed` = 2.5 µL.
        If not specified, it is assumed that each sample is 1 µL, and that the total number of samples
        taken is the number of entries in `absorbance`.
        If `absorbance` is a single volume (e.g., ``float``, ``int``, ``str``, ``Quantity[Decimal]``),
        then it is assumed the number of samples is 1 (i.e., `vol_removed` = 1 µL),
        otherwise if `absorbance` is a list, then the length of the list is assumed to be the
        number of samples taken, each at 1 µL.
    :return:
        The pair (current concentration of DNA sample, volume to add to reach `target_conc`)
    """
    target_conc_high = parse_conc(target_conc_high)
    target_conc_low = parse_conc(target_conc_low)
    assert target_conc_high > target_conc_low
    vol = hydrate(target_conc=target_conc_high, nmol=nmol)
    actual_start_conc, vol_to_add = measure_conc_and_dilute(
        absorbance=absorbance,
        ext_coef=ext_coef,
        target_conc=target_conc_low,
        vol=vol,
        vol_removed=vol_removed,
    )
    return actual_start_conc, vol_to_add


def key_prefix_in_dataframe(dataframe: pandas.DataFrame, keys: Sequence[str]) -> str:
    # If any key in keys is either in the dataframe, or if it is a prefix of a
    for key in keys:
        if key in dataframe.keys():
            return key

    for key in keys:
        for existing_key in dataframe.keys():
            if existing_key.startswith(key):
                return existing_key

    raise KeyError(
        f"key in {keys} not found in dataframe, "
        f"nor is it a prefix of any key in the dataframe.\n"
        f"dataframe =\n{dataframe}"
    )


def key_to_prop_from_dataframe(
    dataframe: pandas.DataFrame, key: str, prop: str
) -> dict[str, str]:
    key_series = dataframe[key]
    prop_series = dataframe[prop]
    return dict(zip(key_series, prop_series))


def hydrate_and_measure_conc_and_dilute_from_specs(
    filename: str,
    target_conc_high: float | int | str | Quantity[D],
    target_conc_low: float | int | str | Quantity[D],
    absorbances: dict[str, float | int | Sequence[float | int]],
    vols_removed: dict[str, None | float | int | str | Quantity[D]] | None = None,
    enforce_utf8: bool = True,
) -> dict[str, tuple[Quantity[D], Quantity[D]]]:
    """
    Like :func:`hydrate_and_measure_conc_and_dilute`, but works with multiple strands,
    using an IDT spec file to look up nmoles and extinction coefficients.

    The intended usage of this method is to be used in conjunction with the function
    :func:`hydrate_from_specs` as follows.

    >>> from alhambra_mixes.quantitate import hydrate_from_specs, hydrate_and_measure_conc_and_dilute_from_specs
    >>> specs_file = 'path/to/coa.csv'
    >>> target_conc_high = '200 uM'
    >>> target_conc_low = '100 uM'
    >>> hydrate_from_specs(
    ...     filename=specs_file,
    ...     target_conc=target_conc_high,
    ...     strands=['5RF', '3RQ'],
    ... )
    nmoles = 8.9 nmol
    nmoles = 15.7 nmol
    {'5RF': <Quantity(44.5, 'microliter')>,
     '3RQ': <Quantity(78.5, 'microliter')>}
    >>> # now go to the lab and add the above quantities of water/buffer to the dry samples,
    >>> # then measure absorbances, e.g., with a NanoDrop, to populate the dict `absorbances` below
    >>> absorbances = {
    ...     '5RF': [48.46, 48.28],
    ...     '3RQ': [34.36, 34.82],
    ... }
    >>> hydrate_and_measure_conc_and_dilute_from_specs(
    ...     filename=specs_file,
    ...     target_conc_high=target_conc_high,
    ...     target_conc_low=target_conc_low,
    ...     absorbances=absorbances,
    ... )
    {'5RF': (<Quantity(213.931889, 'micromolar')>, <Quantity(48.4210528, 'microliter')>),
     '3RQ': (<Quantity(190.427429, 'micromolar')>, <Quantity(69.176983, 'microliter')>)}

    Note in particular that we do not need to specify the volume prior to the dilution step,
    since it is calculated based on the volume necessary for the first hydration step to
    reach concentration `target_conc_high`.

    For convenience in Jupyter notebooks, there are also versions of these functions beginning with
    ``display_``:
    :func:`display_hydrate_from_specs` and :func:`display_hydrate_and_measure_conc_and_dilute_from_specs`.
    Instead of returning a dictionary, these methods display the result in the Jupyter notebook,
    as nicely-formatted Markdown.

    :param filename:
        path to IDT Excel/CSV spreadsheet with specs of strands (e.g., coa.csv)
    :param target_conc_high:
        target concentration for initial hydration. Should be higher than `target_conc_low`,
    :param target_conc_low:
        the "real" target concentration that we will try to hit after the second
        addition of water/buffer.
    :param absorbances:
        UV absorbances at 260 nm. Is a dict mapping each strand name to an "absorbance" as defined
        in the `absobance` parameter of :func:`hydrate_and_measure_conc_and_dilute`.
        In other words the value to which each strand name maps
        can either be a single float/int, or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    :param vols_removed:
        Total volumes removed from `vol` to measure absorbance;
        is a dict mapping strand names (should be subset of strand names that are keys in `absorbances`).
        Can be None, or can have strictly fewer strand names than in `absorbances`;
        defaults are assumed as explained next for any missing strand name key.
        For example, if two samples were taken, one at 1 µL and one at 1.5 µL, then set
        `vol_removed` = 2.5 µL.
        If not specified, it is assumed that each sample is 1 µL, and that the total number of samples
        taken is the number of entries in `absorbance`.
        If `absorbance` is a single volume (e.g., ``float``, ``int``, ``str``, ``Quantity[Decimal]``),
        then it is assumed the number of samples is 1 (i.e., `vol_removed` = 1 µL),
        otherwise if `absorbance` is a list, then the length of the list is assumed to be the
        number of samples taken, each at 1 µL.
    :param enforce_utf8:
        If `filename` is a text CSV file and this paramter is True, it enforces that `filename` is valid
        UTF-8, raising an exception if not. This helps to avoid accidentally dropping Unicode characters
        such as µ, which would silently convert a volume from µL to L.
        If do not want to convert the specs file to UTF-8 and you are certain that no important Unicode
        characters would be dropped, then you can set this parameter to false.
    :return:
        dict mapping each strand name in keys of `absorbances` to a pair (`conc`, `vol_to_add`),
        where `conc` is the measured concentration according to the absorbance value(s) of that strandm
        and `vol_to_add` is the volume needed to add to reach concentration `target_conc_low`.
    """
    if vols_removed is None:
        vols_removed = {}

    strands = list(absorbances.keys())
    vols_of_strands = hydrate_from_specs(
        filename=filename,
        target_conc=target_conc_high,
        strands=strands,
        enforce_utf8=enforce_utf8,
    )

    name_key = "Sequence Name"
    nmol_key = "nmoles"
    dataframe = _read_dataframe_from_excel_or_csv(filename, enforce_utf8)
    nmols_of_strands = key_to_prop_from_dataframe(dataframe, name_key, nmol_key)

    ext_coef_key = find_extinction_coefficient_key(dataframe)
    ext_coef_of_strand = key_to_prop_from_dataframe(dataframe, name_key, ext_coef_key)

    concs_and_vols_to_add = {}
    for name, vol in vols_of_strands.items():
        vol_removed = vols_removed.get(name)  # None if name not a key in vol_removed
        nmol = nmols_of_strands[name]
        ext_coef_str = ext_coef_of_strand[name]
        ext_coef = float(ext_coef_str)
        absorbance = absorbances[name]
        conc_and_vol_to_add = hydrate_and_measure_conc_and_dilute(
            nmol=nmol,
            target_conc_high=target_conc_high,
            target_conc_low=target_conc_low,
            absorbance=absorbance,
            ext_coef=ext_coef,
            vol_removed=vol_removed,
        )
        concs_and_vols_to_add[name] = conc_and_vol_to_add

    return concs_and_vols_to_add


# from https://stackoverflow.com/a/3114640/5339430
def iterable_is_empty(iterable: Iterable) -> bool:
    return not any(True for _ in iterable)


def hydrate_from_specs(
    filename: str,
    target_conc: float | int | str | Quantity[D],
    strands: Sequence[str] | Sequence[int] | None = None,
    enforce_utf8: bool = True,
) -> dict[str, Quantity[D]]:
    """
    Indicates how much volume to add to a dry DNA sample to reach a particular concentration,
    given data in an Excel file in the IDT format.

    :param filename:
        path to IDT Excel/CSV spreadsheet with specs of strands (e.g., coa.csv)
    :param target_conc:
        target concentration. If float/int, units are µM (micromolar).
    :param strands:
        strands to hydrate. Can be list of strand names (strings), or list of of ints indicating
        which rows in the Excel spreadsheet to hydrate
    :param enforce_utf8:
        If `filename` is a text CSV file and this paramter is True, it enforces that `filename` is valid
        UTF-8, raising an exception if not. This helps to avoid accidentally dropping Unicode characters
        such as µ, which would silently convert a volume from µL to L.
        If do not want to convert the specs file to UTF-8 and you are certain that no important Unicode
        characters would be dropped, then you can set this parameter to false.
    :return:
        dict mapping each strand name to an amount of µL (microliters) of water/buffer
        to pipette to reach `target_conc` concentration for that strand
    """
    if strands is not None and iterable_is_empty(strands):
        raise ValueError("strands cannot be empty")

    name_key = "Sequence Name"
    nmol_key = "nmoles"

    dataframe = _read_dataframe_from_excel_or_csv(filename, enforce_utf8)
    num_rows, num_cols = dataframe.shape

    names_series = dataframe[name_key]
    nmol_series = dataframe[nmol_key]

    if strands is None:
        # if strands not specified, iterate over all of them in sheet
        rows = list(range(num_rows))
    elif strands is not None and isinstance(
        strands[0], str
    ):  # TODO: generalize to iterable
        # if strand names specified, figure out which rows these are
        names = set(cast(Sequence[str], strands))
        name_to_row = {}
        for row in range(num_rows):
            name = names_series[row]
            if name in names:
                names.remove(name)
                name_to_row[name] = row
        if len(names) > 0:
            raise ValueError(
                f"The following strand names were not found in the spreadsheet:\n"
                f'{", ".join(names)}'
            )
        # sort rows by order of strand name in strands
        rows = [name_to_row[name] for name in strands]

    elif strands is not None and isinstance(strands[0], int):
        # is list of indices; subtract 1 to convert them to 0-based indices
        rows = [row - 2 for row in cast(Sequence[int], strands)]
    else:
        raise ValueError(
            "strands must be None, or list of strings, or list of ints\n"
            f"instead its first element is type {type(strands[0])}: {strands[0]}"
        )

    nmols = [nmol_series[row] for row in rows]

    names_list = [names_series[row] for row in rows]

    for nmol, name in zip(nmols, names_list):
        if isinstance(nmol, str) and "RNase-Free Water" in nmol:
            raise ValueError(
                f"cannot hydrate strand {name}: according to IDT, it is already hydrated.\n"
                f'Here is its "nmoles" entry in the file {filename}: "{nmol}"'
            )

    vols = [hydrate(target_conc, nmol) for nmol in nmols]
    return dict(zip(names_list, vols))


def _is_utf8(filename: str) -> bool:
    """Tests if `content` is UTF-8."""
    import codecs

    try:
        f = codecs.open(filename, encoding="utf-8", errors="strict")
        for _ in f:
            pass
        return True
    except UnicodeDecodeError:
        return False


def _read_dataframe_from_excel_or_csv(
    filename: str, enforce_utf8: bool
) -> pandas.DataFrame:
    if filename.lower().endswith(".xls") or filename.lower().endswith(".xlsx"):
        dataframe = pandas.read_excel(filename, 0)
    elif filename.lower().endswith(".csv"):
        if enforce_utf8 and not _is_utf8(filename):
            raise ValueError(
                f"""{filename}
is not a valid UTF-8 file. To avoid accidentally skipping Unicode 
characters such as µ (which would silently convert µL to L, for instance), 
first convert the file to UTF-8 format. Alternately, if you are certain that 
the file contains no important Unicode characters, set the parameter 
enforce_utf8 to False to avoid getting this error."""
            )
        # encoding_errors='ignore' prevents problems with, e.g., µ Unicode symbol
        dataframe = pandas.read_csv(filename, encoding_errors="ignore")
    else:
        raise ValueError(
            f"unrecognized file extension in filename {filename}; "
            f"must be .xls, .xlsx, or .csv"
        )
    # removing rows from a CSV with Excel can actually leave them there with all values as NaN,
    # so let's remove those rows in case it was edited in that way
    dataframe.dropna(how="all", inplace=True)
    return dataframe


def find_extinction_coefficient_key(dataframe: pandas.DataFrame) -> str:
    key = "Extinction Coefficient"
    # in some spec files, "Extinction Coefficient" is followed by units " L/(mole·cm)"
    for column_name in dataframe.columns:
        if key in column_name:
            key = column_name
            break
    return key


def find_volume_key(dataframe: pandas.DataFrame) -> str:
    key = "Volume"
    # can be "Volume" or "Final Volume µL"
    for column_name in dataframe.columns:
        if key in column_name:
            key = column_name
            return key
    raise KeyError(f"no key found with f{key} as a substring; dataframe =\n{dataframe}")


def measure_conc_from_specs(
    filename: str,
    absorbances: dict[str, float | int | Sequence[float] | Sequence[int]],
    enforce_utf8: bool = True,
) -> dict[str, Quantity[D]]:
    """
    Indicates the concentrations of DNA samples, given data in an Excel file in the IDT format and
    measured absorbances from a Nanodrop machine.

    :param filename:
        path to IDT Excel/CSV spreadsheet with specs of strands (e.g., coa.csv)
    :param absorbances:
        dict mapping each strand name to its absorbance value.
        Each absorbance value represents UV absorbance at 260 nm.
        Each can either be a single float/int or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    :param enforce_utf8:
        If `filename` is a text CSV file and this paramter is True, it enforces that `filename` is valid
        UTF-8, raising an exception if not. This helps to avoid accidentally dropping Unicode characters
        such as µ, which would silently convert a volume from µL to L.
        If do not want to convert the specs file to UTF-8 and you are certain that no important Unicode
        characters would be dropped, then you can set this parameter to false.
    :return:
        dict mapping each strand name to a concentration for that strand
    """
    name_key = "Sequence Name"

    dataframe = _read_dataframe_from_excel_or_csv(filename, enforce_utf8)

    ext_coef_key = find_extinction_coefficient_key(dataframe)

    names_series = dataframe[name_key]
    ext_coef_series = dataframe[ext_coef_key]

    # create dict mapping each strand name to its row in the pandas dataframe
    row_of_name = {}
    for row, name in enumerate(names_series):
        if name in absorbances:
            row_of_name[name] = row

    name_to_concs = {}
    for name, absorbance in absorbances.items():
        row = row_of_name[name]
        ext_coef = ext_coef_series[row]
        conc = measure_conc(absorbance, ext_coef)
        name_to_concs[name] = conc

    return name_to_concs


def display_hydrate_and_measure_conc_and_dilute_from_specs(
    filename: str,
    target_conc_high: float | int | str | Quantity[D],
    target_conc_low: float | int | str | Quantity[D],
    absorbances: dict[str, float | int | Sequence[float | int]],
    vols_removed: dict[str, None | float | int | str | Quantity[D]] | None = None,
) -> None:
    """
    Like :meth:`hydrate_and_measure_conc_and_dilute_from_specs`, but displays the value in a Jupyter
    notebook instead of returning it.
    """
    from IPython.display import Markdown, display
    from tabulate import tabulate

    names_to_concs_and_vols_to_add = hydrate_and_measure_conc_and_dilute_from_specs(
        filename=filename,
        target_conc_high=target_conc_high,
        target_conc_low=target_conc_low,
        absorbances=absorbances,
        vols_removed=vols_removed,
    )

    headers = ["name", "measured conc", "volume to add"]
    table_list = [
        (name, round(conc, 2), round(vol_to_add, 2))  # type: ignore
        for name, (conc, vol_to_add) in names_to_concs_and_vols_to_add.items()
    ]
    table = tabulate(table_list, headers=headers, tablefmt="pipe", floatfmt=".2f")
    from alhambra_mixes.mixes import _format_title

    raw_title = "Initial measured concentrations and subsequent dilution volumes"
    title = _format_title(raw_title, level=2, tablefmt="pipe")
    display(Markdown(title + "\n\n" + table))


def display_hydrate_from_specs(
    filename: str,
    target_conc: float | int | str | Quantity[D],
    strands: Sequence[str] | Sequence[int] | None = None,
) -> None:
    """
    Indicates how much volume to add to a dry DNA sample to reach a particular concentration,
    given data in an Excel file in the IDT format,
    displaying the result in a jupyter notebook.

    :param filename:
        path to IDT Excel/CSV spreadsheet with specs of strands (e.g., coa.csv)
    :param target_conc:
        target concentration. If float/int, units are µM (micromolar).
    :param strands:
        strands to hydrate. Can be list of strand names (strings), or list of of ints indicating
        which rows in the Excel spreadsheet to hydrate
    """
    from IPython.display import Markdown, display
    from tabulate import tabulate

    names_to_vols = hydrate_from_specs(
        filename=filename, target_conc=target_conc, strands=strands
    )

    headers = ["name", "volume to add"]
    table_list = []
    for name, vol in names_to_vols.items():
        table_list.append((name, round(vol, 2)))  # type: ignore
    table = tabulate(table_list, headers=headers, tablefmt="pipe", floatfmt=".2f")
    from alhambra_mixes.mixes import _format_title

    raw_title = "Initial hydration volumes"
    title = _format_title(raw_title, level=2, tablefmt="pipe")
    display(Markdown(title + "\n\n" + table))


def display_measure_conc_from_specs(
    filename: str, absorbances: dict[str, float | int | Sequence[float] | Sequence[int]]
) -> None:
    """
    Indicates how much volume to add to a dry DNA sample to reach a particular concentration,
    given data in an Excel/CSV file in the IDT format,
    displaying the result in a jupyter notebook.

    :param filename:
        path to IDT Excel/CSV spreadsheet with specs of strands (e.g., coa.csv)
    :param absorbances:
        dict mapping each strand name to its absorbance value.
        Each absorbance value represents UV absorbance at 260 nm.
        Each can either be a single float/int or a nonempty sequence of floats/ints
        representing repeated measurements; if the latter then an average is taken.
    """
    from IPython.display import Markdown, display
    from tabulate import tabulate

    names_to_concs = measure_conc_from_specs(filename=filename, absorbances=absorbances)

    headers = ["name", "concentration"]
    table = tabulate(list(names_to_concs.items()), headers=headers, tablefmt="pipe")
    display(Markdown(table))
