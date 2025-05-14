import pytest

from alhambra_mixes import Reference


def test_idt():
    r_order = Reference.compile(("tests/data/holes-order.xlsx", "200 µM"))
    r_platespec = Reference.compile("tests/data/holes-platespecs.xlsx")
    r_coa = Reference.compile(["tests/data/holes-coa.csv"])

    dfo = r_order.df.set_index("Name").sort_index()
    dfp = r_platespec.df.set_index("Name").sort_index()
    dfc = r_coa.df.set_index("Name").sort_index()

    assert (dfo == dfp).all().all()

    # COA does not have plate names...
    eq = (dfo == dfp).all()
    print(eq)
    assert eq.loc[["Well", "Sequence", "Concentration (nM)"]].all()


def test_raise_error_if_plate_name_not_found():
    r_order = Reference.compile(("tests/data/holes-order.xlsx", "200 µM"))

    # This should raise an error because the plate name "fake plate name" is not found in the reference
    with pytest.raises(ValueError):
        r_order.plate_map("fake plate name")
