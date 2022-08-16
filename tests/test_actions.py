import pytest

from alhambra_mixes.abbreviated import *


def test_names_and_numbers():
    a = TC(["a", "b", "c"], fixed_concentration="100 µM")
    b = EC(["a", "b", "c"], fixed_volume="100 µL")
    c = FV(["a", "b", "c"], fixed_volume="100 µL")
    d = FC(["a", "b", "c"], fixed_concentration="100 µM")

    assert a.name == b.name == c.name == d.name == "a, b, c"

    cf = FV(["a", "b"], fixed_volume="100 µL", set_name="set name")
    df = FC("a long name", fixed_concentration="100 µM", set_name="set name")
    assert cf.name == "set name" == df.name
    assert cf.number == 2
    assert 1 == df.number


def test_ec_volumes():
    a = C("A", "50 nM")
    b = C("B", "100 nM")
    c = C("C", "50 nM")

    assert EC([a, b], Q_("10", uL), method="min_volume").each_volumes(
        ureg("100 uL")
    ) == [ureg("20 µL"), ureg("10 µL")]

    assert EC([a, b], Q_("10", uL), method="max_volume").each_volumes(
        ureg("100 uL")
    ) == [ureg("10 µL"), ureg("5 µL")]

    assert EC([a, b], Q_("10", uL), method=("max_fill", "Buffer")).each_volumes(
        ureg("100 uL")
    ) == [
        ureg("10 µL"),
        ureg("5 µL"),
    ]  # FIXME

    with pytest.raises(ValueError):
        EC([a, b], Q_("10", uL), method="check").each_volumes(ureg("100 uL"))

    assert EC([a, c], Q_("10", uL), method="max_volume").each_volumes(
        ureg("100 uL")
    ) == [ureg("10 µL"), ureg("10 µL")]
