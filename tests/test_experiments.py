from decimal import Decimal
import math
from alhambra_mixes import *
from alhambra_mixes.abbreviated import *
import pytest
import re


@pytest.fixture
def experiment():
    exp = Exp()

    # We'll make some components:
    c1 = C("c1", "10 µM", plate="plate1", well="A1")
    c2 = C("c2", "10 µM", plate="plate1", well="A2")
    c3 = C("c3", "5 µM", plate="plate1", well="A3")
    c4 = C("c4", "5 µM", plate="plate2", well="B3")

    # Add a mix with add_mix
    exp.add_mix(FV([c2, c3], "10 µL"), "mix1")

    # Add a mix by assignment
    exp["mix2"] = Mix(FC([c1, c4], "1 µM"), fixed_total_volume="10 µL")

    # Add a mix that has a NaN value
    exp.add_mix(Mix(FC([c1], "1 µM"), "mix3"))

    # Add a mix with mixes, by reference
    exp["mixmix"] = Mix(FC(["mix1", "mix2", c3], "100 nM"), fixed_total_volume="200 µL")

    return exp


def test_forward_reference():
    exp = Exp()

    exp.add_mix(Mix(FV(["mix1", "mix2"], "10 µL"), "mix3"))
    exp["mix1"] = Mix(FC(["c1", "c2"], "1 µM"), fixed_total_volume="10 µL")
    exp["mix2"] = Mix(FC(["c1", "c2"], "1 µM"), fixed_total_volume="10 µL")
    exp["c1"] = C("c1", "10 µM", plate="plate1", well="A1")

    exp.resolve_components()

    assert exp["mix3"].actions[0].components == [exp["mix1"], exp["mix2"]]


def test_iterate_mixes(experiment):
    assert len(experiment) == 4
    assert list(experiment) == [
        experiment["mix1"],
        experiment["mix2"],
        experiment["mix3"],
        experiment["mixmix"],
    ]


def test_consumed_and_produced_volumes(experiment):
    cp = experiment.consumed_and_produced_volumes()

    # c1 and mix3 can't be directly compared because they have NaN values
    assert math.isnan(cp["c1"][0].m)
    assert math.isnan(cp["mix3"][1].m)

    del cp["c1"]
    del cp["mix3"]

    assert cp == {
        "mix1": (ureg("4.00000 µL"), ureg("20 µL")),
        "mix2": (ureg("20 µL"), ureg("10 µL")),
        "mixmix": (ureg("0 µL"), 200 * ureg("µL")),
        "c2": (Decimal("10") * ureg("µL"), ureg("0.0 µL")),
        "c3": (Decimal("14") * ureg("µL"), ureg("0.0 µL")),
        "c4": (Decimal("2.0") * ureg("µL"), ureg("0.0 µL")),
        "Buffer": (ureg("179 µL"), ureg("0 µL")),
    }


def test_check_volumes(experiment, capsys):
    experiment.check_volumes(showall=True)
    cvstring = capsys.readouterr().out
    assert re.search(
        r"Making 10 µl of mix2 but need at least 20(\.0*)? µl", cvstring, re.UNICODE
    )
    assert len([x for x in cvstring.splitlines() if x]) == 9


def test_unnamed_mix(experiment):
    with pytest.raises(ValueError):
        experiment.add_mix(Mix(FC(["a"], "1 µM"), fixed_total_volume="10 µL"))
    with pytest.raises(ValueError):
        experiment.add_mix(FC(["a"], "1 µM"), fixed_total_volume="10 µL")


def test_add_mix_already_present(experiment):
    with pytest.raises(ValueError):
        experiment.add_mix(
            Mix(
                FC(["mix1", "mix2", "c3"], "100 nM"),
                "mixmix",
                fixed_total_volume="10 µL",
            )
        )
    # with pytest.raises(ValueError):
    #    experiment['mixmix'] = Mix(FC(["mix1", "mix2", "c3"], "100 nM"), "mixmix", fixed_total_volume="10 µL" )


def test_add_wrong_name(experiment):
    with pytest.raises(ValueError):
        experiment["mixA"] = Mix(
            [FC("mix1", "100 nM")], "mixB", fixed_total_volume="10 µL"
        )


def test_save_load(experiment, tmp_path):
    experiment.save(tmp_path / "test.json")

    e2 = Exp.load(tmp_path / "test.json")

    assert e2 == experiment


def test_save_load_on_stream(experiment, tmp_path):
    with open(tmp_path / "test.json", "w") as f:
        experiment.save(f)

    with open(tmp_path / "test.json", "r") as f:
        e2 = Exp.load(f)

    assert e2 == experiment


def test_save_load_no_suffix(experiment, tmp_path):
    experiment.save(tmp_path / "test")

    assert (tmp_path / "test.json").exists()

    e2 = Exp.load(tmp_path / "test")

    assert e2 == experiment


def test_load_invalid_json(experiment, tmp_path):
    with open(tmp_path / "test.json", "w") as f:
        f.write("{}")

    with pytest.raises(ValueError):
        Exp.load(tmp_path / "test.json")
