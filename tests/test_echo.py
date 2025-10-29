import pytest

from riverine import Component, Experiment, Mix


def test_echo_experiment():
    pytest.importorskip('kithairon')
    from riverine import EchoTargetConcentration
    exp = Experiment()

    # We'll make some components:
    c1 = Component("c1", "10 µM", plate="plate1", well="A1")
    c2 = Component("c2", "10 µM", plate="plate1", well="A2")
    c3 = Component("c3", "5 µM", plate="plate1", well="A3")
    c4 = Component("c4", "5 µM", plate="plate2", well="B3")


    m = Mix(
        [
            EchoTargetConcentration([c1, c2, c3, c4], "1 nM")
        ],
        "testmix", plate="destplate", well="A1"
    )

    mstr = str(m)

    exp.add(m)

    p = exp.generate_picklist()
