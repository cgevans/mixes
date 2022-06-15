import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal as assert_approx

from alhambra_mixes.quantitate import (
    Q_,
    D,
    nM,
    uM,
    ureg,
    hydrate,
    dilute,
    measure_conc,
    measure_conc_and_dilute,
    hydrate_from_specs,
    measure_conc_from_specs,
)


def test_hydrate():
    assert hydrate(target_conc="100 uM", nmol=50) == Q_(D(500), "uL")
    assert hydrate(target_conc="100 mM", nmol=50) == Q_(D(500), "nL")


def test_dilute():
    assert dilute(target_conc="1 uM", start_conc="10 uM", vol="100 uL") == Q_(
        D(900), "uL"
    )
    assert dilute(target_conc="1 uM", start_conc="100 uM", vol="100 uL") == Q_(
        D(9900), "uL"
    )
    assert dilute(target_conc="1 uM", start_conc="100 uM", vol="100 uL") == Q_(
        D("9.9"), "mL"
    )


def test_measure_conc():
    assert measure_conc(absorbance=100, ext_coef=200_000) == Q_(D(500), "uM")


def test_measure_conc_and_dilute():
    # should measure concentration as 500 uM (see previous test)
    # So if we start with 101 uL and use 1 uL to measure absorbance, we'll have 100 uL left.
    # To diluate 500 uM down to 100 uM, we add 400 uL to 100 uL to do 5x dilution.
    assert measure_conc_and_dilute(
        absorbance=100,
        ext_coef=200_000,
        target_conc="100 uM",
        vol="101 uL",
    ) == (Q_(D(500), "uM"), Q_(D(400), "uL"))

    # same thing but we take 3 samples of 1 uL each from 103 uL originally
    assert measure_conc_and_dilute(
        absorbance=[100, 95, 105],
        ext_coef=200_000,
        target_conc="100 uM",
        vol="103 uL",
    ) == (Q_(D(500), "uM"), Q_(D(400), "uL"))

    # this time we explicitly say we took 4 uL despite there being only 3 absorbance samples
    assert measure_conc_and_dilute(
        absorbance=[100, 95, 105],
        ext_coef=200_000,
        target_conc="100 uM",
        vol="104 uL",
        vol_removed="4 uL",
    ) == (Q_(D(500), "uM"), Q_(D(400), "uL"))


def test_hydrate_from_specs():
    """
    first few rows:

    1 Plate Name	Payment Method	Plate Barcode	Sales Order #	Reference #	Well Position	Sequence Name	Sequence	Manufacturing ID	Measured Molecular Weight	Calculated Molecular Weight	OD260	nmoles	µg	Measured Concentration µM 	Final Volume µL 	Extinction Coefficient L/(mole·cm)	Tm	Well Barcode
    2 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218503	A01	mon0	/5Phos/CTC CTG GTT AAG AGA TCG ATA	459782228	6525.3	6525	4.91	23.45	153	200.43	117	209300	51
    3 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218504	A02	mon0_F	TTG GGA TGC GAA GGG ATG GTC TCC TGG TTA AGA GAT CGA TA	459782229	12800.6	12800	9.33	22.75	291	201.33	113	410300	67
    4 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218505	A04	adp0	TAT CGA TCT CTC AAA TAA ATC CTC ATT AAA GC	459782230	9710.2	9710	6.62	21.13	205	201.24	105	313100	55
    5 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218506	B01	mon1	/5Phos/GAC CTC GTA TTG GCA ATT AAT	459782231	6500.3	6500	6.4	31.28	203	200.51	156	204700	51
    """
    # excel file
    excel_filename = "tests/sample_specs.xlsx"

    # index strands by name
    target_conc = "200 uM"
    name_to_vol = hydrate_from_specs(
        filename=excel_filename,
        target_conc=target_conc,
        strands=["mon0", "mon0_F", "mon1"],
    )
    assert len(name_to_vol) == 3
    assert name_to_vol["mon0"] == Q_(D("117.25"), "uL")
    assert name_to_vol["mon0_F"] == Q_(D("113.75"), "uL")
    assert name_to_vol["mon1"] == Q_(D("156.4"), "uL")

    # index strands by row
    name_to_vol = hydrate_from_specs(
        filename=excel_filename,
        target_conc=target_conc,
        strands=[2, 3, 5],  # rows are Excel rows, so "2-based" since row 1 is header
    )
    assert len(name_to_vol) == 3
    assert name_to_vol["mon0"] == Q_(D("117.25"), "uL")
    assert name_to_vol["mon0_F"] == Q_(D("113.75"), "uL")
    assert name_to_vol["mon1"] == Q_(D("156.4"), "uL")

    # csv file with dry strands
    csv_filename = "tests/sample_coa.csv"
    name_to_vol = hydrate_from_specs(
        filename=csv_filename,
        target_conc=target_conc,
        strands=["3RQ", "5RF"],
    )
    assert len(name_to_vol) == 2
    assert name_to_vol["3RQ"] == Q_(D("78.5"), "uL")
    assert name_to_vol["5RF"] == Q_(D("44.5"), "uL")

    # assert error on wet strand
    with pytest.raises(ValueError):
        hydrate_from_specs(
            filename=csv_filename,
            target_conc=target_conc,
            strands=["mon1"],
        )


def test_measure_conc_from_specs():
    """
    first few rows:

    1 Plate Name	Payment Method	Plate Barcode	Sales Order #	Reference #	Well Position	Sequence Name	Sequence	Manufacturing ID	Measured Molecular Weight	Calculated Molecular Weight	OD260	nmoles	µg	Measured Concentration µM 	Final Volume µL 	Extinction Coefficient L/(mole·cm)	Tm	Well Barcode
    2 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218503	A01	mon0	/5Phos/CTC CTG GTT AAG AGA TCG ATA	459782228	6525.3	6525	4.91	23.45	153	200.43	117	209300	51
    3 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218504	A02	mon0_F	TTG GGA TGC GAA GGG ATG GTC TCC TGG TTA AGA GAT CGA TA	459782229	12800.6	12800	9.33	22.75	291	201.33	113	410300	67
    4 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218505	A04	adp0	TAT CGA TCT CTC AAA TAA ATC CTC ATT AAA GC	459782230	9710.2	9710	6.62	21.13	205	201.24	105	313100	55
    5 5 monomer synthesis	DD 2022-02-08 IDT 5-monomer synthesis plate and 5R	15073593	3600597	231218506	B01	mon1	/5Phos/GAC CTC GTA TTG GCA ATT AAT	459782231	6500.3	6500	6.4	31.28	203	200.51	156	204700	51
    """
    excel_filename = "tests/sample_specs.xlsx"
    absorbances = {
        "mon0": 100,
        "mon0_F": [100, 95, 105],
        "mon1": 100,
    }
    name_to_conc = measure_conc_from_specs(
        absorbances=absorbances,
        filename=excel_filename,
    )
    assert len(name_to_conc) == 3

    relerror = D(
        "0.1"
    )  # relative error needs to be Decimal to avoid type error in assert_approx below

    assert_approx(name_to_conc["mon0"], Q_(D("477.78"), "uM"), rtol=relerror)
    assert_approx(name_to_conc["mon0_F"], Q_(D("243.72"), "uM"), rtol=relerror)
    assert_approx(name_to_conc["mon1"], Q_(D("488.52"), "uM"), rtol=relerror)
