#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pytest

import pycf
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM

# Example of Spin Hamiltonian calculations for Ce:YLF, w/ data from
# 10.1016/j.optmat.2015.06.046
MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"


# @pytest.mark.slow  # mark this test as slow, so it can be skipped by default
def test_shfit() -> None:
    ####
    t = ImportSLJM(str(MATEL_BASE))
    coeff = {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
    }
    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64])
    h.set_coeff(coeff)
    # Calculate the Spin Hamiltonian from the CF Hamiltonian.
    sh = cfl.SpinHamiltonian("zeeman", S=1 / 2, level=1)
    sh.set_pro_data([t.MAGX, t.MAGY, t.MAGZ])
    h.set_coeff(coeff)
    print("Spin Hamiltonian w/ input CF parameters:")
    print(np.real(sh.calc_param(h)))
    # Spin Hamiltonian and energy level fitting.
    ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])
    g_tensor = np.array([[1.473, 0, 0], [0, 1.473, 0], [0, 0, 2.765]])
    shx = {"zeeman": g_tensor}
    weights = {"energy": 1, "zeeman": 1e7}
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6)
    res = cfl.esh_fit(
        ["EAVG", "ZETA", "C20", "C40", "C44", "C60", "C64"],
        h,
        sh,
        ex,
        shx,
        weights,
        cfl_min,
        svd_sym=True,
    )
    print(res["summary"])
    fit_coeff = res["coeff"]
    print("Check fitted parameters:")
    expected_coeff = {
        "EAVG": 1528.89,
        "ZETA": 625.78,
        "C20": 308.14,
        "C40": -1257.36,
        "C44": -1183.71,
        "C60": -251.14,
        "C64": -2046.62 + 235.49j,
    }
    # uncomment this line to deliberately make it crash:
    # expected_coeff['EAVG'] = 0
    tolerance = 1e-2
    for label, value in fit_coeff.items():
        print(label, value, " should be equal to ", expected_coeff[label])
        assert value == pytest.approx(expected_coeff[label], rel=tolerance)


if __name__ == "__main__":
    # for running from spyder or as a stand-alone file
    pycf.pycf_info()
    print("\nRun shfit test\n")
    test_shfit()
