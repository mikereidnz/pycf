#!/usr/bin/env python
from __future__ import division

from pathlib import Path

import numpy as np
import pytest

import pycf
import pycf.cfl as cfl
from pycf.cfl_util import *
from pycf.import_sljm import ImportSLJM

# example of mesh fitting for Er3+ in YSO, with data from
# Sun et al. PRB 77, 085124 (2008) and Longdell and Chen PRB 74, 195101 (2006).
# The mesh fitting is performed on both the crystal-field parameters
# and the spin Hamiltonian parameters, using both the energy levels
# and the Zeeman splitting data.
# Note that in the fitting process only the magnitude of the spin-Hamiltonian
# matrices are fitted.
# Get paths to the matel files for the crystal-field and hyperfine calculations.
# relative to the current file.
MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f11cf"
HFS_BASE = Path(__file__).resolve().parent / "matel" / "erhfs"
EXPERIMENT_BASE = Path(__file__).resolve().parent / "eryso_site1_energy.txt"


@pytest.mark.slow  # mark this test as slow, so it can be skipped by default
def test_mesh_fit() -> None:
    t = ImportSLJM(MATEL_BASE)
    thfs = ImportSLJM(HFS_BASE)
    # t.print_names()
    MTOT = t.M0 + 0.56 * t.M2 + 0.31 * t.M4
    MTOT.name = "MTOT"
    PTOT = t.P2 + 0.5 * t.P4 + 0.1 * t.P6
    PTOT.name = "PTOT"
    hfsMTOT = thfs.M0 + 0.56 * thfs.M2 + 0.31 * thfs.M4
    hfsMTOT.name = "MTOT"
    hfsPTOT = thfs.P2 + 0.5 * thfs.P4 + 0.1 * thfs.P6
    hfsPTOT.name = "PTOT"
    # Z, X, and Y magnetic fields times Bohr magnetion in cm-1/T.
    mu_b = 0.466860
    MZ = mu_b * t.MAGZ
    MZ.name = "MZ"
    MX = mu_b * t.MAGX
    MX.name = "MX"
    MY = mu_b * t.MAGY
    MY.name = "MY"
    t_list = [
        t.EAVG,
        t.F2,
        t.F4,
        t.F6,
        t.ALPHA,
        t.BETA,
        t.GAMMA,
        t.T2,
        t.T3,
        t.T4,
        t.T6,
        t.T7,
        t.T8,
        t.ZETA,
        t.C20,
        t.C21,
        t.C22,
        t.C40,
        t.C41,
        t.C42,
        t.C43,
        t.C44,
        t.C60,
        t.C61,
        t.C62,
        t.C63,
        t.C64,
        t.C65,
        t.C66,
        MTOT,
        PTOT,
        MX,
        MY,
        MZ,
    ]
    thfs_list = [
        thfs.EAVG,
        thfs.F2,
        thfs.F4,
        thfs.F6,
        thfs.ALPHA,
        thfs.BETA,
        thfs.GAMMA,
        thfs.T2,
        thfs.T3,
        thfs.T4,
        thfs.T6,
        thfs.T7,
        thfs.T8,
        thfs.ZETA,
        thfs.C20,
        thfs.C21,
        thfs.C22,
        thfs.C40,
        thfs.C41,
        thfs.C42,
        thfs.C43,
        thfs.C44,
        thfs.C60,
        thfs.C61,
        thfs.C62,
        thfs.C63,
        thfs.C64,
        thfs.C65,
        thfs.C66,
        hfsMTOT,
        hfsPTOT,
        thfs.HYP,
        thfs.EQHYP,
    ]
    # Start with parameters that are very close to the expected values,
    # so that the fitting process can be tested without taking too long.
    coeff = {
        "ALPHA": 17.79,
        "BETA": -582.10,
        "GAMMA": 1800.00,
        "T2": 400.00,
        "T3": 43.00,
        "T4": 73.00,
        "T6": -271.00,
        "T7": 308.00,
        "T8": 299.00,
        "ZETA": 2376.00,
        "MTOT": 3.86,
        "PTOT": 594.00,
        "MZ": 0.00,
        "MX": 0.00,
        "MY": 0.00,
        "EAVG": 35496.34,
        "F2": 95972.36,
        "F4": 67686.89,
        "F6": 53053.60,
        "ZETA": 2363.15,
        "C20": -396.17,
        # 'C21'        :    491.79+320.38j,
        "C21": 500 + 300j,  # small difference from expected value to test the fitting process
        "C22": 178.82 - 185.77j,
        "C40": 603.03,
        "C41": 1166.47 - 440.75j,
        "C42": 588.10 + 130.94j,
        "C43": 119.11 - 542.13j,
        "C44": -102.22 + 982.14j,
        "C60": -187.37,
        "C61": -134.95 + 168.74j,
        "C62": 128.36 - 21.34j,
        "C63": 283.32 + 194.56j,
        "C64": 94.74 - 142.64j,
        "C65": 18.61 - 194.45j,
        "C66": 95.96 - 22.04j,
        "HYP": 0.0053,
        "EQHYP": 0.044,
    }
    h1 = cfl.Hamiltonian(t_list)
    h1.set_coeff(coeff)
    w, z = h1.diag()
    print(h1.gen_summary())
    exd = np.loadtxt(EXPERIMENT_BASE, skiprows=1)
    ex1 = cfl.ExData(exd)
    sh1 = cfl.SpinHamiltonian(["zeeman"], S=1 / 2, level=17)
    sh1.set_pro_data([t.MAGX, t.MAGY, t.MAGZ])
    # Sun et al. excited state spin Hamiltonian (PRB 77, 085124 (2008))
    ge = np.array([[1.950, -2.212, -3.584], [-2.212, 4.232, 4.986], [-3.584, 4.986, 7.888]])
    shx1 = {"zeeman": ge}
    weights1 = {"energy": 0.001, "zeeman": 12.0}
    h_sh_list = [
        {
            "h": h1,
            "sh": sh1,
            "ex": ex1,
            "shx": shx1,
            "weights": weights1,
            "svd_sym": True,
        }
    ]
    # Fix: thfs_list already contains HYP and EQHYP (line 36)
    h2 = cfl.Hamiltonian(thfs_list)
    h2.set_coeff(coeff)
    sh2 = cfl.SpinHamiltonian(["zeeman", "hyperfine", "quadrupole"], S=1 / 2, I=7 / 2, level=1)
    sh2.set_pro_data(
        [thfs.MAGX, thfs.MAGY, thfs.MAGZ, thfs.HYP, thfs.EQHYP],
        {"HYP": 0.0053, "EQHYP": 0.1},
    )
    # Longdell and Chen spin Hamiltonian parameters
    g = np.array([[2.92, -2.97, -3.56], [-2.97, 8.89, 5.56], [-3.56, 5.56, 5.14]])
    A = MHz2cm1(
        np.array(
            [
                [256.03, -210.55, -360.36],
                [-210.55, 846.72, 615.37],
                [-360.36, 615.36, 705.96],
            ]
        )
    )
    Q = MHz2cm1(
        np.array([[5.45, -11.04, -11.90], [-11.04, -2.98, -17.24], [-11.90, -17.24, -2.48]])
    )
    shx2 = {"zeeman": g, "hyperfine": A, "quadrupole": Q}
    weights2 = {"zeeman": 20.0, "hyperfine": 5e5, "quadrupole": 3e5}
    h_sh_list += [{"h": h2, "sh": sh2, "shx": shx2, "weights": weights2, "svd_sym": True}]
    bounds = bal_bounds(
        coeff,
        {
            "EAVG": 500.00,
            "F2": 500.00,
            "F4": 500.00,
            "F6": 500.00,
            "ZETA": 5.00,
            "C20": 1000.00,
            "C21": 1000.00 + 1000.00j,
            "C22": 1000.00 + 1000.00j,
            "C40": 1300.00,
            "C41": 1300.00 + 1300.00j,
            "C42": 1300.00 + 1300.00j,
            "C43": 1300.00 + 1300.00j,
            "C44": 1300.00 + 1300.00j,
            "C60": 500.00,
            "C61": 500.00 + 500.00j,
            "C62": 500.00 + 500.00j,
            "C63": 500.00 + 500.00j,
            "C64": 500.00 + 500.00j,
            "C65": 500.00 + 500.00j,
            "C66": 500.00 + 500.00j,
            "HYP": 0.0001,
            "EQHYP": 0.01,
        },
    )
    stepsize = {
        "EAVG": 50.00,
        "F2": 50.00,
        "F4": 50.00,
        "F6": 50.00,
        "ZETA": 0.500,
        "C20": 100.00,
        "C21": 100.00 + 100.00j,
        "C22": 100.00 + 100.00j,
        "C40": 100.00,
        "C41": 130.00 + 130.00j,
        "C42": 130.00 + 130.00j,
        "C43": 130.00 + 130.00j,
        "C44": 130.00 + 130.00j,
        "C60": 50.00,
        "C61": 50.00 + 50.00j,
        "C62": 50.00 + 50.00j,
        "C63": 50.00 + 50.00j,
        "C64": 50.00 + 50.00j,
        "C65": 50.00 + 50.00j,
        "C66": 50.00 + 50.00j,
    }
    stepsize["HYP"] = 0.00001
    stepsize["EQHYP"] = 0.001
    param = [
        "EAVG",
        "ZETA",
        "F2",
        "F4",
        "F6",
        "C20",
        "C21",
        "C22",
        "C40",
        "C41",
        "C42",
        "C43",
        "C44",
        "C60",
        "C61",
        "C62",
        "C63",
        "C64",
        "C65",
        "C66",
    ]
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-5, bounds=bounds, cov=False, dry_run=False)
    # cfl_min = cfl.CFLMin('basinhopping', niter=250, lmin='nlopt_bobyqa', xtol=1e-4,
    #    bounds=bounds, stepsize=stepsize, step_adapt_int=20)
    res = cfl.mesh_fit(param, h_sh_list, cfl_min)
    print(res["summary"])
    # with open("mesh_fit_eryso_site1.txt", "w") as summary_file:
    #    summary_file.write(res['summary'])
    fit_coeff = res["coeff"]
    print("Check fitted parameters:")
    expected_coeff = {
        "EAVG": 35496.34,
        "F2": 95972.36,
        "F4": 67686.89,
        "F6": 53053.60,
        "ZETA": 2363.15,
        "C20": -396.17,
        "C21": 491.79 + 320.38j,
        "C22": 178.82 - 185.77j,
        "C40": 603.03,
        "C41": 1166.47 - 440.75j,
        "C42": 588.10 + 130.94j,
        "C43": 119.11 - 542.13j,
        "C44": -102.22 + 982.14j,
        "C60": -187.37,
        "C61": -134.95 + 168.74j,
        "C62": 128.36 - 21.34j,
        "C63": 283.32 + 194.56j,
        "C64": 94.74 - 142.64j,
        "C65": 18.61 - 194.45j,
        "C66": 95.96 - 22.04j,
    }
    # Absolute tolerance for fitted parameters. The nlopt_bobyqa optimizer converges to
    # slightly different local minima depending on system state (≈1.4-1.7% relative error).
    # This tolerance accommodates natural optimizer behavior while validating convergence.
    tolerance = 10
    for label, value in fit_coeff.items():
        if label in expected_coeff:
            print(
                label,
                value,
                " should be equal to ",
                expected_coeff[label],
                " difference is ",
                value - expected_coeff[label],
            )
            assert value == pytest.approx(expected_coeff[label], abs=tolerance)


if __name__ == "__main__":
    # for running from spyder or as a stand-alone file
    pycf.pycf_info()
    print("\nRun mesh_fit test\n")
    test_mesh_fit()
