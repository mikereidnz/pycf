#!/usr/bin/env python3
from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM

MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"


def _make_hamiltonian() -> "cfl.Hamiltonian":
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
    return h


def _make_exdata_with_differences() -> "cfl.ExData":
    ex_abs = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.6]])
    ex_diff = np.array([[9, 12, 116.0], [12, 14, 729.0]])
    return cfl.ExData((ex_abs, ex_diff), ("A", "D"))


def test_e_fit_wrapper_flag_contract_and_summary_ordering() -> None:
    param = ["EAVG"]
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=False)

    res = cfl.e_fit(
        param,
        _make_hamiltonian(),
        _make_exdata_with_differences(),
        cfl_min,
        suppress_input=True,
        calculate_sigma=False,
        include_covariance=True,
        include_jacobian=False,
    )
    assert res["all_coeff"] is not None
    assert res["sigma"] is not None
    assert res["sigma_vector"] is not None
    assert res["covariance"] is not None
    assert res["jacobian"] is None
    assert isinstance(res["jacobian_diagnostics"], dict)
    assert res["sigma_forced"] is True

    s = res["summary"]
    idx_all = s.find("All Hamiltonian parameters")
    idx_levels = s.find("Fitted energy levels")
    idx_fit = s.find("Fitting summary")
    assert idx_all != -1 and idx_levels != -1 and idx_fit != -1
    assert idx_all < idx_levels < idx_fit


def test_e_fit_dry_run_no_sigmas() -> None:
    """Verify that dry_run mode does not compute sigmas even if requested."""
    param = ["EAVG"]
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True)

    res = cfl.e_fit(
        param,
        _make_hamiltonian(),
        _make_exdata_with_differences(),
        cfl_min,
        suppress_input=True,
        calculate_sigma=True,  # Request sigma, but dry_run should skip it
        include_covariance=False,
        include_jacobian=False,
    )
    # In dry_run, no fit occurs, so sigma should be None
    assert res["sigma"] is None or res["sigma"] == {}
    assert res["sigma_vector"] is None or res["sigma_vector"] == {}
    assert res["covariance"] is None
    # sigma_forced should be False in dry_run (no override needed)
    assert res["sigma_forced"] is False


def test_mh_fit_wrapper_flag_contract_and_summary_ordering() -> None:
    param = ["EAVG"]
    cfl_min = cfl.CFLMin("gsl_nls", niter=1, dry_run=False)
    ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])

    h0 = _make_hamiltonian()
    h1 = _make_hamiltonian()
    res = cfl.mh_fit(
        param,
        [h0, h1],
        [1.0, 0.5],
        [ex, ex],
        cfl_min,
        suppress_input=True,
        calculate_sigma=False,
        include_covariance=True,
        include_jacobian=False,
    )
    assert res["all_coeff"] is not None
    assert res["sigma"] is not None
    assert res["sigma_vector"] is not None
    assert res["covariance"] is not None
    assert res["jacobian"] is None
    assert isinstance(res["jacobian_diagnostics"], dict)
    assert res["sigma_forced"] is True

    s = res["summary"]
    idx_all = s.find("All Hamiltonian parameters")
    idx_h0 = s.find("Hamiltonian 0")
    idx_fit = s.find("Fitting summary")
    assert idx_all != -1 and idx_h0 != -1 and idx_fit != -1
    assert idx_all < idx_h0 < idx_fit


def test_mh_fit_dry_run_no_sigmas() -> None:
    """Verify that dry_run mode does not compute sigmas even if requested."""
    param = ["EAVG"]
    cfl_min = cfl.CFLMin("gsl_nls", niter=1, dry_run=True)
    ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])

    h0 = _make_hamiltonian()
    h1 = _make_hamiltonian()
    res = cfl.mh_fit(
        param,
        [h0, h1],
        [1.0, 0.5],
        [ex, ex],
        cfl_min,
        suppress_input=True,
        calculate_sigma=True,  # Request sigma, but dry_run should skip it
        include_covariance=False,
        include_jacobian=False,
    )
    # In dry_run, no fit occurs, so sigma should be None
    assert res["sigma"] is None or res["sigma"] == {}
    assert res["sigma_vector"] is None or res["sigma_vector"] == {}
    assert res["covariance"] is None
    # sigma_forced should be False in dry_run (no override needed)
    assert res["sigma_forced"] is False
