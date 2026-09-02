"""Tests for :class:`pycf.pyfit.PyFit`."""

from __future__ import annotations

import numpy as np
import pytest

from pycf import cfl
from pycf.pyfit import PyFit


def _state_labels(n: int) -> "cfl.StateLabels":
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name: str, diag) -> "cfl.Tensor":
    n = len(diag)
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def _make_efit(diag, ex, params=("EAVG",)):
    t = _diag_tensor("EAVG", diag)
    h = cfl.Hamiltonian([t])
    h.set_coeff({"EAVG": 1.0})
    return cfl.EFit(list(params), h, ex)


def test_pyfit_residuals_match_efit_chi2():
    """sum(residuals**2) at x0 equals fit.eval at x0 (and EData.chi2)."""
    diag = np.array([0.0, 5.0, 12.0])
    ex = cfl.ExData(np.array([[1, 0.5], [3, 11.5]]))
    efit = _make_efit(diag, ex)

    py = PyFit(efit)
    r = py.residuals(py.x0)
    chi2_py = float(np.dot(r, r))
    chi2_eval = float(efit.eval({})[0])
    assert chi2_py == pytest.approx(chi2_eval, rel=1e-12)
    assert py.chi2(py.x0) == pytest.approx(chi2_eval, rel=1e-12)


def test_pyfit_residuals_track_x():
    """Changing x changes residuals predictably."""
    diag = np.array([0.0, 5.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 4.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    # At x = 1.0, e_calc = (0, 5); residuals = (sqrt(1)*0, sqrt(1)*1).
    r1 = py.residuals(np.array([1.0]))
    np.testing.assert_allclose(r1, [0.0, 1.0], atol=1e-12)

    # At x = 0.8, the EAVG coefficient scales the diagonal by 0.8:
    # e_calc = (0, 4); residuals = (0, 0).
    r2 = py.residuals(np.array([0.8]))
    np.testing.assert_allclose(r2, [0.0, 0.0], atol=1e-12)


def test_pyfit_does_not_mutate_underlying_x0():
    """A residual evaluation must not change the wrapped fit's state."""
    diag = np.array([0.0, 5.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 4.0]]))
    efit = _make_efit(diag, ex)
    saved_x0 = np.asarray(efit.x0).copy()

    py = PyFit(efit)
    py.residuals(np.array([2.5]))
    py.chi2(np.array([0.7]))
    np.testing.assert_array_equal(np.asarray(efit.x0), saved_x0)


def test_pyfit_lm_finds_minimum_for_linear_problem():
    """lm method recovers the parameter that zeros all residuals."""
    diag = np.array([0.0, 5.0, 10.0])
    # observed energies are exactly 0.7 * diag -> optimal x = 0.7.
    ex = cfl.ExData(np.array([[1, 0.0], [2, 3.5], [3, 7.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit(method="lm")
    assert res.success or res.status > 0
    assert res.x[0] == pytest.approx(0.7, rel=1e-6)
    assert py.chi2(res.x) == pytest.approx(0.0, abs=1e-18)


def test_pyfit_works_with_mhfit():
    """PyFit handles MHFit (multi-Hamiltonian) just like EFit."""
    h0 = _diag_tensor("EAVG", [0.0, 5.0])
    h1 = _diag_tensor("EAVG", [0.0, 7.0])
    H0 = cfl.Hamiltonian([h0])
    H0.set_coeff({"EAVG": 1.0})
    H1 = cfl.Hamiltonian([h1])
    H1.set_coeff({"EAVG": 1.0})
    ex0 = cfl.ExData(np.array([[1, 0.0], [2, 3.5]]))
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 4.9]]))
    mhfit = cfl.MHFit(["EAVG"], [H0, H1], [1.0, 1.0], [ex0, ex1])

    py = PyFit(mhfit)
    chi2_py = py.chi2(py.x0)
    chi2_eval = float(mhfit.eval({}).sum())
    assert chi2_py == pytest.approx(chi2_eval, rel=1e-12)

    res = py.fit(method="lm")
    # observed both data sets satisfy 0.7 * diag exactly.
    assert res.x[0] == pytest.approx(0.7, rel=1e-6)


def test_pyfit_handles_state_label_data():
    """PyFit picks up AS/DS support transparently from get_edata."""
    diag = np.array([0.0, 5.0])
    abs_sl = np.array([[0, 0, 0, 0.0], [0, 0, 2, 5.0]], dtype=np.float64)
    ex = cfl.ExData(abs_sl, key="AS", label_key="LJM")
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    # At x=1.0, residuals are zero.
    r = py.residuals(np.array([1.0]))
    np.testing.assert_allclose(r, [0.0, 0.0], atol=1e-12)


def test_pyfit_rejects_non_fit_argument():
    with pytest.raises(TypeError, match="EFit or MHFit"):
        PyFit(object())


def test_pyfit_bounds_with_trf():
    """Bounds-aware method runs and respects the bound."""
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 3.5], [3, 7.0]]))  # opt x = 0.7
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    # Force x to stay >= 0.9; the optimum is then on the boundary.
    res = py.fit(method="trf", bounds=([0.9], [2.0]))
    assert res.x[0] == pytest.approx(0.9, abs=1e-6)


def test_pyfit_jacobian_matches_residual_fd():
    """PyFit.jacobian(x) approximates the FD Jacobian of residuals."""
    diag = np.array([0.0, 5.0, 12.0])
    ex = cfl.ExData(np.array([[1, 0.5], [2, 4.6], [3, 11.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    x = np.array([0.95])
    J = py.jacobian(x)
    assert J.shape == (3, 1)

    # Compare with central differences on residuals().
    h = 1e-5
    rp = py.residuals(x + h)
    rm = py.residuals(x - h)
    J_ref = (rp - rm)[:, None] / (2 * h)
    np.testing.assert_allclose(J, J_ref, atol=1e-7, rtol=1e-7)


def test_pyfit_fit_with_pycf_jacobian():
    """jac='pycf' converges to the same minimum as default '2-point'."""
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 3.5], [3, 7.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit(method="lm", jac="pycf")
    assert res.x[0] == pytest.approx(0.7, rel=1e-6)
    assert py.chi2(res.x) == pytest.approx(0.0, abs=1e-18)
    assert py.last_result is res


def test_pyfit_covariance_matches_underlying_fit():
    """PyFit.covariance delegates to fit.covariance() at the optimum."""
    diag = np.array([0.0, 5.0, 10.0])
    # add a small offset so the fit isn't exactly zero residual.
    ex = cfl.ExData(np.array([[1, 0.05], [2, 3.45], [3, 7.05]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit(method="lm")
    cov_py, sigma_py, _ = py.covariance()
    cov_ref, sigma_ref, _ = efit.covariance(x=res.x)
    np.testing.assert_allclose(cov_py, cov_ref, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(sigma_py, sigma_ref, rtol=1e-8, atol=1e-12)


def test_pyfit_stderr_shape_and_positive():
    """stderr() returns a length-n_p_real non-negative vector."""
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.05], [2, 3.45], [3, 7.05]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    py.fit(method="lm")
    sigma = py.stderr()
    assert sigma.shape == (py.n_p_real,)
    assert np.all(sigma >= 0.0)


def test_pyfit_fit_res_returns_e_fit_like_payload():
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 3.5], [3, 7.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit_res(method="lm", jac="pycf", max_levels=2)
    assert "coeff" in res
    assert "all_coeff" in res
    assert "sigma" in res
    assert "summary" in res
    assert "optimizer_result" in res
    assert "All Hamiltonian parameters" in res["summary"]


def test_pyfit_fit_res_covariance_and_jacobian_opt_in():
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.05], [2, 3.45], [3, 7.05]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit_res(method="lm", include_covariance=True, include_jacobian=True)
    assert res["covariance"] is not None
    assert res["jacobian"] is not None
    assert "rank" in res["jacobian_diagnostics"]


def test_pyfit_fit_res_prints_pycf_details(capsys):
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 3.5], [3, 7.0]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    py.fit_res(method="lm")
    out = capsys.readouterr().out
    assert "pycf details" in out
    assert "Calculation started at:" in out
    assert "Calculation completed at:" in out
    # PyFit calls gen_pycf_summary(suppress_input=True) internally
    # (pyfit.py itself is the immediate caller, not the user's script), so
    # no "File: ..." line should ever be echoed here -- otherwise every
    # fit summary would leak pycf's own pyfit.py path as if it were the
    # user's input file.
    assert "File: " not in out


def test_pyfit_fit_res_forces_sigma_for_covariance(capsys):
    diag = np.array([0.0, 5.0, 10.0])
    ex = cfl.ExData(np.array([[1, 0.05], [2, 3.45], [3, 7.05]]))
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit_res(method="lm", calculate_sigma=False, include_covariance=True)
    assert res["sigma"] is not None
    assert res["sigma_vector"] is not None
    assert res["covariance"] is not None
    assert res["sigma_forced"] is True
    assert "calculate_sigma assumed True" in capsys.readouterr().out


def test_pyfit_fit_res_mhfit_produces_multi_hamiltonian_summary():
    """PyFit.fit_res on MHFit exercises the multi-Hamiltonian summary path.

    Covers ``pycf/pyfit.py`` lines 311-318 (initial_coeff sourced from
    ``efit.h_list[0]``) and 469-500 (multi-Hamiltonian summary block).
    """
    h0_t = _diag_tensor("EAVG", [0.0, 5.0])
    h1_t = _diag_tensor("EAVG", [0.0, 7.0])
    H0 = cfl.Hamiltonian([h0_t])
    H0.set_coeff({"EAVG": 1.0})
    H1 = cfl.Hamiltonian([h1_t])
    H1.set_coeff({"EAVG": 1.0})
    ex0 = cfl.ExData(np.array([[1, 0.0], [2, 3.5]]))
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 4.9]]))
    mhfit = cfl.MHFit(["EAVG"], [H0, H1], [1.0, 1.0], [ex0, ex1])
    py = PyFit(mhfit)

    res = py.fit_res(method="lm")
    assert "coeff" in res
    assert "summary" in res
    summary = res["summary"]
    assert "Multi-Hamiltonian fit" in summary
    assert "Hamiltonian 0" in summary
    assert "Hamiltonian 1" in summary
    assert "All Hamiltonian parameters" in summary
    # initial_coeff should have been picked up from h_list[0].coeff_dict.
    assert "EAVG" in res["coeff"]


def test_pyfit_fit_res_mhfit_with_max_levels():
    """MHFit + ``max_levels`` exercises the ``kwargs["max_levels"]`` branch
    inside the multi-Hamiltonian summary loop (pyfit.py line 495-496)."""
    h0_t = _diag_tensor("EAVG", [0.0, 5.0, 10.0])
    h1_t = _diag_tensor("EAVG", [0.0, 7.0, 14.0])
    H0 = cfl.Hamiltonian([h0_t])
    H0.set_coeff({"EAVG": 1.0})
    H1 = cfl.Hamiltonian([h1_t])
    H1.set_coeff({"EAVG": 1.0})
    ex0 = cfl.ExData(np.array([[1, 0.0], [2, 3.5]]))
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 4.9]]))
    mhfit = cfl.MHFit(["EAVG"], [H0, H1], [1.0, 1.0], [ex0, ex1])
    py = PyFit(mhfit)

    res = py.fit_res(method="lm", max_levels=2)
    assert "Multi-Hamiltonian fit" in res["summary"]


def test_pyfit_rejects_object_missing_get_edata():
    """PyFit requires a get_edata() accessor on the fit object (pyfit.py L82-86)."""

    class _FakeFit:
        n_p_real = 1
        x0 = np.array([1.0])

    with pytest.raises(TypeError, match="get_edata"):
        PyFit(_FakeFit())


def test_pyfit_fit_res_with_difference_data():
    """Single-Hamiltonian fit_res with difference data exercises the
    ``self.efit.ex.n_d != 0`` branch (pyfit.py lines 443-457)."""
    diag = np.array([0.0, 5.0, 12.0])
    diff = np.array([[1, 2, 5.0], [1, 3, 12.0]])
    ex = cfl.ExData(diff, key="D")
    efit = _make_efit(diag, ex)
    py = PyFit(efit)

    res = py.fit_res(method="lm", max_levels=3)
    assert "Fitted energy levels" in res["summary"]
    # gen_summary header from h.gen_summary() should appear.
    assert "Hamiltonian" in res["summary"]
