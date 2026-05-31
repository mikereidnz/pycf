"""Tests for EFit.covariance / MHFit.covariance (S6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pycf import cfl


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name, diag):
    n = len(diag)
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def _make_efit():
    diag_a = np.array([0.0, 1.0, 2.0, 3.0])
    diag_b = np.array([0.5, 0.25, 0.125, 0.0625])
    ta = _diag_tensor("A", diag_a)
    tb = _diag_tensor("B", diag_b)
    h = cfl.Hamiltonian([ta, tb])
    h.set_coeff({"A": 1.0, "B": 1.0})
    ex = cfl.ExData(np.array([[1, 0.0], [2, 1.5], [3, 2.5], [4, 3.5]]))
    return cfl.EFit(["A", "B"], h, ex)


def test_covariance_shapes_and_unscaled_matches_pinv():
    fit = _make_efit()
    J = fit.fd_jacobian(check_swaps=False)
    cov, sigma, edata = fit.covariance(scale="unscaled", jacobian=J)
    n_p = len(fit.x0)
    assert cov.shape == (n_p, n_p)
    assert sigma.shape == (n_p,)
    assert edata.arr.shape[0] == 4
    W = np.asarray(edata.arr["weight"])
    expected = np.linalg.pinv(J.T @ (W[:, None] * J))
    np.testing.assert_allclose(cov, expected)
    np.testing.assert_allclose(sigma, np.sqrt(np.clip(np.diag(expected), 0, None)))


def test_covariance_reduced_chi2_scales_with_chi2():
    fit = _make_efit()
    J = fit.fd_jacobian(check_swaps=False)
    cov_u, _, edata = fit.covariance(scale="unscaled", jacobian=J)
    cov_r, _, _ = fit.covariance(scale="reduced_chi2", jacobian=J)
    chi2 = float(np.sum(np.asarray(edata.arr["wresidual"]) ** 2))
    n_obs = edata.arr.shape[0]
    n_p = len(fit.x0)
    factor = chi2 / max(n_obs - n_p, 1)
    np.testing.assert_allclose(cov_r, factor * cov_u)


def test_covariance_uses_last_jacobian_after_gsl_fit():
    fit = _make_efit()
    cmin = cfl.CFLMin("gsl_nls", niter=50)
    fit.fit(cmin)
    assert fit.last_jacobian is not None
    cov, sigma, _ = fit.covariance(scale="unscaled")
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))


def test_covariance_rank_deficient_warns():
    # Two parameters that produce identical Jacobian columns:
    # A and A2 with the same diag => J columns are identical => rank deficient.
    diag_a = np.array([0.0, 1.0, 2.0, 3.0])
    ta = _diag_tensor("A", diag_a)
    ta2 = _diag_tensor("A2", diag_a)
    h = cfl.Hamiltonian([ta, ta2])
    h.set_coeff({"A": 1.0, "A2": 0.5})
    ex = cfl.ExData(np.array([[1, 0.0], [2, 1.5], [3, 3.0], [4, 4.5]]))
    fit = cfl.EFit(["A", "A2"], h, ex)
    J = fit.fd_jacobian(check_swaps=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cov, sigma, _ = fit.covariance(scale="unscaled", jacobian=J)
    assert any("rank-deficient" in str(w.message) for w in caught)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))


def test_covariance_invalid_scale_raises():
    fit = _make_efit()
    with pytest.raises(ValueError, match="scale"):
        fit.covariance(scale="bogus")


def test_covariance_explicit_x_recomputes_jacobian():
    fit = _make_efit()
    x_alt = np.array([1.1, 1.0])
    cov, sigma, edata = fit.covariance(x=x_alt, scale="unscaled")
    assert cov.shape == (2, 2)
    # State must be restored after the call.
    np.testing.assert_allclose(fit.x0, [1.0, 1.0])
