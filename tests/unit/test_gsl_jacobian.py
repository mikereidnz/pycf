"""Tests for the GSL-extracted Jacobian on EFit/MHFit (S5a)."""

from __future__ import annotations

import numpy as np

from pycf import cfl


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name, diag):
    n = len(diag)
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def test_efit_gsl_nls_populates_last_jacobian():
    diag_a = np.array([0.0, 1.0, 2.0, 3.0])
    diag_b = np.array([0.5, 0.25, 0.125, 0.0625])
    ta = _diag_tensor("A", diag_a)
    tb = _diag_tensor("B", diag_b)
    h = cfl.Hamiltonian([ta, tb])
    h.set_coeff({"A": 1.0, "B": 1.0})
    ex = cfl.ExData(np.array([[1, 0.0], [2, 1.5], [3, 2.5], [4, 3.5]]))
    fit = cfl.EFit(["A", "B"], h, ex)
    # Compare FD jacobian vs GSL jacobian after running the fit. Since
    # the system is diagonal/linear, both should agree at the optimum.

    cmin = cfl.CFLMin("gsl_nls", niter=50)
    fit.fit(cmin)
    assert fit.last_jacobian is not None
    assert fit.last_jacobian.shape == (4, 2)
    # The GSL Jacobian rows are weighted residual derivatives; with unit
    # weights this is just the energy Jacobian (sign aside; gsl_nls
    # uses (calc - obs), so dRes/dx = dE/dx). Magnitudes should match
    # the FD energy jacobian at the converged solution to within FD
    # precision.
    fd_at_xstar = fit.fd_jacobian(check_swaps=False)
    np.testing.assert_allclose(
        np.abs(fit.last_jacobian),
        np.abs(fd_at_xstar),
        atol=1e-5,
    )
