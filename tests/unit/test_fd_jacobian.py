"""Tests for EFit.fd_jacobian() and MHFit.fd_jacobian() (S5)."""

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


# --- EFit ----------------------------------------------------------------


def _make_efit(diag_a, diag_b, ex):
    """One Hamiltonian with two diagonal tensors A,B (no shared off-diag)."""
    ta = _diag_tensor("A", diag_a)
    tb = _diag_tensor("B", diag_b)
    h = cfl.Hamiltonian([ta, tb], label="H")
    h.set_coeff({"A": 1.0, "B": 1.0})
    fit = cfl.EFit(["A", "B"], h, ex)
    return fit, h


def test_fd_jacobian_diagonal_real_params():
    # Diagonal Hamiltonian: E_i = a*A_i + b*B_i  =>  dE/da = A_i, dE/db = B_i.
    # Choose A,B with no degeneracies in A+B at (a=b=1) so that the
    # sort order is unambiguous and FD picks each eigenvalue cleanly.
    diag_a = np.array([0.0, 1.0, 2.0, 3.0])
    diag_b = np.array([0.5, 0.25, 0.125, 0.0625])
    ex_arr = np.array([[1, 0.0], [2, 0.0], [3, 0.0], [4, 0.0]])
    ex = cfl.ExData(ex_arr)
    fit, _ = _make_efit(diag_a, diag_b, ex)
    e0 = diag_a + diag_b
    order = np.argsort(e0)
    A_sorted = diag_a[order]
    B_sorted = diag_b[order]

    J = fit.fd_jacobian(check_swaps=False)
    assert J.shape == (4, 2)
    np.testing.assert_allclose(J[:, 0], A_sorted, atol=1e-7)
    np.testing.assert_allclose(J[:, 1], B_sorted, atol=1e-7)
    # last_jacobian stored.
    np.testing.assert_array_equal(fit.last_jacobian, J)


def test_fd_jacobian_restores_state_on_exit():
    diag_a = np.array([0.0, 1.0, 2.0])
    diag_b = np.array([3.0, 2.0, 1.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0]]))
    fit, h = _make_efit(diag_a, diag_b, ex)
    coeff_before = dict(fit.coeff)
    h_coeff_before = dict(h.coeff_dict)
    x0_before = fit.x0.copy()
    fit.fd_jacobian(check_swaps=False)
    assert fit.coeff == coeff_before
    assert h.coeff_dict == h_coeff_before
    np.testing.assert_array_equal(fit.x0, x0_before)


def test_fd_jacobian_default_uses_x0():
    diag_a = np.array([0.0, 1.0, 2.0])
    diag_b = np.array([3.0, 2.0, 1.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0]]))
    fit, _ = _make_efit(diag_a, diag_b, ex)
    J_default = fit.fd_jacobian(check_swaps=False)
    J_explicit = fit.fd_jacobian(x=fit.x0.copy(), check_swaps=False)
    np.testing.assert_allclose(J_default, J_explicit)


def test_fd_jacobian_difference_observation():
    # 'D' rows have e_calc = |w[fld]-w[ild]|; J row should be the diff
    # of per-eigenvalue derivatives (signed by the calc convention).
    diag_a = np.array([0.0, 1.0, 2.0, 4.0])
    diag_b = np.array([0.5, 0.25, 0.125, 0.0625])
    # Two rows: one absolute (level 1) + one diff (between levels 3 and 1).
    abs_data = np.array([[1, 0.0]])
    diff_data = np.array([[3, 1, 2.0]])
    ex = cfl.ExData((abs_data, diff_data), key=("A", "D"))
    fit, _ = _make_efit(diag_a, diag_b, ex)
    J = fit.fd_jacobian(check_swaps=False)
    assert J.shape == (2, 2)
    # Eigenvalues are diag_a + diag_b sorted; check that derivatives
    # for difference row equal A_sorted[2] - A_sorted[0] etc.
    e0 = diag_a + diag_b
    order = np.argsort(e0)
    A_sorted = diag_a[order]
    np.testing.assert_allclose(J[0, 0], A_sorted[0], atol=1e-7)
    np.testing.assert_allclose(J[1, 0], A_sorted[2] - A_sorted[0], atol=1e-7)


def test_fd_jacobian_validates_x_shape():
    diag = np.array([0.0, 1.0, 2.0])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0], [3, 0.0]]))
    diag_b = np.array([0.5, 0.25, 0.125])
    fit, _ = _make_efit(diag, diag_b, ex)
    with pytest.raises(ValueError, match="x must have shape"):
        fit.fd_jacobian(x=np.zeros(7))


def test_fd_jacobian_validates_delta_positive():
    diag = np.array([0.0, 1.0, 2.0])
    diag_b = np.array([0.5, 0.25, 0.125])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0], [3, 0.0]]))
    fit, _ = _make_efit(diag, diag_b, ex)
    with pytest.raises(ValueError, match="must be strictly positive"):
        fit.fd_jacobian(delta=0.0)
    with pytest.raises(ValueError, match="must be strictly positive"):
        fit.fd_jacobian(delta=np.array([1e-3, -1e-3]))


def test_fd_jacobian_swap_warning_triggers_on_huge_step():
    # With a huge step, max|J|*delta easily exceeds the energy spread.
    diag_a = np.array([0.0, 1.0, 2.0, 3.0])
    diag_b = np.array([0.5, 0.25, 0.125, 0.0625])
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0], [3, 0.0], [4, 0.0]]))
    fit, _ = _make_efit(diag_a, diag_b, ex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit.fd_jacobian(delta=1e6, check_swaps=True)
    assert any("fd_jacobian" in str(w.message) for w in caught)


def test_fd_jacobian_complex_param_real_imag_split():
    # Build H with one complex-coefficient diagonal tensor T:
    # H = c * diag, c complex; eigenvalues are c*diag_i, but we observe
    # absolute-value (real) eigenvalues from a Hermitian Hamiltonian.
    # Use H = c*T + conj(c)*T_dag where T is symmetric -> easier to
    # just use an off-diagonal scheme... For simplicity here, use one
    # real tensor + one complex param that ends up multiplying a
    # Hermitian-zero contribution (so dE/dRe = dE/dIm = 0) and a real
    # tensor that gives a non-zero dE/d(real param).
    n = 3
    real_diag = np.array([0.0, 1.0, 2.0])
    real_t = _diag_tensor("R", real_diag)
    # Complex tensor: skew-Hermitian off-diagonal pair using two CSR
    # entries [0,1] = i and [1,0] = -i would not work for our diagonal
    # helper. Simpler: another diagonal tensor; coeff is complex but
    # only its real part affects Hermitian eigenvalues if we wrap with
    # symmetrisation. Easier: declare the param as complex and keep
    # imag part contributing 0.
    # We do this by giving the second tensor diagonal of zeros so that
    # any complex coefficient still yields a Hermitian H with real
    # eigenvalues independent of Im(c).
    zero_diag = np.zeros(n)
    zero_t = _diag_tensor("Z", zero_diag)
    h = cfl.Hamiltonian([real_t, zero_t])
    h.set_coeff({"R": 1.0, "Z": complex(0.5, 0.25)})
    ex = cfl.ExData(np.array([[1, 0.0], [2, 0.0], [3, 0.0]]))
    fit = cfl.EFit(["R", "Z"], h, ex)
    # x0 layout is [R, Re(Z), Im(Z)] -> n_p_real == 3
    assert fit.n_p_real == 3
    J = fit.fd_jacobian(check_swaps=False)
    assert J.shape == (3, 3)
    # dE/dR equals sorted real_diag.
    np.testing.assert_allclose(J[:, 0], np.sort(real_diag), atol=1e-7)
    # dE/dRe(Z), dE/dIm(Z) are both ~0 (Z tensor is the zero matrix).
    np.testing.assert_allclose(J[:, 1], 0.0, atol=1e-7)
    np.testing.assert_allclose(J[:, 2], 0.0, atol=1e-7)


# --- MHFit ---------------------------------------------------------------


def _make_mhfit(diags_a_per_h, diags_b_per_h, ex_per_h, weights):
    h_list = []
    for i, (da, db) in enumerate(zip(diags_a_per_h, diags_b_per_h)):
        ta = _diag_tensor("A", da)
        tb = _diag_tensor("B", db)
        h = cfl.Hamiltonian([ta, tb], label="H%d" % i)
        h.set_coeff({"A": 1.0, "B": 1.0})
        h_list.append(h)
    fit = cfl.MHFit(["A", "B"], h_list, list(weights), list(ex_per_h))
    return fit, h_list


def test_mhfit_fd_jacobian_concatenates_per_h_blocks():
    da0 = np.array([0.0, 1.0, 2.0])
    db0 = np.array([0.5, 0.25, 0.125])
    da1 = np.array([0.0, 5.0])
    db1 = np.array([0.3, 0.1])
    ex0 = cfl.ExData(np.array([[1, 0.0], [2, 0.0]]))
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 0.0]]))
    fit, h_list = _make_mhfit([da0, da1], [db0, db1], [ex0, ex1], [1.0, 1.0])

    J = fit.fd_jacobian(check_swaps=False)
    assert J.shape == (4, 2)
    e0 = da0 + db0
    order0 = np.argsort(e0)
    e1 = da1 + db1
    order1 = np.argsort(e1)
    expected_a = np.concatenate([da0[order0][:2], da1[order1][:2]])
    expected_b = np.concatenate([db0[order0][:2], db1[order1][:2]])
    np.testing.assert_allclose(J[:, 0], expected_a, atol=1e-7)
    np.testing.assert_allclose(J[:, 1], expected_b, atol=1e-7)


def test_mhfit_fd_jacobian_restores_state():
    da0 = np.array([0.0, 1.0])
    db0 = np.array([2.0, 1.0])
    da1 = np.array([0.0, 3.0])
    db1 = np.array([1.0, 0.0])
    ex0 = cfl.ExData(np.array([[1, 0.0]]))
    ex1 = cfl.ExData(np.array([[1, 0.0]]))
    fit, h_list = _make_mhfit([da0, da1], [db0, db1], [ex0, ex1], [1.0, 2.0])
    coeff_before = dict(fit.coeff)
    x0_before = fit.x0.copy()
    h_coeffs_before = [dict(h.coeff_dict) for h in h_list]
    fit.fd_jacobian(check_swaps=False)
    assert fit.coeff == coeff_before
    np.testing.assert_array_equal(fit.x0, x0_before)
    for h, before in zip(h_list, h_coeffs_before):
        assert h.coeff_dict == before
