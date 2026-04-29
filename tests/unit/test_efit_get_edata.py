"""Tests for EFit.get_edata() (S3)."""

from __future__ import annotations

import numpy as np
import pytest

from pycf import cfl
from pycf.cfl_util import EData


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name, diag, n=None):
    """Return a Tensor whose matrix is diagonal(diag)."""
    n = len(diag) if n is None else n
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def _make_fit_data(diag_vals, ex_data, params=("EAVG",), label=None):
    """Build an EFit with one diagonal tensor scaled by 1.0."""
    t = _diag_tensor("EAVG", diag_vals)
    h = cfl.Hamiltonian([t], label=label) if label else cfl.Hamiltonian([t])
    h.set_coeff({"EAVG": 1.0})
    fit = cfl.EFit(list(params), h, ex_data)
    return fit, h


def test_get_edata_absolute_only():
    # 4 levels with eigenvalues 0,10,20,30 — already sorted.
    diag = np.array([0.0, 10.0, 20.0, 30.0])
    ex_arr = np.array(
        [[1, 0.0],
         [2, 11.0],
         [4, 31.0]]
    )
    ex = cfl.ExData(ex_arr)
    fit, _h = _make_fit_data(diag, ex, label="Ground")

    edata = fit.get_edata()
    assert isinstance(edata, EData)
    assert len(edata) == 3
    assert all(edata.arr["kind"] == "A")
    assert list(edata.arr["i_lo"]) == [1, 2, 4]
    assert all(edata.arr["i_hi"] == 0)
    np.testing.assert_allclose(edata.arr["e_calc"], [0.0, 10.0, 30.0])
    np.testing.assert_allclose(edata.arr["e_obs"], [0.0, 11.0, 31.0])
    np.testing.assert_allclose(edata.arr["residual"], [0.0, -1.0, -1.0])
    assert all(edata.arr["h_index"] == 0)
    assert all(s == "Ground" for s in edata.arr["h_label"])
    # chi2 = sum w * residual^2 = 0 + 1 + 1 = 2 with unit weights.
    assert edata.chi2() == pytest.approx(2.0)


def test_get_edata_difference_only():
    diag = np.array([0.0, 5.0, 12.0, 20.0])
    # Differences: 2-1, 3-1, 4-2 expected = 5, 12, 15.
    diff = np.array(
        [[1, 2, 5.5],
         [1, 3, 12.0],
         [2, 4, 15.5]]
    )
    ex = cfl.ExData(diff, key="D")
    fit, _h = _make_fit_data(diag, ex)

    edata = fit.get_edata()
    assert len(edata) == 3
    assert all(edata.arr["kind"] == "D")
    assert list(edata.arr["i_lo"]) == [1, 1, 2]
    assert list(edata.arr["i_hi"]) == [2, 3, 4]
    np.testing.assert_allclose(edata.arr["e_calc"], [5.0, 12.0, 15.0])
    np.testing.assert_allclose(edata.arr["e_obs"], [5.5, 12.0, 15.5])
    # All h_label fall back to 'H[0]' since no label was set.
    assert all(s == "H[0]" for s in edata.arr["h_label"])


def test_get_edata_difference_uses_abs():
    """e_calc must use abs(w[hi-1]-w[lo-1]) to match cfl_h_fit.c fabs.

    With lo > hi, the signed difference is negative; the absolute value
    is what the C objective squares.
    """
    diag = np.array([0.0, 5.0, 12.0])
    # i_lo=2, i_hi=1: signed w[0]-w[1] = -5 -> abs(-5) = 5.
    diff = np.array([[2, 1, 5.0]])
    ex = cfl.ExData(diff, key="D")
    fit, _h = _make_fit_data(diag, ex)

    edata = fit.get_edata()
    assert int(edata.arr["i_lo"][0]) == 2
    assert int(edata.arr["i_hi"][0]) == 1
    np.testing.assert_allclose(edata.arr["e_calc"], [5.0])
    np.testing.assert_allclose(edata.arr["residual"], [0.0])


def test_get_edata_mixed_a_and_d_ordering():
    diag = np.array([0.0, 5.0, 12.0, 20.0])
    abs_data = np.array([[1, 0.5], [3, 12.5]])
    diff_data = np.array([[1, 2, 5.0], [2, 4, 15.5]])
    ex = cfl.ExData((abs_data, diff_data), key=("A", "D"))
    fit, _h = _make_fit_data(diag, ex)

    edata = fit.get_edata()
    # n_a==2 then n_d==2.
    assert len(edata) == 4
    assert list(edata.arr["kind"]) == ["A", "A", "D", "D"]
    assert list(edata.arr["i_lo"]) == [1, 3, 1, 2]
    assert list(edata.arr["i_hi"]) == [0, 0, 2, 4]
    np.testing.assert_allclose(edata.arr["e_calc"], [0.0, 12.0, 5.0, 15.0])
    np.testing.assert_allclose(edata.arr["e_obs"], [0.5, 12.5, 5.0, 15.5])


def test_get_edata_chi2_matches_efit_eval():
    diag = np.array([0.0, 10.0, 20.0, 30.0])
    ex_arr = np.array([[1, 0.5], [2, 10.5], [3, 22.0]])
    ex = cfl.ExData(ex_arr)
    fit, _h = _make_fit_data(diag, ex)

    edata = fit.get_edata()
    chi2_eval = float(fit.eval({})[0])
    assert edata.chi2() == pytest.approx(chi2_eval, rel=1e-12)


def test_get_edata_state_label_data_raises():
    diag = np.array([0.0, 5.0])
    # State-label-indexed AS data triggers sl_index==1 inside ExData.
    abs_sl = np.array([[0, 0, 0, 0.0],
                       [0, 0, 2, 5.0]], dtype=np.float64)
    ex = cfl.ExData(abs_sl, key="AS", label_key="LJM")
    fit, _h = _make_fit_data(diag, ex)

    with pytest.raises(NotImplementedError, match="state-label"):
        fit.get_edata()


def test_get_edata_uses_hamiltonian_label():
    diag = np.array([0.0, 5.0])
    ex_arr = np.array([[1, 0.0], [2, 4.5]])
    ex = cfl.ExData(ex_arr)
    fit, h = _make_fit_data(diag, ex, label="Site A")
    assert fit.get_edata().arr["h_label"][0] == "Site A"

    # Falls back to H[0] when label is None.
    h.label = None
    assert fit.get_edata().arr["h_label"][0] == "H[0]"
