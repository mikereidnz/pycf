"""Tests for MHFit.get_edata() (S4)."""

from __future__ import annotations

import numpy as np
import pytest

from pycf import cfl
from pycf.cfl_util import EData


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name, diag):
    n = len(diag)
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def _make_ham(name, diag, label=None):
    t = _diag_tensor(name, diag)
    h = cfl.Hamiltonian([t], label=label) if label else cfl.Hamiltonian([t])
    h.set_coeff({name: 1.0})
    return h


def test_mhfit_get_edata_concatenates_in_h_order():
    h0 = _make_ham("EAVG", [0.0, 5.0, 10.0], label="Site A")
    h1 = _make_ham("EAVG", [0.0, 7.0, 14.0], label="Site B")
    ex0 = cfl.ExData(np.array([[1, 0.0], [3, 10.5]]))  # 2 abs rows
    ex1 = cfl.ExData(np.array([[2, 7.5], [3, 13.5]]))  # 2 abs rows
    fit = cfl.MHFit(["EAVG"], [h0, h1], [1.0, 1.0], [ex0, ex1])

    edata = fit.get_edata()
    assert isinstance(edata, EData)
    assert len(edata) == 4
    assert list(edata.arr["h_index"]) == [0, 0, 1, 1]
    assert list(edata.arr["h_label"]) == ["Site A", "Site A", "Site B", "Site B"]
    assert list(edata.arr["i_lo"]) == [1, 3, 2, 3]
    np.testing.assert_allclose(edata.arr["e_calc"], [0.0, 10.0, 7.0, 14.0])
    np.testing.assert_allclose(edata.arr["e_obs"], [0.0, 10.5, 7.5, 13.5])


def test_mhfit_get_edata_chi2_matches_eval():
    # Use distinct per-Hamiltonian weights to confirm scaling is applied.
    h0 = _make_ham("EAVG", [0.0, 5.0, 10.0])
    h1 = _make_ham("EAVG", [0.0, 7.0, 14.0])
    ex0 = cfl.ExData(np.array([[1, 0.5], [3, 10.0]]))
    ex1 = cfl.ExData(np.array([[2, 6.0], [3, 14.5]]))
    weights = [2.0, 0.5]
    fit = cfl.MHFit(["EAVG"], [h0, h1], weights, [ex0, ex1])

    edata = fit.get_edata()
    chi2_per_h = fit.eval({})
    assert edata.chi2() == pytest.approx(float(chi2_per_h.sum()), rel=1e-12)


def test_mhfit_get_edata_mixed_a_d_per_hamiltonian():
    h0 = _make_ham("EAVG", [0.0, 5.0, 12.0, 20.0])
    h1 = _make_ham("EAVG", [0.0, 8.0, 18.0])
    # h0: 1 abs + 1 diff; h1: 2 abs.
    ex0 = cfl.ExData(
        (np.array([[1, 0.0]]), np.array([[2, 4, 14.5]])),
        key=("A", "D"),
    )
    ex1 = cfl.ExData(np.array([[2, 8.5], [3, 17.5]]))
    fit = cfl.MHFit(["EAVG"], [h0, h1], [1.0, 1.0], [ex0, ex1])

    edata = fit.get_edata()
    assert len(edata) == 4
    assert list(edata.arr["kind"]) == ["A", "D", "A", "A"]
    np.testing.assert_allclose(edata.arr["e_calc"], [0.0, 15.0, 8.0, 18.0])


def test_mhfit_get_edata_state_label_mixed_with_index():
    """Mix AS-mode (sl_index=1) and A-mode (sl_index=0) ExData per Hamiltonian."""
    h0 = _make_ham("EAVG", [0.0, 5.0])
    abs_sl = np.array([[0, 0, 0, 0.0], [0, 0, 2, 5.0]], dtype=np.float64)
    ex0 = cfl.ExData(abs_sl, key="AS", label_key="LJM")
    h1 = _make_ham("EAVG", [0.0, 5.0])
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 5.0]]))
    fit = cfl.MHFit(["EAVG"], [h0, h1], [1.0, 1.0], [ex0, ex1])

    edata = fit.get_edata()
    assert list(edata.arr["kind"]) == ["AS", "AS", "A", "A"]
    assert list(edata.arr["i_lo"]) == [1, 2, 1, 2]
    np.testing.assert_allclose(edata.arr["e_calc"], [0.0, 5.0, 0.0, 5.0])
    assert edata.chi2() == pytest.approx(float(fit.eval({}).sum()), rel=1e-12)


def test_mhfit_get_edata_per_h_weight_applied():
    h0 = _make_ham("EAVG", [0.0, 5.0])
    h1 = _make_ham("EAVG", [0.0, 5.0])
    ex0 = cfl.ExData(np.array([[1, 0.0], [2, 4.0]]))  # residual = (0, +1)
    ex1 = cfl.ExData(np.array([[1, 0.0], [2, 6.0]]))  # residual = (0, -1)
    fit = cfl.MHFit(["EAVG"], [h0, h1], [3.0, 7.0], [ex0, ex1])

    edata = fit.get_edata()
    # weights should be 3.0 for first H (rows 0,1) and 7.0 for second (2,3).
    np.testing.assert_allclose(edata.arr["weight"], [3.0, 3.0, 7.0, 7.0])
    # chi2 = 3*1^2 + 7*1^2 = 10.
    assert edata.chi2() == pytest.approx(10.0)


def test_mhfit_get_edata_falls_back_to_h_index_label():
    h0 = _make_ham("EAVG", [0.0])
    h1 = _make_ham("EAVG", [0.0])
    ex = cfl.ExData(np.array([[1, 0.0]]))
    fit = cfl.MHFit(["EAVG"], [h0, h1], [1.0, 1.0], [ex, ex])
    edata = fit.get_edata()
    assert list(edata.arr["h_label"]) == ["H[0]", "H[1]"]
