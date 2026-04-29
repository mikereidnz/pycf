"""Tests for the Hamiltonian-label heading on gen_e_summary (S7)."""

from __future__ import annotations

import numpy as np

from pycf import cfl, cfl_util


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _diag_tensor(name, diag):
    n = len(diag)
    row_ptr = np.arange(n + 1, dtype=np.intc)
    col_in = np.arange(n, dtype=np.intc)
    val = np.asarray(diag, dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def test_gen_e_summary_h_label_kwarg():
    w = np.array([0.0, 1.0, 2.0])
    z = np.eye(3)
    labels = [[0, 0, 0], [0, 0, 2], [0, 0, 4]]
    s = cfl_util.gen_e_summary(w, z, labels, "LJM", h_label="Site 1")
    assert "Hamiltonian: Site 1" in s

    s_no = cfl_util.gen_e_summary(w, z, labels, "LJM")
    assert "Hamiltonian:" not in s_no


def test_hamiltonian_e_summary_uses_label():
    t = _diag_tensor("A", [0.0, 1.0, 2.0])
    h = cfl.Hamiltonian([t])
    h.set_coeff({"A": 1.0})
    h.label = "B || c"
    h.diag()
    s = h.gen_summary()
    assert "Hamiltonian: B || c" in s

    h2 = cfl.Hamiltonian([t])
    h2.set_coeff({"A": 1.0})
    h2.diag()
    s2 = h2.gen_summary()
    assert "Hamiltonian:" not in s2
