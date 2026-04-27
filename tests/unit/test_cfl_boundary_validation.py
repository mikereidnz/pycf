"""Boundary validation tests for the Cython/C CFL interface."""

import numpy as np
import pytest

from pycf import cfl


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _tensor(name="T", n=2):
    row_ptr = np.array([0, 1, 2], dtype=np.intc)
    col_in = np.array([0, 1], dtype=np.intc)
    val = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def test_tensor_rejects_empty_row_ptr():
    with pytest.raises(ValueError, match="row_ptr"):
        cfl.Tensor(
            "BAD",
            np.array([], dtype=np.intc),
            np.array([], dtype=np.intc),
            np.array([], dtype=np.complex128),
            _state_labels(0),
        )


def test_tensor_rejects_inconsistent_csr_lengths():
    with pytest.raises(ValueError, match="lengths"):
        cfl.Tensor(
            "BAD",
            np.array([0, 2], dtype=np.intc),
            np.array([0], dtype=np.intc),
            np.array([1.0 + 0.0j], dtype=np.complex128),
            _state_labels(1),
        )


def test_tensor_rejects_out_of_range_column_index():
    with pytest.raises(ValueError, match="column indices"):
        cfl.Tensor(
            "BAD",
            np.array([0, 1], dtype=np.intc),
            np.array([1], dtype=np.intc),
            np.array([1.0 + 0.0j], dtype=np.complex128),
            _state_labels(1),
        )


def test_hamiltonian_rejects_empty_tensor_list():
    with pytest.raises(ValueError, match="at least one Tensor"):
        cfl.Hamiltonian([])


def test_hamiltonian_rejects_dimension_mismatch():
    t1 = _tensor("T1", n=2)
    t2 = cfl.Tensor(
        "T2",
        np.array([0, 1, 2, 3], dtype=np.intc),
        np.array([0, 1, 2], dtype=np.intc),
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128),
        _state_labels(3),
    )

    with pytest.raises(ValueError, match="same dimension"):
        cfl.Hamiltonian([t1, t2])
