"""Tests for the optional Hamiltonian.label attribute."""

from __future__ import annotations

import numpy as np
import pytest

from pycf import cfl


def _state_labels(n):
    return cfl.StateLabels("LJM", [[0, 0, 2 * i] for i in range(n)])


def _tensor(name="DIAG", n=2):
    row_ptr = np.array([0, 1, 2], dtype=np.intc)
    col_in = np.array([0, 1], dtype=np.intc)
    val = np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex128)
    return cfl.Tensor(name, row_ptr, col_in, val, _state_labels(n))


def test_default_label_is_none():
    h = cfl.Hamiltonian([_tensor()])
    assert h.label is None


def test_constructor_accepts_label():
    h = cfl.Hamiltonian([_tensor()], label="Ground state")
    assert h.label == "Ground state"


def test_label_is_writable():
    h = cfl.Hamiltonian([_tensor()])
    h.label = "After fit"
    assert h.label == "After fit"
    h.label = None
    assert h.label is None


def test_non_string_label_raises():
    with pytest.raises(TypeError):
        cfl.Hamiltonian([_tensor()], label=42)


def test_label_is_keyword_only():
    # Positional second argument must remain forbidden so future
    # positional additions don't collide.
    with pytest.raises(TypeError):
        cfl.Hamiltonian([_tensor()], "Ground state")
