"""Unit tests for :class:`pycf.import_sljm.ImportTensors`.

These tests build small Hamiltonians directly from numpy / scipy matrices,
without going through any of the legacy jmcalc text files, exercising the
generic in-memory path that ``ImportTensors`` provides.
"""

import contextlib
import io

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, triu

import pycf
import pycf.cfl as cfl
from pycf.import_sljm import ImportTensors

# ---------------------------------------------------------------------------
# Fixtures: small Hermitian operators
# ---------------------------------------------------------------------------

# Pauli matrices acting on |+1/2>, |-1/2>; M labels are doubled, so M = +1, -1.
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
STATES_2 = np.array([[1], [-1]], dtype=np.int32)

# A 4x4 Hermitian operator for shape tests.
_DIAG4 = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
_OFF4 = np.zeros((4, 4), dtype=complex)
_OFF4[0, 1] = 0.5
_OFF4[1, 0] = 0.5  # Hermitian
_OFF4[2, 3] = 0.7j
_OFF4[3, 2] = -0.7j  # Hermitian
H4 = _DIAG4 + _OFF4
STATES_4 = np.array([[2, 1], [2, -1], [4, 1], [4, -1]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Construction and storage convention
# ---------------------------------------------------------------------------


def test_construct_from_dense_full():
    it = ImportTensors("M", STATES_2, {"SX": SX, "SZ": SZ})
    assert set(it.tensors) == {"SX", "SZ"}
    # ImportTensors stores only the upper triangle (Hermitian-CSR convention
    # inherited from the underlying C layer). get_matel() returns that upper
    # triangle verbatim; consumers that need the full Hermitian completion
    # do it themselves.
    assert np.allclose(it.SX.get_matel(), np.triu(SX))
    assert np.allclose(it.SZ.get_matel(), np.triu(SZ))


def test_construct_from_full_sparse():
    sx_sp = csr_matrix(SX)
    it = ImportTensors("M", STATES_2, {"SX": sx_sp})
    assert np.allclose(it.SX.get_matel(), np.triu(SX))


def test_construct_from_upper_csr():
    sx_upper = triu(csr_matrix(SX), format="csr")
    it = ImportTensors("M", STATES_2, {"SX": sx_upper})
    assert np.allclose(it.SX.get_matel(), np.triu(SX))


def test_full_and_upper_inputs_agree():
    """Same operator passed as full and as upper-only gives identical storage."""
    it_full = ImportTensors("M", STATES_2, {"SX": SX})
    sx_upper = triu(csr_matrix(SX), format="csr")
    it_upper = ImportTensors("M", STATES_2, {"SX": sx_upper})
    assert np.allclose(it_full.SX.get_matel(), it_upper.SX.get_matel())


def test_accepts_non_csr_sparse():
    sx_coo = coo_matrix(SX)
    it = ImportTensors("M", STATES_2, {"SX": sx_coo})
    assert np.allclose(it.SX.get_matel(), np.triu(SX))


def test_mixed_dense_and_sparse():
    it = ImportTensors("M", STATES_2, {"SX": SX, "SZ": csr_matrix(SZ)})
    assert np.allclose(it.SX.get_matel(), np.triu(SX))
    assert np.allclose(it.SZ.get_matel(), np.triu(SZ))


# ---------------------------------------------------------------------------
# End-to-end: synthetic Hamiltonian eigenvalues
# ---------------------------------------------------------------------------


def test_synthetic_hamiltonian_eigenvalues_match_numpy():
    """Diagonalised ImportTensors output matches numpy.linalg.eigvalsh."""
    it = ImportTensors("M", STATES_2, {"SX": SX, "SY": SY, "SZ": SZ})
    h = cfl.Hamiltonian([it.SX, it.SY, it.SZ])
    bx, by, bz = 0.3, -0.5, 0.7
    h.set_coeff({"SX": bx, "SY": by, "SZ": bz})
    w, _ = h.diag()
    expected = np.linalg.eigvalsh(bx * SX + by * SY + bz * SZ)
    assert np.allclose(np.sort(w), np.sort(expected), atol=1e-10)


def test_synthetic_hamiltonian_4x4():
    it = ImportTensors("JM", STATES_4, {"H": H4})
    h = cfl.Hamiltonian([it.H])
    h.set_coeff({"H": 1.0})
    w, _ = h.diag()
    expected = np.linalg.eigvalsh(H4)
    assert np.allclose(np.sort(w), np.sort(expected), atol=1e-10)


# ---------------------------------------------------------------------------
# Eigenvector phase-matched comparison vs numpy
#
# Mirrors tests/integration/spin-half/test_spin-half.py: build a 1S spin-half
# Hamiltonian via ImportTensors and check that real, imaginary, and complex
# coefficient combinations all reproduce numpy.linalg.eig() eigenvectors
# after a per-column phase normalisation. The phase reference is the
# component of largest magnitude in each column, which is robust when some
# components are exactly zero.
#
# Historical note: the original spin-half test was written to catch a bug
# in which cfl was effectively diagonalising the transpose of the supplied
# matrix; the fix was a complex conjugation added to
# ``solve_hermitian_block`` in ``cfl/src/cfl_h.c``. The imaginary and complex
# parametrisations exercise exactly the path that bug affected, so this
# test acts as a regression guard for that conjugation. Non-degenerate
# eigenvalues are sufficient — degenerate eigenspaces would require a
# projector-based comparison and are not needed for this regression.
# ---------------------------------------------------------------------------


def _norm_eig_largest(z, tol=1e-10):
    """Phase-match each eigenvector using its largest-magnitude component.

    Returns a copy of `z` with every column rotated so that its anchor
    component is positive real. The anchor is the *first* index whose
    magnitude is within `tol` of the column's maximum magnitude; this
    deterministic tie-breaking is essential when several components have
    equal magnitudes (e.g. Pauli eigenvectors), where bare ``argmax``
    would pick different indices in pycf vs numpy due to floating-point
    noise.
    """
    z_n = np.zeros_like(z)
    for j in range(z.shape[1]):
        col = z[:, j]
        absc = np.abs(col)
        m = absc.max()
        k = int(np.argmax(absc >= m - tol))
        y = col[k]
        z_n[:, j] = np.conj(y) * col / np.abs(y)
    return z_n


def _sorted_phase_matched(w, z):
    """Sort eigenpairs by ascending energy, then phase-match eigenvectors."""
    w = np.asarray(w).real
    idx = np.argsort(w)
    return w[idx], _norm_eig_largest(z[:, idx])


@pytest.mark.parametrize("data_sel", ["real", "imag", "complex"])
def test_spin_half_eigenvectors_match_numpy(data_sel):
    """ImportTensors round-trip for a spin-half system with real, imaginary,
    and complex Hamiltonians.

    The test constructs SX, SY, SZ from numpy arrays via ImportTensors,
    builds a cfl.Hamiltonian, sets coefficients selecting a real, imaginary,
    or complex linear combination, diagonalises, and compares both
    eigenvalues and (phase-matched) eigenvectors against numpy.linalg.eig
    of the same dense matrix.
    """
    it = ImportTensors("M", STATES_2, {"SX": SX, "SY": SY, "SZ": SZ})
    h = cfl.Hamiltonian([it.SX, it.SY, it.SZ])

    coeff = {"SX": 0.0, "SY": 0.0, "SZ": 0.0}
    if data_sel == "real":
        coeff["SX"] = 1.0
    elif data_sel == "imag":
        coeff["SY"] = 1.0
    elif data_sel == "complex":
        coeff["SX"] = 1.0
        coeff["SY"] = 1.0

    h.set_coeff(coeff)
    w_pycf, z_pycf = h.diag()

    H_dense = coeff["SX"] * SX + coeff["SY"] * SY + coeff["SZ"] * SZ
    w_np, z_np = np.linalg.eig(H_dense)

    w_pycf, z_pycf = _sorted_phase_matched(w_pycf, z_pycf)
    w_np, z_np = _sorted_phase_matched(w_np, z_np)

    assert np.allclose(w_pycf, w_np, atol=1e-10)
    assert np.allclose(z_pycf, z_np, atol=1e-10)


# ---------------------------------------------------------------------------
# Alias synthesis
# ---------------------------------------------------------------------------


def test_aliases_off_by_default():
    it = ImportTensors("M", STATES_2, {"MAG10": SZ, "MAG11": SX})
    assert "MAGX" not in it.tensors
    assert "MAGY" not in it.tensors
    assert "MAGZ" not in it.tensors


def test_aliases_synthesised_when_enabled():
    it = ImportTensors("M", STATES_2, {"MAG10": SZ, "MAG11": SX}, add_aliases=True)
    assert {"MAGX", "MAGY", "MAGZ"} <= set(it.tensors)
    assert it.MAGZ.name == "MAGZ"
    # MAGX = -1/sqrt(2) * MAG11
    expected_magx = (-1.0 / np.sqrt(2)) * SX
    assert np.allclose(it.MAGX.get_matel(), np.triu(expected_magx))


def test_hyp_alias_synthesised():
    it = ImportTensors("M", STATES_2, {"AHYP": SX, "BHYP": SZ}, add_aliases=True)
    assert "HYP" in it.tensors
    expected = SX - np.sqrt(10) * SZ
    assert np.allclose(it.HYP.get_matel(), np.triu(expected))


def test_alias_collision_raises():
    with pytest.raises(ValueError, match="overwrite"):
        ImportTensors(
            "M",
            STATES_2,
            {"MAG10": SZ, "MAG11": SX, "MAGX": SX},
            add_aliases=True,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_label_key_rejected():
    with pytest.raises(ValueError, match="label_key"):
        ImportTensors("", STATES_2, {"SX": SX})


def test_states_wrong_columns_rejected():
    bad_states = np.array([[1, 1], [-1, -1]], dtype=np.int32)
    with pytest.raises(ValueError, match="label_key"):
        ImportTensors("M", bad_states, {"SX": SX})


def test_states_3d_rejected():
    bad = np.zeros((2, 1, 1), dtype=np.int32)
    with pytest.raises(ValueError, match="2-D"):
        ImportTensors("M", bad, {"SX": SX})


def test_states_1d_special_case_for_single_label():
    """A 1-D states array is allowed only when label_key has one character."""
    states_1d = np.array([1, -1], dtype=np.int32)
    it = ImportTensors("M", states_1d, {"SX": SX})
    assert np.allclose(it.SX.get_matel(), np.triu(SX))


def test_empty_states_rejected():
    with pytest.raises(ValueError, match="at least one row"):
        ImportTensors("M", np.zeros((0, 1), dtype=np.int32), {"SX": SX})


def test_non_square_tensor_rejected():
    bad = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="square"):
        ImportTensors("M", STATES_2, {"X": bad})


def test_tensor_shape_mismatch_rejected():
    bad = np.zeros((3, 3), dtype=complex)
    with pytest.raises(ValueError, match="shape"):
        ImportTensors("M", STATES_2, {"X": bad})


def test_non_hermitian_dense_accepted_lower_dropped():
    """Non-Hermitian dense input is accepted; the strict lower triangle is
    silently dropped. This documents the upper-triangle-only storage
    contract (callers responsible for the input being meaningful)."""
    bad = np.array([[0, 1], [2, 0]], dtype=complex)
    it = ImportTensors("M", STATES_2, {"X": bad})
    expected = np.array([[0, 1], [0, 0]], dtype=complex)
    assert np.allclose(it.X.get_matel(), expected)


# ---------------------------------------------------------------------------
# Round-trip through get_matel(): the upper triangle of the input is
# preserved verbatim.
#
# The underlying Hermitian-CSR storage (zhcsr_alloc in cfl/src/cfl_csr.c)
# discards strictly-lower entries by design — only upper-triangular
# content is stored. A strictly-upper input therefore round-trips
# faithfully, while a strictly-lower input is lost. Crystal-field T_kq
# tensors with q<0 (which physically live below the diagonal) must be
# folded into the upper triangle before being passed to ImportTensors;
# this is what ImportSLJM does internally for SLJM matrix-element files.
# ---------------------------------------------------------------------------


def test_strictly_upper_tensor_round_trips():
    """A q>0 tensor (upper-triangle only) survives the import round-trip."""
    t_plus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    it = ImportTensors("M", STATES_2, {"TPLUS": t_plus})
    assert np.allclose(it.TPLUS.get_matel(), t_plus)


def test_tensors_must_be_mapping():
    with pytest.raises(TypeError, match="mapping"):
        ImportTensors("M", STATES_2, [SX])


def test_empty_tensor_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        ImportTensors("M", STATES_2, {"": SX})


# ---------------------------------------------------------------------------
# Reserved names and expose_attrs
# ---------------------------------------------------------------------------


def test_reserved_name_rejected_when_exposing_attrs():
    with pytest.raises(ValueError, match="reserved"):
        ImportTensors("M", STATES_2, {"tensors": SX})


def test_reserved_name_allowed_when_not_exposing_attrs():
    it = ImportTensors("M", STATES_2, {"tensors": SX}, expose_attrs=False)
    # The user-supplied "tensors" key lives in the dict, but self.tensors
    # is still the dict mapping (not shadowed).
    assert isinstance(it.tensors, dict)
    assert "tensors" in it.tensors


# ---------------------------------------------------------------------------
# Zero-tensor warning
# ---------------------------------------------------------------------------


def test_zero_tensor_warning():
    zero = np.zeros((2, 2), dtype=complex)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ImportTensors("M", STATES_2, {"Z": zero})
    assert "all matrix elements of Z are zero" in buf.getvalue()


def test_zero_tensor_warning_suppressed():
    zero = np.zeros((2, 2), dtype=complex)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ImportTensors("M", STATES_2, {"Z": zero}, warn_zero=False)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------


def test_import_tensors_reexported_from_pycf():
    assert pycf.ImportTensors is ImportTensors


def test_iteration_and_print_names(capsys):
    it = ImportTensors("M", STATES_2, {"SX": SX, "SZ": SZ})
    yielded = [t.name for t in it]
    assert set(yielded) == {"SX", "SZ"}
    it.print_names()
    captured = capsys.readouterr().out
    assert "SX" in captured and "SZ" in captured
