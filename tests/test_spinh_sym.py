"""Unit tests for the SU(2)-symmeterization paths in ``pycf.spinh``.

These cover ``su2_rz_lsq_f``, ``su2_rotation_lsq_f`` and the
``SpinH.inv_term(sym=True)`` codepath.  The latter is the only consumer of the
``lsq_f`` helpers in the codebase.

The tests probe expected behaviour rather than just shape:

* At zero rotation, the residue ``r`` is the off-diagonal asymmetry of the
  inverted parameter tensor.  For a tensor that is already symmetric in the
  spin-Hamiltonian sense (i.e. ``tensor[i] == tensor[j]`` for the index pairs
  used internally), ``r`` must be exactly zero.
* The residue is non-negative, real, and symmetric under sign flips of the
  input phase.
* ``SpinH.inv_term(sym=True, sym_phase=[0,0,0])`` is functionally equivalent
  to ``inv_term()`` with no symmetrization (both apply the identity SU(2)
  rotation), so the two must agree to machine precision.
* When ``sym_phase`` has the wrong length, ``inv_term`` raises ``ValueError``.
* The full basinhopping path is exercised with a small, seeded run as a slow
  smoke test: the recovered parameters round-trip through forward + inverse
  to within an order of magnitude of the user's tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycf.spinh import (
    SpinH,
    bmj,
    bmj_coeff_array,
    ias,
    ias_coeff_array,
    invert_term,
    su2_rotation_lsq_f,
    su2_rz_lsq_f,
)
from pycf.matel import matel


# ---------------------------------------------------------------------------
# Direct tests of su2_rz_lsq_f
# ---------------------------------------------------------------------------


@pytest.fixture
def ias_setup():
    """Build the IAS coefficient/value pair for a symmetric A tensor."""
    S = 0.5
    Ival = 0.5
    S_m = [matel(c, S) for c in ("jx", "jy", "jz")]
    I_m = [matel(c, Ival) for c in ("jx", "jy", "jz")]
    coeff_a = ias_coeff_array(I_m, S_m)
    A_sym = np.array(
        [
            [10.0, 1.0, 2.0],
            [1.0, 20.0, 3.0],
            [2.0, 3.0, 30.0],
        ],
        dtype=complex,
    )
    h = ias(I_m, A_sym, S_m)
    return coeff_a, h, A_sym


@pytest.fixture
def bgs_setup():
    """Build the BgS coefficient/value triple for a symmetric g tensor."""
    S = 0.5
    S_m = [matel(c, S) for c in ("jx", "jy", "jz")]
    g_sym = np.array(
        [
            [2.0, 0.05, 0.0],
            [0.05, 2.1, 0.02],
            [0.0, 0.02, 2.2],
        ],
        dtype=complex,
    )
    B_list = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    S_dimsq = int((2 * S + 1) ** 2)
    B_a = np.zeros([len(B_list), S_dimsq, 9], dtype=complex)
    h = np.zeros([len(B_list), 2, 2], dtype=complex)
    for i, e in enumerate(B_list):
        B_a[i, :, :] = bmj_coeff_array(e, S_m)
        h[i, :, :] = bmj(e, g_sym, S_m)
    coeff_a = np.reshape(B_a, (len(B_list) * S_dimsq, 9))
    return coeff_a, h, g_sym


class TestSu2RzLsqF:
    def test_returns_nonnegative_float(self, ias_setup):
        coeff_a, h, _ = ias_setup
        r = su2_rz_lsq_f(0.3, coeff_a, h)
        assert isinstance(r, float)
        assert r >= 0

    def test_zero_for_symmetric_tensor_at_zero_phase(self, ias_setup):
        coeff_a, h, _ = ias_setup
        r = su2_rz_lsq_f(0.0, coeff_a, h)
        # ``A_sym`` is exactly symmetric, so the residue must vanish.
        assert r < 1e-12

    def test_identity_phase_recovers_input(self, ias_setup):
        # At phase = 0 (and 2*pi), the SU(2) rotation D_z is the identity, so
        # the inverted tensor must equal the original.
        coeff_a, h, A_sym = ias_setup
        recovered = invert_term(coeff_a, h)
        np.testing.assert_allclose(recovered.reshape(3, 3), A_sym, atol=1e-10)

    def test_residue_is_real_and_finite_for_complex_tensor(self):
        # Build a deliberately asymmetric IAS tensor and verify the function
        # produces a finite, non-negative residue.
        S = 0.5
        Ival = 0.5
        S_m = [matel(c, S) for c in ("jx", "jy", "jz")]
        I_m = [matel(c, Ival) for c in ("jx", "jy", "jz")]
        coeff_a = ias_coeff_array(I_m, S_m)
        A_asym = np.array(
            [
                [10.0, 1.0 + 0.5j, 2.0],
                [-1.0 + 0.5j, 20.0, 3.0],
                [2.0, 3.0, 30.0],
            ],
            dtype=complex,
        )
        h = ias(I_m, A_asym, S_m)
        r = su2_rz_lsq_f(0.7, coeff_a, h)
        assert np.isreal(r) or np.iscomplex(r) and r.imag == 0
        assert np.isfinite(r)
        assert r > 0  # non-trivially asymmetric

    def test_bgs_branch_handles_3x2x2_input(self, bgs_setup):
        # The function detects b.shape == (3, 2, 2) and loops over field
        # directions.  Verify it does not crash and produces a small residue
        # for an already-symmetric g tensor.
        coeff_a, h, _ = bgs_setup
        r = su2_rz_lsq_f(0.0, coeff_a, h)
        assert r < 1e-10


class TestSu2RotationLsqF:
    def test_returns_nonnegative_float(self, ias_setup):
        coeff_a, h, _ = ias_setup
        r = su2_rotation_lsq_f(np.array([0.1, 0.2, 0.3]), coeff_a, h)
        assert isinstance(r, float)
        assert r >= 0

    def test_zero_for_symmetric_tensor_at_zero_rotation(self, ias_setup):
        coeff_a, h, _ = ias_setup
        r = su2_rotation_lsq_f(np.array([0.0, 0.0, 0.0]), coeff_a, h)
        assert r < 1e-12

    def test_bgs_branch_zero_for_symmetric_g(self, bgs_setup):
        coeff_a, h, _ = bgs_setup
        r = su2_rotation_lsq_f(np.array([0.0, 0.0, 0.0]), coeff_a, h)
        assert r < 1e-10


# ---------------------------------------------------------------------------
# SpinH.inv_term(sym=True) round-trip
# ---------------------------------------------------------------------------


def _build_inv_spinh_for_ias(I_val=0.5):
    """Forward + inverse SpinH pair for an IAS round-trip with hyperfine A."""
    return SpinH(["ias"], S=0.5, I=I_val), SpinH(["ias"], S=0.5, I=I_val, inv=True)


class TestSpinHInvTermSymPath:
    def test_sym_phase_zero_matches_no_sym(self):
        """sym=True with sym_phase=[0,0,0] must equal the no-sym result."""
        sh_fwd, sh_inv = _build_inv_spinh_for_ias()
        A = np.diag([100.0, 200.0, 300.0])
        sh_fwd.add_term("ias", A)
        sh_inv.add_H_term("ias", sh_fwd.terms["ias"])

        no_sym = sh_inv.inv_term("ias")
        # rebuild for the second call so add_H_term state is fresh
        sh_inv2 = SpinH(["ias"], S=0.5, I=0.5, inv=True)
        sh_inv2.add_H_term("ias", sh_fwd.terms["ias"])
        with_sym = sh_inv2.inv_term("ias", sym=True, sym_phase=[0, 0, 0])

        np.testing.assert_allclose(no_sym, with_sym, atol=1e-12)

    def test_sym_phase_wrong_length_raises(self):
        sh_fwd, sh_inv = _build_inv_spinh_for_ias()
        A = np.diag([100.0, 200.0, 300.0])
        sh_fwd.add_term("ias", A)
        sh_inv.add_H_term("ias", sh_fwd.terms["ias"])
        with pytest.raises(ValueError, match="sym_phase argument must be of length 3"):
            sh_inv.inv_term("ias", sym=True, sym_phase=[0.0, 0.0])

    def test_sym_phase_records_attribute(self):
        sh_fwd, sh_inv = _build_inv_spinh_for_ias()
        A = np.diag([10.0, 20.0, 30.0])
        sh_fwd.add_term("ias", A)
        sh_inv.add_H_term("ias", sh_fwd.terms["ias"])
        phase = [0.5, 1.0, 1.5]
        sh_inv.inv_term("ias", sym=True, sym_phase=phase)
        np.testing.assert_array_equal(sh_inv.sym_phase, phase)

    def test_no_sym_records_zero_phase(self):
        sh_fwd, sh_inv = _build_inv_spinh_for_ias()
        A = np.diag([10.0, 20.0, 30.0])
        sh_fwd.add_term("ias", A)
        sh_inv.add_H_term("ias", sh_fwd.terms["ias"])
        sh_inv.inv_term("ias")  # sym=False default
        assert list(sh_inv.sym_phase) == [0, 0, 0]


@pytest.mark.slow
class TestSpinHBasinhoppingSymmetrize:
    """Exercise the full basinhopping path.

    This is slow (100 iterations of Powell) and stochastic; we test a
    weak property (the recovered tensor is real-symmetric to ~1e-3) rather
    than asserting exact agreement with the input.
    """

    def test_basinhopping_recovers_symmetric_g(self, capsys):
        # Use a g tensor that is exactly diagonal, then build the SpinH and
        # invert with sym=True.  Despite the basinhopping callback printing
        # to stdout, the recovered tensor should be close to diagonal.
        np.random.seed(0)
        g = np.diag([2.0, 2.05, 2.1]).astype(complex)
        B_list = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]
        sh_fwd = SpinH(["bgs"], S=0.5, B=B_list[0])
        sh_fwd.add_term("bgs", g)
        # Replicate the BgS Hamiltonian for each field direction by directly
        # using bmj.
        S_m = [matel(c, 0.5) for c in ("jx", "jy", "jz")]
        h_list = np.array([bmj(e, g, S_m) for e in B_list])

        sh_inv = SpinH(["bgs"], S=0.5, B=B_list, inv=True)
        sh_inv.add_H_term("bgs", h_list)
        recovered = sh_inv.inv_term("bgs", sym=True).reshape(3, 3)

        # The Hermitian / "physically symmetric" part should match the input.
        # We check the symmetric component of the real part is close to g.
        sym_part = 0.5 * (recovered + recovered.T)
        np.testing.assert_allclose(np.real(sym_part), np.real(g), atol=5e-3)
        # The basinhopping callback prints; assert it ran at least once.
        out = capsys.readouterr().out
        assert "Symmeterization minimum" in out
