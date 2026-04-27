"""Round-trip tests for pycf.spinh.SpinH.

The existing tests/eryso/test_spinh.py round-trips a g-tensor (bgs term)
through SpinH and asserts on g, but does NOT assert on the recovered A
(ias) or Q (iqi) matrices. These tests fill that gap and add coverage
for the bi (nuclear Zeeman, no inverse) error path.

A round-trip test:
1. Constructs a SpinH with a chosen term and a known parameter matrix.
2. Builds the full Hamiltonian via add_term + the public terms dict.
3. Feeds that Hamiltonian into a fresh SpinH (with inv=True) via
   add_H_term.
4. Calls inv_term to recover the original parameter matrix.
5. Asserts agreement with the original matrix to high precision.

These tests do not require f-electron physics knowledge — they exercise
the linear-algebra round-trip purely through the public SpinH API.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycf.spinh import SpinH

ROUNDTRIP_TOL = 1e-8


# ---------------------------------------------------------------------------
# bgs (Zeeman) round-trip — already covered by tests/eryso/test_spinh.py
# but repeated here as a stand-alone unit test with synthetic data.
# ---------------------------------------------------------------------------

def test_spinh_bgs_roundtrip_diagonal_g():
    """Diagonal g-tensor must round-trip exactly."""
    g = np.diag([2.0, 2.5, 1.8])
    B_list = [np.eye(3)[:, i] for i in range(3)]

    bgs = []
    for i in range(3):
        sh = SpinH(["bgs"], B=B_list[i], S=1 / 2)
        sh.add_term("bgs", g)
        bgs.append(sh.terms["bgs"])

    sh_inv = SpinH(["bgs"], B=B_list, S=1 / 2, inv=True)
    sh_inv.add_H_term("bgs", bgs)
    g_recovered = np.reshape(sh_inv.inv_term("bgs"), (3, 3))

    np.testing.assert_allclose(g_recovered, g, atol=ROUNDTRIP_TOL)


def test_spinh_bgs_roundtrip_full_g():
    """Off-diagonal symmetric g-tensor must round-trip."""
    g = np.array(
        [
            [2.92, -0.30, -0.40],
            [-0.30, 2.10, 0.55],
            [-0.40, 0.55, 1.80],
        ]
    )
    B_list = [np.eye(3)[:, i] for i in range(3)]

    bgs = []
    for i in range(3):
        sh = SpinH(["bgs"], B=B_list[i], S=1 / 2)
        sh.add_term("bgs", g)
        bgs.append(sh.terms["bgs"])

    sh_inv = SpinH(["bgs"], B=B_list, S=1 / 2, inv=True)
    sh_inv.add_H_term("bgs", bgs)
    g_recovered = np.reshape(sh_inv.inv_term("bgs"), (3, 3))

    np.testing.assert_allclose(g_recovered, g, atol=ROUNDTRIP_TOL)


# ---------------------------------------------------------------------------
# ias (hyperfine A) round-trip — NEW coverage, not asserted in existing tests.
# ---------------------------------------------------------------------------

def test_spinh_ias_roundtrip_diagonal_A():
    """Diagonal hyperfine matrix must round-trip exactly."""
    A = np.diag([100.0, 200.0, 350.0])
    sh = SpinH(["ias"], S=1 / 2, I=7 / 2)
    sh.add_term("ias", A)

    sh_inv = SpinH(["ias"], S=1 / 2, I=7 / 2, inv=True)
    sh_inv.add_H_term("ias", sh.terms["ias"])
    A_recovered = np.reshape(sh_inv.inv_term("ias"), (3, 3))

    np.testing.assert_allclose(A_recovered, A, atol=ROUNDTRIP_TOL)


def test_spinh_ias_roundtrip_full_A():
    """Full hyperfine matrix from Guillot-Noël et al., PhysRevB.74.214409.

    This is the same data as in tests/eryso/test_spinh.py, but here we
    assert on the recovered matrix (the eryso test asserts only on g).
    """
    A = np.array(
        [
            [69.35, -580.73, -248.83],
            [-580.73, 696.30, 682.49],
            [-248.83, 682.49, 495.54],
        ]
    )
    sh = SpinH(["ias"], S=1 / 2, I=7 / 2)
    sh.add_term("ias", A)

    sh_inv = SpinH(["ias"], S=1 / 2, I=7 / 2, inv=True)
    sh_inv.add_H_term("ias", sh.terms["ias"])
    A_recovered = np.reshape(sh_inv.inv_term("ias"), (3, 3))

    np.testing.assert_allclose(A_recovered, A, atol=1e-6)


@pytest.mark.parametrize("I_val", [0.5, 1.5, 2.5, 3.5])
def test_spinh_ias_roundtrip_various_I(I_val):
    """Round-trip with varying nuclear spin values."""
    A = np.array(
        [
            [50.0, 10.0, 5.0],
            [10.0, 80.0, -3.0],
            [5.0, -3.0, 120.0],
        ]
    )
    sh = SpinH(["ias"], S=1 / 2, I=I_val)
    sh.add_term("ias", A)

    sh_inv = SpinH(["ias"], S=1 / 2, I=I_val, inv=True)
    sh_inv.add_H_term("ias", sh.terms["ias"])
    A_recovered = np.reshape(sh_inv.inv_term("ias"), (3, 3))

    np.testing.assert_allclose(A_recovered, A, atol=ROUNDTRIP_TOL)


# ---------------------------------------------------------------------------
# iqi (quadrupole Q) round-trip — NEW coverage.
# ---------------------------------------------------------------------------

def test_spinh_iqi_roundtrip_diagonal_Q():
    """Diagonal quadrupole matrix must round-trip exactly."""
    Q = np.diag([15.0, -5.0, -10.0])
    sh = SpinH(["iqi"], I=7 / 2)
    sh.add_term("iqi", Q)

    sh_inv = SpinH(["iqi"], I=7 / 2, inv=True)
    sh_inv.add_H_term("iqi", sh.terms["iqi"])
    Q_recovered = np.reshape(sh_inv.inv_term("iqi"), (3, 3))

    np.testing.assert_allclose(Q_recovered, Q, atol=ROUNDTRIP_TOL)


def test_spinh_iqi_roundtrip_full_Q():
    """Full quadrupole matrix from Guillot-Noël et al., PhysRevB.74.214409."""
    Q = np.array(
        [
            [21.40, -8.18, -15.27],
            [-8.18, 3.79, 0.60],
            [-15.27, 0.60, -25.20],
        ]
    )
    sh = SpinH(["iqi"], I=7 / 2)
    sh.add_term("iqi", Q)

    sh_inv = SpinH(["iqi"], I=7 / 2, inv=True)
    sh_inv.add_H_term("iqi", sh.terms["iqi"])
    Q_recovered = np.reshape(sh_inv.inv_term("iqi"), (3, 3))

    np.testing.assert_allclose(Q_recovered, Q, atol=1e-6)


# ---------------------------------------------------------------------------
# Combined ias + iqi round-trip on full Hamiltonian (matches user-script
# usage pattern in eryso_site1_bobyqa_try_ab-initio.py).
# ---------------------------------------------------------------------------

def test_spinh_combined_ias_iqi_roundtrip():
    """Construct ias + iqi together; recover both matrices."""
    A = np.array([[50.0, 5.0, 0.0], [5.0, 80.0, 0.0], [0.0, 0.0, 120.0]])
    Q = np.array([[10.0, 0.5, 0.0], [0.5, -3.0, 0.0], [0.0, 0.0, -7.0]])

    sh = SpinH(["ias", "iqi"], S=1 / 2, I=7 / 2)
    sh.add_term("ias", A)
    sh.add_term("iqi", Q)

    sh_inv = SpinH(["ias", "iqi"], S=1 / 2, I=7 / 2, inv=True)
    sh_inv.add_H_term("ias", sh.terms["ias"])
    sh_inv.add_H_term("iqi", sh.terms["iqi"])

    A_recovered = np.reshape(sh_inv.inv_term("ias"), (3, 3))
    Q_recovered = np.reshape(sh_inv.inv_term("iqi"), (3, 3))

    np.testing.assert_allclose(A_recovered, A, atol=ROUNDTRIP_TOL)
    np.testing.assert_allclose(Q_recovered, Q, atol=ROUNDTRIP_TOL)


# ---------------------------------------------------------------------------
# get_H sanity check — full Hamiltonian must be Hermitian.
# ---------------------------------------------------------------------------

def test_spinh_get_H_is_hermitian():
    """The full Hamiltonian assembled by SpinH.get_H() must be Hermitian."""
    A = np.array(
        [
            [69.35, -580.73, -248.83],
            [-580.73, 696.30, 682.49],
            [-248.83, 682.49, 495.54],
        ]
    )
    Q = np.array(
        [
            [21.40, -8.18, -15.27],
            [-8.18, 3.79, 0.60],
            [-15.27, 0.60, -25.20],
        ]
    )
    sh = SpinH(["ias", "iqi"], S=1 / 2, I=7 / 2)
    sh.add_term("ias", A)
    sh.add_term("iqi", Q)
    H = sh.get_H()
    np.testing.assert_allclose(H, H.conj().T, atol=ROUNDTRIP_TOL)


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------

def test_spinh_bi_cannot_be_inverted():
    """The 'bi' (nuclear Zeeman) term has no parameter matrix and so
    cannot be inverted; constructing with inv=True must raise."""
    with pytest.raises(ValueError, match="Nuclear Zeeman cannot be inverted"):
        SpinH(["bi"], B=np.array([1.0, 0.0, 0.0]), I=7 / 2, inv=True)


def test_spinh_invalid_term_name():
    """Unknown term names must be rejected at construction."""
    with pytest.raises(ValueError, match="Invalid element in terms list"):
        SpinH(["nonexistent"], S=1 / 2)


def test_spinh_missing_S_kwarg():
    """bgs/ias terms require S; absence must raise."""
    with pytest.raises(ValueError, match="Missing keyword argument S"):
        SpinH(["ias"], I=7 / 2)


def test_spinh_missing_I_kwarg():
    """ias/iqi/bi/bmi terms require I; absence must raise."""
    with pytest.raises(ValueError, match="Missing keyword argument I"):
        SpinH(["iqi"])


def test_spinh_missing_B_kwarg():
    """bgs/bi/bmi terms require B; absence must raise."""
    with pytest.raises(ValueError, match="Missing keyword argument B"):
        SpinH(["bgs"], S=1 / 2)


def test_spinh_add_term_unknown():
    """add_term must reject terms not declared at construction."""
    sh = SpinH(["ias"], S=1 / 2, I=7 / 2)
    with pytest.raises(ValueError, match="not instantiated"):
        sh.add_term("iqi", np.eye(3))


def test_spinh_add_H_term_without_inv():
    """add_H_term without inv=True must raise."""
    sh = SpinH(["ias"], S=1 / 2, I=7 / 2)  # inv defaults to absent (not True)
    H_dummy = np.zeros((4, 4), dtype=complex)
    with pytest.raises((TypeError, AttributeError)):
        sh.add_H_term("ias", H_dummy)


def test_spinh_inv_term_without_inv():
    """inv_term without inv=True must raise."""
    sh = SpinH(["ias"], S=1 / 2, I=7 / 2)
    with pytest.raises((TypeError, AttributeError)):
        sh.inv_term("ias")
