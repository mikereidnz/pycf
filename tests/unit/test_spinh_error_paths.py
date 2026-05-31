"""Error-path coverage for :class:`pycf.spinh.SpinH`.

These tests exercise the defensive raises and validation branches in
``SpinH.__init__``, ``add_H_term``, ``inv_term`` and ``get_H`` which
were previously untested.  They are pure-Python algebraic checks; no
f-electron physics input is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycf.spinh import SpinH

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_term_name_raises():
    with pytest.raises(ValueError, match="Invalid element in terms list"):
        SpinH(["not_a_term"])


def test_missing_B_kwarg_raises():
    with pytest.raises(ValueError, match="Missing keyword argument B"):
        SpinH(["bgs"], S=1 / 2)


def test_missing_S_kwarg_raises():
    with pytest.raises(ValueError, match="Missing keyword argument S"):
        SpinH(["bgs"], B=np.array([1.0, 0.0, 0.0]))


def test_missing_I_kwarg_raises():
    with pytest.raises(ValueError, match="Missing keyword argument I"):
        SpinH(["iqi"])


def test_invalid_inv_value_raises():
    """``inv`` must be exactly True or False (line 656)."""
    with pytest.raises(ValueError, match="Invalid value for keyword argument 'inv'"):
        SpinH(
            ["bgs"],
            B=[np.eye(3)[:, i] for i in range(3)],
            S=1 / 2,
            inv="yes",
        )


def test_inv_true_with_bi_term_raises():
    """Nuclear Zeeman has no parameter matrix to invert (line 628)."""
    with pytest.raises(ValueError, match="Nuclear Zeeman cannot be inverted"):
        SpinH(["bi"], B=[np.eye(3)[:, i] for i in range(3)], I=1 / 2, inv=True)


def test_inv_true_bgs_requires_B_list():
    """When ``inv=True`` and 'bgs' is in terms, ``B`` must be a list (line 633)."""
    with pytest.raises(TypeError, match="B must be a"):
        SpinH(["bgs"], B=np.eye(3)[:, 0], S=1 / 2, inv=True)


def test_inv_true_bmi_requires_B_list():
    """Same constraint for the bmi term (line 647)."""
    with pytest.raises(TypeError, match="B must be a"):
        SpinH(["bmi"], B=np.eye(3)[:, 0], I=1 / 2, kramers=False, inv=True)


# ---------------------------------------------------------------------------
# add_H_term validation
# ---------------------------------------------------------------------------


def test_add_H_term_without_inv_raises():
    """``add_H_term`` only works when SpinH was constructed with inv=True."""
    sh = SpinH(["bgs"], B=np.eye(3)[:, 0], S=1 / 2)
    with pytest.raises(TypeError, match="does not support add_H_term"):
        sh.add_H_term("bgs", [np.zeros((2, 2), dtype=complex)])


def test_add_H_term_unknown_term_raises():
    sh = SpinH(["bgs"], B=[np.eye(3)[:, i] for i in range(3)], S=1 / 2, inv=True)
    with pytest.raises(ValueError, match="not instantiated"):
        sh.add_H_term("ias", np.zeros((4, 4), dtype=complex))


# ---------------------------------------------------------------------------
# inv_term validation
# ---------------------------------------------------------------------------


def test_inv_term_without_inv_raises():
    sh = SpinH(["bgs"], B=np.eye(3)[:, 0], S=1 / 2)
    with pytest.raises(TypeError, match="does not support inv_term"):
        sh.inv_term("bgs")


def test_inv_term_unknown_term_raises():
    sh = SpinH(["bgs"], B=[np.eye(3)[:, i] for i in range(3)], S=1 / 2, inv=True)
    with pytest.raises(ValueError, match="not instantiated"):
        sh.inv_term("ias")


def test_inv_term_bad_sym_phase_length_raises():
    """``sym=True`` with ``sym_phase`` of wrong length (line 807)."""
    sh = SpinH(["bgs"], B=[np.eye(3)[:, i] for i in range(3)], S=1 / 2, inv=True)
    g = np.diag([2.0, 2.5, 1.8])
    # Build a valid bgs payload first so inv_term gets past structural checks.
    bgs_list = []
    for i in range(3):
        sh_fwd = SpinH(["bgs"], B=np.eye(3)[:, i], S=1 / 2)
        sh_fwd.add_term("bgs", g)
        bgs_list.append(sh_fwd.terms["bgs"])
    sh.add_H_term("bgs", bgs_list)
    with pytest.raises(ValueError, match="sym_phase argument must be of length 3"):
        sh.inv_term("bgs", sym=True, sym_phase=[0.1, 0.2])


# ---------------------------------------------------------------------------
# get_H validation
# ---------------------------------------------------------------------------


def test_get_H_without_add_term_raises():
    """Calling ``get_H`` before populating ``self.terms`` for every t (line 836)."""
    sh = SpinH(["bgs"], B=np.eye(3)[:, 0], S=1 / 2)
    with pytest.raises(ValueError, match="does not have data for the bgs"):
        sh.get_H()
