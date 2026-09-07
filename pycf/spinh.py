#!/usr/bin/env python
# Filename = spinh.py
"""
Spin Hamiltonian extraction and manipulation utilities.

This module provides functions for:

- Building spin Hamiltonian matrices from tensor operators
- Extracting spin Hamiltonian parameters from crystal field calculations
- Inverting spin Hamiltonians to recover effective parameters
- Manipulating and formatting spin Hamiltonian terms

The spin Hamiltonian formalism reduces complex rare-earth ion interactions to
effective magnetic operators acting on the lowest Kramers doublet. This module
bridges crystal field theory (implemented in cfl) to spin Hamiltonian models
commonly used in magnetic resonance and magnetism studies.

Key workflow:

  1. Calculate crystal field Hamiltonian and eigenstates (cfl module)
  2. Project high-level (J=5/2...) multiplet onto Kramers doublet
  3. Extract effective spin Hamiltonian parameters
  4. Verify by comparing predicted vs. experimental properties
"""

# Copyright (C) 2013-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import List

import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import block_diag, svd  # type: ignore[import-untyped]
from scipy.optimize import basinhopping  # type: ignore[import-untyped]

from pycf.matel import matel


def bmj(v: np.ndarray, m: np.ndarray, t: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `BgS` or `BMI` term, an array of size `(2 \times j + 1)` by `(2 \times j
    + 1)`, with `j` the angular momentum of the tensor `S` or `I`.

    Parameters
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    m : numpy.ndarray
        The `3` by `3` Zeeman parameter matrix `g`.
    t : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : array
    """
    tl = len(t[0])
    matrix_len = len(t)
    result = np.zeros([tl, tl], dtype=complex)
    # All states of t are iterated through by the outer two loops.  The
    # contribution due to each term in v cdot m cdot t is computed by
    # the inner two loops.  This consists of a matrix multiplication of
    # the form transpose(v) * m * t_element, where t_element is the
    # matrix element corresponding to the state enumerated by tc and tr.
    for tr in range(tl):
        for tc in range(tl):
            elem = 0
            for i in range(matrix_len):
                for j in range(matrix_len):
                    elem += v[i] * m[i, j] * t[j][tr, tc]
            result[tr, tc] = elem
    return result


def ias(t1: List[np.ndarray], m: np.ndarray, t2: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `IAS` term, an array of size `(2 \times j_1+1) \times (2 \times
    j_2+1)` by `(2 \times j_1+1) \times (2 \times j_2+1)`, with `j_1` and `j_2`
    the angular momentum of the rank one tensors `A` and `I`, respectively.

    Parameters
    ----------
    t1 : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    m : numpy.ndarray
        The `3` by `3` dipole parameter matrix `A`.
    t2 : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : array
    """
    t1l = len(t1[0])
    t2l = len(t2[0])
    matrix_len = len(t1)
    result = np.zeros([t1l * t2l, t1l * t2l], dtype=complex)
    # All states of t1 and t2 are iterated through by the outer four loops.
    # The contribution of each term in the t1 cdot m cdot t2 construct is
    # computed by the inner two loops.  This is effectively a matrix
    # multiplication, but for each term the multiplicative factors due to t1
    # and t2 are looked up from the matrix element determined by the indices
    # t1c, t1r, t2c and t2r.
    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    elem = 0
                    for i in range(matrix_len):
                        for j in range(matrix_len):
                            elem += t1[i][t1r, t1c] * m[i, j] * t2[j][t2r, t2c]
                    result[t1r + t1l * t2r, t1c + t1l * t2c] = elem
    return result


def iqi(t: List[np.ndarray], m: np.ndarray) -> np.ndarray:
    r"""
    Generate the `IQI` term, an array of size `(2 \times j + 1)` by `(2 \times j
    + 1)`, with `j` the angular momentum of the rank one tensor `I`.

    Parameters
    ----------
    t : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    m : numpy.ndarray
        The `3` by `3` quadrupole parameter matrix `Q`.

    Returns
    -------
    result : array
    """
    tl = len(t[0])
    matrix_len = len(t)
    result = np.zeros([tl, tl], dtype=complex)
    # All states of t are iterated through by the outer two loops.  The
    # contribution due to each term in t cdot m cdot t is computed by
    # the inner three loops.  The i and j loops iterate over x, y and z
    # to yield a 3 by 3 matrix, with the sqi loop evaluating the matrix
    # elements for the square t * t operator.
    for tr in range(tl):
        for tc in range(tl):
            elem = 0
            for i in range(matrix_len):
                for j in range(matrix_len):
                    components = 0
                    for ci in range(tl):
                        components += t[i][tr, ci] * t[j][ci, tc]
                    elem += m[i, j] * components
            result[tr, tc] = elem
    return result


def bi(v: np.ndarray, t: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `BI` term, an array of size `(2 \times j + 1)` by `(2 \times j
    + 1)`, with `j` the angular momentum of the rank one tensor `I`.

    Parameters
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    t : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.

    Returns
    -------
    result : array
    """
    tl = len(t[0])
    matrix_len = len(t)
    result = np.zeros([tl, tl], dtype=complex)
    # All states of t are iterated through by the outer two loops.  The
    # contribution due to each term in v cdot t is computed by the inner loop.
    # Unlike bmj, ias, and iqi which compute v^T * M * t and require two inner
    # loops (i for v, j for t, linked by the coupling matrix M), bi computes the
    # dot product v^T * t (identity coupling), so only a single inner loop is
    # needed: t_element is selected by the same index i as v[i].
    for tr in range(tl):
        for tc in range(tl):
            elem = 0
            for i in range(matrix_len):
                elem += v[i] * t[i][tr, tc]
            result[tr, tc] = elem
    return result


def bmj_coeff_array(v: np.ndarray, t: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `BgS` or `BMI` coefficient array.  This consists of a `2j+1
    \times 2j+1` by `3 \times 3` array containing the matrix elements of the
    terms `B_a J_b`, with `a,b \in \{x, y, z\}` and `j` the angular momentum of
    the rank one tensor `S` or `I`.  Here the rows enumerate the `2j+1 \times
    2j+1` different state combinations while the columns enumerate all
    combinations of `a` and `b`.  This array is independent of `S`/`I` and is
    intended to be computed once, then employed with numpy's :func:`lstsq`
    function to calculate `g` or `M` given a `BgS` or `BMI` matrix.

    Parameters
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    t : list
        Elements consist of the matrix elements of `J_x`, `J_y` and `J_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j+1 \times 2j+1` by `3 \times 3` array.
    """
    tl = len(t[0])
    matrix_len = len(t)
    bmj_a = np.zeros([tl, tl, matrix_len, matrix_len], dtype=complex)
    for tr in range(tl):
        for tc in range(tl):
            for i in range(matrix_len):
                for j in range(matrix_len):
                    bmj_a[tr, tc, i, j] = v[i] * t[j][tr, tc]
    return np.reshape(bmj_a, (tl * tl, matrix_len * matrix_len))


def ias_coeff_array(t1: List[np.ndarray], t2: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `IAS` coefficient array.  This consists of a `2j_1+1 \times
    2j_2+1` by `3 \times 3` array containing the matrix elements of the
    operators `I_a S_b`, with `a,b \in \{x, y, z\}` and `j_1` and `j_2` the
    angular momentum of the rank one tensors `I` and `S`, respectively.  Here
    the rows enumerate the `2j_1+1 \times 2j_2+1` different state combinations
    while the columns enumerate all combinations of `a` and `b`.  This array is
    independent of `A` and is intended to be computed once, then employed with
    numpy's :func:`lstsq` function to calculate `A` given an `IAS` matrix.

    Parameters
    ----------
    t1 : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    t2 : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j_1+1 \times 2j_2+1` by `3 \times 3` array.
    """
    t1l = len(t1[0])
    t2l = len(t2[0])
    matrix_len = len(t1)
    ias_a = np.zeros([t1l * t2l, t1l * t2l, matrix_len, matrix_len], dtype=complex)
    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(matrix_len):
                        for j in range(matrix_len):
                            ias_a[t1r + t1l * t2r, t1c + t1l * t2c, i, j] = (
                                t1[i][t1r, t1c] * t2[j][t2r, t2c]
                            )
    return np.reshape(ias_a, (t1l * t2l * t1l * t2l, matrix_len * matrix_len))


def iqi_coeff_array(t: List[np.ndarray]) -> np.ndarray:
    r"""
    Generate the `IQI` coefficient array.  This consists of a `2j+1 \times 2j+1`
    by `3 \times 3` array containing the matrix elements of the operators `I_a
    I_b`, with `a,b \in \{x, y, z\}` and `j` the angular momentum of the rank
    one tensor `I`.  Here the rows enumerate the `2j+1 \times 2j+1` different
    state combinations while the columns enumerate all combinations of `a` and
    `b`.  This array is independent of `Q` and is intended to be computed once,
    then employed with numpy's :func:`lstsq` function to calculate `Q` given an
    `IQI` matrix.

    Parameters
    ----------
    t : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j+1 \times 2j+1` by `3 \times 3` array.
    """
    tl = len(t[0])
    matrix_len = len(t)
    iqi_a = np.zeros([tl, tl, matrix_len, matrix_len], dtype=complex)
    for tr in range(tl):
        for tc in range(tl):
            for i in range(matrix_len):
                for j in range(matrix_len):
                    components = 0
                    for ci in range(tl):
                        components += t[i][tr, ci] * t[j][ci, tc]
                    iqi_a[tr, tc, i, j] = components
    return np.reshape(iqi_a, (tl * tl, matrix_len * matrix_len))


def invert_term(coeff_a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
    Invert a spin Hamiltonian term.

    Parameters
    ----------
    term : np.ndarray
        The matrix elements of the term.
    coeff_a : np.ndarray
        The appropriate coefficient array, generated with either
        :func:`bmj_coeff_array`, :func:`ias_coeff_array` or
        :func:`iqi_coeff_array`.
    b : numpy.ndarray
        For a 'BgS' or 'BMI' term, b must be a :math:`3 \times (2j + 1)
        \times (2j + 1)` array :math:`(j = S \mathrm{\ or\ } I)`, corresponding
        to individual 'BgS'/'BMI' matrix elements for a field along three
        linearly independent directions. For an 'IAS' term, b must be a
        :math:`2 (I + 1) \times 2` np.ndarray corresponding to the `IAS`
        matrix elements.

    Returns
    -------
    result : numpy.ndarray
        A `9` by `1` vector, consisting of stacked rows of the 3 by 3 term
        parameter matrix.
    """
    # Reshape the matel array to a vector.
    b = b.flatten()
    # Solve the equation coeff_a * x = b for x.
    # Fix: explicitly set rcond to suppress FutureWarning
    return np.real(lstsq(coeff_a, b, rcond=None)[0])


def su2_rz(p: float, m: np.ndarray) -> np.ndarray:
    """
    Apply an SU(2) rotation about the z-axis of the spin-half matrix elements of a
    spin Hamiltonian term, specifically, a Zeeman interaction term or a magnetic
    dipole hyperfine interaction term.

    Parameters
    ----------
    p : float
        The phase ``phi`` of an SU(2) rotation D_z(phi).
    m : ndarray
        A 2x2 Zeeman interaction term, or a 2*(I+1) x 2*(I+1)
        magnetic dipole hyperfine interaction term.

    Returns
    -------
    m : ndarray
        The transformed matrix.
    """
    t = np.exp(complex(0, 1) * p)
    mp = np.array(m, dtype=complex)
    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements of the 2 by 2 spin-half blocks.
    mp[0::2, 1::2] = m[0::2, 1::2] * t
    mp[1::2, 0::2] = m[1::2, 0::2] * np.conj(t)
    return mp


def su2_rz_lsq_f(p: float, coeff_a: np.ndarray, b: np.ndarray) -> float:
    """
    Helper function for least squares fitting of the SU(2) rotation required to
    symmetrize spin Hamiltonian terms containing spin half matrix elements.

    Parameters
    ----------
    p : float
        The phase to be varied.
    coeff_a : np.ndarray
        The appropriate coefficient array, generated with either
        :func:`bgs_coeff_array` or :func:`ias_coeff_array`.
    b : numpy.ndarray
        For a 'BgS' term, b must be a `3 \times 2 \times 2` array, corresponding
        to individual 'BgS' matrix elements for a field along three linearly
        independent directions. For an 'IAS' term, b must be a `2 (I + 1) \times
        2` np.ndarray corresponding to the `IAS` matrix elements.

    Returns
    -------
    r : float
        The residue; calculated from the differences between the off diagonal
        elements of the spin Hamiltonian tensor.
    """
    # Check whether 'BgS', or 'IAS', since we need to loop through each field
    # direction for the former.
    if b.shape == (3, 2, 2):
        b_p = np.zeros((3, 2, 2), dtype=complex)
        for i in range(3):
            b_p[i, :, :] = su2_rz(p, b[i, :, :])
    else:
        b_p = su2_rz(p, b)
    tensor = invert_term(coeff_a, b_p)
    r = 0
    for i_a, i_b in [(1, 3), (2, 6), (5, 7)]:
        r += np.abs(tensor[i_a] - tensor[i_b])
    return r


def su2_rotation(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Apply an SU(2) rotation an about the z, y, and x axes respectively to the
    spin-half matrix elements of a spin Hamiltonian term, specifically, a Zeeman
    interaction term or a magnetic dipole hyperfine interaction term.

    Parameters
    ----------
    p : array
        The three rotation angles.
    m : ndarray
        A `2 \times 2` Zeeman interaction term, or a `2 \times (I+1) \times 2`
        by `2 \times (I+1) \times 2` magnetic dipole hyperfine interaction term.

    Returns
    -------
    m : ndarray
        The transformed matrix.
    """
    a = p[0]
    b = p[1]
    c = p[2]
    rotation = np.array(
        [
            [
                np.exp(-1j * (a) / 2)
                * (np.cos(b / 2) * np.cos(c / 2) + 1j * np.sin(b / 2) * np.sin(c / 2)),
                -np.exp(-1j * (a) / 2)
                * (1j * np.cos(b / 2) * np.sin(c / 2) + np.sin(b / 2) * np.cos(c / 2)),
            ],
            [
                np.exp(1j * (a) / 2)
                * (np.sin(b / 2) * np.cos(c / 2) - 1j * np.cos(b / 2) * np.sin(c / 2)),
                np.exp(1j * (a) / 2)
                * (-1j * np.sin(b / 2) * np.sin(c / 2) + np.cos(b / 2) * np.cos(c / 2)),
            ],
        ]
    )
    # Allocate as complex (mirroring su2_rz) so the complex rotation result is
    # never truncated when the input m happens to be real-dtyped.
    rm = np.array(m, dtype=complex)
    # Use integer floor division (//) rather than float division + int() cast.
    for i in range(m.shape[0] // 2):
        for j in range(m.shape[1] // 2):
            rm[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = np.dot(
                np.dot(np.conj(rotation.T), m[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]),
                rotation,
            )
    return rm


def su2_rotation_lsq_f(p: np.ndarray, coeff_a: np.ndarray, b: np.ndarray) -> float:
    """
    Helper function for least squares fitting of the SU(2) rotation required to
    symmetrize spin Hamiltonian terms containing spin half matrix elements.

    Parameters
    ----------
    p : array
        The three rotation angles, with entries for rotations about the z, y,
        and x axes respectively.
    coeff_a : np.ndarray
        The appropriate coefficient array, generated with either
        :func:`bgs_coeff_array` or :func:`ias_coeff_array`.
    b : numpy.ndarray
        For a 'BgS' term, b must be a `3 \times 2 \times 2` array, corresponding
        to individual 'BgS' matrix elements for a field along three linearly
        independent directions. For an 'IAS' term, b must be a `2 (I + 1) \times
        2` np.ndarray corresponding to the `IAS` matrix elements.

    Returns
    -------
    r : float
        The residue; calculated from the differences between the off diagonal
        elements of the spin Hamiltonian tensor.
    """
    # Check whether 'BgS', or 'IAS', since we need to loop through each field
    # direction for the former.
    if b.shape == (3, 2, 2):
        b_p = np.zeros((3, 2, 2), dtype=complex)
        for i in range(3):
            b_p[i, :, :] = su2_rotation(p, b[i, :, :])
    else:
        b_p = su2_rotation(p, b)
    tensor = invert_term(coeff_a, b_p)
    r = 0
    for i_a, i_b in [(1, 3), (2, 6), (5, 7)]:
        r += np.abs(tensor[i_a] - tensor[i_b])
    return r


def param_ten_svd(t: np.ndarray) -> np.ndarray:
    r"""
    Apply a symmetrising SVD rotation to a parameter tensor.

    Computes the SVD ``t = U @ diag(s) @ Vh`` and returns
    ``t' = t @ Vh.T @ U.T``, which leaves the singular values of ``t``
    unchanged while rotating the input/output bases. Used inside the
    spin-Hamiltonian inversion to bring the parameter matrix into a canonical
    frame before further analysis.

    Parameters
    ----------
    t : numpy.ndarray
        Square real or complex parameter matrix.

    Returns
    -------
    numpy.ndarray
        Rotated parameter matrix with the same singular values as ``t``.
    """
    U, s, Vh = svd(t)
    t = np.dot(t, Vh.T).dot(U.T)
    return t


class SpinH(object):
    r"""
    Container for holding data about the spin Hamiltonian.  Can either be used
    to calculate the full spin Hamiltonian from individual terms, or to invert
    the spin Hamiltonian and recover the spin Hamiltonian parameters.  If used
    for the former, the object is instantiated and then spin Hamiltonian terms
    are added with the :func:`add_term` method.  The full Hamiltonian can then
    be returned using the :func:`get_H` method.  If used for the latter, terms
    are added as arrays with dimensions of the full Hamiltonian using the
    :func:`add_H_term` method.  The spin Hamiltonian parameters can then be
    calculated using the :func:`inv_term` method.
    Note: units are Tesla for magnetic field values and MHz for energies.

    Parameters
    ----------
    terms : list
        Elements are strings with possible values 'bgs', 'ias', 'iqi', 'bi', and
        'bmi'.  The choice of elements affects what other keyword arguments are
        required; see below.
    kramers : bool
        Set to True for Kramers ions and False for non-Kramers ions. Default is
        True.
    B : numpy.ndarray or list with numpy.ndarray elements
        A `3` by `1` vector containing values for the magnetic field strengths
        `B_x`, `B_y` and `B_z`; if ``terms`` contains 'bgs' or 'bi' this keyword
        argument must be specified.  Furthermore, if ``'inv' = True``, this must
        be a list of magnetic field strength vectors, with a minimum of three
        linearly independent vectors required for a fully determined solution.
    S : float
        The spin projection `S_z`; if ``terms`` contains 'bgs' or 'ias' this
        keyword argument must be specified.
    I : float
        The nuclear spin projection `I_z`; if ``terms`` contains 'ias', 'iqi',
        'bi', or 'bmi' this keyword argument must be specified.
    inv : boolean, optional
        If True, the coefficient arrays for term inversion are pre-computed.

    Returns
    -------
    object : SpinH
    """

    def __init__(self, terms, **kwargs):
        # Bohr magneton in MHz/T
        # http://physics.nist.gov/cgi-bin/cuu/Value?mubshhz|search_for=bohr+magneton
        self.mu_b = 13.996245042e3
        # Nuclear magneton in MHz/T
        # http://physics.nist.gov/cgi-bin/cuu/Value?munshhz|search_for=nuclear+magneton
        self.mu_n = 7.622593285
        for t in terms:
            if t not in ["bgs", "ias", "iqi", "bi", "bmi"]:
                raise ValueError(
                    "Invalid element in terms list: {}. Allowed"
                    " values are 'bgs', 'ias', 'iqi', 'bi', 'bmi'.".format(terms)
                )
            else:
                self.t_list = terms
        self.terms = {}
        # Calculate matrix elements for the specified terms.
        j_l = ["jx", "jy", "jz"]
        if "bgs" in terms or "bi" in terms or "bmi" in terms:
            try:
                B = kwargs["B"]
                self.B = B
            except KeyError:
                raise ValueError("Missing keyword argument B.")
        else:
            B = None
        if "bgs" in terms or "ias" in terms:
            try:
                S = kwargs["S"]
            except KeyError:
                raise ValueError("Missing keyword argument S.")
            # Calculate the matrix elements for spin.
            S_m = [None, None, None]
            for i in range(3):
                S_m[i] = matel(j_l[i], S)
            self.S = S
            self.S_m = S_m
        else:
            S_m = None
        if "ias" in terms or "iqi" in terms or "bi" in terms or "bmi" in terms:
            try:
                Ival = kwargs["I"]
            except KeyError:
                raise ValueError("Missing keyword argument I.")
            # Calculate the matrix elements for nuclear spin.
            I_m = [None, None, None]
            for i in range(3):
                I_m[i] = matel(j_l[i], Ival)
            self.I = Ival  # noqa: E741 - public attribute; physics symbol for nuclear spin
            self.I_m = I_m
        else:
            I_m = None
        if "kramers" not in kwargs:
            self.kramers = True
        else:
            self.kramers = kwargs["kramers"]
        # Determine Hamiltonian dimension.
        if self.kramers:
            if "bgs" in terms:
                if I_m is None:
                    # Only the bgs term.
                    H_dim = 2 * S + 1
                else:
                    # The bgs and/or ias/iqi/bi terms.
                    H_dim = (2 * S + 1) * (2 * Ival + 1)
            elif S_m is None:
                # Only the iqi and/or bi term.
                H_dim = 2 * Ival + 1
            else:
                # S_m != None and no bgs -> ias term.
                H_dim = (2 * S + 1) * (2 * Ival + 1)
        else:
            H_dim = 2 * Ival + 1
        self.H_dim = int(H_dim)
        # Calculate the coefficient arrays.
        if "inv" in kwargs:
            # Use identity comparison rather than == for booleans.
            if kwargs["inv"] is True:
                if "bi" in terms:
                    raise ValueError("Nuclear Zeeman cannot be inverted" " (no parameter matrix).")
                self.H_terms = {}
                self.coeff_a = {}
                if "bgs" in terms:
                    if not isinstance(B, list):
                        raise TypeError(
                            "When passing inv = True, B must be a" "list of numpy.ndarrays."
                        )
                    S_dimsq = int((2 * S + 1) ** 2)
                    B_a = np.zeros([len(B), S_dimsq, 9], dtype=complex)
                    for i, e in enumerate(B):
                        B_a[i, :, :] = bmj_coeff_array(e, S_m)
                    self.coeff_a["bgs"] = np.reshape(B_a, (len(B) * S_dimsq, 9))
                if "ias" in terms:
                    self.coeff_a["ias"] = ias_coeff_array(I_m, S_m)
                if "iqi" in terms:
                    self.coeff_a["iqi"] = iqi_coeff_array(I_m)
                if "bmi" in terms:
                    if not isinstance(B, list):
                        raise TypeError(
                            "When passing inv = True, B must be a" "list of numpy.ndarrays."
                        )
                    I_dimsq = int((2 * Ival + 1) ** 2)
                    B_a = np.zeros([len(B), I_dimsq, 9], dtype=complex)
                    for i, e in enumerate(B):
                        B_a[i, :, :] = bmj_coeff_array(e, I_m)
                    self.coeff_a["bmi"] = np.reshape(B_a, (len(B) * I_dimsq, 9))
            elif kwargs["inv"] is not False:
                raise ValueError(
                    "Invalid value for keyword argument 'inv'; "
                    "valid values are either True or False."
                )
            self.inv = kwargs["inv"]
        else:
            self.inv = False

    def add_term(self, term, m):
        r"""
        Add the specified parameter matrix to the spin Hamiltonian.

        Parameters
        ----------
        term : string
            Specifies the parameter type; must be one of the values of the
            ``term`` list provided when the SpinH object was instantiated.
        m : numpy.ndarray or float
            A `3` by `3` matrix consisting of the parameters for 'bgs', 'ias',
            'iqi', or 'bmi' terms.  For 'ib', this should be a float specifying
            the nuclear g-factor.
        """
        if term not in self.t_list:
            raise ValueError(
                "This SpinH object was not instantiated "
                "with support for the specified term: {}".format(term)
            )

        def bgs_helper(mat, hdim):
            """
            Create matrix with dimension of hdim, with elements of mat in
            diagonal blocks of size hdim/len(mat).
            """
            matd = len(mat)  # dim of bgs
            bd = hdim // matd  # block dim
            H = np.zeros([hdim, hdim], dtype=complex)
            d = np.diag([1] * bd)
            for r in range(matd):
                for c in range(matd):
                    H[r * bd : (r + 1) * bd, c * bd : (c + 1) * bd] += d * mat[r, c]
            return H

        if term == "bgs":
            # Create list of H_dim/(2*S + 1) length and block diagonalize.
            self.terms["bgs"] = bgs_helper(bmj(self.B, m, self.S_m), self.H_dim)
        elif term == "ias":
            # ias term is of correct dimension.
            self.terms["ias"] = ias(self.I_m, m, self.S_m)
        elif term == "iqi":
            # Create list of H_dim/(2*I + 1) length and block diagonalize.
            # self.I is a half-integer float, so cast the denominator to int
            # before dividing to get an integer result for list repetition.
            n = self.H_dim // int(2 * self.I + 1)
            self.terms["iqi"] = block_diag(*[iqi(self.I_m, m)] * n)
        elif term == "bi":
            # Create list of H_dim/(2*I + 1) length and block diagonalize.
            n = self.H_dim // int(2 * self.I + 1)
            self.terms["bi"] = block_diag(*[m * bi(self.B, self.I_m)] * n)
        elif term == "bmi":
            # bmi is of correct dimension for non-Kramers ions.
            self.terms["bmi"] = bmj(self.B, m, self.I_m)

    def add_H_term(self, term, val):
        r"""
        Extract the elements of the specified term from a spin Hamiltonian with
        full dimension and update the appropriate term value of the SpinH
        object.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinH object was instantiated.
        val : numpy.ndarray or list
            The value of the specified term.  For 'bgs' this must be a list of
            numpy.ndarrays, with elements in the list in the same order as the
            ``B`` list used to instantiate the SpinH object.
        """
        if not self.inv:
            raise TypeError(
                "This spectrum object does not support add_H_term "
                "method calls; to enable this pass the inv = True argument "
                "to the constructor."
            )
        if term not in self.t_list:
            raise ValueError(
                "This SpinH object was not instantiated "
                "with support for the specified term: {}".format(term)
            )
        if term == "bgs":
            S_dim = int(2 * self.S + 1)
            self.H_terms["bgs"] = np.array([v[:S_dim, :S_dim] for v in val])
        elif term == "ias":
            self.H_terms["ias"] = val
        elif term == "iqi":
            I_dim = int(2 * self.I + 1)
            self.H_terms["iqi"] = val[:I_dim, :I_dim]
        elif term == "bmi":
            self.H_terms["bmi"] = np.array(val)

    def inv_term(self, term, sym=False, sym_phase=None):
        r"""
        Invert the specified term of this spin Hamiltonian.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinH object was instantiated.
        sym : bool
            Set to True to enable spin Hamiltonian parameter symmeterization.
        sym_phase : list
            List with three elements specifying the symmeterization angles.  If
            this argument is provided, no fit will be performed and the
            symmeterization will be applied with the specified angles.

        Returns
        -------
        term_parameters : numpy.ndarray
            A `9` by `1` vector consisting of stacked rows of the corresponding
            `3` by `3` term parameter matrix.
        """
        if not self.inv:
            raise TypeError(
                "This SpinH object does not support inv_term method"
                " calls; to enable this pass the inv = True argument to the"
                " constructor."
            )
        elif term not in self.t_list:
            raise ValueError(
                "This SpinH object was not instantiated "
                "with support for the specified term: {}".format(term)
            )
        if sym:

            def print_fun(x, f, accepted):
                """Basinhopping callback: print each accepted symmetrisation step."""
                print("Symmeterization minimum %.4f accepted %d" % (f, int(accepted)))

            if sym_phase is None:

                def fmin(p):
                    """Residual function minimised by basinhopping over SU(2) angles."""
                    return su2_rotation_lsq_f(p, self.coeff_a[term], self.H_terms[term])

                r = basinhopping(
                    fmin,
                    [0, 0, 0],
                    minimizer_kwargs={"method": "Powell"},
                    callback=print_fun,
                    niter=100,
                )
                self.sym_phase = r["x"]
            else:
                if len(sym_phase) != 3:
                    raise ValueError("The sym_phase argument must be of length 3.")
                self.sym_phase = sym_phase
            # Check whether 'BgS', or 'IAS', since we need to loop through each
            # field direction for the former.
            if self.H_terms[term].shape == (3, 2, 2):
                for i in range(3):
                    self.H_terms[term][i, :, :] = su2_rotation(
                        self.sym_phase, self.H_terms[term][i, :, :]
                    )
            else:
                self.H_terms[term] = su2_rotation(self.sym_phase, self.H_terms[term])
        else:
            self.sym_phase = [0, 0, 0]
        m = invert_term(self.coeff_a[term], self.H_terms[term])
        return m

    def get_H(self):
        r"""
        Calculate the full Hamiltonian and return the result.
        """
        H = complex(0, 0)
        for t in self.t_list:
            try:
                if t == "bgs" or t == "bmi":
                    H += self.terms[t] * self.mu_b
                elif t == "bi":
                    H += self.terms[t] * self.mu_n
                else:
                    H += self.terms[t]
            except KeyError:
                raise ValueError(
                    "This object does not have data for the {} "
                    "term.  Have you run the 'add_term' method?".format(t)
                )
        return H
