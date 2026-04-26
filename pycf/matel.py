#!/usr/bin/env python
# Filename = matel.py
"""
Angular momentum matrix element calculations.
Provides efficient methods for calculating matrix elements of angular momentum
operators and tensor operators, essential for crystal field Hamiltonian
construction.
**Tensor Matrix Elements (t_q):**
- <j1, m1 | T_q^(1) | j2, m2> for rank-1 tensor operators
- Used in coupling different angular momentum spaces
- Implementation: Wigner-Eckart theorem with 3j symbols
**Angular Momentum Components (matel):**
- Matrix elements of Jx, Jy, Jz in basis |j, m>
- Built using Wigner symbol algebra
- Used for dipole operators and spin Hamiltonian construction
Key features:
- Supports half-integer j (including rare-earth ions like j=5/2)
- Hermitian matrices (Jz, Jx) and rank-raising/lowering (J+, J-)
- Efficient numeric computation with factorial optimization
Used throughout pycf for building tensor operators in the crystal field
Hamiltonian and for intensity calculations.
"""

# Copyright (C) 2013 Sebastian Horvath (sebastian.horvath@gmail.com)
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
import numpy as np

import pycf.njsymbols as nj


def t_q(
    j1: float | int, j2: float | int, m1: float | int, m2: float | int, q: float | int
) -> complex:
    r"""
    Calculate the matrix element `\langle j_1 m_1 | T_q^{(1)} | j_2 m_2
    \rangle`, where `T_q^{(1)}` is a rank one tensor.
    Parameters
    ----------
    j1 : integer or half-integer
        The value of `j_1`.
    j2 : integer or half-integer
        The value of `j_2`.
    m2 : integer or half-integer
        The value of `m_2`.
    m2 : integer or half-integer
        The value of `m_2`.
    q : integer or half-integer
        The value of `q`.
    Returns
    -------
    element : complex
       The matrix element `\langle j_1 m_1 | T_q^{(1)} | j_2 m_2 \rangle`.
    """

    def delta(a, ap):
        """
        A delta function, returns 1 iff a == ap, else returns 0.
        """
        if a == ap:
            return 1
        else:
            return 0

    element = (
        complex(-1) ** (j1 - m1)
        * nj.wigner_3j(j1, 1, j2, -m1, q, m2)
        * np.sqrt((2 * j1 + 1) * (j1 + 1) * j1)
        * delta(j1, j2)
    )
    return element


def matel(c: str, j: float | int) -> np.ndarray:
    r"""
    Calculate the matrix elements for `\langle j m_1 | J_a | j m_2 \rangle`,
    where `a \in \{x, y, z\}` for `m_1 = j, j-1 \ldots -j` and `m_2 = j, j-1,
    \ldots -j`.
    Parameters
    ----------
    j : integer or half-integer
        The value of `j`.
    c : string
        The component label, a value of either 'jx', 'jy', or 'jz'.
    Returns
    -------
    matel : numpy array
        A `2j + 1` by `2j + 1` array, with rows and columns enumerated by `m_1`
        and `m_2` respectively.
    """
    if c == "jx":

        def f(m1, m2):
            return -1 / np.sqrt(2) * (t_q(j, j, m1, m2, 1) - t_q(j, j, m1, m2, -1))

    elif c == "jy":

        def f(m1, m2):
            return complex(0, 1) / np.sqrt(2) * (t_q(j, j, m1, m2, 1) + t_q(j, j, m1, m2, -1))

    elif c == "jz":

        def f(m1, m2):
            return t_q(j, j, m1, m2, 0)

    else:
        raise ValueError(
            "Invalid component argument '{}'.  Permitted values "
            "are 'jx', 'jy', or 'jz'.".format(c)
        )
    size = int(2 * j + 1)
    matel = np.zeros([size, size], dtype=complex)
    for row in range(size):
        for col in range(size):
            matel[row, col] = f(j - row, j - col)
    return matel
