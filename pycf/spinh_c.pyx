# cython: profile=True
# filename = spinh_c.pyx

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

from __future__ import division
import warnings
import numpy as np
from scipy.optimize import minimize


# Symmeterization of spin Hamiltonian tensors makes heavy use of QR
# factorization; therefore, we directly call the LAPACK zgels function using
# cython.
cpdef zgels(np.ndarray[NPCOMPLEX_t,ndim=2] A, np.ndarray[NPCOMPLEX_t,ndim=1] b):
    """
    Wrap the zgels function to make it cython callable.

    Parameters
    ----------
    A : ndarray
        The coefficient matrix.
    b : ndarray
        The r.h.s. of A x = b. 

    Returns
    -------
    x : ndarray
        The solution vector.
    """
    cdef lapack_int m = A.shape[0]
    cdef lapack_int n = A.shape[1]
    cdef lapack_int nrhs = 1
    cdef lapack_int lda = m
    cdef lapack_int ldb = m
    cdef lapack_int info
    
    # LAPACKE handles work space query and memory allocation.
    info = LAPACKE_zgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, <complex*>A.data, lda,
            <complex*>b.data, ldb)
    
    cdef lapack_int zero = 0
    if info != zero:
        raise RuntimeError("Non-zero info returned by ZGELS.")
    return(b[:n])


cpdef np.ndarray[complex, ndim=2] su2_rz(np.ndarray[complex, ndim=2] m, float p):
    """
    Apply a rotation about the z-axis in the SU(2) matrix representation.
    
    Parameters
    ----------
    m : ndarray
        The `2 \times 2` SU(2) matrix. 
    p : float
        The phase `\phi` of an SU(2) rotation `\mathcal{D}_z(\phi)`.
    
    Returns
    -------
    mp : ndarry
        The transformed `2 \times 2` matrix.
    """
    cdef complex t = cexp(_Complex_I*p)
    cdef np.ndarray[complex, ndim=2] mp = np.zeros((2, 2), dtype=np.complex)
    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements.
    mp[0, 0] = m[0, 0]
    mp[0, 1] = m[0, 1] * t
    mp[1, 0] = m[1, 0] * conj(t)
    mp[1, 1] = m[1, 1]

    return(mp)


cpdef inline su2_rz_ias(m, p):
    """
    Apply a rotation about the z-axis of the spin-half matrix elements of a
    magnetic dipole spin Hamiltonian term.

    Parameters
    ----------
    m : ndarray
        The `IAS` term of dimension `2 \times (I+1) \times 2` by `2 \times (I+1)
        \times 2`.
    p : float 
        The phase `\phi` of an SU(2) rotation `\mathcal{D}_z(\phi)`.
    
    Returns
    -------
    mp : ndarray
        The transformed `IAS` term. 
    """
    t = np.exp(np.complex(0,1)*p)
    mp = np.array(m, dtype=np.complex)
    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements of the 2 by 2 spin-half blocks.
    mp[0::2, 1::2] = m[0::2, 1::2] * t
    mp[1::2, 0::2] = m[1::2, 0::2] * np.conj(t)
    return(mp)


cpdef inline su2_rz_lsq_f(p, sh, H_sh, term):
    """
    Helper function for least squares fitting of the SU(2) rotation required to
    symmetrize spin Hamiltonian terms containing spin half matrix elements.

    Parameters
    ----------
    p : float
        The phase to be varied. 
    sh : SpinHamiltonian
        Must have been instantiated with the 'inv' = True kwarg and contain a
        term with spin half matrix element, i.e., either `BgS` or `IAS`.   
    H_sh : list or numpy.ndarray
        If term='bgs' H_sh must be a list of `2 \times 2` ndarrays corresponding
        to the `BgS` spin Hamiltonian term.  If term = 'ias' H_sh must be a `2
        (I + 1) \times 2` ndarray corresponding to the `IAS` spin Hamiltonian
        term.
    term : string
        Specify whether to symmetrize using the `BgS` or the `IAS` term using
        values 'bgs' and 'ias', respectively.
       
    Returns
    -------
    r : float
        The residue; calculated from the differences between the off diagonal
        elements of the spin Hamiltonian tensor.
    """
    sh.add_H_term(term, H_sh, phase=p[0])
    tensor = sh.inv_term(term)

    sym_index = [(1, 3), (2, 6), (5, 7)]
    r = 0
    for i in sym_index:
        r += np.abs(tensor[i[0]] - tensor[i[1]])
    return r
