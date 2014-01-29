# filename = spinh_c.pxd

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

# The spinh_c module is used to collect functions used by the spinh module which
# are optimized using cython.  These are kept in a separate module so that one
# does not have to sacrifice python tracebacks and easy of modifications in the
# main spinh module.

import numpy as np
cimport numpy as np

ctypedef np.int_t NPINT_t
ctypedef np.float64_t NPFLOAT_t
ctypedef np.complex128_t NPCOMPLEX_t
ctypedef int integer

cdef extern from "complex.h":
    double complex conj(double complex z)
    double complex cexp(double complex z)
    double complex _Complex_I

cdef extern from "lapacke.h":
    ctypedef long int lapack_int
    lapack_int LAPACKE_zgels(lapack_int matrix_order, char trans, lapack_int m,
            lapack_int n, lapack_int nrhs, complex* a, lapack_int lda, complex*
            b, lapack_int ldb ) 
    lapack_int LAPACK_COL_MAJOR

cpdef zgels(np.ndarray[NPCOMPLEX_t,ndim=2] A, np.ndarray[NPCOMPLEX_t,ndim=1] b)
cpdef np.ndarray[complex, ndim=2] su2_rz(np.ndarray[complex, ndim=2] m, float p)
cpdef inline su2_rz_ias(m, p)
cpdef inline su2_rz_lsq_f(p, sh, H_sh, term)
