# -*- coding: utf-8 -*-
# filename = pycfl.pyx
#cython: c_string_encoding=ascii
#cython: embedsignature=True

#   Copyright (C) 2014-2017 Sebastian Horvath (sebastian.horvath@gmail.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division

cimport cython
cimport numpy as np

from pycf cimport cfl

import copy
import sys
import warnings
from contextlib import contextmanager
from numbers import Number

import numpy as np
from numpy.lib.stride_tricks import as_strided

from cpython cimport Py_DECREF, Py_INCREF
from cpython.pycapsule cimport *
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from pycf.cfl_util import *
from pycf.matel import matel

# Global storage for Python error handler callback
_python_error_handler = None


# C callback wrapper for Python error handlers
cdef void _c_error_handler_wrapper(const char *func, const char *file,
    int line, const char *message) noexcept nogil:
    """
    C callback that bridges to Python error handler.
    Must acquire GIL before calling Python code.
    """
    global _python_error_handler
    with gil:
        if _python_error_handler is not None:
            _python_error_handler(
                func.decode('utf-8') if func else "",
                file.decode('utf-8') if file else "",
                line,
                message.decode('utf-8') if message else ""
            )


def set_error_handler(handler):
    """
    Register custom error handler for CFL library errors.

    The error handler receives:
    - func: C function name where error occurred
    - file: Source file name
    - line: Line number
    - message: Error message

    Parameters
    ----------
    handler : callable or None
        Error handler function with signature:
            handler(func: str, file: str, line: int, message: str) -> None

        If None, restores default printf() error reporting.

    Examples
    --------
    >>> def log_error(func, file, line, msg):
    ...     print(f"ERROR in {func} ({file}:{line}): {msg}")
    >>> pycf.cfl.set_error_handler(log_error)

    >>> # Restore default behavior
    >>> pycf.cfl.set_error_handler(None)

    Notes
    -----
    The handler function should be fast and non-blocking, as it may be called
    from performance-critical code paths. Avoid heavy computation or I/O.

    The handler will be called even for non-fatal warnings and informational
    messages, depending on the error type.
    """
    global _python_error_handler
    _python_error_handler = handler

    if handler is not None:
        cfl.cfl_set_error_handler(_c_error_handler_wrapper)
    else:
        cfl.cfl_set_error_handler(NULL)


cdef inline void* _capsule_get_pointer(object cap, const char* name):
    """
    Safe capsule pointer extraction with validation.
    Raises TypeError if capsule is invalid or has wrong name.
    """
    if cap is None:
        raise TypeError("Capsule is None")
    if not PyCapsule_IsValid(cap, name):
        raise TypeError("Invalid capsule: either deleted or wrong type")
    return PyCapsule_GetPointer(cap, name)


cdef class StateLabels:
    r"""
    State label type for tensors and spin Hamiltonians.  State labels are
    generally not entered manually but should be generated with
    :class:`import_sljm.ImportSLJM`.

    Parameters
    ----------
    label_key : string
        String identifying the type of each state label.  Valid keys are: S, L,
        J, M and I, and the order in which they are listed must correspond to
        the order used in the list for each state in the labels list.
    labels : list
        List of strings, with each string corresponding to a specific state, and
        string elements indicating the respective label values of that state.
        The order of the labels in state strings is specified using the label_key
        argument.  To avoid half integers, label values are always stored as
        twice their real value.
    """
    cdef cfl.sl *cfl_sl
    cdef public object sl_cap
    cdef public list labels
    cdef public str label_key
    def __cinit__(self, label_key, labels):
        cdef size_t n
        cdef char *key
        cdef bytes key_b
        cdef np.ndarray[int, ndim=1, mode='c'] clabels
        cdef int **l_a

        self.label_key = label_key
        self.labels = labels

        n = <size_t> len(labels)
        key_b = label_key.encode('utf-8')
        key = key_b
        l_a = <int **>malloc(len(labels)*sizeof(int *))
        if l_a == NULL:
            raise MemoryError("l_a malloc failed")

        try:
            nplabels = []
            for i,l in enumerate(labels):
                nplabels += [np.ascontiguousarray(np.array(labels[i], dtype=np.int32))]
                clabels = nplabels[i]
                l_a[i] = &clabels[0]

            # sl_alloc copies both the label key and the label arrays (via memcpy),
            # so nplabels and key_b can safely go out of scope after this call.
            self.cfl_sl = cfl.sl_alloc(n, key, l_a)
            if self.cfl_sl == NULL:
                raise MemoryError("cfl_sl alloc failed")
            self.sl_cap = PyCapsule_New(<void *>self.cfl_sl, "pycfl.StateLabels", NULL)
        finally:
            free(l_a)

    def __dealloc__(self):
        if self.cfl_sl != NULL:
            cfl.sl_free(self.cfl_sl)

cdef class Tensor:
    r"""
    The Tensor class provides an interface for the creation of cfl zt objects.
    They are employed for the creation of both complete Hamiltonians and the
    projection of spin Hamiltonian interactions from complete Hamiltonians.
    Objects of type Tensor support standard arithmetic operations and can be
    added, subtracted, and scaled to yield new Tensor objects.

    Tensors should typically not be created manually but imported from emp sljm
    output files using :class:`import_sljm.ImportSLJM`.

    Parameters
    ----------
    name : string
        A string that uniquely identifies the tensor.
    a : np.ndarray
        A two dimensional array containing the matrix elements of the tensor.

    Returns
    -------
    t : Tensor

    """
    cdef public object t_cap
    cdef public str name
    cdef public str arith_name
    cdef public int n
    cdef public StateLabels states
    def __cinit__(self, name, np.ndarray[int, ndim=1, mode='c'] row_ptr,
            np.ndarray[int, ndim=1, mode='c'] col_in, np.ndarray[double complex, ndim=1, mode='c'] val,
            states, object data_tuple=None):
        cdef cfl.zt *t
        cdef cfl.zt *t1
        cdef cfl.zt *t2
        cdef int *col_ptr = NULL
        cdef double complex *val_ptr = NULL
        cdef Py_ssize_t n
        cdef Py_ssize_t expected_nnz
        cdef Py_ssize_t i
        cdef bytes name_b
        cdef char *name_c
        self.states = states

        # Hold an explicit bytes buffer so the C string outlives the calls
        # below; passing <char *> on a Python str directly produces a
        # temporary whose lifetime is fragile.
        if isinstance(name, bytes):
            name_b = name
            name_str = name.decode('utf-8')
        else:
            name_str = name
            name_b = (<str>name).encode('utf-8')
        name_c = name_b

        if (data_tuple is None):
            if not isinstance(states, StateLabels):
                raise TypeError("states must be a StateLabels instance")
            if len(row_ptr) < 2:
                raise ValueError("row_ptr must contain at least two entries")

            n = len(row_ptr)-1
            if n > 2147483647:
                raise ValueError("Tensor dimension is too large for the C API")
            if n != len(states.labels):
                raise ValueError("row_ptr dimension must match StateLabels length")
            if row_ptr[0] != 0:
                raise ValueError("row_ptr must start at 0")
            for i in range(n):
                if row_ptr[i] > row_ptr[i+1]:
                    raise ValueError("row_ptr entries must be nondecreasing")

            expected_nnz = row_ptr[n]
            if expected_nnz < 0:
                raise ValueError("row_ptr cannot contain negative indices")
            if len(col_in) != expected_nnz or len(val) != expected_nnz:
                raise ValueError(
                    "col_in and val lengths must match row_ptr[-1]"
                )
            for i in range(expected_nnz):
                if col_in[i] < 0 or col_in[i] >= n:
                    raise ValueError(
                        "column indices must be in the range [0, n)"
                    )
            if expected_nnz > 0:
                col_ptr = &col_in[0]
                val_ptr = &val[0]

            self.name = name_str
            self.arith_name = None
            self.n = n
            t = cfl.zt_csr_alloc(name_c, <int>n, &row_ptr[0], col_ptr, val_ptr,
                    <cfl.sl *>PyCapsule_GetPointer(states.sl_cap, "pycfl.StateLabels"))

        elif (len(data_tuple)==3):
            # Addition or subtraction of tensors; we use the arithmetic name for
            # zt_sa.
            self.name = None
            self.arith_name = name_str
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t2 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[1].t_cap, "pycfl.Tensor")
            t = cfl.zt_sa(name_c, t1, t2, 1, data_tuple[2])

        else:
            # Scaling of a tensor; we use the arithmetic name for zt_sa.
            self.name = None
            self.arith_name = name_str
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t = cfl.zt_s(name_c, t1, <double complex> data_tuple[1])

        if t is NULL:
            self.t_cap = None
            raise MemoryError("Cannot alloc zt memory")
        else:
            self.t_cap = PyCapsule_New(<void *>t, "pycfl.Tensor", NULL)

    def __dealloc__(self):
        if self.t_cap is not None:
            try:
                if PyCapsule_IsValid(self.t_cap, "pycfl.Tensor"):
                    cfl.zt_free(<cfl.zt *>PyCapsule_GetPointer(self.t_cap, "pycfl.Tensor"))
            except (TypeError, ValueError):
                pass

    def __add__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if t1.name is None:
            t1name = t1.arith_name
        else:
            t1name = t1.name
        if t2.name is None:
            t2name = t2.arith_name
        else:
            t2name = t2.name
        tmp_name = "{0}+{1}".format(t1name, t2name)
        d = (t1, t2, 1)
        return Tensor(tmp_name, None, None, None, t1.states, data_tuple=d)

    def __sub__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be subtracted from Tensors")
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if t1.name is None:
            t1name = t1.arith_name
        else:
            t1name = t1.name
        if t2.name is None:
            t2name = t2.arith_name
        else:
            t2name = t2.name
        tmp_name = "{0}-{1}".format(t1name, t2name)
        d = (t1, t2, -1)
        return Tensor(tmp_name, None, None, None, t1.states, data_tuple=d)

    def __mul__(x, y):
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if isinstance(x, Number):
            if isinstance(y, Tensor):
                if y.name is None:
                    yname = y.arith_name
                else:
                    yname = y.name
                tmp_name = "{0:.2f}*{1}".format(x, yname)
                d = (y, x)
                return Tensor(tmp_name, None, None, None, y.states, data_tuple=d)
        elif isinstance(x, Tensor):
            if isinstance(y, Number):
                if x.name is None:
                    xname = x.arith_name
                else:
                    xname = x.name
                tmp_name = "{0:.2f}x{1}".format(y, xname)
                d = (x, y)
                return Tensor(tmp_name, None, None, None, x.states, data_tuple=d)
        else:
            raise TypeError("Tensors can only be multiplied by scalar numbers")

    def __rmul__(self, other):
        if isinstance(other, Number):
            if self.name is None:
                selfname = self.arith_name
            else:
                selfname = self.name
            tmp_name = "{0:.2f}*{1}".format(other, selfname)
            d = (self, other)
            return Tensor(tmp_name, None, None, None, self.states, data_tuple=d)
        raise TypeError("Tensors can only be multiplied by scalar numbers")

    def get_name(self):
        """
        Get the name of this Tensor.  If it has not been explicitly named (by
        instantiation or setting of the name attribute post creation by
        arithmetic), we set the Tensor's name to the name of the variable that
        this tensor is assigned to in the __main__ namespace.  This is useful,
        since often new tensors are created by the scaling or addition of other
        Tensors, which, with out recourse to this hack, would require us to
        explicitly name such tensors after instantiation.  If more than one
        variable points to the same Tensor object, get_name cannot guarantee a
        unique name and will raise a RuntimeError.
        """
        if self.name is not None:
            return self.name
        else:
            name = [k for k,v in sys.modules['__main__'].__dict__.items() if v is self]
            if len(name) == 1:
                return name[0]
            else:
                raise RuntimeError("Found multiple variable names pointing to "\
                        "the same Tensor.  This occurs when the same Tensor "\
                        "object is assigned to multiple variables in the "\
                        "interpreters namespace, which means the "\
                        "Tensor.get_name() method can no longer guarantee a "\
                        "unique name for this tensor.  To solve this problem, "\
                        "either ensure all Tensors are only assigned to a single "\
                        "variable, or explicitly set the Tensor.name attribute of "\
                        "all Tensor objects created by Tensor arithmetic.  All "\
                        "tensors imported with ImportSLJM automatically have "\
                        "their name attribute set.")
    def get_matel(self):
        """
        Returns the matrix elements of this tensor in a dense array.
        """
        cdef np.ndarray[double complex, ndim=2, mode="c"] matel

        matel = np.ascontiguousarray(np.zeros((self.n,self.n), dtype=np.complex128))
        cfl.zt_get_matel(<cfl.zt *>PyCapsule_GetPointer(self.t_cap, "pycfl.Tensor"), &matel[0,0])

        return matel

cdef class Hamiltonian:
    r"""
    The crystal field Hamiltonian class.  Creates a cfl zh object and provides
    an interface for diagonalizing zh.  Can be used to calculate:

    * energy levels given a list of :class:`Tensor` objects and corresponding
      coefficients;
    * spin Hamiltonian parameters from crystal field parameters;
    * crystal field parameters by fitting to either energy levels or both
      energy levels and spin Hamiltonian parameters.

    A summary of calculated energy levels can be generated with
    :func:`cfl_util.gen_e_summary`.

    Hamiltonians are iterable, returning the Tensor objects from which it is composed.

    **Parameters for (mu, n) fitting format:**

    For fitting with experimental data in (mu, n) format (see :class:`ExData`),
    two optional parameters control the mu/n conversion:

    - ``minimum_q``: The smallest non-zero q value in your crystal field expansion.
      Common values are 2 (for C20, C22) or 4 (for higher-order terms).
      **REQUIRED** if using (mu, n) experimental data format. Defaults to ``None``.

    - ``half_integer_states``: Set to ``True`` if your system has half-integer 
      magnetic quantum numbers m stored as doubled integers (e.g., ±1, ±3, ±5
      representing ±1/2, ±3/2, ±5/2). Use ``True`` for f-electrons and other systems
      with J = 5/2, 7/2, etc. Set to ``False`` for integer m values. 
      Defaults to ``False``.

    **When to use (mu, n) format:**

    The (mu, n) parametrization is useful when symmetry is reduced and magnetic 
    quantum numbers become ambiguous. Example: Ce:YLF (f-electrons) where magnetic 
    decoherence mixes |m⟩ states and only folded combinations have physical meaning.

    **Example usage:**

    .. code-block:: python

        import pycf
        
        # For f-electron system (Ce:YLF)
        h = pycf.cfl.Hamiltonian(tensors)
        h.minimum_q = 2              # C20, C22 terms
        h.half_integer_states = True # f-electrons: m = ±1/2, ±3/2, ±5/2
        
        # For d-electron system (Er:YSO with integer m)
        h2 = pycf.cfl.Hamiltonian(tensors2)
        h2.minimum_q = 4              # Higher-order expansion
        h2.half_integer_states = False # integer m values

    Parameters
    ----------
    tensors : list
        A list with components of type Tensor; this specifies the type of
        interactions modeled by the Hamiltonian.
    label : str, optional
        A human-readable label for this Hamiltonian (e.g. ``"Ground state"``,
        ``"B || c"``).  Surfaced in summary output produced by
        :func:`cfl_util.gen_e_summary` and the EData/covariance helpers, and
        used to disambiguate Hamiltonians in multi-Hamiltonian fits
        (:class:`MHFit`).  Defaults to ``None`` (no label printed).

    Returns
    -------
    h : Hamiltonian

    """
    cdef cfl.zh *cfl_zh
    cdef cfl.zt **tensor_array
    cdef public int n
    cdef public int nt
    cdef public list tensors
    cdef public dict coeff_dict
    cdef public np.ndarray coeff
    cdef public np.ndarray w
    cdef public np.ndarray z
    cdef public object h_cap
    cdef public object label
    cdef public object minimum_q
    cdef public object half_integer_states
    cdef int diag_run
    def __cinit__(self, tensors, *, label=None):

        if len(tensors) == 0:
            raise ValueError("Hamiltonian requires at least one Tensor")
        for i,t in enumerate(tensors):
            if not isinstance(t, Tensor):
                raise TypeError("Hamiltonian inputs must be Tensor objects")

        n = tensors[0].n
        for i,t in enumerate(tensors):
            if t.n != n:
                raise ValueError(
                    "All tensors in a Hamiltonian must have the same dimension"
                )
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
        self.coeff_dict = None
        self.diag_run = 0
        if label is not None and not isinstance(label, str):
            raise TypeError("Hamiltonian label must be a str or None")
        self.label = label
        self.minimum_q = None
        self.half_integer_states = False

        # Create array of tensors and array of character arrays to be passed to
        # the zh_set cfl function.
        tensor_array = <cfl.zt **>malloc(len(tensors)*sizeof(cfl.zt *))
        if tensor_array is NULL:
            raise MemoryError("tensor_array alloc failed")

        self.tensor_array = tensor_array
        for i,t in enumerate(tensors):
            tensor_array[i] = <cfl.zt *> PyCapsule_GetPointer(t.t_cap, "pycfl.Tensor")

        # Allocate storage for zh.
        self.cfl_zh = cfl.zh_alloc(n, self.nt, tensor_array)
        if self.cfl_zh is NULL:
            free(tensor_array)
            # NULL self.tensor_array so __dealloc__ does not attempt a
            # second free of the same pointer after the exception propagates.
            self.tensor_array = NULL
            raise MemoryError("cfl_zh alloc failed")
        else:
            self.h_cap = PyCapsule_New(<void *>self.cfl_zh, "pycfl.Hamiltonian", NULL)

    def __dealloc__(self):
        if self.cfl_zh is not NULL:
            cfl.zh_free(self.cfl_zh)

        if self.tensor_array is not NULL:
            free(self.tensor_array)

    def __iter__(self):
        for t in self.tensors:
            yield t

    def __contains__(self, tensor):
        if isinstance(tensor, Tensor):
            return tensor in self.tensors
        elif isinstance(tensor, str):
            return tensor in [t.get_name() for t in self.tensors]
        else:
            raise ValueError("Membership test is only available for Tensor type "\
                    "objects and tensor names (strings)")

    def index(self, tensor):
        if isinstance(tensor, Tensor):
            try:
                return self.tensors.index(tensor)
            except ValueError:
                raise ValueError("The tensor {} is not an element of this "\
                        "Hamiltonian".format(tensor.get_name()))

        elif isinstance(tensor, str):
            try:
                return [t.get_name() for t in self.tensors].index(tensor)
            except ValueError:
                raise ValueError("The tensor {} is not an element of this "\
                        "Hamiltonian".format(tensor))
        else:
            raise ValueError("The index method is only available for Tensor type "\
                "objects and tensor names (strings)")


    cpdef set_coeff(self, coeff):
        r"""
        Set the tensor coefficients.

        Parameters
        ----------
        coeff : dict
            Must contain an element for each tensor specified when the
            Hamiltonian object was instantiated.  Keys have to be the same as
            tensor names.
        """
        cdef np.ndarray[double complex, ndim=1, mode='c'] co

        if not isinstance(coeff, dict):
            raise TypeError("coeff is not a dictionary.")

        # Keep copy of dict; fitting routines need to know the original type of
        # coeff elements to determine whether a parameter is real or complex.
        self.coeff_dict = copy.deepcopy(coeff)

        self.coeff = np.array([], dtype=np.complex128)
        for t in self.tensors:
            try:
                coeff_val = coeff[t.get_name()]
                # Validate that coefficient is numeric (int, float, or complex)
                if not isinstance(coeff_val, (int, float, complex, np.number)):
                    raise TypeError("Coefficient for tensor '%s' must be numeric, got %s" %
                                    (t.get_name(), type(coeff_val).__name__))
                self.coeff = np.append(self.coeff, coeff_val)
            except KeyError:
                raise KeyError("Missing coefficient for tensor: %s" % t.get_name())

        co = <np.ndarray[double complex, ndim=1, mode='c']> self.coeff
        cfl.zh_set_coeff(self.cfl_zh, &co[0])

        return None


    cpdef update_coeff(self, coeff):
        r"""
        Update the tensor coefficients for a subset of tensors.  This method can
        only be called after an initial set_coeff call has been made.

        Parameters
        ----------
        coeff : dict
            Keys have to be the same as tensor names.
        """
        cdef np.ndarray[double complex, ndim=1, mode='c'] co

        if self.coeff_dict is None:
            raise ValueError("Hamiltonian must have coefficients set prior to call of" \
                    "update_coeff.")
        elif not isinstance(coeff, dict):
            raise TypeError("coeff is not a dictionary.")

        # Note: coeff_dict may contain extra keys beyond self.tensors because
        # the same dict is often shared between multiple Hamiltonians built
        # from different tensor lists. set_coeff only reads the keys matching
        # self.tensors; the KeyError below catches any genuinely missing one.
        self.coeff_dict.update(copy.deepcopy(coeff))
        self.coeff = np.array([], dtype=np.complex128)
        for t in self.tensors:
            try:
                self.coeff = np.append(self.coeff, self.coeff_dict[t.get_name()])
            except KeyError:
                raise KeyError("Tensor '%s' not found in coefficient dictionary" % t.get_name())

        co = <np.ndarray[double complex, ndim=1, mode='c']> self.coeff
        cfl.zh_set_coeff(self.cfl_zh, &co[0])

        return None


    @cython.boundscheck(False)
    cpdef diag(self):
        r"""
        Diagonalize the Hamiltonian.

        Returns
        -------
        (w, z) : tuple
            The eigenvalues and eigenvectors, respectively, of the diagonalized
            Hamiltonian.

        """
        cdef cfl.zh *h = self.cfl_zh
        cdef cfl.zhd_w *hd_w
        cdef np.ndarray[double, ndim=1, mode="c"] w
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] z

        self.w = np.ascontiguousarray(np.zeros(self.n, dtype=np.float64))
        self.z = np.asfortranarray(np.zeros((self.n,self.n), dtype=np.complex128))
        w = <np.ndarray[double, ndim=1, mode="c"]> self.w
        z = <np.ndarray[double complex, ndim=2, mode="fortran"]> self.z

        if self.coeff_dict is None:
            raise ValueError("Hamiltonian must have coefficients set prior to diagonalization.")
        hd_w = cfl.zhd_w_alloc('V', self.cfl_zh)
        if hd_w is NULL:
            free(self.tensor_array)
            cfl.zh_free(self.cfl_zh)
            raise MemoryError("hd_w alloc failed")

        try:
            with nogil:
                cfl.zhd('V', &w[0], &z[0,0], h, hd_w)
        finally:
            cfl.zhd_w_free(hd_w)

        self.diag_run = 1

        return (w, z)

    def gen_summary(self, **kwargs):
        r"""
        Generate an energy level summary resulting from a diagonalization.

        Returns
        -------
        ex : np.ndarray, optional
            A 2 by m array, specifying the experimental energy levels, with m the
            number of available experimental levels.  The first column specifies the
            index of the corresponding entry in the complete eigenvalue vector, and
            the second column contains the energy level values.
        nstates : int, optional
            The number of constituent states to display for mixed states.
        chi2 : float, optional
            The final chi2 value of the fit.
        ndof : int, optional
            The number of degrees of freedom of the fit; that is, the number of
            observables minus the number of parameters.
        weighting : float, optional
            The weighting applied to during the chi2 fit.  This needs to be
            provided if ndof is set.
        e_shift : bool, optional
            Shift entire eigenvalue spectrum s.t. the first eigenvalue is zero.
        """
        if self.diag_run:
            if "h_label" not in kwargs and self.label is not None:
                kwargs["h_label"] = self.label
            # Pass minimum_q and half_integer_states if set on Hamiltonian
            if self.minimum_q is not None and "minimum_q" not in kwargs:
                kwargs["minimum_q"] = self.minimum_q
            if "half_integer_states" not in kwargs:
                kwargs["half_integer_states"] = self.half_integer_states
            return gen_e_summary(self.w, self.z, self.tensors[0].states.labels,
                    self.tensors[0].states.label_key, **kwargs)
        else:
            raise ValueError("Hamiltonian must have run diag prior to summary generation.")

    def validate_mu_parameters(self) -> None:
        r"""
        Validate that mu/n fitting parameters are correctly configured.

        This method checks that `minimum_q` and `half_integer_states` are properly
        set before attempting to use (mu, n) experimental data format. Call this
        before creating an :class:`EFit` with mu/n data to catch configuration
        errors early with clear guidance.

        Raises
        ------
        ValueError
            If `minimum_q` is None (not set).
        TypeError
            If `half_integer_states` is not a bool.

        Examples
        --------
        Validate before fitting with (mu, n) data:

        .. code-block:: python

            h = pycf.cfl.Hamiltonian(tensors)
            h.minimum_q = 2              # Required: set smallest non-zero q
            h.half_integer_states = True # f-electrons use True, d-electrons use False
            h.set_coeff(coefficients)
            h.diag()

            # Optional: validate before using mu/n data
            h.validate_mu_parameters()

            # Now safe to fit with (mu, n) format
            exdata = pycf.cfl.ExData((mu_n_pairs, energies),
                                    key=('mu', 'n', 'energy'))
            fit = pycf.cfl.EFit(h, exdata)
        """
        if self.minimum_q is None:
            raise ValueError(
                "Hamiltonian.minimum_q must be set before using (mu, n) fitting format. "
                "Set it to the smallest non-zero q value in your crystal field expansion "
                "(typically 2 for C20, C22 terms or 4 for higher-order expansions)."
            )
        if not isinstance(self.half_integer_states, bool):
            raise TypeError(
                f"Hamiltonian.half_integer_states must be a bool, got {type(self.half_integer_states).__name__}. "
                "Set to True for f-electrons (half-integer m values) or False for integer m values."
            )


cpdef zeeman_sh_coeff(v, t):
    r"""
    Generate the Zeeman interaction spin Hamiltonian 'coefficient array'.  This
    consists of a `2j+1 \times 2j+1` by `3 \times 3` array containing the matrix
    elements of the terms `B_a J_b`, with `a,b \in \{x, y, z\}` and `j` the
    angular momentum of the rank one tensor `J` (either 'S' or 'I').  Here the
    rows enumerate the `2j+1 \times 2j+1` different state combinations while the
    columns enumerate all combinations of `a` and `b`.

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
    # Validate input arrays
    if not isinstance(v, np.ndarray):
        raise TypeError("v must be a numpy.ndarray")
    if len(v) != 3:
        raise ValueError("v must have length 3 (B_x, B_y, B_z)")

    if not isinstance(t, (list, tuple)) or len(t) != 3:
        raise ValueError("t must be a sequence of 3 matrices (J_x, J_y, J_z)")

    for i, mat in enumerate(t):
        if not isinstance(mat, np.ndarray):
            raise TypeError("t[%d] must be a numpy.ndarray" % i)
        if mat.ndim != 2:
            raise ValueError("t[%d] must be 2-dimensional" % i)
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("t[%d] must be square" % i)

    # Verify all matrices have same dimensions
    tl = t[0].shape[0]
    for i in range(1, 3):
        if t[i].shape[0] != tl or t[i].shape[1] != tl:
            raise ValueError("All matrices in t must have same dimensions")

    l = len(t)
    a = np.zeros([tl, tl, l, l], dtype = np.complex128)

    for tr in range(tl):
        for tc in range(tl):
            for i in range(l):
                for j in range(l):
                    a[tr, tc, i, j] = v[i] * t[j][tr, tc]

    return(np.reshape(a, (tl*tl, l*l)))


cpdef hyperfine_sh_coeff(t1, t2):
    r"""
    Generate the hyperfine interaction spin Hamiltonian 'coefficient array'.
    This consists of a `(2j_1+1 \times 2j_2+1) \times (2j_1+1 \times 2j_2+1)` by
    `3 \times 3` array containing the matrix elements of the operators `I_a
    S_b`, with `a,b \in \{x, y, z\}` and `j_1` and `j_2` the angular momentum of
    the rank one tensors `I` and `S`, respectively.  Here the rows enumerate the
    `(2j_1+1 \times 2j_2+1) \times (2j_1+1 \times 2j_2+1)` different state
    combinations while the columns enumerate all combinations of `a` and `b`.

    Parameters
    ----------
    t1 : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    t2 : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : numpy.ndarray
        A `(2j_1+1 \times 2j_2+1) \times (2j_1+1 \times 2j_2+1)` by `3 \times 3`
        array.
    """

    t1l = len(t1[0])
    t2l = len(t2[0])
    l = len(t1)

    a = np.zeros([t1l * t2l, t1l * t2l, l, l], dtype = np.complex128)

    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(l):
                        for j in range(l):
                            a[t1r+t1l*t2r,t1c+t1l*t2c,i,j]=t1[i][t1r,t1c]*t2[j][t2r,t2c]

    return(np.reshape(a, (t1l*t2l*t1l*t2l, l*l)))


cpdef quadrupole_sh_coeff(t):
    r"""
    Generate the quadrupole interaction spin Hamiltonian 'coefficient array'.
    This consists of a `2j+1 \times 2j+1` by `3 \times 3` array containing the
    matrix elements of the operators `I_a I_b`, with `a,b \in \{x, y, z\}` and
    `j` the angular momentum of the rank one tensor `I`.  Here the rows
    enumerate the `2j+1 \times 2j+1` different state combinations while the
    columns enumerate all combinations of `a` and `b`.

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
    l = len(t)
    a = np.zeros([tl, tl, l, l], dtype = np.complex128)

    for tr in range(tl):
        for tc in range(tl):
            for i in range(l):
                for j in range(l):
                    components = 0
                    for ci in range(tl):
                        components +=t[i][tr, ci] * t[j][ci, tc]
                    a[tr, tc, i, j] = components

    return(np.reshape(a, (tl*tl, l*l)))


cdef sh_hpro_helper(h, sh):
    """
    Add small magnetic field along z for state-label sorting if not already
    present, and test whether a separate projection Hamiltonian is required.

    Parameters
    ----------
    h : Hamiltonian
    sh : SpinHamiltonian

    Returns
    -------
    hpro : Hamiltonian
        The dedicated projection Hamiltonian, if required; otherwise, None will
        be returned.
    """
    # If not present, add small magnetic field to Hamiltonian to order
    # states.
    if not any([t.get_name() == 'MAGZS' for t in h.tensors]):
        for t in sh.tensors:
            if t.get_name() == 'MAGZ':
                magzs = 1e-6 * t
                magzs.name = 'MAGZS'

        # Fix: copy coeff_dict before modifying to avoid mutating the caller's dict.
        tmp_h_coeff = dict(h.coeff_dict)
        tmp_h_coeff['MAGZS'] = 1
        try:
            h_new = Hamiltonian([magzs] + h.tensors)
            h_new.set_coeff(tmp_h_coeff)
            h = h_new
        except:
            # If new Hamiltonian creation fails, use original h
            h.update_coeff({'MAGZS' : 1})
            raise
    else:
        h.update_coeff({'MAGZS' : 1})

    # Check whether the provided Hamiltonian contains spin Hamiltonian
    # interaction matrix elements, in which case we create a separate
    # Hamiltonian to perform the spin Hamiltonian projection which has these
    # matrix elements removed.
    pro_tensor_list = ['MAGX', 'MAGY', 'MAGZ', 'HYP', 'EQHYP']
    pro_h_tensors = []
    create_pro_h = False
    for t in h:
        if t.get_name() not in pro_tensor_list:
            pro_h_tensors += [t]
        else:
            create_pro_h = True

    if create_pro_h:
        tmp_coeff = h.coeff_dict
        hpro = Hamiltonian(pro_h_tensors)
        hpro.set_coeff(tmp_coeff)
    else:
        hpro = None

    return (h, hpro)


cpdef sh_svd(m):
    r"""
    Use a singular value decomposition to symmeterize a 3 by 3 spin Hamiltonian
    parameter array. The intended use of this function is to allow any
    experimental parameter values to be transformed to the same basis as the
    projected parameter matrices.

    Parameters
    ----------
    m : np.ndarray
        The 3 by 3 spin Hamiltonian parameter array.
    """
    cdef cfl.svd_sym_w *work
    cdef np.ndarray[double, ndim=1, mode="c"] cm

    if m.shape != (3,3):
        raise ValueError("m must be a 3 by 3 array.")

    cm = np.ascontiguousarray(m.flatten(), dtype=np.float64)
    work = cfl.svd_sym_w_alloc()
    if work == NULL:
        raise MemoryError("Failed to allock SVD workspace")

    cfl.svd_sym(&cm[0], work)
    cfl.svd_sym_w_free(work)

    return cm.reshape(3,3)


cdef class SpinHamiltonian:
    r"""
    Abstraction for spin Hamiltonian data.  Objects of type SpinHamiltonian are
    used for calculating spin Hamiltonian parameters from crystal field
    parameters in conjunction with :class:`Hamiltonian` objects.

    The type of data that a SpinHamiltonian object represents depends on the
    specified interactions, but can be loosly thought of as the matrix elements
    of for all specified interactions; for Zeeman interactions, this will be
    three sets of matrix elements.  Objects of this type are used by the
    function :func:`esh_fit` to fit crystal field parameters to spin Hamiltonian
    data.

    Parameters
    ----------
    interactions : list
        Elements are strings which specify the interactions of the spin
        Hamiltonian.  Possible values are: 'zeeman', 'hyperfine', and
        'quadrupole'.
    level : int
        The level of the complete Hamiltonian for which to project the spin
        Hamiltonian; uses 1 as base index (ground state = 1).
    S : float
        The spin projection `S_z`; if ``interactions`` contains 'zeeman' or
        'hyperfine' this keyword argument must be specified.
    I : float
        The nuclear spin projection `I_z`; if ``interactions`` contains
        'hyperfine' or 'quadrupole' this keyword argument must be specified.
    kramers : bool
        Default is True; specifies whether this is a Kramers ion spin
        Hamiltonian.

    Returns
    -------
    object : SpinHamiltonian
    """
    cdef cfl.zsh *cfl_zsh
    cdef public list interactions
    cdef public list required_tensors
    cdef public int level
    cdef public int nsh
    cdef public int n_obs
    cdef public float Sz
    cdef public list S_matel
    cdef public float Iz
    cdef public list I_matel
    cdef list inv_data
    cdef double complex **inv_data_ptrs
    cdef char **inter_array
    cdef public object sh_cap
    cdef public list tensors
    cdef public int pro_data_set
    cdef public dict coeff_dict
    cdef int dz
    cdef int dh
    cdef int dq
    cdef int kramers
    def __init__(self, interactions, **kwargs):
        cdef int csz
        cdef int ciz
        cdef int ckramers
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] a

        if not isinstance(interactions, list):
            interactions = [interactions]
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list: '{}'.".format(i))
        self.interactions = interactions
        if 'kramers' in kwargs:
            self.kramers = int(kwargs['kramers'])
        else:
            self.kramers = 1

        if 'level' not in kwargs:
            raise KeyError("SpinHamiltonian: missing keyword argument 'level'.")
        elif kwargs['level'] < 1:
            raise ValueError("level kwarg uses 1 base index.")
        self.level = kwargs['level']-1  # we use zero base index internally.

        # Calculate matrix elements for the specified interactions.
        j_l = ['jx', 'jy', 'jz']
        if self.kramers:
            if 'zeeman' in interactions or 'hyperfine' in interactions:
                try:
                    self.Sz = kwargs['S']
                except KeyError:
                    raise KeyError("SpinHamiltonian: missing keyword argument S.")
                # Calculate the matrix elements of spin operator.
                self.S_matel = [matel(j_l[i], self.Sz) for i in range(3)]
            else:
                self.S_matel = None

            if 'hyperfine' in interactions or 'quadrupole' in interactions:
                try:
                    self.Iz = kwargs['I']
                except KeyError:
                    raise KeyError("SpinHamiltonian: missing keyword argument I.")
                # Calculate the matrix elements of nuclear spin operator.
                self.I_matel = [matel(j_l[i], self.Iz) for i in range(3)]
            else:
                self.I_matel = None
        else:
            if 'zeeman' in interactions or 'quadrupole' in interactions:
                try:
                    self.Iz = kwargs['I']
                except KeyError:
                    raise KeyError("SpinHamiltonian: missing keyword argument I.")
                # Calculate the matrix elements of nuclear spin operator.
                self.I_matel = [matel(j_l[i], self.Iz) for i in range(3)]
            else:
                self.I_matel = None

        # Calculate the coefficient arrays and alloc spin Hamiltonian.
        n_inter = len(interactions)

        self.inter_array = <char **>malloc(n_inter*sizeof(char *))
        if self.inter_array == NULL:
            raise MemoryError("inter_array malloc failed")

        self.inv_data_ptrs = <double complex **>malloc(len(interactions)*sizeof(double complex *))
        if self.inv_data_ptrs == NULL:
            raise MemoryError("inv_data_ptrs malloc failed")

        self.nsh = 0
        self.n_obs = 0
        self.required_tensors = []
        self.inv_data = []
        for i,inter in enumerate(interactions):
            if inter == 'zeeman':
                # Coefficient arrays are calculated for three B fields in x, y,
                # and z directions, respectively.
                if self.kramers:
                    self.dz = int(2*self.Sz+1)
                    B_a = np.zeros([3, self.dz**2, 9], dtype = np.complex128)
                    for j in range(3):
                        B_a[j, :, :] = zeeman_sh_coeff(np.eye(3,3)[j,:], self.S_matel)
                else:
                    self.dz = int(2*self.Iz+1)
                    B_a = np.zeros([3, self.dz**2, 9], dtype = np.complex128)
                    for j in range(3):
                        B_a[j, :, :] = zeeman_sh_coeff(np.eye(3,3)[j,:], self.I_matel)

                self.inv_data += [np.asfortranarray(np.reshape(B_a, (3 * self.dz**2, 9)),
                    dtype=np.complex128)]
                self.nsh += 3
                # Three g-values plus three Euler rotation parameters.
                self.n_obs += 6
                self.required_tensors += ['MAGX', 'MAGY', 'MAGZ']

            if inter == 'hyperfine':
                self.dh = int((2*self.Sz+1) * (2*self.Iz+1))
                # The ordering of I_matel and S_matel are such that Iz is the
                # 'fast', that is the inner, index, while Sz is the 'slow', that
                # is the outer, index.
                self.inv_data += [np.asfortranarray(hyperfine_sh_coeff(self.I_matel, self.S_matel),
                    dtype=np.complex128)]
                self.nsh += 1
                # Three hyperfine values plus three Euler rotation parameters.
                self.n_obs += 6
                self.required_tensors += ['HYP']

            if inter == 'quadrupole':
                self.dq = int(2*self.Iz+1)
                self.inv_data += [np.asfortranarray(quadrupole_sh_coeff(self.I_matel), dtype=np.complex128)]
                self.nsh += 1
                # Two quadrupole values plus three Euler rotation parameters.
                self.n_obs += 5
                self.required_tensors += ['EQHYP']

            a = <np.ndarray[double complex, ndim=2, mode='fortran']> self.inv_data[i]
            self.inv_data_ptrs[i] = &a[0,0]
            self.inter_array[i] = inter

        csz = int(2*self.Sz)
        ciz = int(2*self.Iz)
        ckramers = int(self.kramers)
        self.cfl_zsh = cfl.zsh_alloc(self.inter_array, len(interactions), csz, ciz, ckramers, self.inv_data_ptrs);
        if self.cfl_zsh == NULL:
            raise MemoryError("Failed to alloc zsh")
        else:
            self.sh_cap = PyCapsule_New(<void *>self.cfl_zsh, "pycfl.SpinHamiltonian", NULL)

        self.tensors = None
        self.pro_data_set = 0

    def __dealloc__(self):
        if self.cfl_zsh != NULL:
            try:
                if PyCapsule_IsValid(self.sh_cap, "pycfl.SpinHamiltonian"):
                    cfl.zsh_free(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"))
            except TypeError:
                # Capsule was already invalidated; skip cleanup
                pass
        if self.inv_data_ptrs != NULL:
            free(self.inv_data_ptrs)
        if self.inter_array != NULL:
            free(self.inter_array)

    def __iter__(self):
        for t in self.tensors:
            yield t

    def __contains__(self, tensor):
        if (isinstance(tensor, Tensor)):
            return tensor in self.tensors
        elif (isinstance(tensor, str)):
            return tensor in [t.get_name() for t in self.tensors]
        else:
            raise ValueError("SpinHamiltonian: membership test is only available "\
                    "for Tensor type objects and tensor names (strings)")

    def index(self, tensor):
        if isinstance(tensor, Tensor):
            try:
                return self.tensors.index(tensor)
            except ValueError:
                raise ValueError("SpinHamiltonian: the tensor {} is not an "\
                        "element of this Hamiltonian".format(tensor.get_name()))

        elif isinstance(tensor, str):
            try:
                return [t.get_name() for t in self.tensors].index(tensor)
            except ValueError:
                raise ValueError("SpinHamiltonian: the tensor {} is not an "\
                        "element of this Hamiltonian".format(tensor))
        else:
            raise ValueError("SpinHamiltonian: the index method is only "\
                    "available for Tensor type objects and tensor names "\
                    "(strings)")


    def set_pro_data(self, tensors, coupling_constants={}):
        r"""
        Set the projection data for all spin Hamiltonian interactions.

        Parameters
        ----------
        tensor : list
            Elements must be of type Tensor.  The list must contain Tensors for
            every interaction specified when the SpinHamiltonian was created.
            These must have the following name attributes: 'MAGX', 'MAGY', and
            'MAGZ' for Zeeman interactions; 'HYP' for hyperfine interactions;
            'EQHYP' for quadrupole interactions.  Finally, even if the
            SpinHamiltonian does not include Zeeman interactions the 'MAGZ'
            tensor must be provided for state-label sorting.
        coupling_constants : dict, optional
            If hyperfine or quadrupole interactions are present, this dictionary
            has to be provided, which specifies the nuclear dipole and nuclear
            quadrupole coupling constants, using keys 'HYP' and 'EQHYP',
            respectively.
        """
        cdef cfl.zt **t_array
        cdef np.ndarray[double, ndim=1, mode="c"] cc
        cdef double *ccptr

        if not isinstance(tensors, list):
            raise TypeError("The tensors argument of set_pro_data must be a list "\
                    "of Tensor objects, not an object of type %s." % type(tensors))
        if not all((isinstance(t, Tensor) for t in tensors)):
            raise TypeError("The tensors argument of set_pro_data must be a list of Tensor objects")

        # Validate that the level is within bounds for the given tensors
        if tensors:
            nstates = tensors[0].n
            if self.level >= nstates:
                raise ValueError("level parameter (%d) must be less than number of states (%d)" %
                                (self.level + 1, nstates))

        t_array = <cfl.zt **>malloc(len(self.required_tensors)*sizeof(cfl.zt *))
        if t_array == NULL:
            raise MemoryError("t_array malloc failed")

        # Ensure all tensors required for projecting the interactions of this
        # spin Hamiltonian are provided.
        cc_list = []
        for i,rt in enumerate(self.required_tensors):
            try:
                t_array[i] = <cfl.zt *>PyCapsule_GetPointer(
                        next((t for t in tensors if t.get_name() == rt)).t_cap, "pycfl.Tensor")
            except StopIteration:
                free(t_array)
                raise ValueError("Missing tensor %s in tensors list in set_pro_data call." % rt)
            if rt == 'HYP':
                try:
                    cc_list += [coupling_constants['HYP']]
                except KeyError:
                    free(t_array)
                    raise KeyError("Missing the nuclear dipole coupling constant in set_pro_data call.")
            elif rt == 'EQHYP':
                try:
                    cc_list += [coupling_constants['EQHYP']]
                except KeyError:
                    free(t_array)
                    raise KeyError("Missing the nuclear quadrupole coupling constant in set_pro_data call.")

        self.coeff_dict = coupling_constants
        cc = np.array(cc_list, dtype=np.float64)
        if len(cc):
            ccptr = &cc[0]
        else:
            ccptr = NULL

        retval = zsh_set_pro(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"),
                t_array, self.level, ccptr)

        free(t_array)
        if retval != 1:
            raise ValueError("zsh_set_pro failed.  See the cfl error message for details.")

        self.tensors = tensors
        self.pro_data_set = 1


    def calc_param(self, h, matel=False, svd_sym=False):
        r"""
        Calculate the spin Hamiltonian parameters given a crystal-field
        Hamiltonian.

        Parameters
        ----------
        h : Hamiltonian
            The corresponding crystal-field Hamiltonian.
        matel : bool, optional
            If true, a dictionary containing the spin Hamiltonian matrix
            elements is returned.
        svd_sym : bool, optional
            Symmeterize spin Hamiltonian parameter tensors by applying an SVD
            transformation.


        Returns
        -------
        res : tuple
            If matel=False, a list of nd.arrays is returned.  These correspond
            to the spin Hamiltonian tensors of interactions specified when the
            spin Hamiltonian object was instantiated.
        """

        cdef cfl.zshp_w *shp_w
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] cz
        cdef np.ndarray[double, ndim=1, mode="c"] a
        cdef np.ndarray[double complex, ndim=1, mode="c"] b
        cdef Py_ssize_t d_inter = 0

        if not self.pro_data_set:
            raise ValueError("The spin Hamiltonian interaction is missing projection data.")

        # Add small magnetic field for state-label sorting; generate hpro, if
        # required.
        (h, hpro) = sh_hpro_helper(h, self)
        if hpro is not None:
            h = hpro

        # Check whether to perform SVD symmeterization.
        if svd_sym:
            svd = <char> 'S'
        else:
            svd = <char> 'N'

        (w, z) = h.diag()
        cz = <np.ndarray[double complex, ndim=2, mode="fortran"]> z
        shp_w = cfl.zshp_w_alloc(svd, <cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"))
        a = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(9, dtype=np.float64)

        result_list = []
        sh_matel = {}
        for i,inter in enumerate(self.interactions):
            if inter == 'zeeman':
                # Factor of 3 accounts for the orientations magx, magy, magz.
                d_inter = 3*self.dz**2
            elif inter == 'hyperfine':
                d_inter = self.dh**2
            elif inter == 'quadrupole':
                d_inter = self.dq**2
            else:
                raise ValueError("Unsupported interaction: %s" % inter)

            b = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(d_inter, dtype=np.complex128)

            cfl.zshp(&a[0], &b[0], &cz[0,0], i, <cfl.zsh *>PyCapsule_GetPointer(self.sh_cap,
                "pycfl.SpinHamiltonian"), shp_w)
            result_list += [np.copy(a.reshape(3,3))]

            if inter == 'zeeman':
                sh_matel['magx'] = b[0:self.dz*self.dz].reshape(self.dz, self.dz)
                sh_matel['magy'] = b[self.dz*self.dz:2*self.dz*self.dz].reshape(self.dz, self.dz)
                sh_matel['magz'] = b[2*self.dz*self.dz:3*self.dz*self.dz].reshape(self.dz, self.dz)
            elif inter == 'hyperfine':
                sh_matel['hyperfine'] = b.reshape(self.dh, self.dh)
            elif inter == 'quadrupole':
                sh_matel['quadrupole'] = b.reshape(self.dq, self.dq)

        cfl.zshp_w_free(shp_w)

        # Set small magnetic field that was used for state-label sorting to
        # zero.
        h.update_coeff({'MAGZS': 0})

        if matel:
            return ((result_list, sh_matel))
        else:
            return result_list


cdef class ExData(object):
    r"""
    Experimental energy level data for Hamiltonians.  If both absolute and
    difference energy levels are present, then ex.e will be ordered such that
    all absolute energy level values are before the difference energy levels.

    Parameters
    ----------
    data : np.ndarray or tuple
        If data is of type np.ndarray it assumed to be a 2 by n dimensional,
        with the first column containing energy level indices starting at 1, and
        the second column containing the absolute experimental energy of the
        corresponding level.  If data is of type tuple each element must be a
        np.ndarray of type specified using the key argument.  Implemented types
        are:

        - Absolute energy level data with level index (default).
        - Difference energy level data with level index; np.ndarray must be
          3 by n dimensions, where the first column specifies the initial
          energy level index, the second column specifies the final energy
          level index, and the third column corresponds to the energy
          difference.
        - Absolute energy level data with state label index; of dimension
          m+1 by n, where m is the number of state labels.  The first m
          elements are state labels in LS coupling with the type of label of
          each element specified by the label_key argument.  The (m+1)th
          entry contains the absolute experimental energy of the
          corresponding level.
        - Difference energy level data with state label index; of
          dimension 2m+1 by n, where m is the number of state labels.  The
          first m elements are state labels in LS coupling specifying the
          initial energy level.  The next m elements are state labels in LS
          coupling specifying the final energy level.  The final entry
          corresponds to the energy difference.  The type of state label
          elements are given by label_key.

        Note: mixing level index data with state label index data is not
        supported.
    key : str or tuple, optional
        If data is of type np.ndarray this argument is optional; otherwise it
        must be specified and be of the same length as the data tuple.  This
        argument is used to specify the type of data. Available keys are:

        - 'A', absolute energy data with level index;
        - 'D', difference energy data with level index;
        - 'AS', absolute energy level data with state label index;
        - 'DS', difference energy level data with state label index.
    label_key : str, optional
        This argument is only required if experimental data with state label
        indices is to be used.  In this case, each element of label_key
        specifies the type of each of the m state label entries passed via the
        data argument.  It must match the label_key of Hamiltonian to be fit to
        this experimental data.
    weights : np.ndarray or tuple, optional
        This argument can be used to specify a relative weighting between energy
        levels.  The overall weighting, used in fitting functions such as MHFit
        or ESHFit still scales the weighting of relative Hamiltonians, but this
        allows finer control within the energy level data specific to a single
        Hamiltonian.  If it is a tuple, each element must be an np.ndarray with
        a one-to-one correspondence to the energy levels in provided via the
        data parameter.
    """
    cdef public int sl_index
    cdef public int n_obs
    cdef public int n_a
    cdef public int n_d
    cdef public np.ndarray a_states
    cdef public np.ndarray id_states
    cdef public np.ndarray fd_states
    cdef public np.ndarray e
    cdef public np.ndarray w
    cdef public np.ndarray la
    cdef public np.ndarray ild
    cdef public np.ndarray fld
    cdef public np.ndarray lah
    cdef public np.ndarray ildh
    cdef public np.ndarray fldh
    # Storage for AMu/DMu raw data (not converted to level indices)
    cdef public np.ndarray mu_n_abs
    cdef public np.ndarray mu_n_diff
    cdef public bint has_mu_n
    def __init__(self, data, key=None, label_key=None, weights=None):
        cdef np.ndarray[int, ndim=1, mode='c'] clabels

        # Initialize mu_n attributes
        self.has_mu_n = False
        self.mu_n_abs = np.zeros((0, 2), dtype=np.float64)
        self.mu_n_diff = np.zeros((0, 4), dtype=np.float64)

        if not (isinstance(data, np.ndarray) or isinstance(data, tuple)):
            raise TypeError("The ex data argument must either be of type np.ndarray or " \
                    "tuple, not %s." % type(data))
        if weights is None:
            if isinstance(data, np.ndarray):
                weights = np.ones(data.shape[0], dtype=np.float64)
            else:
                # Handle both 1-element and 2-element tuples
                if len(data) == 1:
                    weights = (np.ones(data[0].shape[0], dtype=np.float64),)
                else:
                    weights = (np.ones(data[0].shape[0], dtype=np.float64),
                            np.ones(data[1].shape[0], dtype=np.float64))
        else:
            if isinstance(weights, np.ndarray):
                if data.shape[0] != len(weights):
                    raise ValueError("Weights must be of the same length as data.")

        if not (isinstance(weights, np.ndarray) or isinstance(weights, tuple)):
            raise TypeError("The weights argument must either be of type np.ndarray or " \
                    "tuple, not %s." % type(weights))
        if not (isinstance(key, str) or isinstance(key, tuple) or key is None):
            raise TypeError("The key argument must either be of type np.ndarray or tuple, " \
                    "not %s." % type(key))
        if isinstance(data, tuple):
            if key is None:
                raise ValueError("Missing key argument; this must be specified if data is a tuple.")
            elif len(data) < 1 or len(data) > 2:
                raise ValueError("The data argument must contain 1 or 2 elements.")
            elif not all(isinstance(e, np.ndarray) for e in data):
                raise TypeError("Elements of the data tuple must be of type np.ndarray.")
            elif not isinstance(key, tuple):
                raise TypeError("If the data argument is a tuple, the key argument must also be a tuple.")
            elif len(key) != len(data):
                raise ValueError("The key tuple must have the same length as the data tuple.")
            elif not all(isinstance(k, str) for k in key):
                raise TypeError("Elements of the key tuple must be of type str.")
            elif not isinstance(weights, tuple):
                raise TypeError("Weights must be of type tuple if data is specified as a tuple")
            elif not all(isinstance(e, np.ndarray) for e in weights):
                raise TypeError("Elements of the weights tuple must be of type np.ndarray.")
            elif len(weights) != len(data):
                raise ValueError("The weights argument must have the same length as the data tuple.")
        if key is not None:
            if not isinstance(key, tuple):
                # A single key is provided; we therefore make both data and key
                # iterable for key checks below, and such that key check is
                # forced in case of state labels (type(data) != np.ndarray).
                key = [key]
                data = [data]
                weights = [weights]

            if any(d.ndim != 2 for d in data):
                raise ValueError("All data arrays must be two dimensional.")

            for k in key:
                if k == 'A' or k == 'D':
                    if 'AS' in key or 'DS' in key or 'AMu' in key or 'DMu' in key:
                        raise ValueError("Mixed data types are not supported; use either " \
                                "(A/D), (AS/DS), or (AMu/DMu).")
                    self.sl_index = 0
                else:
                    if 'A' in key or 'D' in key:
                        raise ValueError("Mixed data types are not supported; use either " \
                                "(A/D), (AS/DS), or (AMu/DMu).")
                    elif 'AS' in key or 'DS' in key:
                        if 'AMu' in key or 'DMu' in key:
                            raise ValueError("Cannot mix state label types; use either " \
                                    "(AS/DS) or (AMu/DMu).")
                        # SLJM-based state labels
                        if type(label_key) != str:
                            raise TypeError("The label_key argument must be of type str and is " \
                                    "mandatory for state label indices.")
                        self.sl_index = 1
                    elif 'AMu' in key or 'DMu' in key:
                        # mu/n-based data: don't use state label hashes
                        self.sl_index = 0
                    else:
                        # Other state label types (shouldn't reach here currently)
                        if type(label_key) != str:
                            raise TypeError("The label_key argument must be of type str and is " \
                                    "mandatory for state label indices.")
                        self.sl_index = 1

                if k not in ['A', 'D', 'AS', 'DS', 'AMu', 'DMu']:
                    raise ValueError("Invalid key argument; allowed options are 'A', 'D', " \
                            "'AS', 'DS', 'AMu', and 'DMu'.")
        if isinstance(data, np.ndarray):
            if not data.shape[1] == 2:
                raise ValueError("Incorrect ex data shape; expected a two column array.")
            # No energy level differences.
            self.n_a = data.shape[0]
            self.n_d = 0

            self.e = np.ascontiguousarray(data[:, 1], dtype=np.float64)
            self.w = np.ascontiguousarray(weights, dtype=np.float64)

            # Subtract one, since we need an index starting at zero, whereas ex
            # levels start at 1.
            self.la = np.ascontiguousarray(data[:, 0]-1, dtype=np.int32)


            if len(self.la) != len(set(self.la)):
                raise ValueError("ex data contains duplicate absolute state label entries.")
        else:
            if self.sl_index:
                ll = len(label_key)

                if 'AS' in key:
                    if data[key.index('AS')].shape[1] != ll + 1:
                        raise ValueError("The dimension of absolute energy level array " \
                                "is inconsistent with the specified label_key.")

                    self.n_a = len(data[key.index('AS')])
                    self.a_states = np.zeros((self.n_a, ll), dtype=np.int32)
                    lah = np.zeros(self.n_a, dtype=np.int32)
                    for i in range(len(lah)):
                        self.a_states[i, :] = data[key.index('AS')][i, :ll]
                        clabels = np.ascontiguousarray(self.a_states[i, :], dtype=np.int32)
                        lah[i] = <int> cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))

                    self.la = np.zeros(self.n_a, dtype=np.int32)
                    self.lah = np.ascontiguousarray(lah, dtype=np.int32)
                    if len(self.lah) != len(set(self.lah)):
                        raise ValueError("ex data contains duplicate absolute index entries.")

                if 'DS' in key:
                    if data[key.index('DS')].shape[1] != 2*ll + 1:
                        raise ValueError("The dimension of difference energy level array " \
                                "is inconsistent with the specified label_key.")
                    self.n_d = len(data[key.index('DS')])
                    self.id_states = np.zeros((self.n_d, ll), dtype=np.int32)
                    self.fd_states = np.zeros((self.n_d, ll), dtype=np.int32)
                    ildh = np.zeros(self.n_d, dtype=np.int32)
                    fldh = np.zeros(self.n_d, dtype=np.int32)
                    for i in range(len(ildh)):
                        self.id_states[i, :] = data[key.index('DS')][i, :ll]
                        self.fd_states[i, :] = data[key.index('DS')][i, ll:2*ll]
                        clabels = np.ascontiguousarray(self.id_states[i, :], dtype=np.int32)
                        ildh[i] = <int> cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))
                        clabels = np.ascontiguousarray(self.fd_states[i, :], dtype=np.int32)
                        fldh[i] = <int> cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))

                    self.ild = np.zeros(self.n_d, dtype=np.int32)
                    self.fld = np.zeros(self.n_d, dtype=np.int32)
                    self.ildh = np.ascontiguousarray(ildh, dtype=np.int32)
                    self.fldh = np.ascontiguousarray(fldh, dtype=np.int32)

                if len(key) == 2:
                    # Both abs. and diff. levels present; energies and weights
                    # are stacked with all abs. values before the diff. values.
                    if 'AS' in key:
                        self.e = np.ascontiguousarray(np.hstack((data[key.index('AS')][:, ll],
                            data[key.index('DS')][:, 2*ll])), dtype=np.float64)
                        self.w = np.ascontiguousarray(np.hstack((weights[key.index('AS')],
                            weights[key.index('DS')])), dtype=np.float64)

                elif key[0] == 'AS':
                    self.e = np.ascontiguousarray(data[key.index('AS')][:, ll], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('AS')], dtype=np.float64)
                    self.n_d = 0

                elif key[0] == 'DS':
                    self.e = np.ascontiguousarray(data[key.index('DS')][:, 2*ll], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('DS')], dtype=np.float64)
                    self.n_a = 0
                    self.la = np.zeros(0)
            
            # Handle mu/n-based data (sl_index=0)
            elif 'AMu' in key or 'DMu' in key:
                if 'AMu' in key:
                    # Store raw (mu, n) data without conversion
                    if data[key.index('AMu')].shape[1] != 3:
                        raise ValueError("AMu data must have 3 columns: (mu, n, energy).")
                    self.n_a = len(data[key.index('AMu')])
                    self.mu_n_abs = np.ascontiguousarray(data[key.index('AMu')][:, :2], dtype=np.float64)
                    self.has_mu_n = True

                if 'DMu' in key:
                    # Store raw (mu, n) data for initial and final states
                    if data[key.index('DMu')].shape[1] != 5:
                        raise ValueError("DMu data must have 5 columns: (mu_i, n_i, mu_f, n_f, energy_diff).")
                    self.n_d = len(data[key.index('DMu')])
                    self.mu_n_diff = np.ascontiguousarray(data[key.index('DMu')][:, :4], dtype=np.float64)
                    self.has_mu_n = True

                if len(key) == 2:
                    # Both abs. and diff. levels present; energies and weights
                    # are stacked with all abs. values before the diff. values.
                    self.e = np.ascontiguousarray(np.hstack((data[key.index('AMu')][:, 2],
                        data[key.index('DMu')][:, 4])), dtype=np.float64)
                    self.w = np.ascontiguousarray(np.hstack((weights[key.index('AMu')],
                        weights[key.index('DMu')])), dtype=np.float64)
                    # Initialize la, ild, fld for mixed case; will be filled by conversion
                    self.la = np.zeros(self.n_a, dtype=np.int32)
                    self.ild = np.zeros(self.n_d, dtype=np.int32)
                    self.fld = np.zeros(self.n_d, dtype=np.int32)

                elif key[0] == 'AMu':
                    self.e = np.ascontiguousarray(data[key.index('AMu')][:, 2], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('AMu')], dtype=np.float64)
                    self.n_d = 0
                    # Initialize la; will be filled by mu_n_to_level conversion
                    self.la = np.zeros(self.n_a, dtype=np.int32)

                elif key[0] == 'DMu':
                    self.e = np.ascontiguousarray(data[key.index('DMu')][:, 4], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('DMu')], dtype=np.float64)
                    self.n_a = 0
                    self.mu_n_abs = np.zeros((0, 2), dtype=np.float64)
                    # Initialize ild and fld; will be filled by mu_n_to_level conversion
                    self.ild = np.zeros(self.n_d, dtype=np.int32)
                    self.fld = np.zeros(self.n_d, dtype=np.int32)
            else:
                if 'A' in key:
                    if data[key.index('A')].shape[1] != 2:
                        raise ValueError("Incorrect ex data shape for absolute energies; " \
                                "expected a two column array.")

                    self.n_a = len(data[key.index('A')])
                    self.la = np.ascontiguousarray(data[key.index('A')][:, 0]-1, dtype=np.int32)
                    if len(self.la) != len(set(self.la)):
                        raise ValueError("ex data contains duplicate absolute index entries.")
                if 'D' in key:
                    if data[key.index('D')].shape[1] != 3:
                        raise ValueError("Incorrect ex data shape for energy differences; " \
                                "expected a three column array.")

                    self.n_d = len(data[key.index('D')])
                    self.ild = np.ascontiguousarray(data[key.index('D')][:, 0]-1, dtype=np.int32)
                    self.fld = np.ascontiguousarray(data[key.index('D')][:, 1]-1, dtype=np.int32)
                if len(key) == 2:
                    # Both abs. and diff. levels present; energies and weights
                    # are stacked with all abs. values before the diff. values.
                    self.e = np.ascontiguousarray(np.hstack((data[key.index('A')][:, 1],
                        data[key.index('D')][:, 2])), dtype=np.float64)
                    self.w = np.ascontiguousarray(np.hstack((weights[key.index('A')],
                        weights[key.index('D')])), dtype=np.float64)
                elif key[0] == 'A':
                    self.e = np.ascontiguousarray(data[key.index('A')][:, 1], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('A')], dtype=np.float64)
                    self.n_d = 0
                elif key[0] == 'D':
                    self.e = np.ascontiguousarray(data[key.index('D')][:, 2], dtype=np.float64)
                    self.w = np.ascontiguousarray(weights[key.index('D')], dtype=np.float64)
                    self.n_a = 0
                    self.la = np.zeros(0)

        self.n_obs = self.n_a + self.n_d


cdef exdata_alloc_helper(ex, double weight=1.0):
    """
    Takes care of creating the cfl.ex_data c struct and returns it via a PyCapsule.

    Parameters
    ----------
    ex : ExData
        Experimental energy level data object.
    weight: float, optional
        Specifies the absolute weighting factor of this ex dataset with respect
        to other ex datasets, or spin Hamiltonians.
    """
    cdef np.ndarray[double, ndim=1, mode="c"] ex_e
    cdef np.ndarray[double, ndim=1, mode="c"] ex_w
    cdef np.ndarray[int, ndim=1, mode="c"] ex_la
    cdef np.ndarray[int, ndim=1, mode="c"] ex_ild
    cdef np.ndarray[int, ndim=1, mode="c"] ex_fld
    cdef np.ndarray[int, ndim=1, mode="c"] ex_lah
    cdef np.ndarray[int, ndim=1, mode="c"] ex_ildh
    cdef np.ndarray[int, ndim=1, mode="c"] ex_fldh
    cdef cfl.ex_data *ex_data

    ex_data = <cfl.ex_data *>malloc(sizeof(cfl.ex_data))
    if ex_data == NULL:
        raise MemoryError("ex_data alloc failed")

    ex_data.n_obs = ex.n_obs
    ex_data.n_a = ex.n_a
    ex_data.n_d = ex.n_d
    ex_e = <np.ndarray[double, ndim=1, mode="c"]> ex.e
    # Perform global weighting; ex.w is array of ones unless specified otherwise
    # to ExData constructor.  The multiplication always creates a new array
    # (even when weight==1.0), so we keep ex_w_np alive and return it to the
    # caller; without it ex_data.w would be a dangling pointer.
    ex_w_np = np.ascontiguousarray(ex.w * weight, dtype=np.float64)
    ex_w = <np.ndarray[double, ndim=1, mode="c"]> ex_w_np
    # Set to NULL ptr if it's an empty energy array.
    if ex.n_obs:
        ex_data.e = &ex_e[0]
        ex_data.w = &ex_w[0]
    else:
        ex_data.e = NULL
        ex_data.w = NULL

    if ex.n_a:
        ex_la = <np.ndarray[int, ndim=1, mode="c"]> ex.la
        ex_data.la = &ex_la[0]
    else:
        # There are no absolute energy level observables.
        ex_data.la = NULL
    if ex.n_d:
        ex_ild = <np.ndarray[int, ndim=1, mode="c"]> ex.ild
        ex_fld = <np.ndarray[int, ndim=1, mode="c"]> ex.fld
        ex_data.ild = &ex_ild[0]
        ex_data.fld = &ex_fld[0]
    else:
        # There are no energy level difference observables.
        ex_data.ild = NULL
        ex_data.fld = NULL
    if ex.sl_index:
        if ex.n_a:
            ex_lah = <np.ndarray[int, ndim=1, mode="c"]> ex.lah
            ex_data.lah = &ex_lah[0]
        else:
            ex_data.lah = NULL

        if ex.n_d:
            ex_ildh = <np.ndarray[int, ndim=1, mode="c"]> ex.ildh
            ex_fldh = <np.ndarray[int, ndim=1, mode="c"]> ex.fldh
            ex_data.ildh = &ex_ildh[0]
            ex_data.fldh = &ex_fldh[0]
        else:
            ex_data.ildh = NULL
            ex_data.fldh = NULL
    else:
        ex_data.lah = NULL
        ex_data.ildh = NULL
        ex_data.fldh = NULL

    ex_data_cap = PyCapsule_New(<void *>ex_data, "pycfl.ExData", NULL)

    # Return capsule and backing weight array together; caller must store
    # ex_w_np to prevent ex_data.w from becoming a dangling pointer.
    return (ex_data_cap, ex_w_np)

cdef set_param_helper(fit_obj):
    "Helper for updating real-valued parameter array of Fit objects"
    ii = 0
    for p in fit_obj.parameters:
        if fit_obj.param_types[p] == 'c':
            fit_obj.x0[ii] = np.real(fit_obj.coeff[p])
            fit_obj.x0[ii+1] = np.imag(fit_obj.coeff[p])
            ii += 2
        else:
            fit_obj.x0[ii] = fit_obj.coeff[p]
            ii += 1


def _x_to_coeff_dict(fit_obj, x):
    """Convert a real-valued parameter vector to a coeff dict.

    Splits each complex parameter back into a (Re, Im) pair following
    the same indexing convention as ``set_param_helper`` (the inverse
    operation).
    """
    coeff = {}
    ii = 0
    for p in fit_obj.parameters:
        if fit_obj.param_types[p] == 'c':
            coeff[p] = complex(float(x[ii]), float(x[ii + 1]))
            ii += 2
        else:
            coeff[p] = float(x[ii])
            ii += 1
    return coeff


def _fit_hamiltonians(fit_obj):
    """Return the list of Hamiltonians a fit object owns."""
    if hasattr(fit_obj, "h_list") and fit_obj.h_list is not None:
        return list(fit_obj.h_list)
    return [fit_obj.h]


@contextmanager
def _temporary_x(fit_obj, x):
    """Temporarily set fit parameters to ``x``; restore on exit.

    Round-trips both the Python state (``fit_obj.coeff``, ``fit_obj.x0``,
    each Hamiltonian's ``coeff_dict``) and the C-side coefficients in
    each Hamiltonian (via :py:meth:`Hamiltonian.update_coeff`) so that a
    subsequent ``h.diag()`` evaluates at the perturbed point and the
    original state is fully restored on context exit (even on
    exceptions).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.shape != (fit_obj.n_p_real,):
        raise ValueError(
            "x must have shape (%d,); got shape %s" % (
                fit_obj.n_p_real, x_arr.shape,
            )
        )
    h_list = _fit_hamiltonians(fit_obj)
    saved_coeff = copy.deepcopy(fit_obj.coeff)
    saved_x0 = np.asarray(fit_obj.x0, dtype=np.float64).copy()
    saved_h_coeffs = [copy.deepcopy(h.coeff_dict) for h in h_list]
    new_coeff = _x_to_coeff_dict(fit_obj, x_arr)
    try:
        fit_obj.coeff.update(new_coeff)
        np.asarray(fit_obj.x0)[:] = x_arr  # in-place; preserves buffer.
        for h in h_list:
            sub = {k: v for k, v in new_coeff.items() if k in h}
            if sub:
                h.update_coeff(sub)
        yield
    finally:
        for h, saved in zip(h_list, saved_h_coeffs):
            if saved is not None:
                h.set_coeff(saved)
        fit_obj.coeff.clear()
        fit_obj.coeff.update(saved_coeff)
        np.asarray(fit_obj.x0)[:] = saved_x0


def _fd_jacobian_impl(fit_obj, x=None, *, delta=None,
                      rel_delta=1e-5, atol=1e-8, check_swaps=True):
    """Shared finite-difference Jacobian implementation.

    Computes ``J_E`` (the unweighted energy Jacobian, see the project
    plan section 5.3) by central differences, returning an
    ``(n_obs, n_p_real)`` ``np.ndarray``.  ``check_swaps`` raises a
    ``UserWarning`` for any column whose magnitude suggests an
    eigenvalue swap across the FD step.

    On exit the fit's parameter state matches its state on entry.
    """
    n_p = fit_obj.n_p_real
    n_obs = fit_obj.n_obs
    if x is None:
        x_base = np.asarray(fit_obj.x0, dtype=np.float64).copy()
    else:
        x_base = np.asarray(x, dtype=np.float64).copy()
    if x_base.shape != (n_p,):
        raise ValueError(
            "x must have shape (%d,); got shape %s" % (n_p, x_base.shape)
        )

    if delta is None:
        delta_vec = np.maximum(rel_delta * np.abs(x_base), atol)
    elif np.isscalar(delta):
        delta_vec = np.full(n_p, float(delta), dtype=np.float64)
    else:
        delta_vec = np.asarray(delta, dtype=np.float64).copy()
        if delta_vec.shape != (n_p,):
            raise ValueError(
                "delta must be a scalar or shape (%d,) array; got %s" % (
                    n_p, delta_vec.shape,
                )
            )
    if np.any(delta_vec <= 0):
        raise ValueError("All FD step sizes must be strictly positive")

    # Capture the baseline calculated energies (used for the swap check).
    with _temporary_x(fit_obj, x_base):
        E_base = fit_obj.get_edata().arr["e_calc"].astype(np.float64, copy=True)

    J = np.zeros((n_obs, n_p), dtype=np.float64)
    for alpha in range(n_p):
        d = float(delta_vec[alpha])
        x_p = x_base.copy()
        x_p[alpha] = x_base[alpha] + d
        x_m = x_base.copy()
        x_m[alpha] = x_base[alpha] - d
        with _temporary_x(fit_obj, x_p):
            E_p = fit_obj.get_edata().arr["e_calc"]
        with _temporary_x(fit_obj, x_m):
            E_m = fit_obj.get_edata().arr["e_calc"]
        J[:, alpha] = (E_p - E_m) / (2.0 * d)

    if check_swaps and n_obs > 0:
        e_range = float(np.ptp(E_base)) if n_obs > 1 else 1.0
        if e_range > 0.0:
            for alpha in range(n_p):
                col_max = float(np.max(np.abs(J[:, alpha]))) if n_obs > 0 else 0.0
                if col_max * delta_vec[alpha] > 0.5 * e_range:
                    warnings.warn(
                        "fd_jacobian: column %d has max|J|*delta=%.3g, "
                        "comparable to the energy spread (%.3g). This may "
                        "indicate an eigenvalue swap or near-degeneracy "
                        "across the FD step; consider reducing delta for "
                        "this parameter." % (
                            alpha, col_max * delta_vec[alpha], e_range,
                        ),
                        UserWarning,
                        stacklevel=3,
                    )

    fit_obj.last_jacobian = J
    return J


def _covariance_impl(fit_obj, x=None, *, jacobian=None,
                     scale="reduced_chi2", **fd_kwargs):
    """Shared covariance implementation for EFit/MHFit.

    Returns ``(cov, sigma, edata)`` where ``cov`` has shape
    ``(n_p_real, n_p_real)``, ``sigma = sqrt(diag(cov))``, and ``edata``
    is the :class:`EData` snapshot used to compute the residuals.
    """
    if scale not in ("reduced_chi2", "unscaled"):
        raise ValueError(
            "scale must be 'reduced_chi2' or 'unscaled', got %r" % (scale,))

    if jacobian is None:
        J = getattr(fit_obj, "last_jacobian", None)
        if J is None or x is not None:
            J = _fd_jacobian_impl(fit_obj, x=x, **fd_kwargs)
    else:
        J = np.ascontiguousarray(jacobian, dtype=np.float64)

    # Snapshot edata at the requested x (or current state).
    if x is None:
        edata = fit_obj.get_edata()
    else:
        with _temporary_x(fit_obj, np.asarray(x, dtype=np.float64)):
            edata = fit_obj.get_edata()

    arr = edata.arr
    weights = np.asarray(arr["weight"], dtype=np.float64)
    n_obs = arr.shape[0]
    n_p_real = J.shape[1]
    if J.shape[0] != n_obs:
        raise ValueError(
            "Jacobian row count %d does not match n_obs=%d"
            % (J.shape[0], n_obs))

    JtWJ = J.T @ (weights[:, None] * J)
    rank = np.linalg.matrix_rank(JtWJ)
    if rank < n_p_real:
        warnings.warn(
            "Normal matrix is rank-deficient (rank %d < %d); "
            "returning Moore-Penrose pseudo-inverse covariance."
            % (rank, n_p_real),
            UserWarning,
            stacklevel=3,
        )
    N = np.linalg.pinv(JtWJ)

    if scale == "unscaled":
        cov = N
    else:
        chi2 = float(np.sum(np.asarray(arr["wresidual"]) ** 2))
        ndof = max(n_obs - n_p_real, 1)
        cov = (chi2 / ndof) * N

    diag = np.diag(cov)
    sigma = np.sqrt(np.where(diag > 0, diag, 0.0))
    return cov, sigma, edata


cdef class EFit(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    energy level data.

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of each
    coefficient when they are set also determines whether that coefficient is
    fit as real or complex parameter.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels.
    ex : np.ndarray or ExData
        Either a 2 by n dimensional np.ndarray or an ExData type object.  In the
        former case, n is the number of energy levels, with the first column
        containing energy level indices starting at 1, and the second column
        containing the absolute experimental energy of the corresponding level.
        In order to specify energy level differences, or specify energies
        according to their SLJM state labels, use the ExData interface.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    cdef public Hamiltonian h
    cdef public dict coeff
    cdef int n_p
    cdef public list parameters
    cdef public int n_p_real
    cdef public int n_obs
    cdef public dict param_types
    cdef cfl.ex_data *ex_data
    cdef public ExData ex
    cdef cfl.param_type **param_array
    cdef public np.ndarray x0
    cdef public np.ndarray wts
    cdef cfl.efit_data *efit_data
    cdef public object obj_f_cap
    cdef public object nls_f_cap
    cdef public object fit_data_cap
    cdef public np.ndarray chi2
    cdef public object last_jacobian
    cdef object _ex_w_backing   # keeps ex_data.w buffer alive (see exdata_alloc_helper)
    def __init__(self, parameters, h, ex, **kwargs):
        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters

        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")

        # Create a local copy of coefficients for each parameter and an array of
        # arrays specifying the parameters specific to each Hamiltonian.
        if h.coeff_dict is None:
            raise ValueError("Hamiltonian must have coefficients set prior to efit.")
        else:
            self.coeff = copy.deepcopy(h.coeff_dict)

        self.param_types = {}
        self.n_p_real = 0
        for p in parameters:
            if p not in h:
                raise ValueError("Parameter %s not found in the Hamiltonian." % p)
            if isinstance(self.coeff[p], complex):
                self.n_p_real += 2
                self.param_types[p] = "c"
            else:
                self.n_p_real += 1
                self.param_types[p] = "r"

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False

        # Parse the energy level data, if required.
        if not isinstance(ex, ExData):
            self.ex = ExData(ex)
        else:
            self.ex = ex
        
        # Convert mu/n data to level indices if present
        if self.ex.has_mu_n:
            # Validate required parameters for mu/n fitting
            if self.h.minimum_q is None:
                raise ValueError(
                    "Hamiltonian.minimum_q must be set before fitting with AMu/DMu data. "
                    "Set it to the smallest non-zero q value in your expansion "
                    "(typically 2 for C20/C22 terms).")
            if not isinstance(self.h.half_integer_states, bool):
                raise ValueError(
                    "Hamiltonian.half_integer_states must be explicitly set to True or False "
                    "before fitting with AMu/DMu data. "
                    "Use True if m values are half-integers (e.g., f-electrons with J=5/2), "
                    "False if m values are integers.")
            
            from pycf.cfl_util import mu_n_to_level
            
            # Convert absolute mu/n data to level indices
            if self.ex.n_a > 0 and self.ex.mu_n_abs is not None and len(self.ex.mu_n_abs) > 0:
                level_indices = mu_n_to_level(
                    self.h, self.ex.mu_n_abs, self.h.minimum_q, self.h.half_integer_states
                )
                self.ex.la = np.ascontiguousarray(level_indices - 1, dtype=np.int32)
            
            # Convert difference mu/n data to level indices
            if self.ex.n_d > 0 and self.ex.mu_n_diff is not None and len(self.ex.mu_n_diff) > 0:
                mu_n_initial = self.ex.mu_n_diff[:, :2]
                mu_n_final = self.ex.mu_n_diff[:, 2:4]
                
                initial_levels = mu_n_to_level(
                    self.h, mu_n_initial, self.h.minimum_q, self.h.half_integer_states
                )
                final_levels = mu_n_to_level(
                    self.h, mu_n_final, self.h.minimum_q, self.h.half_integer_states
                )
                
                self.ex.ild = np.ascontiguousarray(initial_levels - 1, dtype=np.int32)
                self.ex.fld = np.ascontiguousarray(final_levels - 1, dtype=np.int32)
        
        self.n_obs = self.ex.n_obs

        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of "\
                    " parameters, %i, exceeds the number of observables, %i." %
                    (self.n_p_real, self.n_obs))

        cap, self._ex_w_backing = exdata_alloc_helper(self.ex)
        self.ex_data = <cfl.ex_data *>PyCapsule_GetPointer(cap, "pycfl.ExData")

        # Weights array for GSL nonlinear least-squares; since individual energy
        # level weighting isn't really implemented, we could in principle forego
        # this alloc... but to make self.wts interoperable we just use ones.
        self.wts = np.ascontiguousarray(np.ones(self.n_obs), dtype=np.float64)


        # Prepare array of pointers to parameter data structs.
        self.x0 = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_array = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
        if param_array == NULL:
            free(self.ex_data)
            # NULL self.ex_data so __dealloc__ does not double-free it.
            self.ex_data = NULL
            raise MemoryError("param_array alloc failed")

        ii = 0
        for i,p in enumerate(parameters):
            param_array[i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                # NULL self.ex_data so __dealloc__ does not double-free it.
                self.ex_data = NULL
                # free(self.param_array) would be a no-op here because
                # self.param_array has not been assigned yet; free the local
                # param_array instead to avoid leaking the outer array.
                free(param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))

            param_array[i].type = ord(self.param_types[p])
            param_array[i].ci = h.index(p)
            param_array[i].xi = ii

            if self.param_types[p] == 'c':
                self.x0[ii] = np.real(self.coeff[p])
                self.x0[ii+1] = np.imag(self.coeff[p])
                ii += 2
            else:
                self.x0[ii] = self.coeff[p]
                ii += 1

        self.param_array = param_array

        if self.ex.sl_index:
            self.efit_data = cfl.efit_data_alloc('S', <cfl.zh *>PyCapsule_GetPointer(
                h.h_cap, "pycfl.Hamiltonian"), self.ex_data, self.n_p, self.param_array);
        else:
            self.efit_data = cfl.efit_data_alloc('N', <cfl.zh *>PyCapsule_GetPointer(
                h.h_cap, "pycfl.Hamiltonian"), self.ex_data, self.n_p, self.param_array);
        if self.efit_data is NULL:
            for i in range(self.n_p):
                free(self.param_array[i])
            free(self.param_array)
            # NULL both members so __dealloc__ does not attempt second frees.
            self.param_array = NULL
            free(self.ex_data)
            self.ex_data = NULL
            raise MemoryError("efit_data_alloc failed")

        self.fit_data_cap = PyCapsule_New(<void *>self.efit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.efit_obj, "pycfl.MinObjF", NULL)
        self.nls_f_cap = PyCapsule_New(<void *>&cfl.efit_nls, "pycfl.NlsObjF", NULL)

    def __dealloc__(self):
        if self.ex_data != NULL:
            free(self.ex_data)
        if self.param_array != NULL:
            for i in range(self.n_p):
                free(self.param_array[i])
            free(self.param_array)
        if self.efit_data != NULL:
            cfl.efit_data_free(self.efit_data)

    def __iter__(self):
        for p in self.parameters:
            yield p

    def fit(self, min_object):
        r"""
        Run the fit using the provided minimization object.

        Parameters
        ----------
        min_object : CFLMin
            The minimization object to be used, which sets the optimization
            algorithm, bounds and other settings as applicable to the selected
            algorithm.

        Returns
        -------
        result : tuple
            The first element is a np.ndarray containing complex coefficients
            while the second entry contains the final value of the objective
            function.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        fmin = min_object.minimize(self, x)

        # If the minimiser produced a Jacobian (gsl_nls path), expose it
        # for downstream callers (e.g. covariance helper).
        if 'jac' in min_object.kwargs:
            self.last_jacobian = np.array(min_object.kwargs['jac'], copy=True)

        coeff = self.coeff.copy()
        params = {}
        ri = 0

        for p in self:
            if (self.param_types[p] == 'c'):
                params[p] = complex(x[ri], x[ri+1])
                ri += 2
            else:
                params[p] = x[ri]
                ri += 1

        chi2 = np.ascontiguousarray(np.zeros(1, dtype=np.float64))
        cfl.efit_chi2(&x[0], self.efit_data, &chi2[0])
        self.chi2 = chi2

        return(params, fmin)

    @cython.boundscheck(False)
    def eval(self, coeff):
        r"""
        Return chi2 obtained for the provided coefficients.

        Parameters
        ----------
        coeff : dict
            The usual coeff dict; if coefficients for only a subset of tensors
            are provided, the remainder are held at their initial value.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        self.coeff.update(copy.deepcopy(coeff))
        set_param_helper(self)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        chi2 = np.ascontiguousarray(np.zeros(1, dtype=np.float64))
        with nogil:
            cfl.efit_chi2(&x[0], self.efit_data, &chi2[0])

        return chi2

    def get_edata(self):
        r"""
        Return an :class:`~pycf.cfl_util.EData` table describing this fit's
        observations and the values currently produced by the Hamiltonian.

        The Hamiltonian is (re-)diagonalised at its current coefficients so
        that ``self.h.w`` matches the eigenvalues used to compute
        ``e_calc`` for each row.  Row order matches the order in which the
        C objective concatenates residuals: all ``'A'`` rows first, then
        all ``'D'`` rows.

        Returns
        -------
        edata : EData
        """
        self.h.diag()
        return _build_edata_for_ex(self.h, self.ex, h_index=0)


    def fd_jacobian(self, x=None, *, delta=None, rel_delta=1e-5,
                    atol=1e-8, check_swaps=True):
        r"""
        Finite-difference energy Jacobian.

        Computes :math:`J_E[i, \alpha] = \partial E_i / \partial x_\alpha`
        by central differences, where :math:`E_i` is the calculated value
        of the i-th row of :py:meth:`get_edata` (an eigenvalue for ``'A'``
        rows or :math:`|\lambda_f - \lambda_i|` for ``'D'`` rows) and
        :math:`x` is the real-valued parameter vector with the same
        layout as :attr:`x0` (complex parameters split into Re/Im).

        Parameters
        ----------
        x : array_like, optional
            Parameter vector at which to evaluate the Jacobian.  Defaults
            to the current :attr:`x0`.
        delta : float or array_like, optional
            Per-parameter FD step.  If ``None``, ``max(rel_delta * |x|,
            atol)`` is used.
        rel_delta : float, optional
            Relative step used when ``delta`` is ``None``.
        atol : float, optional
            Minimum absolute step used when ``delta`` is ``None``.
        check_swaps : bool, optional
            Emit a :class:`UserWarning` for any column whose magnitude
            suggests an eigenvalue swap or near-degeneracy across the
            FD step.

        Returns
        -------
        J : np.ndarray, shape ``(n_obs, n_p_real)``
            The unweighted energy Jacobian.  Also stored on
            :attr:`last_jacobian`.

        Notes
        -----
        On exit the fit's parameter state matches its state on entry,
        even if an exception is raised.  Cost is ``2 * n_p_real``
        diagonalisations.
        """
        return _fd_jacobian_impl(self, x=x, delta=delta,
                                 rel_delta=rel_delta, atol=atol,
                                 check_swaps=check_swaps)

    def covariance(self, x=None, *, jacobian=None,
                   scale="reduced_chi2", **fd_kwargs):
        r"""
        Variance-covariance matrix for the real-valued parameter vector.

        Builds :math:`N = (J_E^T W J_E)^+` (Moore-Penrose pseudo-inverse)
        and applies the requested ``scale``:

        - ``scale="unscaled"`` returns ``N`` directly (matches the GSL
          ``gsl_multifit_nlinear_covar`` convention).
        - ``scale="reduced_chi2"`` (default) returns
          ``(chi2 / max(N_obs - M, 1)) * N`` where ``M = n_p_real``.

        Parameters
        ----------
        x : array_like, optional
            Parameter vector at which to evaluate.  Defaults to the
            current ``x0``.  If supplied, a fresh FD Jacobian is
            computed at ``x`` (ignoring any cached ``last_jacobian``).
        jacobian : array_like, optional
            Pre-computed energy Jacobian of shape ``(n_obs, n_p_real)``.
            If omitted, ``self.last_jacobian`` is used when available
            and ``x is None``; otherwise an FD Jacobian is computed.
        scale : {"reduced_chi2", "unscaled"}, optional
            Scaling convention.
        **fd_kwargs
            Forwarded to :py:meth:`fd_jacobian` when an FD Jacobian is
            computed.

        Returns
        -------
        cov : np.ndarray, shape ``(n_p_real, n_p_real)``
        sigma : np.ndarray, shape ``(n_p_real,)``
            ``sqrt(diag(cov))`` (clipped at 0).
        edata : EData
            Snapshot used to weight the normal matrix.
        """
        return _covariance_impl(self, x=x, jacobian=jacobian,
                                scale=scale, **fd_kwargs)


cdef _build_edata_for_ex(Hamiltonian h, ExData ex, int h_index, double h_weight=1.0):
    """Construct an EData table for a single (Hamiltonian, ExData) pair.

    Assumes ``h.diag()`` has already populated ``h.w`` and ``h.z``.
    ``h_weight`` is the per-Hamiltonian scalar weight used by MHFit
    (default 1.0 for EFit); the C side bakes ``ex.w * h_weight`` into
    the value it squares, so we mirror that here so that
    ``EData.chi2()`` matches the C objective.

    For ``sl_index == 0`` ExData the level indices ``ex.la``/
    ``ex.ild``/``ex.fld`` are taken directly.  For ``sl_index == 1``
    ExData (``'AS'``/``'DS'`` modes), the matching mirrors the C
    routine ``find_sort_indices`` (cfl_h_fit.c:702): for each
    eigenvector the principal-component basis state is identified, and
    the requested state label is matched against those basis labels.
    The implementation reuses :py:func:`pycf.cfl_util.ex_parse_abs`
    and :py:func:`pycf.cfl_util.ex_parse_diff` which already encode
    that same logic in NumPy.
    """
    cdef np.ndarray w = h.w
    n_a = ex.n_a
    n_d = ex.n_d
    n_obs = ex.n_obs
    label = h.label if h.label is not None else "H[%d]" % h_index

    arr = np.zeros(n_obs, dtype=EData.DTYPE)
    arr["h_index"] = h_index
    arr["h_label"] = label

    if ex.sl_index:
        # State-label-indexed observations; resolve level indices via
        # principal-component matching.
        labels = h.tensors[0].states.labels
        a_kind = "AS"
        d_kind = "DS"
        if n_a > 0:
            parsed_a = ex_parse_abs(ex, h.z, labels)
            la = np.asarray(parsed_a[:, 0], dtype=np.int64)
        else:
            la = np.empty(0, dtype=np.int64)
        if n_d > 0:
            parsed_d = ex_parse_diff(ex, h.z, labels)
            ild = np.asarray(parsed_d[:, 0], dtype=np.int64)
            fld = np.asarray(parsed_d[:, 1], dtype=np.int64)
        else:
            ild = np.empty(0, dtype=np.int64)
            fld = np.empty(0, dtype=np.int64)
    else:
        a_kind = "A"
        d_kind = "D"
        la = np.asarray(ex.la, dtype=np.int64) if n_a > 0 else np.empty(0, dtype=np.int64)
        ild = np.asarray(ex.ild, dtype=np.int64) if n_d > 0 else np.empty(0, dtype=np.int64)
        fld = np.asarray(ex.fld, dtype=np.int64) if n_d > 0 else np.empty(0, dtype=np.int64)

    # Absolute rows: indices 0 .. n_a-1 in the residual vector.
    if n_a > 0:
        arr["kind"][:n_a] = a_kind
        arr["i_lo"][:n_a] = la + 1
        arr["i_hi"][:n_a] = 0
        arr["e_calc"][:n_a] = w[la]

    # Difference rows: indices n_a .. n_a+n_d-1.
    if n_d > 0:
        sl = slice(n_a, n_a + n_d)
        arr["kind"][sl] = d_kind
        arr["i_lo"][sl] = ild + 1
        arr["i_hi"][sl] = fld + 1
        # Match the C objective's fabs(...) (see cfl_h_fit.c:661,1112).
        arr["e_calc"][sl] = np.abs(w[fld] - w[ild])

    arr["e_obs"][:] = np.asarray(ex.e, dtype=np.float64)[:n_obs]
    arr["weight"][:] = np.asarray(ex.w, dtype=np.float64)[:n_obs] * h_weight
    arr["residual"][:] = arr["e_calc"] - arr["e_obs"]
    arr["wresidual"][:] = np.sqrt(arr["weight"]) * arr["residual"]

    return EData(arr)


cdef class MHFit(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    multiple Hamiltonians.  Typically, this would consist of one Hamiltonian at
    zero field without hyperfine or quadrupole interactions, complemented by a
    set of Hamiltonians at linearly independent magnetic field orientations and
    possibly containing hyperfine interactions.  The associated additional
    eigenvalues can either be measured or synthetically calculated for specific
    crystal field levels from spin Hamiltonian data.

    The Hamiltonians must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of each
    coefficient when they are set also determines whether that coefficient is
    fit as real or complex parameter, thus they must be consistent among each
    Hamiltonian.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h_list : list
        A list of Hamiltonians, each containing the interactions required to
        match the corresponding experimental energy level data.
    weights_list : list
        A list of floating point weights that determine the weighting added to
        the chi^2 contribution of each eigenvalue vector.
    ex : np.ndarray or ExData
        Either a list of 2 by n dimensional np.ndarrays or a list of ExData type
        objects.  In the former case, n is the number of energy levels, with the
        first column of each array containing energy level indices starting at
        1, and the second column containing the absolute experimental energy of
        the corresponding level.  In order to specify energy level differences,
        or specify energies according to their SLJM state labels, use the ExData
        interface.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    cdef int n_h
    cdef public Hamiltonian h
    cdef public dict coeff
    cdef public list h_list
    cdef int n_p
    cdef public list parameters
    cdef public int n_p_real
    cdef public int n_obs
    cdef public dict param_types
    cdef list h_param_list
    cdef cfl.zh **ha
    cdef cfl.ex_data **ex_data
    cdef list ex_list
    cdef np.ndarray n_zx
    cdef cfl.param_type ***param_arrays
    cdef public np.ndarray x0
    cdef public np.ndarray wts
    cdef cfl.mhfit_data *mhfit_data
    cdef np.ndarray job_a
    cdef public object obj_f_cap
    cdef public object nls_f_cap
    cdef public object fit_data_cap
    cdef public np.ndarray chi2
    cdef public list weights_list
    cdef public object last_jacobian
    cdef list _ex_w_backing     # keeps each ex_data.w buffer alive (see exdata_alloc_helper)
    def __init__(self, parameters, h_list, weights_list, ex_list, **kwargs):
        cdef np.ndarray[int, ndim=1, mode="c"] n_zx
        cdef np.ndarray[char, ndim=1, mode="c"] job_a

        self.n_h = len(h_list)
        self.n_p = len(parameters)
        self.h_list = h_list
        self.parameters = parameters
        self.weights_list = weights_list

        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")

        self.coeff = {}                                # Local copy of all coefficients of any H/SH.
        h_param_list = []                              # Array of arrays specifying parameters of each H.
        self.n_zx = np.empty(self.n_h, dtype=np.int32) # Number of complex valued params for each Hamiltonian
        for i,h in enumerate(h_list):
            if h.coeff_dict is None:
                raise ValueError("Hamiltonian must have coefficients set prior to mhfit.")
            else:
                self.coeff.update(copy.deepcopy(h.coeff_dict))
            h_param_list += [[p for p in parameters if p in h]]
            self.n_zx[i] = len(h_param_list[i])

        self.h_param_list = h_param_list

        # Create cython copy for passing to c func call.
        n_zx = <np.ndarray[int, ndim=1, mode="c"]> self.n_zx

        self.param_types = {}       # The type of each parameter (real, complex, or imag).
        self.n_p_real = 0           # The total number of real parameters (two for each complex number).
        x0_index = {}               # Index of each parameter in the real-valued param array.
        for p in parameters:
            if all((p not in h for h in h_list)):
                raise ValueError("Parameter %s not found in any Hamiltonian." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                x0_index[p] = self.n_p_real
                self.n_p_real += 2
                self.param_types[p] = "c"
            else:
                x0_index[p] = self.n_p_real
                self.param_types[p] = "r"
                self.n_p_real += 1

        # Parse the energy level data.
        self.n_obs = 0
        self.ex_list = []
        if (len(ex_list) != self.n_h):
            raise ValueError("The number of Hamiltonians does not match the \
                    number of elements in ex_list.")
        for i,ex in enumerate(ex_list):
            if not isinstance(ex, ExData):
                self.ex_list += [ExData(ex)]
            else:
                self.ex_list += [ex]
            self.n_obs += self.ex_list[i].n_obs

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False

        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of \
                    parameters, %i, exceeds the number of observables, %i." %
                    (self.n_p_real, self.n_obs))

        # Weights for GSL nonlinear least-squares.  nls_echisq already encodes
        # the full combined weight (per-level ex.w * per-Hamiltonian scalar) into
        # each residual as sqrt(w)*(calc-obs), so GSL must not apply additional
        # weighting on top.  Pass unit weights to gsl_multifit_nlinear_winit so
        # that it minimises sum(y_i^2) = echisq without double-counting.
        self.wts = np.ones(self.n_obs, dtype=np.float64)

        # Hamiltonian array
        self.ha = <cfl.zh **>malloc(self.n_h*sizeof(cfl.zh *))
        if self.ha == NULL:
            raise MemoryError("ha alloc failed")

        for i in range(self.n_h):
            self.ha[i] = <cfl.zh *>PyCapsule_GetPointer(h_list[i].h_cap, "pycfl.Hamiltonian")
        self.ex_data = <cfl.ex_data **>malloc(self.n_h*sizeof(cfl.ex_data *))
        if self.ex_data == NULL:
            free(self.ha)
            # NULL self.ha so __dealloc__ does not attempt a second free.
            self.ha = NULL
            raise MemoryError("exa alloc failed")

        self._ex_w_backing = []
        for i in range(self.n_h):
            try:
                cap, ex_w_np = exdata_alloc_helper(self.ex_list[i], weights_list[i])
                self.ex_data[i] = <cfl.ex_data *>PyCapsule_GetPointer(cap, "pycfl.ExData")
                self._ex_w_backing.append(ex_w_np)
            except BaseException:
                for j in range(i):
                    free(self.ex_data[j])
                free(self.ex_data)
                free(self.ha)
                # NULL both members so __dealloc__ does not double-free.
                self.ex_data = NULL
                self.ha = NULL
                raise

        # Prepare array of pointers to parameter data structs.
        param_arrays = <cfl.param_type ***>malloc(self.n_h*sizeof(cfl.param_type **))
        if param_arrays == NULL:
            for i in range(self.n_h):
                free(self.ex_data[i])
            free(self.ex_data)
            free(self.ha)
            # NULL both members so __dealloc__ does not double-free.
            self.ex_data = NULL
            self.ha = NULL
            raise MemoryError("param_arrays alloc failed")

        for i in range(self.n_h):
            param_arrays[i] = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
            if param_arrays[i] == NULL:
                for j in range(i):
                    free(param_arrays[j])
                free(param_arrays)
                for j in range(self.n_h):
                    free(self.ex_data[j])
                free(self.ex_data)
                free(self.ha)
                # NULL both members so __dealloc__ does not double-free.
                self.ex_data = NULL
                self.ha = NULL
                raise MemoryError("param_arrays[{}] alloc failed".format(i))

        for hi,h in enumerate(h_list):
            for i,p in enumerate(h_param_list[hi]):
                param_arrays[hi][i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
                if param_arrays[hi][i] is NULL:
                    for hj in range(hi):
                        for j in range(len(h_param_list[hj])):
                            free(param_arrays[hj][j])
                    for j in range(i):
                        free(param_arrays[hi][j])
                    for hj in range(self.n_h):
                        free(param_arrays[hj])
                    free(param_arrays)
                    for j in range(self.n_h):
                        free(self.ex_data[j])
                    free(self.ex_data)
                    free(self.ha)
                    # NULL both members so __dealloc__ does not double-free.
                    # self.param_arrays is not yet assigned, so no need to null it.
                    self.ex_data = NULL
                    self.ha = NULL
                    raise MemoryError("param_arrays[{0}][{1}] alloc failed".format(hi, i))

                param_arrays[hi][i].type = ord(self.param_types[p])
                param_arrays[hi][i].ci = h_list[hi].index(p)
                param_arrays[hi][i].xi = x0_index[p]

        self.param_arrays = param_arrays

        # Set initial values.
        self.x0 = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        set_param_helper(self)

        self.job_a = np.empty(self.n_h, dtype=np.dtype('S'))
        for i,ex in enumerate(self.ex_list):
            if ex.sl_index:
                self.job_a[i] = 'S'
            else:
                self.job_a[i] = 'N'
        job_a = self.job_a
        self.mhfit_data = mhfit_data_alloc(&job_a[0], self.n_h, self.ha, self.ex_data,
                &n_zx[0], self.param_arrays)
        if self.mhfit_data is NULL:
            for hi in range(self.n_h):
                for i in range(len(self.h_param_list[hi])):
                    free(self.param_arrays[hi][i])
                free(self.param_arrays[hi])
            free(param_arrays)
            # NULL self.param_arrays so __dealloc__ does not double-free it.
            self.param_arrays = NULL
            for j in range(self.n_h):
                free(self.ex_data[j])
            free(self.ex_data)
            free(self.ha)
            # NULL both members so __dealloc__ does not double-free them.
            self.ex_data = NULL
            self.ha = NULL
            raise MemoryError("mhfit_data_alloc failed")

        self.fit_data_cap = PyCapsule_New(<void *>self.mhfit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.mhfit_obj, "pycfl.MinObjF", NULL)
        self.nls_f_cap = PyCapsule_New(<void *>&cfl.mhfit_nls, "pycfl.NlsObjF", NULL)

    def __dealloc__(self):
        if self.ha != NULL:
            free(self.ha)
        if self.ex_data != NULL:
            for i in range(self.n_h):
                free(self.ex_data[i])
            free(self.ex_data)

        if self.param_arrays != NULL:
            for hi in range(self.n_h):
                for i in range(len(self.h_param_list[hi])):
                    free(self.param_arrays[hi][i])
                free(self.param_arrays[hi])
            free(self.param_arrays)

        if self.mhfit_data != NULL:
            cfl.mhfit_data_free(self.mhfit_data)

    def __iter__(self):
        for p in self.parameters:
            yield p

    def fit(self, min_object):
        r"""
        Run the fit using the provided minimization object.

        Parameters
        ----------
        min_object : CFLMin
            The minimization object to be used, which sets the optimization
            algorithm, bounds and other settings as applicable to the selected
            algorithm.

        Returns
        -------
        result : tuple
            The first element is a np.ndarray containing complex coefficients
            while the second entry contains the final value of the objective
            function.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        fmin = min_object.minimize(self, x)

        if 'jac' in min_object.kwargs:
            self.last_jacobian = np.array(min_object.kwargs['jac'], copy=True)

        params = {}
        ri = 0

        for p in self:
            if (self.param_types[p] == 'c'):
                params[p] = complex(x[ri], x[ri+1])
                ri += 2
            else:
                params[p] = x[ri]
                ri += 1

        chi2 = np.ascontiguousarray(np.zeros(self.n_h, dtype=np.float64))
        cfl.mhfit_chi2(&x[0], self.mhfit_data, &chi2[0])
        self.chi2 = chi2

        return(params, fmin)

    @cython.boundscheck(False)
    def eval(self, coeff):
        r"""
        Return chi2 obtained for the provided coefficients.

        Parameters
        ----------
        coeff : dict
            The usual coeff dict; if coefficients for only a subset of tensors
            are provided, the remainder are held at their initial value.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        self.coeff.update(copy.deepcopy(coeff))
        set_param_helper(self)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        chi2 = np.ascontiguousarray(np.zeros(self.n_h, dtype=np.float64))
        with nogil:
            cfl.mhfit_chi2(&x[0], self.mhfit_data, &chi2[0])

        return chi2

    def get_edata(self):
        r"""
        Return an :class:`~pycf.cfl_util.EData` table aggregating the
        observations of every Hamiltonian in this fit.

        Each Hamiltonian is (re-)diagonalised at its current coefficients
        and rows are concatenated in fit-evaluation order
        ``(h_list[0], h_list[1], ...)``, with each Hamiltonian's rows in
        the same internal order produced by :py:meth:`EFit.get_edata`
        (all ``'A'`` rows then all ``'D'`` rows).  Row index in the
        returned table aligns with column index of the residual vector
        the C minimiser sees, which is also the column index of the
        Jacobian.

        The per-Hamiltonian scalar weight passed in ``weights_list`` is
        applied to each row's ``weight`` field so that
        ``EData.chi2()`` matches the value the C objective squares.
        (Internally, the C side bakes that scalar into the ``ex_data.w``
        buffer it reads, while the Python ``ExData.w`` attribute keeps
        the original per-level weights.)

        Returns
        -------
        edata : EData
        """
        parts = []
        for i, h in enumerate(self.h_list):
            h.diag()
            parts.append(
                _build_edata_for_ex(
                    h, self.ex_list[i], h_index=i,
                    h_weight=float(self.weights_list[i]),
                ).arr
            )

        if len(parts) == 0:
            return EData.empty(0)
        return EData(np.concatenate(parts))


    def fd_jacobian(self, x=None, *, delta=None, rel_delta=1e-5,
                    atol=1e-8, check_swaps=True):
        r"""
        Finite-difference energy Jacobian for a multi-Hamiltonian fit.

        Computes :math:`J_E[i, \alpha] = \partial E_i / \partial x_\alpha`
        by central differences across all Hamiltonians.  Row order
        matches :py:meth:`get_edata`; ``x`` and the column order match
        :attr:`x0`.  See :py:meth:`EFit.fd_jacobian` for parameter
        details.

        Returns
        -------
        J : np.ndarray, shape ``(n_obs, n_p_real)``
            The unweighted energy Jacobian.  Also stored on
            :attr:`last_jacobian`.

        Notes
        -----
        On exit the fit's parameter state matches its state on entry.
        Cost is ``2 * n_p_real`` diagonalisations of every Hamiltonian.
        """
        return _fd_jacobian_impl(self, x=x, delta=delta,
                                 rel_delta=rel_delta, atol=atol,
                                 check_swaps=check_swaps)

    def covariance(self, x=None, *, jacobian=None,
                   scale="reduced_chi2", **fd_kwargs):
        r"""
        Variance-covariance matrix for the real-valued parameter vector.

        See :py:meth:`EFit.covariance` for the convention; for ``MHFit``
        the Jacobian rows correspond to the concatenated per-Hamiltonian
        observation list returned by :py:meth:`get_edata`.
        """
        return _covariance_impl(self, x=x, jacobian=jacobian,
                                scale=scale, **fd_kwargs)


cdef shxdata_alloc_helper(sh, shx, weights):
    """
    Generate cfl.shx_data array of spin Hamiltonian experimental data.

    Parameters
    ----------
    sh : SpinHamiltonian
        The pycf spin Hamiltonian object.
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Is used to specify the chi squared weighting of each spin Hamiltonian
        interaction.  Valid keys are: zeeman, hyperfine, and quadrupole, and
        omitted values are set to unity.
    Returns
    -------
    ret : tuple
        First entry is a PyCapsule containing a pointer to the created
        cfl.shx_data array, and the second entry is a list numpy arrays.  The
        numpy arrays point to the pa arrays of each shx_array element and
        consequently a reference to shx_list should be kept for as long as the
        shx_data array is required in order to prevent the GC from deallocing
        corresponding pa chunks of memory.
    """
    cdef np.ndarray[double, ndim=1, mode="c"] shx_pa

    shx_list = []
    shx_array = <cfl.shx_data **>malloc(len(sh.interactions)*sizeof(cfl.shx_data *))
    if shx_array == NULL:
        raise MemoryError("shx_array alloc failed")

    for i,inter in enumerate(sh.interactions):
        if inter not in shx:
            for j in range(i):
                free(shx_array[j])
            free(shx_array)
            raise ValueError("The spin Hamiltonian experimental data dictionary "
                    "is missing data for the {} interaction.".format(inter))
        elif not isinstance(shx[inter], np.ndarray):
            for j in range(i):
                free(shx_array[j])
            free(shx_array)
            raise TypeError("exp_tensor must be a np.ndarray.")
        elif shx[inter].shape == (3, 3):
            shx_list += [np.ascontiguousarray(shx[inter].flatten(), dtype=np.float64)]
        elif shx[inter].shape == (9,):
            shx_list += [np.ascontiguousarray(shx[inter], dtype=np.float64)]
        else:
            for j in range(i):
                free(shx_array[j])
            free(shx_array)
            raise ValueError("exp_tensor must either be a (3, 3) or (9, 1) array.")
        shx_array[i] = <cfl.shx_data *>malloc(sizeof(cfl.shx_data))
        if shx_array[i] == NULL:
            for j in range(i):
                free(shx_array[j])
            free(shx_array)
            raise MemoryError("shx_array[{}] alloc failed".format(i))
        shx_pa = <np.ndarray[double, ndim=1, mode="c"]> shx_list[i]
        shx_array[i].pa = &shx_pa[0]
        try:
            wi = weights[inter]
        except KeyError:
            wi = 1.0
        shx_array[i].chisq_weight = wi

    shx_array_cap = PyCapsule_New(<void *>shx_array, "pycfl.ShxArray", NULL)

    return (shx_array_cap, shx_list)


cdef class ESHFit(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    energy level and spin Hamiltonian data.

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of each
    coefficient when they are set also determines whether that coefficient is
    fit as real or complex parameter.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels.
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit.  Must have projection data set
        with the set_pro_data method.  If it contains hyperfine or quadrupole
        interactions, the respective coupling constants will automatically be
        added to the parameters.
    ex : np.ndarray or ExData
        Either a 2 by n dimensional np.ndarray or an ExData type object.  In the
        former case, n is the number of energy levels, with the first column
        containing energy level indices starting at 1, and the second column
        containing the absolute experimental energy of the corresponding level.
        In order to specify energy level differences, or specify energies
        according to their SLJM state labels, use the ExData interface.
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.  Any omitted values will be set
        to unity.
    svd_sym : bool, optional
        Symmeterize spin Hamiltonian parameter tensors by applying an SVD
        transformation.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    cdef SpinHamiltonian sh
    cdef public Hamiltonian h
    cdef Hamiltonian hpro
    cdef public dict coeff
    cdef int n_p
    cdef public list parameters
    cdef public int n_p_real
    cdef public int n_obs
    cdef public dict param_types
    cdef cfl.ex_data *ex_data
    cdef public ExData ex
    cdef cfl.param_type **param_array
    cdef cfl.shx_data **shx_array
    cdef list shx_list
    cdef public np.ndarray x0
    cdef cfl.eshfit_data *eshfit_data
    cdef public object obj_f_cap
    cdef public object nls_f_cap
    cdef public object fit_data_cap
    cdef public np.ndarray chi2
    cdef dict weights
    cdef object _ex_w_backing   # keeps ex_data.w buffer alive (see exdata_alloc_helper)
    def __init__(self, parameters, h, sh, ex, shx, weights, **kwargs):
        self.n_p = len(parameters)
        self.parameters = parameters
        self.sh = sh

        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")

        if h.coeff_dict is None:
            raise ValueError("Hamiltonian must have coefficients set prior to eshfit.")
        else:
            self.coeff = copy.deepcopy(h.coeff_dict)

        if not sh.pro_data_set:
            raise ValueError("Spin Hamiltonian must have projection data set prior to eshfit.")

        # Add small magnetic field for state-label sorting; generate hpro, if
        # required.
        (h, self.hpro) = sh_hpro_helper(h, sh)
        self.h = h

        # Determine the type of each parameter.
        self.param_types = {}
        # The number of real parameters.
        self.n_p_real = 0
        for i,p in enumerate(parameters):
            if all((p not in hh for hh in [h, sh])):
                raise ValueError("Parameter %s not found in any Hamiltonian." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                self.n_p_real += 2
                self.param_types[p] = "c"
            elif p == 'HYP':
                self.n_p_real += 1
                self.param_types[p] = "h"
            elif p == 'EQHYP':
                self.n_p_real += 1
                self.param_types[p] = "q"
            else:
                self.param_types[p] = "r"
                self.n_p_real += 1

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False

        # Parse the energy level data, if required.
        if not isinstance(ex, ExData):
            self.ex = ExData(ex)
        else:
            self.ex = ex
        self.n_obs = self.ex.n_obs

        self.n_obs += sh.nsh
        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of \
                    parameters, %i, exceeds the number of observables, %i." %
                    (self.n_p_real, self.n_obs))

        if 'energy' not in weights:
            weights['energy'] = 1.0
        self.weights = weights

        cap, self._ex_w_backing = exdata_alloc_helper(self.ex, weights['energy'])
        self.ex_data = <cfl.ex_data *>PyCapsule_GetPointer(cap, "pycfl.ExData")

        # Prepare array of pointers to parameter data structs.
        param_array = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
        if param_array == NULL:
            free(self.ex_data)
            # NULL self.ex_data so __dealloc__ does not double-free it.
            self.ex_data = NULL
            raise MemoryError("param_array alloc failed")
        self.param_array = param_array

        self.x0 = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        ii = 0
        for i,p in enumerate(parameters):
            param_array[i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                # NULL both members so __dealloc__ does not double-free them.
                self.ex_data = NULL
                free(self.param_array)
                self.param_array = NULL
                raise MemoryError("param_array[{}] alloc failed".format(i))

            param_array[i].type = ord(self.param_types[p])

            # Set the coeff index of parameters that are present in the CF
            # Hamiltonian.
            try:
                param_array[i].ci = self.h.index(p)
            except KeyError:
                continue

            # Set the index of the ith param in the x array.
            param_array[i].xi = ii

            if self.param_types[p] == 'c':
                self.x0[ii] = np.real(self.coeff[p])
                self.x0[ii+1] = np.imag(self.coeff[p])
                ii += 2
            else:
                self.x0[ii] =  self.coeff[p]
                ii += 1

        # Check the SVD kwarg...
        if 'svd_sym' in kwargs:
            if kwargs['svd_sym']:
                svd = <char> 'S'
                # Ensure any input spin Hamiltonian parameters are in the
                # singular value decomposition basis.
                for inter in shx:
                    # Disable SVD for quad, since there's no S matrix elements.
                    if inter != 'quadrupole':
                        shx[inter] = sh_svd(shx[inter])
            else:
                svd = <char> 'N'
        else:
            svd = <char> 'N'

        # Create array of experimental spin Hamiltonian data.
        try:
            (shx_array_cap, self.shx_list) = shxdata_alloc_helper(sh, shx, weights)
        except BaseException:
            for i in range(self.n_p):
                free(param_array[i])
            free(self.ex_data)
            # NULL both members so __dealloc__ does not double-free them.
            self.ex_data = NULL
            free(param_array)
            self.param_array = NULL
            raise
        shx_array = <cfl.shx_data **>PyCapsule_GetPointer(shx_array_cap, "pycfl.ShxArray")
        self.shx_array = shx_array

        if (self.hpro is not None):
            if self.ex.sl_index:
                self.eshfit_data = eshfit_data_alloc('S', svd, <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap,
                    "pycfl.Hamiltonian"),
                    <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"),
                    self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.param_array)
            else:
                self.eshfit_data = eshfit_data_alloc('N', svd, <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap,
                    "pycfl.Hamiltonian"),
                    <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"),
                    self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.param_array)
            if self.eshfit_data is NULL:
                for i in range(len(self.sh.interactions)):
                    free(self.shx_array[i])
                # Free and NULL the outer shx_array so __dealloc__ skips it.
                free(self.shx_array)
                self.shx_array = NULL
                for i in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                # NULL all members freed above so __dealloc__ does not double-free.
                self.ex_data = NULL
                free(param_array)
                self.param_array = NULL
                raise MemoryError("eshfit_data_alloc failed")

            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_obj, "pycfl.MinObjF", NULL)

        else:
            if self.ex.sl_index:
                self.eshfit_data = eshfit_data_alloc('S', svd, <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap,
                    "pycfl.Hamiltonian"),
                    NULL, self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.param_array)
            else:
                self.eshfit_data = eshfit_data_alloc('N', svd, <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap,
                    "pycfl.Hamiltonian"),
                    NULL, self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.param_array)
            if self.eshfit_data is NULL:
                for i in range(len(self.sh.interactions)):
                    free(self.shx_array[i])
                # Free and NULL the outer shx_array so __dealloc__ skips it.
                free(self.shx_array)
                self.shx_array = NULL
                for i in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                # NULL all members freed above so __dealloc__ does not double-free.
                self.ex_data = NULL
                free(param_array)
                self.param_array = NULL
                raise MemoryError("eshfit_data_alloc failed")

            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_obj, "pycfl.MinObjF", NULL)

        self.fit_data_cap = PyCapsule_New(<void *>self.eshfit_data, "pycfl.MinData", NULL)
        self.nls_f_cap = None

    def __dealloc__(self):
        if self.ex_data != NULL:
            free(self.ex_data)
        if self.param_array != NULL:
            for i in range(self.n_p):
                free(self.param_array[i])
            free(self.param_array)
        if self.shx_array != NULL:
            for i in range(len(self.sh.interactions)):
                free(self.shx_array[i])
            free(self.shx_array)
        if self.eshfit_data != NULL:
            cfl.eshfit_data_free(self.eshfit_data)

    def __iter__(self):
        for p in self.parameters:
            yield p

    def fit(self, min_object):
        r"""
        Run the fit using the provided minimization object.

        Parameters
        ----------
        min_object : CFLMin
            The minimization object to be used, which sets the optimization
            algorithm, bounds and other settings as applicable to the selected
            algorithm.

        Returns
        -------
        result : tuple
            The first element is a np.ndarray containing complex coefficients
            while the second entry contains the final value of the objective
            function.

        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0
        fmin = min_object.minimize(self, x)

        params = {}
        ri = 0
        for p in self:
            if (self.param_types[p] == 'c'):
                params[p] = complex(x[ri], x[ri+1])
                ri += 2
            else:
                params[p] = x[ri]
                ri += 1
        chi2 = np.ascontiguousarray(np.zeros(len(self.sh.interactions)+1, dtype=np.float64))
        if (self.hpro is not None):
            cfl.eshfit_hpro_chi2(&x[0], self.eshfit_data, &chi2[0])
        else:
            cfl.eshfit_chi2(&x[0], self.eshfit_data, &chi2[0])
        self.chi2 = chi2

        return(params, fmin)

    @cython.boundscheck(False)
    def eval(self, coeff):
        r"""
        Return chi2 obtained for the provided coefficients.

        Parameters
        ----------
        coeff : dict
            The usual coeff dict; if coefficients for only a subset of tensors
            are provided, the remainder are held at their initial value.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        self.coeff.update(copy.deepcopy(coeff))
        set_param_helper(self)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        chi2 = np.ascontiguousarray(np.zeros(len(self.sh.interactions)+1, dtype=np.float64))
        # Use hpro variant when a projection Hamiltonian is present, matching fit().
        if (self.hpro is not None):
            with nogil:
                cfl.eshfit_hpro_chi2(&x[0], self.eshfit_data, &chi2[0])
        else:
            with nogil:
                cfl.eshfit_chi2(&x[0], self.eshfit_data, &chi2[0])

        return chi2


cdef class MESHFit(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    multiple Hamiltonians and spin Hamiltonians.  For now, this is restricted to
    a single spin Hamiltonian per CF Hamiltonian.  Thus, one can fit one excited
    state spin Hamiltonian, excluding hyperfine, in conjunction with electronic
    energy level data.  This is can then combined with a hyperfine spin
    Hamiltonian for the ground state.

    The Hamiltonians must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of each
    coefficient when they are set also determines whether that coefficient is
    fit as real or complex parameter, thus they must be consistent among each
    Hamiltonian.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h_sh_list : list
        Each element should be a dictionary with the following keys: 'h', 'sh',
        'ex', 'shx', 'weights', and svd_sym.  For descriptions of each element,
        see the ESHFit docstring.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    cdef int n_h
    cdef public Hamiltonian h
    cdef public dict coeff
    cdef public list h_list
    cdef public list hpro_list
    cdef public list sh_list
    cdef int n_p
    cdef public list parameters
    cdef public int n_p_real
    cdef public int n_obs
    cdef public dict param_types
    cdef list h_param_list
    cdef public list ex_list
    cdef list ex_data
    cdef list _ex_w_backing     # keeps each ex_data.w buffer alive (see exdata_alloc_helper)
    cdef list param_arrays
    cdef list shx_list
    cdef list shx_arrays
    cdef public np.ndarray x0
    cdef cfl.eshfit_data **eshfit_array
    cdef cfl.meshfit_data *meshfit_data
    cdef public object obj_f_cap
    cdef public object nls_f_cap
    cdef public object fit_data_cap
    cdef public np.ndarray chi2
    cdef public list weights_list
    def __init__(self, parameters, h_sh_list, **kwargs):
        self.n_h = len(h_sh_list)
        self.n_p = len(parameters)
        h_list = []
        hpro_list = []
        sh_list = []
        ex_list = []
        shx_list = []
        weights_list = []
        svd_list = []

        self.coeff = {}             # Local copy of all coefficients of any H/SH.
        h_param_list = []           # Array of arrays specifying parameters of each H.
        n_zxa = np.zeros(self.n_h)  # The number of complex parameters of each H/SH pair.
        self.n_obs = 0              # The number of observables.
        n_ex = 0                    # The number of experimental electronic energy level sets.
        ex_job_list = []            # Specifies whether: state-label sort, standard ex, or no ex.
        for i,d in enumerate(h_sh_list):
            try:
                h = d['h']
            except KeyError:
                raise KeyError("Each h_sh_list element must be a dictionary containing "\
                        "an 'h' key that points to a Hamiltonian object.")
            if h.coeff_dict is None:
                raise ValueError("Hamiltonian must have coefficients set prior to meshfit.")
            else:
                self.coeff.update(copy.deepcopy(h.coeff_dict))

            h_param_list += [[p for p in parameters if p in h]]
            n_zxa[i] += len(h_param_list[i])

            try:
                sh = d['sh']
            except KeyError:
                raise KeyError("Each h_sh_list element must be a dictionary containing "\
                        "an 'sh' key that points to a SpinHamiltonian object.")
            if not sh.pro_data_set:
                raise ValueError("Spin Hamiltonian must have projection data set prior to eshfit.")
            self.n_obs += sh.n_obs

            # Add small magnetic field for state-label sorting; generate hpro, if
            # required.
            (h, hpro) = sh_hpro_helper(h, sh)

            if 'ex' in d:
                ex = d['ex']
                if not isinstance(ex, ExData):
                    ex = ExData(ex)

                ex_list += [ex]
                self.n_obs += ex_list[i].n_obs
                n_ex += 1
                if ex.sl_index:
                    ex_job_list += [<char> 'S']
                else:
                    ex_job_list += [<char> 'N']
            else:
                # No energy level data; passing an empty array to ExData sets
                # n_obs attribute to 0, which disables energy level chi2 fitting
                # in cfl.
                ex_list += [ExData(np.empty((0,2)))]
                ex_job_list += [<char> 'N']
            try:
                shx_list += [d['shx']]
            except KeyError:
                raise KeyError("Each h_sh_list element must be a dict containing an 'shx' "\
                        "key that points to a dict of experimental spin Hamiltonian data.")
            if any(inter not in shx_list[i] for inter in sh.interactions):
                raise ValueError("Missing experimental spin Hamiltonian data for one or more interactions.")
            try:
                weights_list += [d['weights']]
            except KeyError:
                weights_list += [{}]
            # Set default weights to unity.
            for w in ['energy'] + sh.interactions:
                if w not in weights_list[i]:
                    weights_list[i][w] = 1
            if 'svd_sym' in d:
                if d['svd_sym']:
                    svd_list += [<char> 'S']
                    for inter in shx_list[i]:
                        # Disable SVD for quad, since there's no S matrix
                        # elements.
                        if inter != 'quadrupole':
                            shx_list[i][inter] = sh_svd(shx_list[i][inter])
                else:
                    svd_list += [<char> 'N']
            else:
                svd_list += [<char> 'N']

            h_list += [h]
            hpro_list += [hpro]
            sh_list += [sh]

        self.h = h_list[0]
        self.h_list = h_list
        self.hpro_list = hpro_list
        self.sh_list = sh_list
        self.h_param_list = h_param_list
        self.parameters = parameters
        self.ex_list = ex_list
        self.weights_list = weights_list

        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")

        self.param_types = {}       # The type of each parameter (real, complex, or imag).
        self.n_p_real = 0           # The total number of real parameters (two for each complex number).
        x0_index = {}               # Index of each parameter in the real-valued param array.
        for i,p in enumerate(parameters):
            if all((p not in hh for hh in (h_list + sh_list) )):
                raise ValueError("Parameter %s not found in any Hamiltonian or spin Hamiltonian." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                x0_index[p] = self.n_p_real
                self.n_p_real += 2
                self.param_types[p] = "c"
            # n_zxa is the total number of complex parameters for each H/SH
            # pair; therefore, we have account for any parameters that are only
            # in SH.
            elif p == 'HYP':
                x0_index[p] = self.n_p_real
                self.n_p_real += 1
                self.param_types[p] = "h"
                for j,h in enumerate(h_list):
                    if p not in h and p in sh_list[j]:
                        n_zxa[j] += 1
            elif p == 'EQHYP':
                x0_index[p] = self.n_p_real
                self.n_p_real += 1
                self.param_types[p] = "q"
                for j,h in enumerate(h_list):
                    if p not in h and p in sh_list[j]:
                        n_zxa[j] += 1
            else:
                x0_index[p] = self.n_p_real
                self.param_types[p] = "r"
                self.n_p_real += 1

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False

        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
           raise ValueError("The total (real and imaginary) number of \
                    parameters, %i, exceeds the number of observables, %i." %
                    (self.n_p_real, self.n_obs))

        ex_data = []
        ex_w_backing = []
        for i in range(self.n_h):
            if ex_list[i] is not None:
                try:
                    cap, ex_w_np = exdata_alloc_helper(ex_list[i], weights_list[i]['energy'])
                except BaseException:
                    for j in range(i):
                        free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[j], "pycfl.ExData"))
                    raise
                ex_data += [cap]
                ex_w_backing += [ex_w_np]
            else:
                ex_data += [PyCapsule_New(<void *>NULL, "pycfl.ExData", NULL)]
                ex_w_backing += [None]
        self.ex_data = ex_data
        self._ex_w_backing = ex_w_backing

        # Prepare array of pointers to parameter data structs.
        param_arrays = []
        for hi,h in enumerate(h_list):
            pa_hi = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
            if pa_hi is NULL:
                for hj in range(hi):
                    pa_hj = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[hj], "pycfl.ParamArrays")
                    for j in range(len(h_param_list[hj])):
                        free(pa_hj[j])
                    free(pa_hj)
                for j in range(self.n_h):
                    free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[j], "pycfl.ExData"))
                raise MemoryError("param_arrays[{}] alloc failed".format(hi))

            for i,p in enumerate(h_param_list[hi]):
                pa_hi[i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
                if pa_hi[i] is NULL:
                    for j in range(i):
                        free(pa_hi[j])
                    free(pa_hi)
                    for hj in range(hi):
                        pa_hj = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[hj], "pycfl.ParamArrays")
                        for j in range(len(h_param_list[hj])):
                            free(pa_hj[j])
                        free(pa_hj)
                    for j in range(self.n_h):
                        free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[j], "pycfl.ExData"))
                    raise MemoryError("param_arrays[{0}][{1}] alloc failed".format(hi, i))

                pa_hi[i].type = ord(self.param_types[p])
                pa_hi[i].ci = h_list[hi].index(p)
                pa_hi[i].xi = x0_index[p]

            param_arrays += [PyCapsule_New(<void *>pa_hi, "pycfl.ParamArrays", NULL)]

        # Set initial values.
        self.x0 = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        set_param_helper(self)
        self.param_arrays = param_arrays

        # Create list of experimental spin Hamiltonian data arrays.
        shx_arrays = []
        for shi, sh in enumerate(sh_list):
            try:
                (shx_ptr, self.shx_list) = shxdata_alloc_helper(sh, shx_list[shi], weights_list[shi])
            except BaseException:
                for shj in range(shi):
                    shx_j = <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[shj], "pycfl.ShxArray")
                    for j in range(len(sh_list[shj].interactions)):
                        free(shx_j[j])
                    free(shx_j)
                for hi in range(self.n_h):
                    pa_hi = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[hi], "pycfl.ParamArrays")
                    for i in range(len(h_param_list[hi])):
                        free(pa_hi[i])
                    free(pa_hi)
                for i in range(self.n_h):
                    free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[i], "pycfl.ExData"))
                raise

            shx_arrays += [shx_ptr]

        self.shx_arrays = shx_arrays
        self.eshfit_array = <cfl.eshfit_data **>malloc(self.n_h*sizeof(cfl.eshfit_data *))
        if self.eshfit_array is NULL:
            for i in range(self.n_h):
                pa_hi = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[i], "pycfl.ParamArrays")
                for j in range(len(h_param_list[i])):
                    free(pa_hi[j])
                free(pa_hi)
                free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[i], "pycfl.ExData"))
                shx_i = <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[i], "pycfl.ShxArray")
                for j in range(len(sh_list[i].interactions)):
                    free(shx_i[j])
                free(shx_i)
            raise MemoryError("eshfit_array alloc failed")

        for i in range(self.n_h):
            if hpro_list[i] is not None:
                self.eshfit_array[i] = cfl.eshfit_data_alloc(ex_job_list[i], svd_list[i],
                    <cfl.zh *>PyCapsule_GetPointer(h_list[i].h_cap, "pycfl.Hamiltonian"),
                    <cfl.zh *>PyCapsule_GetPointer(hpro_list[i].h_cap, "pycfl.Hamiltonian"),
                    <cfl.ex_data *>PyCapsule_GetPointer(ex_data[i], "pycfl.ExData"),
                    <cfl.zsh *>PyCapsule_GetPointer(sh_list[i].sh_cap, "pycfl.SpinHamiltonian"),
                    <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[i], "pycfl.ShxArray"), n_zxa[i],
                    <cfl.param_type **>PyCapsule_GetPointer(param_arrays[i], "pycfl.ParamArrays"))
                if self.eshfit_array[i] is NULL:
                    for j in range(i):
                        cfl.eshfit_data_free(self.eshfit_array[j])
                    free(self.eshfit_array)
                    self.eshfit_array = NULL
                    for ii in range(self.n_h):
                        pa_hi = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[ii], "pycfl.ParamArrays")
                        for j in range(len(h_param_list[ii])):
                            free(pa_hi[j])
                        free(pa_hi)
                        free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[ii], "pycfl.ExData"))
                        shx_i = <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[ii], "pycfl.ShxArray")
                        for j in range(len(sh_list[ii].interactions)):
                            free(shx_i[j])
                        free(shx_i)
                    raise MemoryError("eshfit_data_alloc failed")
            else:
                self.eshfit_array[i] = cfl.eshfit_data_alloc(ex_job_list[i], svd_list[i],
                    <cfl.zh *>PyCapsule_GetPointer(h_list[i].h_cap, "pycfl.Hamiltonian"), NULL,
                    <cfl.ex_data *>PyCapsule_GetPointer(ex_data[i], "pycfl.ExData"),
                    <cfl.zsh *>PyCapsule_GetPointer(sh_list[i].sh_cap, "pycfl.SpinHamiltonian"),
                    <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[i], "pycfl.ShxArray"), n_zxa[i],
                    <cfl.param_type **>PyCapsule_GetPointer(param_arrays[i], "pycfl.ParamArrays"))
                if self.eshfit_array[i] is NULL:
                    for j in range(i):
                        cfl.eshfit_data_free(self.eshfit_array[j])
                    free(self.eshfit_array)
                    self.eshfit_array = NULL
                    for ii in range(self.n_h):
                        pa_hi = <cfl.param_type **>PyCapsule_GetPointer(param_arrays[ii], "pycfl.ParamArrays")
                        for j in range(len(h_param_list[ii])):
                            free(pa_hi[j])
                        free(pa_hi)
                        free(<cfl.ex_data *>PyCapsule_GetPointer(ex_data[ii], "pycfl.ExData"))
                        shx_i = <cfl.shx_data **>PyCapsule_GetPointer(shx_arrays[ii], "pycfl.ShxArray")
                        for j in range(len(sh_list[ii].interactions)):
                            free(shx_i[j])
                        free(shx_i)
                    raise MemoryError("eshfit_data_alloc failed")

        self.meshfit_data = meshfit_data_alloc(self.n_h, self.eshfit_array)
        self.fit_data_cap = PyCapsule_New(<void *>self.meshfit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.meshfit_obj, "pycfl.MinObjF", NULL)
        self.nls_f_cap = None

    def __dealloc__(self):
        for i in range(self.n_h):
            ex_i = <cfl.ex_data *>PyCapsule_GetPointer(self.ex_data[i], "pycfl.ExData")
            if ex_i != NULL:
                free(ex_i)
            pa_hi = <cfl.param_type **>PyCapsule_GetPointer(self.param_arrays[i], "pycfl.ParamArrays")
            if pa_hi != NULL:
                for j in range(len(self.h_param_list[i])):
                    free(pa_hi[j])
                free(pa_hi)
            if self.shx_arrays[i] is not None:
                shx_i = <cfl.shx_data **>PyCapsule_GetPointer(self.shx_arrays[i], "pycfl.ShxArray")
                if shx_i != NULL:
                    for j in range(len(self.sh_list[i].interactions)):
                        free(shx_i[j])
                    free(shx_i)
        if self.eshfit_array != NULL:
            for i in range(self.n_h):
                cfl.eshfit_data_free(self.eshfit_array[i])
            free(self.eshfit_array)

        if self.meshfit_data != NULL:
            cfl.meshfit_data_free(self.meshfit_data)

    def __iter__(self):
        for p in self.parameters:
            yield p

    def fit(self, min_object):
        r"""
        Run the fit using the provided minimization object.

        Parameters
        ----------
        min_object : CFLMin
            The minimization object to be used, which sets the optimization
            algorithm, bounds and other settings as applicable to the selected
            algorithm.

        Returns
        -------
        result : tuple
            The first element is a np.ndarray containing complex coefficients
            while the second entry contains the final value of the objective
            function.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0
        fmin = min_object.minimize(self, x)

        params = {}
        ri = 0

        for p in self:
            if (self.param_types[p] == 'c'):
                params[p] = complex(x[ri], x[ri+1])
                ri += 2
            else:
                params[p] = x[ri]
                ri += 1

        nchi2 = 0
        for sh in self.sh_list:
            nchi2 += len(sh.interactions) + 1      # +1 for each energy level chi2.
        chi2 = np.ascontiguousarray(np.zeros(nchi2, dtype=np.float64))
        cfl.meshfit_chi2(&x[0], self.meshfit_data, &chi2[0])
        self.chi2 = chi2

        return(params, fmin)

    @cython.boundscheck(False)
    def eval(self, coeff):
        r"""
        Return chi2 obtained for the provided coefficients.

        Parameters
        ----------
        coeff : dict
            The usual coeff dict; if coefficients for only a subset of tensors
            are provided, the remainder are held at their initial value.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[double, ndim=1, mode="c"] chi2

        self.coeff.update(copy.deepcopy(coeff))
        set_param_helper(self)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.x0

        nchi2 = 0
        for sh in self.sh_list:
            nchi2 += len(sh.interactions) + 1      # +1 for each energy level chi2.
        chi2 = np.ascontiguousarray(np.zeros(nchi2, dtype=np.float64))
        with nogil:
            cfl.meshfit_chi2(&x[0], self.meshfit_data, &chi2[0])

        return chi2


cdef class CFLMin:
    r"""
    Object for initializing and configuring minimization routines to be passed
    to e_fit or esh_fit.

    Parameters
    ----------
    method : string
        The minimization routine to employ.  Available options are:

            - 'basinhopping'
            - 'siman'
            - 'gsl_nmsimplex2rand'
            - 'gsl_nmsimplex2'
            - 'gsl_conjugate_fr'
            - 'gsl_conjugate_pr'
            - 'gsl_vector_bfgs2'
            - 'nlopt_cobyla'
            - 'nlopt_bobyqa'
            - 'nlopt_sbplx'
            - 'nlopt_crs2_lm'
            - 'nlopt_esch'

        For simulated annealing ('siman'), all accepted steps are returned as
        part of the fitting result dictionary with keyword argument 'xaccept'.
        This can be used estimate the posterior distribution and get a handle on
        the uncertainty of ones parameters.

        It is also possible to use the GSL nonlinear least-squares method, which
        will use a finite difference method to estimate the Jacobian and,
        accordingly, return the covariance matrix.  This assumes that the
        solution landscape can be approximated by a well conditioned function
        near the minimum.  The corresponding method argument is 'gsl_nls'.

    bounds : dict, optional
        Parameter bounds, for supported algorithms (nlopt, basinhopping, siman).
        Keys specify the tensor name (note that tensors created by tensor
        arithmetic should have their name attribute set explicitly), while
        values correspond to tuples, the first entry of which is the lower bound
        and the second entry the upper bound.  The number of elements in bounds
        must match the length of the parameters list.
    lmin : CFLMin, optional
        The local minimization routine, applicable only for basinhopping
        algorithm; defaults to nlopt_bobyqa.  Implemented options fall into two
        categories, routines from gsl, and routines from nlopt.  For the former,
        available algorithms are:

            - 'gsl_nmsimplex2rand'
            - 'gsl_nmsimplex2'
            - 'gsl_conjugate_fr'
            - 'gsl_conjugate_pr'
            - 'gsl_vector_bfgs2'

        For the latter, available algorithms are:

            - 'nlopt_cobyla'
            - 'nlopt_bobyqa'
            - 'nlopt_sbplx'.

    stepsize : dict, optional
        The stepsize for parameter variation; this argument is only used if the
        basinhopping or siman algorithm is selected.  Keys specify the tensor
        name, while values correspond to the stepsize.  For basinhopping, if
        adaptive stepsize is enabled, then this dictionary is used as the
        starting stepsize, and all step sizes are scaled by the same factor in
        order to achieve the target acceptance rate.  In other words, this kwarg
        is then used to set the relative proportion between the step sizes.  For
        simulated annealing, this is a multiplicative factor A for a stepsize of
        magnitude A*(u*2-1) with u a random number in the interval (0...1],
        specified for each parameter.
    niter : int, optional
        The number of iterations to complete, used by basinhopping, siman, and
        gsl_nls; defaults to 100, 1e6, and 100, respectively.
    target_accept_rate : float, optional
        The target acceptance rate for basinhopping steps; used for adaptive
        stepsize tuning.  To disable adaptive stepsize, set this parameter to 0.
        The default is 0.5.
    step_adapt_int : int, optional
        The number of iterations between adaptive stepsize checks; defaults to 20.
    Tstart : float
        Starting temperature for simulated annealing schedule; defaults to 1e3.
    Tmin : float
        Minimum temperature for simulated annealing; defaults to 1e-3.
    muT : float
        The damping constant for the simulated annealing cooling schedule.  For
        consecutive iterations, the temperature is decreased by a factor of
        1/muT until the minimum temperature is reached.  Defaults to 1.000005,
        but this will need to be adjusted depending on the initial fmin value.
    k : float
        Boltzmann constant; used in exp(-E/kT) to decide whether a step should
        be accepted.  Defaults to unity.
    xtol : float, optional
        If either the global optimization or a local basinhopping minimization
        routine is from nlopt, the ``xtol`` argument can be used to set the
        relative tolerance in parameters x to be used as a stopping criteria.
        Defaults to 1e-5.
    maxtime : float, optional
        The maximum wall time in seconds.  Only used by nlopt routines and
        forces the optimization to return if exceeded, providing the best
        solution so far.  This time is not a hard limit, and it may take a
        little longer to return depending on the evaluation time of each
        iteration.
    dry_run : bool, optional
        Don't run the actual minimization, but perform all the data prep and
        generate a summary with the initial parameters.  Useful for checking how
        well a set parameters fit the prepared input data.
    """
    cdef public str method
    cdef public dict kwargs
    cdef int niter
    cdef size_t nx
    cdef double xtol
    cdef double maxtime
    cdef double Tstart
    cdef double Tmin
    cdef double muT
    cdef double k
    cdef cfl.cfl_min_bounds *cfl_bounds
    cdef cfl.cfl_min_obj *min_obj
    cdef cfl.cfl_min_obj *bh_lmin_obj

    def __cinit__(self, method, **kwargs):
        if method == 'basinhopping':
            if 'niter' in kwargs:
                self.niter = int(kwargs['niter'])
            else:
                self.niter = 100
        elif method == 'gsl_nmsimplex2rand':
            pass
        elif method == 'gsl_nmsimplex2':
            pass
        elif method == 'gsl_conjugate_fr':
            pass
        elif method == 'gsl_conjugate_pr':
            pass
        elif method == 'gsl_vector_bfgs2':
            pass
        elif method == 'nlopt_cobyla':
            pass
        elif method == 'nlopt_bobyqa':
            pass
        elif method == 'nlopt_sbplx':
            pass
        elif method == 'nlopt_crs2_lm':
            pass
        elif method == 'nlopt_esch':
            pass
        elif method == 'gsl_nls':
            if 'niter' in kwargs:
                self.niter = int(kwargs['niter'])
            else:
                self.niter = 100
        elif method == 'siman':
            if 'niter' in kwargs:
                self.niter = int(kwargs['niter'])
            else:
                self.niter = int(1e6)
        else:
            raise NotImplementedError("Method '%s' is not an existing option." % method)

        self.method = method
        self.kwargs = kwargs

    def __dealloc__(self):
        if self.cfl_bounds != NULL:
            free(self.cfl_bounds)
        if self.min_obj != NULL:
            cfl.cfl_min_free(self.min_obj)
        if self.bh_lmin_obj != NULL:
            cfl.cfl_min_free(self.bh_lmin_obj)

    @cython.boundscheck(False)
    cpdef minimize(self, fit_obj, x0):
        r"""
        Run the minimization.

        Parameters
        ----------
        fit_obj : EFit or ESHFit
            The object for which to perform the fit.
        x0 : np.ndarray
            Real valued vector.  Upon entry, these are the initial guesses for
            the parameters; if minimization is successful, x0 will be
            overwritten with the solution.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] cx0
        cdef size_t cnx
        cdef double cxtol = 0
        cdef double cgtol = 0
        cdef double cftol = 0
        cdef double cmaxtime = -1
        cdef double cTstart = 0
        cdef double cTmin = 0
        cdef double cmuT = 0
        cdef double ck = 0
        cdef double (*obj_f_ptr)(size_t, double *, double *, void *) noexcept
        cdef void (*nls_f_ptr)(double *, void *, double *) noexcept
        cdef void *data_ptr = NULL
        cdef double *covar_ptr = NULL
        cdef double *jac_ptr = NULL
        cdef double *wts_ptr = NULL
        cdef np.ndarray[double, ndim=1, mode="c"] cwts
        cdef np.ndarray[double, ndim=2, mode="c"] covar
        cdef np.ndarray[double, ndim=2, mode="c"] jac
        cdef double *chi2accept_ptr = NULL
        cdef double *xaccept_ptr = NULL
        cdef np.ndarray[double, ndim=1, mode="c"] chi2accept
        cdef np.ndarray[double, ndim=2, mode="c"] xaccept
        cdef cfl.cfl_min_obj *min_obj = NULL
        cdef cfl.cfl_min_obj *lmin_obj = NULL
        cdef double fmin = 0
        cdef np.ndarray[double, ndim=1, mode="c"] lb
        cdef np.ndarray[double, ndim=1, mode="c"] ub
        cdef np.ndarray[double, ndim=1, mode="c"] cstepsize
        cdef double *stepsize_ptr = NULL
        cdef float target_accept_rate = 0.5
        cdef int step_adapt_int = 20
        cdef int retval = 0

        nls_f_ptr = NULL
        cnx = <size_t> len(x0)
        obj_f_ptr = <double (*)(size_t, double *, double *, void *) noexcept>PyCapsule_GetPointer(
                fit_obj.obj_f_cap, "pycfl.MinObjF")
        data_ptr = <void *>PyCapsule_GetPointer(fit_obj.fit_data_cap, "pycfl.MinData")

        # If bounds are specified, convert them to real valued lists the order of
        # which matches the order of the real valued parameter lists.
        if 'bounds' in self.kwargs:
            lb = np.zeros(fit_obj.n_p_real)
            ub = np.zeros(fit_obj.n_p_real)
            rpi = 0

            bounds = self.kwargs['bounds']
            for p in fit_obj:
                if fit_obj.param_types[p] == 'c':
                    try:
                        if not isinstance(bounds[p][0], complex) or \
                                not isinstance(bounds[p][1], complex):
                            raise ValueError("%s bounds are not complex, yet the "
                                    "corresponding coefficient in the Hamiltonian is." % p)
                    except KeyError:
                        raise KeyError("Missing bounds key %s." % p)
                    lb[rpi] = np.real(bounds[p][0])
                    lb[rpi+1] = np.imag(bounds[p][0])
                    ub[rpi] = np.real(bounds[p][1])
                    ub[rpi+1] = np.imag(bounds[p][1])
                    if np.real(fit_obj.coeff[p]) < lb[rpi]:
                        raise ValueError("The real part of the %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p)
                    elif np.imag(fit_obj.coeff[p]) < lb[rpi+1]:
                        raise ValueError("The imaginary part of the %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p)
                    elif np.real(fit_obj.coeff[p]) > ub[rpi]:
                        raise ValueError("The real part of the %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p)
                    elif np.imag(fit_obj.coeff[p]) > ub[rpi+1]:
                        raise ValueError("The imaginary part of the %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p)
                    rpi += 2
                else:
                    try:
                        lb[rpi] = np.real(bounds[p][0])
                        ub[rpi] = np.real(bounds[p][1])
                    except KeyError:
                        raise KeyError("Missing bounds key %s." % p)
                    if fit_obj.coeff[p] < lb[rpi]:
                        raise ValueError("The %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p)
                    elif fit_obj.coeff[p] > ub[rpi]:
                        raise ValueError("The %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p)
                    rpi += 1

            cfl_bounds = <cfl.cfl_min_bounds *>malloc(sizeof(cfl.cfl_min_bounds))
            cfl_bounds.l = &lb[0]
            cfl_bounds.u = &ub[0]
            self.cfl_bounds = cfl_bounds
        else:
            self.cfl_bounds = NULL

        # Create real valued stepsize list, if stepsize is provided.
        if 'stepsize' in self.kwargs:
            cstepsize = np.zeros(fit_obj.n_p_real)
            rpi = 0

            stepsize = self.kwargs['stepsize']
            for p in fit_obj:
                if fit_obj.param_types[p] == 'c':
                    try:
                        if not isinstance(stepsize[p], complex):
                            raise ValueError("%s stepsize is not complex, yet the "
                                    "corresponding Hamiltonian coefficient is." % p)
                    except KeyError:
                        raise KeyError("Missing stepsize key %s." % p)
                    cstepsize[rpi] = np.real(stepsize[p])
                    cstepsize[rpi+1] = np.imag(stepsize[p])
                    rpi += 2
                else:
                    try:
                        cstepsize[rpi] = np.real(stepsize[p])
                    except KeyError:
                        raise KeyError("Missing stepsize key %s." % p)
                    rpi += 1

        # Disable maxtime if not provided.
        if 'maxtime' in self.kwargs:
            cmaxtime = self.kwargs['maxtime']
        else:
            cmaxtime = -1

        # Allocate covariance matrix for non-linear least-squares fit, and set
        # nlls specific tolerances.
        if self.method == 'gsl_nls':
            if fit_obj.nls_f_cap is None:
                raise NotImplementedError("gls_nls is not an existing option for requested fitting mode.")
            else:
                nls_f_ptr = <void (*)(double *, void *, double *) noexcept>PyCapsule_GetPointer(
                        fit_obj.nls_f_cap, "pycfl.NlsObjF")
            if 'xtol' in self.kwargs:
                cxtol = self.kwargs['xtol']
            else:
                cxtol = 1e-8
            if 'gtol' in self.kwargs:
                cgtol = self.kwargs['gtol']
            else:
                cgtol = 1e-8
            if 'ftol' in self.kwargs:
                cftol = self.kwargs['ftol']
            else:
                cftol = 0

            covar = np.ascontiguousarray(np.zeros([fit_obj.n_p_real, fit_obj.n_p_real]),
                    dtype=np.float64)
            self.kwargs['covar'] = covar
            covar_ptr = &covar[0,0]
            jac = np.ascontiguousarray(np.zeros([fit_obj.n_obs, fit_obj.n_p_real]),
                    dtype=np.float64)
            self.kwargs['jac'] = jac
            jac_ptr = &jac[0,0]
            cwts = <np.ndarray[double, ndim=1, mode="c"]>fit_obj.wts
            wts_ptr = &cwts[0]
        else:
            # Set xtol to default if not provided.
            if 'xtol' in self.kwargs:
                cxtol = self.kwargs['xtol']
            else:
                cxtol = 1e-5

        if self.method == 'basinhopping':
            if 'stepsize' in self.kwargs:
                stepsize_ptr = &cstepsize[0]
            else:
                stepsize_ptr = NULL

            if 'target_accept_rate' in self.kwargs:
                target_accept_rate = self.kwargs['target_accept_rate']
            else:
                target_accept_rate = 0.5

            if 'step_adapt_int' in self.kwargs:
                step_adapt_int = self.kwargs['step_adapt_int']
            else:
                step_adapt_int = 20

            if 'lmin' in self.kwargs:
                lmin = self.kwargs['lmin']
                if lmin == 'gsl_nmsimplex2rand':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_nmsimplex2rand)
                elif lmin == 'gsl_nmsimplex2':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_nmsimplex2)
                elif lmin == 'gsl_conjugate_fr':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_conjugate_fr)
                elif lmin == 'gsl_conjugate_pr':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_conjugate_pr)
                elif lmin == 'gsl_vector_bfgs2':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_vector_bfgs2)
                elif lmin == 'nlopt_cobyla':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_cobyla,
                            cxtol, cmaxtime, self.cfl_bounds)
                elif lmin == 'nlopt_bobyqa':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                            cxtol, cmaxtime, self.cfl_bounds)
                elif lmin == 'nlopt_sbplx':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_sbplx,
                            cxtol, cmaxtime, self.cfl_bounds)
                else:
                    raise ValueError("Unknown lmin argument: %s" % lmin)
            else:
                lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                        cxtol, cmaxtime, self.cfl_bounds)

            min_obj = cfl_bh_min_setup(self.niter, stepsize_ptr, target_accept_rate, step_adapt_int,
                    self.cfl_bounds, lmin_obj)
            self.bh_lmin_obj = lmin_obj
        elif self.method == 'siman':
            if 'stepsize' not in self.kwargs:
                cstepsize = np.ones(fit_obj.n_p_real)
            stepsize_ptr = &cstepsize[0]
            if 'Tstart' in self.kwargs:
                cTstart = self.kwargs['Tstart']
            else:
                cTstart = 1e3
            if 'Tmin' in self.kwargs:
                cTmin = self.kwargs['Tmin']
            else:
                cTmin = 1e-3
            if 'muT' in self.kwargs:
                cmuT = self.kwargs['muT']
            else:
                cmuT = 1.000005
            if 'k' in self.kwargs:
                ck = self.kwargs['k']
            else:
                ck = 1.0

            self.Tstart = cTstart
            self.Tmin = cTmin
            self.muT = cmuT
            self.k = ck

            chi2accept = np.ascontiguousarray(np.zeros([self.niter]), dtype=np.float64)
            self.kwargs['chi2accept'] = chi2accept
            chi2accept_ptr = &chi2accept[0]

            xaccept = np.ascontiguousarray(np.zeros([self.niter, fit_obj.n_p_real]),
                    dtype=np.float64)
            self.kwargs['xaccept'] = xaccept
            xaccept_ptr = &xaccept[0,0]

            min_obj = cfl_siman_min_setup(obj_f_ptr, cnx, data_ptr, self.niter, self.cfl_bounds,
                    stepsize_ptr, cTstart, cTmin, cmuT, ck, chi2accept_ptr, xaccept_ptr, cmaxtime)
        elif self.method == 'nlopt_cobyla':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_cobyla,
                    cxtol, cmaxtime, self.cfl_bounds)
        elif self.method == 'nlopt_bobyqa':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                    cxtol, cmaxtime, self.cfl_bounds)
        elif self.method == 'nlopt_sbplx':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_sbplx,
                    cxtol, cmaxtime, self.cfl_bounds)
        elif self.method == 'nlopt_crs2_lm':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_crs2_lm,
                    cxtol, cmaxtime, self.cfl_bounds)
        elif self.method == 'nlopt_esch':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cnx, data_ptr, nlopt_esch,
                    cxtol, cmaxtime, self.cfl_bounds)
        elif self.method == 'gsl_nmsimplex2rand':
            min_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_nmsimplex2rand)
        elif self.method == 'gsl_nmsimplex2':
            min_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_nmsimplex2)
        elif self.method == 'gsl_conjugate_fr':
            min_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_conjugate_fr)
        elif self.method == 'gsl_conjugate_pr':
            min_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_conjugate_pr)
        elif self.method == 'gsl_vector_bfgs2':
            min_obj = cfl_gsl_min_setup(obj_f_ptr, cnx, data_ptr, gsl_vector_bfgs2)
        elif self.method == 'gsl_nls':
            min_obj = cfl_gsl_nls_setup(nls_f_ptr, fit_obj.n_obs,
                    fit_obj.n_p_real, data_ptr, wts_ptr, cxtol, cgtol, cftol,
                    covar_ptr, jac_ptr, self.niter)
        else:
            raise ValueError("Unknown minimization method: %s" % self.method)

        # Assign to self to guarantee there exists a reference to these
        # objects until the CFLMin destructor is called.
        self.xtol = cxtol
        self.maxtime = cmaxtime
        self.min_obj = min_obj
        self.nx = cnx

        cx0 = <np.ndarray[double, ndim=1, mode="c"]> x0
        if 'dry_run' in self.kwargs:
            if self.kwargs['dry_run']:
                fmin = obj_f_ptr(cnx, &cx0[0], NULL, data_ptr)
                retval = 0
            else:
                with nogil:
                    retval = cfl.cfl_min(&cx0[0], &fmin, min_obj)
        else:
            with nogil:
                retval = cfl.cfl_min(&cx0[0], &fmin, min_obj)

        if self.method == 'siman':
            # If siman, trim the returned chi2accept and xaccept array.
            self.kwargs['xaccept'] = self.kwargs['xaccept'][:retval, :]
            self.kwargs['chi2accept'] = self.kwargs['chi2accept'][:retval]

        # Assign some kwargs to self for summary printing.
        self.kwargs['retval'] = retval
        self.kwargs['n_obs'] = fit_obj.n_obs
        self.kwargs['n_param'] = fit_obj.n_p_real

        return fmin


def e_fit(parameters, h, ex, cfl_min, suppress_input=False, **kwargs):
    r"""
    Fit parameters to energy level data.

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of
    coefficients when they are set also determines whether they are fit as real
    or complex parameters.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels.
    ex : np.ndarray
        Either a 2 by n dimensional np.ndarray or an ExData type object.  In the
        former case, n is the number of energy levels, with the first column
        containing energy level indices starting at 1, and the second column
        containing the absolute experimental energy of the corresponding level.
        In order to specify energy level differences, or specify energies
        according to their SLJM state labels, use the ExData interface.
    cfl_min : CFLMin
        The minimization object which sets the optimization algorithm and
        corresponding options.
    suppress_input : bool, optional
        If True, omit the input file echo from the fit summary (default: False).
        Useful when running multiple fits to reduce verbose output.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    started_at = datetime.now()
    summary = "=============\n"
    summary+= "e_fit summary\n"
    summary+= "=============\n"
    summary += gen_pycf_summary(started_at, suppress_input=suppress_input)
    print_pycf_details(started_at)

    efit = EFit(parameters, h, ex, **kwargs)
    (x, fmin) = efit.fit(cfl_min)
    completed_at = datetime.now()
    summary += gen_completed_str(completed_at)
    print_completed_str(completed_at)

    h.update_coeff(x)
    (w, z) = h.diag()

    # Fix: ndof is the number of observables minus fitted real parameters.
    ndof = max(efit.n_obs - efit.n_p_real, 1)

    if efit.ex.n_d != 0:
        summary += h.gen_summary() + "\n\n"
        # Pass minimum_q and half_integer_states from Hamiltonian to gen_e_summary_trunc
        summary_kwargs = {"ex": efit.ex, "name": "Fitted energy levels", "chi2": efit.chi2[0], "ndof": ndof, "weighting": 1}
        if h.minimum_q is not None:
            summary_kwargs["minimum_q"] = h.minimum_q
            summary_kwargs["half_integer_states"] = h.half_integer_states
        summary += gen_e_summary_trunc(h.w, h.z, h.tensors[0].states.labels,
                h.tensors[0].states.label_key, **summary_kwargs)
    else:
        summary += h.gen_summary(ex=efit.ex, chi2=efit.chi2[0], ndof=ndof, weighting=1)

    summary += "\n"
    summary += gen_fit_summary(x, efit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary, **cfl_min.kwargs}



def mh_fit(parameters, h_list, weights_list, ex_list, cfl_min, suppress_input=False, **kwargs):
    r"""
    Fit to multiple Hamiltonians simultaneously.  Typically, this would consist
    of one vector of energy levels at zero field without hyperfine or quadrupole
    interactions, complemented by a set of eigenvalue vectors at linearly
    independent magnetic field orientations and possibly containing hyperfine
    interactions.  These additional eigenvalues can either be measured or
    synthetically calculated for specific crystal field levels from spin
    Hamiltonian data.

    The Hamiltonians must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of
    coefficients when they are set also determines whether they are fit as real
    or complex parameters, thus they must be consistent among each Hamiltonian.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h_list : list
        A list of Hamiltonians, each containing the interactions required to
        match the corresponding experimental energy level data.
    weights_list : list
        A list of floating point weights that determine the weighting added to
        the chi^2 contribution of each eigenvalue vector.
    ex_list : list
        Either a list of 2 by n dimensional np.ndarrays or a list of ExData type
        objects.  In the former case, n is the number of energy levels, with the
        first column of each array containing energy level indices starting at
        1, and the second column containing the absolute experimental energy of
        the corresponding level.  In order to specify energy level differences,
        or specify energies according to their SLJM state labels, use the ExData
        interface.
    suppress_input : bool, optional
        If True, omit the input file echo from the fit summary (default: False).
        Useful when running multiple fits to reduce verbose output.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    started_at = datetime.now()
    summary = "==============\n"
    summary+= "mh_fit summary\n"
    summary+= "==============\n"
    summary += gen_pycf_summary(started_at, suppress_input=suppress_input)
    print_pycf_details(started_at)

    mhfit = MHFit(parameters, h_list, weights_list, ex_list, **kwargs)
    (x, fmin) = mhfit.fit(cfl_min)
    completed_at = datetime.now()
    summary += gen_completed_str(completed_at)
    print_completed_str(completed_at)

    # Fix: ndof is the number of observables minus fitted real parameters.
    ndof = max(mhfit.n_obs - mhfit.n_p_real, 1)
    h = mhfit.h_list[0]
    h.update_coeff(x)
    (w, z) = h.diag()
    summary += h.gen_summary() + "\n\n"
    for i,h in enumerate(mhfit.h_list):
        h.update_coeff(x)
        (w, z) = h.diag()

        name = "Hamiltonian %i" % i
        # Pass minimum_q and half_integer_states from Hamiltonian to gen_e_summary_trunc
        summary_kwargs = {"ex": mhfit.ex_list[i], "name": name, "chi2": mhfit.chi2[i], "ndof": ndof, "weighting": mhfit.weights_list[i]}
        if h.minimum_q is not None:
            summary_kwargs["minimum_q"] = h.minimum_q
            summary_kwargs["half_integer_states"] = h.half_integer_states
        summary += gen_e_summary_trunc(h.w, h.z, h.tensors[0].states.labels, h.tensors[0].states.label_key, **summary_kwargs)

        summary += "\n"

    summary += gen_fit_summary(x, mhfit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary, **cfl_min.kwargs}


def esh_fit(parameters, h, sh, ex, shx, weights, cfl_min, suppress_input=False, **kwargs):
    r"""
    Fit parameters to energy level data and spin Hamiltonian data
    simultaneously.

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of
    coefficients when they are set also determines whether they are fit as real
    or complex parameters.


    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels.
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit.
    ex : np.ndarray
        Either a 2 by n dimensional np.ndarray or an ExData type object.  In the
        former case, n is the number of energy levels, with the first column
        containing energy level indices starting at 1, and the second column
        containing the absolute experimental energy of the corresponding level.
        In order to specify energy level differences, or specify energies
        according to their SLJM state labels, use the ExData interface.
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.  Any omitted values will be set
        to unity.
    cfl_min : CFLMin
        The minimization object which sets the optimization algorithm and
        corresponding options.
    suppress_input : bool, optional
        If True, omit the input file echo from the fit summary (default: False).
        Useful when running multiple fits to reduce verbose output.
    svd_sym : bool, optional
        Symmeterize spin Hamiltonian parameter tensors by applying an SVD
        transformation.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    started_at = datetime.now()
    summary = "===============\n"
    summary+= "esh_fit summary\n"
    summary+= "===============\n"
    summary += gen_pycf_summary(started_at, suppress_input=suppress_input)
    print_pycf_details(started_at)

    eshfit = ESHFit(parameters, h, sh, ex, shx, weights, **kwargs)
    (x, fmin) = eshfit.fit(cfl_min)
    completed_at = datetime.now()
    summary += gen_completed_str(completed_at)
    print_completed_str(completed_at)

    h.update_coeff(x)
    (w, z) = h.diag()

    # Fix: ndof is the number of observables minus fitted real parameters.
    ndof = max(eshfit.n_obs - eshfit.n_p_real, 1)

    if 'svd_sym' in kwargs:
        svd = kwargs['svd_sym']
    else:
        svd = False
    sh_param = sh.calc_param(h, svd_sym=svd)

    if eshfit.ex.n_d != 0:
        summary += h.gen_summary() + "\n\n"
        # Pass minimum_q and half_integer_states from Hamiltonian to gen_e_summary_trunc
        summary_kwargs = {"ex": eshfit.ex, "name": "Fitted energy levels", "chi2": eshfit.chi2[0], "ndof": ndof, "weighting": eshfit.weights['energy']}
        if h.minimum_q is not None:
            summary_kwargs["minimum_q"] = h.minimum_q
            summary_kwargs["half_integer_states"] = h.half_integer_states
        summary += gen_e_summary_trunc(h.w, h.z, h.tensors[0].states.labels,
                h.tensors[0].states.label_key, **summary_kwargs)
    else:
        summary += h.gen_summary(ex=eshfit.ex, chi2=eshfit.chi2[0], ndof=ndof,
                weighting=eshfit.weights['energy'])

    #summary += h.gen_summary(ex=eshfit.ex, chi2=eshfit.chi2[0], ndof=ndof,
    #        weighting=eshfit.weights['energy'])
    summary += "\n"
    summary += gen_sh_summary(sh_param, sh, shx=shx, chi2=eshfit.chi2[1:],
            ndof=ndof, weighting=eshfit.weights)
    summary += "\n"
    summary += gen_fit_summary(x, eshfit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary, **cfl_min.kwargs}


def mesh_fit(parameters, h_sh_list, cfl_min, suppress_input=False, **kwargs):
    r"""
    Fit multiple crystal-field Hamiltonians and spin Hamiltonians
    simultaneously.  For now, this is restricted to a single spin Hamiltonian
    per CF Hamiltonian.  Thus, one can fit one excited state spin Hamiltonian,
    excluding hyperfine, in conjunction with electronic energy level data.  This
    is can then combined with a hyperfine spin Hamiltonian for the ground state.

    The Hamiltonians must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of each
    coefficient when they are set also determines whether that coefficient is
    fit as real or complex parameter, thus they must be consistent among each
    Hamiltonian.

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor.
    h_sh_list : list
        Each element should be a dictionary with the following keys: 'h', 'sh',
        'ex', 'shx', 'weights', and svd_sym.  For descriptions of each element,
        see the ESHFit docstring.
    cfl_min : CFLMin
        The minimization object which sets the optimization algorithm and
        corresponding options.
    suppress_input : bool, optional
        If True, omit the input file echo from the fit summary (default: False).
        Useful when running multiple fits to reduce verbose output.
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    started_at = datetime.now()
    summary = "================\n"
    summary+= "mesh_fit summary\n"
    summary+= "================\n"
    summary += gen_pycf_summary(started_at, suppress_input=suppress_input)
    print_pycf_details(started_at)

    meshfit = MESHFit(parameters, h_sh_list, **kwargs)
    (x, fmin) = meshfit.fit(cfl_min)
    completed_at = datetime.now()
    summary += gen_completed_str(completed_at)
    print_completed_str(completed_at)

    h = meshfit.h_list[0]
    h.update_coeff(x)
    (w, z) = h.diag()

    # Fix: ndof is the number of observables minus fitted real parameters.
    ndof = max(meshfit.n_obs - meshfit.n_p_real, 1)

    summary += h.gen_summary()
    summary += "\n"

    chi2_offset = 0
    for i,h in enumerate(meshfit.h_list):
        h.update_coeff(x)
        (w, z) = h.diag()

        name = "Hamiltonian %i" % i
        # Pass minimum_q and half_integer_states from Hamiltonian to gen_e_summary_trunc
        summary_kwargs = {"ex": meshfit.ex_list[i], "name": name, "chi2": meshfit.chi2[chi2_offset], "ndof": ndof, "weighting": meshfit.weights_list[i]['energy']}
        if h.minimum_q is not None:
            summary_kwargs["minimum_q"] = h.minimum_q
            summary_kwargs["half_integer_states"] = h.half_integer_states
        summary += gen_e_summary_trunc(h.w, h.z, h.tensors[0].states.labels,
                h.tensors[0].states.label_key, **summary_kwargs)
        chi2_offset += 1
        summary += "\n"

        if 'svd_sym' in h_sh_list[i]:
            svd = h_sh_list[i]['svd_sym']
        else:
            svd = False
        name = "Spin Hamiltonian %i" % i
        sh_param = meshfit.sh_list[i].calc_param(h, svd_sym=svd)

        ni = len(meshfit.sh_list[i].interactions)   # The number of interactions for this sh.
        summary += gen_sh_summary(sh_param, meshfit.sh_list[i], shx=h_sh_list[i]['shx'], name = name,
                chi2=meshfit.chi2[chi2_offset:chi2_offset+ni], ndof=ndof, weighting=meshfit.weights_list[i])
        chi2_offset += ni
        summary += "\n"

    summary += gen_fit_summary(x, meshfit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary, **cfl_min.kwargs}


cdef class ZEFOZSearch:
    r"""
    Perform search for ZEFOZ points.

    Parameters
    ----------
    h : Hamiltonian
        The Hamiltonian for which to perform the ZEFOZ search.
    xtol : float
        If the total difference between the three field components of
        consecutive iterations is less than this value, then the field value is
        returned as a ZEFOZ point.
    init_size : int
        The initial size of the ZEFOZ point storage array.
    """
    cdef list zmatel_list
    cdef double complex **zmatel
    cdef cfl.zefoz_d *cfl_zd
    cdef cfl.zefoz_a *cfl_za
    cdef np.ndarray zi
    cdef float xtol
    def __cinit__(self, h, xtol, init_size):
        cdef np.ndarray[double complex, ndim=1, mode='c'] zm
        cdef np.ndarray[int, ndim=1, mode='c'] zi

        self.xtol = xtol

        zi = np.ascontiguousarray(np.zeros(3), dtype=np.int32)
        try:
            zi[0] = h.index('MX')
        except KeyError:
            raise KeyError("Missing MX tensor in Hamiltonian.")
        try:
            zi[1] = h.index('MY')
        except KeyError:
            raise KeyError("Missing MY tensor in Hamiltonian.")
        try:
            zi[2] = h.index('MZ')
        except KeyError:
            raise KeyError("Missing MZ tensor in Hamiltonian.")
        self.zi = zi

        self.zmatel = <double complex **>malloc(3*sizeof(double complex *))
        if self.zmatel == NULL:
            raise MemoryError("zmatel malloc failed")

        self.zmatel_list = []
        for i in range(3):
            # ZEFOZ Zeeman operators (MX, MY, MZ) are Hermitian. Hermitian-fill
            # the dense matrix here because cfl_zefoz.c::inprod uses
            # cblas_zgemv (CblasNoTrans) -- a full matrix-vector multiply that
            # reads both triangles. Tensor.get_matel() returns the upper
            # triangle only with the lower triangle zeroed (see
            # zhcsr2zha in cfl/src/cfl_csr.c), so without this completion the
            # off-diagonal contributions to the ZEFOZ gradient and curvature
            # would be silently dropped.
            m = h.tensors[zi[i]].get_matel()
            m = m + np.tril(m.conj().T, k=-1)
            zm = np.ascontiguousarray(m.reshape(h.n**2), dtype=np.complex128)
            self.zmatel[i] = &zm[0]
            self.zmatel_list += [zm]    # Keep reference to avoid GC cleanup.

        self.cfl_zd = cfl.zefoz_d_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"), &zi[0])
        if self.cfl_zd == NULL:
            free(self.zmatel)
            # NULL self.zmatel so __dealloc__ does not attempt a second free.
            self.zmatel = NULL
            raise MemoryError("zefoz_alloc failed")

        self.cfl_za = cfl.zefoz_a_alloc(init_size)
        if self.cfl_za == NULL:
            free(self.zmatel)
            cfl.zefoz_d_free(self.cfl_zd)
            # NULL both members so __dealloc__ does not double-free them.
            self.zmatel = NULL
            self.cfl_zd = NULL
            raise MemoryError("zefoz_a_alloc failed")

    def __dealloc__(self):
        if self.zmatel != NULL:
            free(self.zmatel)
        if self.cfl_zd != NULL:
            cfl.zefoz_d_free(self.cfl_zd)
        if self.cfl_za != NULL:
            cfl.zefoz_a_free(self.cfl_za)

    @cython.boundscheck(False)
    def run_search(self, Bx, By, Bz, k, l):
        """
        Run the ZEFOZ search.

        Parameters
        ----------
        Bx : np.ndarray
            Array of field strengths along x which to traverse.
        By : np.ndarray
            Array of field strengths along y which to traverse.
        Bz : np.ndarray
            Array of field strengths along z which to traverse.
        k : int
            Index of one of the two levels between which the ZEFOZ search is to
            be performed.
        l : int
            The index of the other level for the ZEFOZ search.
        """
        cdef np.ndarray[double, ndim=1, mode='c'] cBx
        cdef np.ndarray[double, ndim=1, mode='c'] cBy
        cdef np.ndarray[double, ndim=1, mode='c'] cBz
        cdef int nx
        cdef int ny
        cdef int nz
        cdef double complex **zmatel
        cdef cfl.zefoz_d *zd
        cdef cfl.zefoz_a *za
        cdef int ck
        cdef int cl
        cdef double xtol
        cdef np.ndarray[double, ndim=1, mode='c'] B
        cdef np.ndarray[double, ndim=1, mode='c'] v

        cBx = np.ascontiguousarray(Bx, dtype=np.float64)
        cBy = np.ascontiguousarray(By, dtype=np.float64)
        cBz = np.ascontiguousarray(Bz, dtype=np.float64)
        nx = len(Bx)
        ny = len(By)
        nz = len(Bz)

        ck = k
        cl = l
        xtol = self.xtol
        zmatel = self.zmatel
        zd = self.cfl_zd
        za = self.cfl_za

        with nogil:
            cfl.zefoz_search(&cBx[0], &cBy[0], &cBz[0], nx, ny, nz, ck, cl, xtol, zmatel, za, zd)

        n = za.ctr
        B = np.ascontiguousarray(np.zeros(3*n, dtype=np.float64))
        v = np.ascontiguousarray(np.zeros(3*n, dtype=np.float64))

        memcpy(&B[0], za.B, 3*n*sizeof(double));
        memcpy(&v[0], za.v, 3*n*sizeof(double));

        return (B, v)


def zefoz(start, stop, num, k, l, h, xtol=0.01, init_size=200):
    """
    Run the ZEFOZ search.

    Parameters
    ----------
    start : list
        List specifying the starting field values along x, y, and z.
    stop : list
        List specifying the stopping field values along x, y, and z.
    num : list
        List specifying the number of steps to take in the x, y, and z
        directions.
    k : int
        Index of one of the two levels between which the ZEFOZ search is to
        be performed.
    l : int
            The index of the other level for the ZEFOZ search.
    h : Hamiltonian
        The Hamiltonian for which to perform the ZEFOZ search.
    xtol : float, optional
        If the total difference between the three field components of
        consecutive iterations is less than this value, then the field value is
        returned as a ZEFOZ point.  Defaults to 0.01 Tesla.
    init_size : int, optional
        The initial size of the ZEFOZ point storage array.  Defaults to 200 and
        doubles in size whenever space runs out.  Perhaps set to some large
        number if there's a lot of expected ZEFOZ points.
    """

    zsearch = ZEFOZSearch(h, xtol, init_size)
    (B, v) = zsearch.run_search(start, stop, num, k, l)

    B=B.reshape(len(B)//3,3)
    v=v.reshape(len(v)//3,3)

    return (B, v)
