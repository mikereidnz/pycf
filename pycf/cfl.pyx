# filename = pycfl.pyx
#cython: c_string_encoding=ascii
#cython: embedsignature=True

#   Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
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
cimport cfl, cython
cimport numpy as np
import numpy as np
import sys
from numbers import Number
from cpython.pycapsule cimport *
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from matel import matel
from cfl_util import *

# TODO: 
#       + Add checks whether efit/eshfit data alloc functions return NULL and
#       corresponding frees.
#       + Python free bug if one does not provide the correct shx data dict
#       (change zeeman to something else). 


cdef class StateLabels:
    r"""
    State label type for tensors and spin Hamiltonians.  State labels are
    generally not entered manually but should be generated with
    :class:`import_sljm.ImportSLJM`.

    Paramters
    ---------
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
    cpdef public list labels
    cpdef public str label_key
    def __cinit__(self, label_key, labels):
        cdef size_t n
        cdef char *key
        cdef np.ndarray[int, ndim=1, mode='c'] clabels
        cdef int **l_a
       
        self.label_key = label_key
        self.labels = labels

        n = <size_t> len(labels)
        key = <char *> label_key
        l_a = <int **>malloc(len(labels)*sizeof(int *))
        if l_a == NULL:
            raise MemoryError("l_a malloc failed")

        nplabels = []
        for i,l in enumerate(labels):
            nplabels += [np.ascontiguousarray(np.array(labels[i], dtype=np.int32))]
            clabels = nplabels[i]
            l_a[i] = &clabels[0]
        
        # sl_alloc copies both the label key and the labels, so we do not need
        # to retain a reference after the alloc call.
        self.cfl_sl = cfl.sl_alloc(n, key, l_a)
        if self.cfl_sl == NULL:
            raise MemoryError("cfl_sl alloc failed")
        else:
            self.sl_cap = PyCapsule_New(<void *>self.cfl_sl, "pycfl.StateLabels", NULL)
        
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
    cpdef public str name
    cpdef public str arith_name
    cpdef public int n
    cdef public StateLabels states
    def __cinit__(self, char *name, np.ndarray[int, ndim=1, mode='c'] row_ptr, 
            np.ndarray[int, ndim=1, mode='c'] col_in, np.ndarray[double complex, ndim=1, mode='c'] val, 
            states, object data_tuple=None):
        cdef cfl.zt *t
        cdef cfl.zt *t1
        cdef cfl.zt *t2
        self.states = states

        if (data_tuple == None):
            self.name = name
            self.arith_name = None
            n = len(row_ptr)-1
            self.n = n
            t = cfl.zt_csr_alloc(name, n, &row_ptr[0], &col_in[0], &val[0], 
                    <cfl.sl *>PyCapsule_GetPointer(states.sl_cap, "pycfl.StateLabels"))
            
        elif (len(data_tuple)==3):
            # Addition or subtraction of tensors; we use the arithmetic name for
            # zt_sa.
            self.name = None
            self.arith_name = name
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t2 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[1].t_cap, "pycfl.Tensor")
            t = cfl.zt_sa(<char *>self.arith_name, t1, t2, 1, data_tuple[2])

        else:
            # Scaling of a tensor; we use the arithmetic name for zt_sa.
            self.name = None
            self.arith_name = name
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t = cfl.zt_s(<char *>self.arith_name, t1, <double complex> data_tuple[1])

        if t is NULL:
            self.t_cap = None
            raise MemoryError("Cannot alloc zt memory")
        else:
            self.t_cap = PyCapsule_New(<void *>t, "pycfl.Tensor", NULL)

    def __dealloc__(self):
        if self.t_cap is not None:
            cfl.zt_free(<cfl.zt *>PyCapsule_GetPointer(self.t_cap, "pycfl.Tensor"))
    
    def __add__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if t1.name == None:
            t1name = t1.arith_name
        else:
            t1name = t1.name
        if t2.name == None:
            t2name = t2.arith_name
        else:
            t2name = t2.name
        tmp_name = "{0}+{1}".format(t1name, t2name)
        d = (t1, t2, 1)
        return Tensor(<char *>tmp_name, None, None, None, t1.states, data_tuple=d) 

    def __sub__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if t1.name == None:
            t1name = t1.arith_name
        else:
            t1name = t1.name
        if t2.name == None:
            t2name = t2.arith_name
        else:
            t2name = t2.name
        tmp_name = "{0}-{1}".format(t1name, t2name)
        d = (t1, t2, -1)
        return Tensor(<char *>tmp_name, None, None, None, t1.states, data_tuple=d) 

    def __mul__(x, y):
        # We check whether name has been explicitly set, otherwise use
        # arithmetic name.
        if isinstance(x, Number):
            if isinstance(y, Tensor):
                if y.name == None:
                    yname = y.arith_name
                else:
                    yname = y.name
                tmp_name = "{0:.2f}*{1}".format(x, yname)
                d = (y, x)
                return Tensor(<char *>tmp_name, None, None, None, y.states, data_tuple=d)
        elif isinstance(x, Tensor):
            if isinstance(y, Number):
                if x.name == None:
                    xname = x.arith_name
                else:
                    xname = x.name
                tmp_name = "{0:.2f}x{1}".format(y, xname)
                d = (x, y)
                return Tensor(<char *>tmp_name, None, None, None, x.states, data_tuple=d)
        else:
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
        if self.name != None:
            return self.name
        else:
            name = [k for k,v in sys.modules['__main__'].__dict__.iteritems() if v is self]
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


cdef class Hamiltonian:
    r"""
    The crystal field Hamiltonian class.  Creates a cfl zh object and provides
    an interface for diagonalizing zh.  Can be used to calculate:

        * energy levels given a list of :class:`Tensor`s and corresponding coefficients;
        * spin Hamiltonian parameters from crystal field parameters;
        * crystal field parameters by fitting to either energy levels or both
        energy levels and spin Hamiltonian parameters.

    A summary of calculated energy levels can be generated with
    :func:`cfl_util.gen_e_summary`.

    Hamiltonians are iterable, returning the Tensor objects from which it is composed.

    Parameters
    ----------
    tensors : list
        A list with components of type Tensor; this specifies the type of
        interactions modeled by the Hamiltonian.

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
    cdef int diag_run
    def __cinit__(self, tensors):

        n = tensors[0].n
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
        self.coeff_dict = None
        self.diag_run = 0
                
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
        self.coeff_dict = coeff

        self.coeff = np.array([], dtype=np.complex128)
        for t in self.tensors:
            try:
                self.coeff = np.append(self.coeff, coeff[t.get_name()])
            except KeyError:
                raise KeyError("Missing coefficient for tensor: %s" % t.get_name())
        
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
        
        self.w = np.ascontiguousarray(np.zeros(self.n), dtype=np.float64)
        self.z = np.asfortranarray(np.zeros(self.n*self.n).reshape((self.n,self.n)), dtype=np.complex128)
        w = <np.ndarray[double, ndim=1, mode="c"]> self.w
        z = <np.ndarray[double complex, ndim=2, mode="fortran"]> self.z

        if self.coeff_dict == None:
            raise ValueError("Hamiltonian must have coefficients set prior to diagonalization.")
        hd_w = cfl.zhd_w_alloc('V', self.cfl_zh)
        if hd_w is NULL:
            free(self.tensor_array)
            cfl.zh_free(self.cfl_zh)
            raise MemoryError("hd_w alloc failed")
        
        with nogil:
            cfl.zhd('V', &w[0], &z[0,0], h, hd_w)
        
        cfl.zhd_w_free(hd_w)
        self.diag_run = 1

        return (w, z)

    cpdef gen_summary(self, ex=None, nstates=2, sigma=None, e_shift=False):
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
        sigma : float, optional
            The standard deviation for the energy level chi^2.
        e_shift : bool, optional
            Shift entire eigenvalue spectrum s.t. the first eigenvalue is zero. 
        """
        if self.diag_run:
            return gen_e_summary(self.w, self.z, self.tensors[0].states.labels,
                    self.tensors[0].states.label_key, ex, nstates, sigma, e_shift)
        else:
            raise ValueError("Hamiltonian must have run diag prior to summary generation.")


cpdef zeeman_sh_coeff(v, t):
    r"""
    Generate the Zeeman interaction spin Hamiltonian 'coefficient array'.  This
    consists of a `2j+1 \times 2j+1` by `3 \times 3` array containing the matrix
    elements of the terms `B_a S_b`, with `a,b \in \{x, y, z\}` and `j` the
    angular momentum of the rank one tensor `S`.  Here the rows enumerate the
    `2j+1 \times 2j+1` different state combinations while the columns enumerate
    all combinations of `a` and `b`.

    Parameters
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    t : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j+1 \times 2j+1` by `3 \times 3` array.
    """

    tl = len(t[0])
    l = len(t)
    a = np.zeros([tl, tl, l, l], dtype = np.complex)

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

    a = np.zeros([t1l * t2l, t1l * t2l, l, l], dtype = np.complex)

    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(l):
                        for j in range(l):
                            a[t1r+t1l*t2r, t1c+t1l*t2c, i, j]= t1[i][t1r, t1c] * t2[j][t2r, t2c]

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
    a = np.zeros([tl, tl, l, l], dtype = np.complex)

    for tr in range(tl):
        for tc in range(tl):
            for i in range(l):
                for j in range(l):
                    components = 0
                    for ci in range(tl):
                        components +=t[i][tr, ci] * t[j][ci, tc]
                    a[tr, tc, i, j] = components

    return(np.reshape(a, (tl*tl, l*l)))


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
        Hamiltonian.
    S : float
        The spin projection `S_z`; if ``interactions`` contains 'zeeman' or
        'hyperfine' this keyword argument must be specified.
    I : float
        The nuclear spin projection `I_z`; if ``interactions`` contains
        'hyperfine' or 'quadrupole' this keyword argument must be specified.

    Returns
    -------
    object : SpinHamiltonian
    """
    cdef cfl.zsh *cfl_zsh
    cdef public list interactions 
    cdef public list required_tensors
    cpdef public int level
    cdef public int nsh
    cdef public int nobs
    cpdef public float Sz
    cpdef public list S_matel
    cpdef public float Iz
    cpdef public list I_matel
    cdef list inv_data
    cdef double complex **inv_data_ptrs
    cdef char **inter_array
    cdef public object sh_cap
    cdef public list tensors
    cdef public int pro_data_set
    cdef public dict coeff_dict
    cdef float dz
    cdef float dh
    cdef float dq

    def __init__(self, interactions, **kwargs):
        cdef int csz
        cdef int ciz
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] a

        if not isinstance(interactions, list):
            interactions = [interactions]
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list: '{}'.".format(i))
        self.interactions = interactions
        if 'S' in kwargs:
            if not any((i in interactions for i in ['zeeman', 'hyperfine'])):
                raise ValueError("SpinHamiltonian: the S_z spin projection was specified yet the "\
                        "interactions list contains neither the zeeman key nor the hyperfine key.")
        if 'I' in kwargs:
            if not any((i in interactions for i in ['hyperfine', 'quadrupole'])):
                raise ValueError("SpinHamiltonian: the I_z spin projection was specified yet the "\
                "interactions list contains neither the hyperfine key nor the quadrupole key.")

        if 'level' not in kwargs:
            raise KeyError("SpinHamiltonian: missing keyword argument 'level'.")
        self.level = kwargs['level']
           
        # Calculate matrix elements for the specified interactions.
        j_l = ['jx', 'jy', 'jz']
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
        
        # Calculate the coefficient arrays and alloc spin Hamiltonian.
        n_inter = len(interactions)
        if 'zeeman' not in interactions:
            # One unspecified interaction, magzs. 
            n_inter += 1

        self.inter_array = <char **>malloc(n_inter*sizeof(char *))
        if self.inter_array == NULL:
            raise MemoryError("inter_array malloc failed")

        self.inv_data_ptrs = <double complex **>malloc(len(interactions)*sizeof(double complex *))
        if self.inv_data_ptrs == NULL:
            raise MemoryError("inv_data_ptrs malloc failed")
        
        self.nsh = 0
        self.nobs = 0
        self.required_tensors = []
        self.inv_data = []
        for i,inter in enumerate(interactions):
            if inter == 'zeeman':
                # Coefficient arrays are calculated for three B fields in \hat{x},
                # \hat{y}, and \hat{z} directions, respectively.
                self.dz = 2*self.Sz+1
                B_a = np.zeros([3, self.dz**2, 9], dtype = np.complex)
                for j in range(3):
                    B_a[j, :, :] = zeeman_sh_coeff(np.eye(3,3)[j,:], self.S_matel)
                self.inv_data += [np.asfortranarray(np.reshape(B_a, (3 * self.dz**2, 9)), dtype=np.complex128)]
                self.nsh += 3
                # Three g-values plus three Euler rotation parameters.
                self.nobs += 6
                self.required_tensors += ['MAGX', 'MAGY', 'MAGZ']

            if inter == 'hyperfine':
                self.dh = (2*self.Sz+1) * (2*self.Iz+1)
                # The ordering of S_matel and I_matel is such that the states
                # are first ordered by Iz, and then order by blocks of 2 by 2 Sz
                # eigenvalues.
                self.inv_data += [np.asfortranarray(hyperfine_sh_coeff(self.S_matel, self.I_matel), 
                    dtype=np.complex128)]
                self.nsh += 1
                # Three hyperfine values plus three Euler rotation parameters.
                self.nobs += 6
                self.required_tensors += ['HYP']

            if inter == 'quadrupole': 
                self.dq = 2*self.Iz+1
                self.inv_data += [np.asfortranarray(quadrupole_sh_coeff(self.I_matel), dtype=np.complex128)]
                self.nsh += 1
                # Two quadrupole values plus three Euler rotation parameters.
                self.nobs += 5
                self.required_tensors += ['EQHYP']

            a = <np.ndarray[double complex, ndim=2, mode='fortran']> self.inv_data[i]
            self.inv_data_ptrs[i] = &a[0,0]
            self.inter_array[i] = inter
        
        # Add magzs to interactions if no Zeeman interaction is specified.
        if 'zeeman' not in interactions:
            self.inter_array[n_inter-1] = 'magzs'
            self.required_tensors += ['MAGZ']
        
        csz = int(2*self.Sz)
        ciz = int(2*self.Iz)
        self.cfl_zsh = cfl.zsh_alloc(self.inter_array, len(interactions), csz, ciz, self.inv_data_ptrs);
        if self.cfl_zsh == NULL:
            raise MemoryError("Failed to alloc zsh")
        else:
            self.sh_cap = PyCapsule_New(<void *>self.cfl_zsh, "pycfl.SpinHamiltonian", NULL)

        self.tensors = None
        self.pro_data_set = 0

    def __dealloc__(self):
        if self.cfl_zsh != NULL:
            cfl.zsh_free(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"))
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
                raise ValueError("SpinHamiltonian: he tensor {} is not an "\
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
            SpinHamiltonian does not describe Zeeman interactions the 'MAGZ'
            tensor must be provided for state-label sorting. 
        coupling_constants : dict, optional
            If hyperfine or quadrupole interactions are present, this dictionary
            has to be provided, which specifies the nuclear dipole and nuclear
            quadrupole coupling constants, using keys 'HYP' and 'QUAD',
            respectively.
        """
        cdef cfl.zt **t_array
        cdef np.ndarray[double, ndim=1, mode="c"] cc
        if not isinstance(tensors, list):
            raise TypeError("The tensors argument of set_pro_data must be a list "\
                    "of Tensor objects, not an object of type %s." % type(tensors))
        if not all((isinstance(t, Tensor) for t in tensors)):
            raise TypeError("The tensors argument of set_pro_data must be a list of Tensor objects")

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
            elif rt == 'QUAD':
                try:
                    cc_list += [coupling_constants['QUAD']]
                except KeyError:
                    free(t_array)
                    raise KeyError("Missing the nuclear quadrupole coupling constant in set_pro_data call.")
            else:
                # Default to unity for Zeeman/magz. 
                cc_list += [1.0]
       
        self.coeff_dict = coupling_constants
        cc = np.array(cc_list, dtype=np.float64)
        retval = zsh_set_pro(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"), 
                t_array, self.level, &cc[0])
        
        free(t_array)
        if retval != 1:
            raise ValueError("zsh_set_pro failed.  See the cfl error message for details.")

        self.tensors = tensors
        self.pro_data_set = 1


    def calc_param(self, h, matel=False):
        r"""
        Calculate the spin Hamiltonian parameters given a crystal-field
        Hamiltonian.

        Parameters
        ----------
        h : Hamiltonian
            The corresponding crystal-field Hamiltonian. 
        matel : bool
            If true, a dictionary containing the spin Hamiltonian matrix
            elements is returned.

        Returns
        -------
        res : tuple
            If matel=False, a list of nd.arrays is returned.  These correspond
            to the spin Hamiltonian tensors of interactions specified when the
            spin Hamiltonian object was instantiated. 
        """

        cdef cfl.zshp_w *shp_w
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] cz
        cdef np.ndarray[double complex, ndim=1, mode="c"] a
        cdef np.ndarray[double complex, ndim=1, mode="c"] b

        if not self.pro_data_set:
            raise ValueError("The spin Hamiltonian interaction is missing projection data.")


        # If not present, add small magnetic field to Hamiltonian to order
        # states.
        if 'MAGZS' not in h.coeff_dict:
            for t in self.tensors:
                if t.get_name() == 'MAGZ':
                    magzs = 0.0001 * t
                    magzs.name = 'MAGZS'
            
            tmp_h_coeff = h.coeff_dict
            tmp_h_coeff['MAGZS'] = 1
            h = Hamiltonian([magzs] + h.tensors)
            h.set_coeff(tmp_h_coeff)


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
            h = Hamiltonian(pro_h_tensors)
            h.set_coeff(tmp_coeff)

        (w, z) = h.diag()
        cz = <np.ndarray[double complex, ndim=2, mode="fortran"]> z
        shp_w = zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"))
        a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(9, dtype=np.complex128)
       
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

            b = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(d_inter, dtype=np.complex128)

            zshp(&a[0], &b[0], &cz[0,0], i, <cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"), shp_w)
            result_list += [np.copy(a.reshape(3,3))]
            
            if inter == 'zeeman':
                sh_matel['magx'] = b[0:4].reshape(2,2)
                sh_matel['magy'] = b[4:8].reshape(2,2)
                sh_matel['magz'] = b[8:12].reshape(2,2)
            elif inter == 'hyperfine':
                sh_matel['hyperfine'] = b.reshape(self.dh, self.dh)
            elif inter == 'quadrupole':
                sh_matel['quadrupole'] = b.reshape(self.dq, self.dq)
        
        if matel:
            return ((result_list, sh_matel))
        else:
            return result_list


cdef parse_ex(ex):
    r"""
    Helper function for parsing energy level data. 
    """
    if not isinstance(ex, np.ndarray):
        raise TypeError("ex data must be of type np.ndarray, not %s." % type(ex))
    elif not (ex.shape[1] == 2 or ex.shape[1] == 3):
        raise ValueError("Incorrect ex shape; expected a two, or three, column array.")
    elif len(ex[:, 0]) != len(set(ex[:, 0])):
        raise ValueError("ex input data contains duplicate entries in the index column.")
    
    if ex.shape[1] == 2:
        # Two dimensional; no energy level differences. 
        n_a = ex.shape[0]
        n_d = 0
        # Subtract one, since we need an index starting at zero, whereas ex
        # levels start at 1. 
        ex_e = np.ascontiguousarray(ex[:, 1], dtype=np.float64)
        ex_la = np.ascontiguousarray(ex[:, 0]-1, dtype=np.int32)
        ex_ild = None
        ex_fld = None
    elif ex.shape[1] == 3:
        # Index of absolute energy levels. 
        ex_a_i = np.where(ex[:, 1] <= -1)[0]
        # Index of difference energy levels. 
        ex_d_i = np.where(ex[:, 1] > -1)[0]

        ex_e = np.ascontiguousarray(np.hstack((ex[ex_a_i, 2], ex[ex_d_i, 2])), dtype=np.float64)
        ex_la = np.ascontiguousarray(ex[ex_a_i, 0]-1, dtype=np.int32)
        ex_ild = np.ascontiguousarray(ex[ex_d_i, 0]-1, dtype=np.int32)
        ex_fld = np.ascontiguousarray(ex[ex_d_i, 1]-1, dtype=np.int32)
        n_d = len(ex_d_i)
        n_a = ex.shape[0] - n_d

    n_obs = ex.shape[0]

    return {'n_obs': n_obs, 'n_a': n_a, 'n_d': n_d, 'ex_e': ex_e, 'ex_la': ex_la, 
            'ex_ild': ex_ild, 'ex_fld': ex_fld}


cdef class ExData(object):
    r"""
    Experimental energy level data for Hamiltonians. 

    Parameters
    ----------
    data : np.ndarray or tuple
        If data is of type np.ndarray it assumed to be a 2 by n dimensional,
        with the first column containing energy level indices starting at 1, and
        the second column containing the absolute experimental energy of the
        corresponding level.  If data is of type tuple each element must be a
        np.ndarray of type specified using the key argument.  Implemented types
        are::
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
        argument is used to specify the type of data. Available keys are::
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
    """
    cpdef public int sl_index
    cpdef public int n_obs
    cpdef public int n_a
    cpdef public int n_d
    cpdef public np.ndarray a_states
    cpdef public np.ndarray id_states
    cpdef public np.ndarray fd_states
    cpdef public np.ndarray e
    cpdef public np.ndarray la
    cpdef public np.ndarray ild
    cpdef public np.ndarray fld
    cpdef public np.ndarray lah
    cpdef public np.ndarray ildh
    cpdef public np.ndarray fldh
    def __init__(self, data, key=None, label_key=None):
        cdef np.ndarray[int, ndim=1, mode='c'] clabels

        if not (isinstance(data, np.ndarray) or isinstance(data, tuple)):
            raise TypeError("The ex data argument must either be of type np.ndarray or tuple, not %s." % type(data))
        if not (isinstance(key, str) or isinstance(key, tuple) or key==None):
            raise TypeError("The key argument must either be of type np.ndarray or tuple, not %s." % type(key))
        if isinstance(data, tuple):
            if key == None:
                raise ValueError("Missing key argument; this must be specified if data is a tuple.")
            elif len(data) != 2:
                raise ValueError("The data argument can at most contain two elements; " \
                        "one absolute energy array and one difference energy array.")
            elif not isinstance(key, tuple):
                raise TypeError("If the data argument is a tuple with two " \
                        "elements the key argument must also be a tuple with two elements.")
            elif len(key) != 2:
                raise ValueError("The key tuple must be the same length as the data tuple.")
            elif not all(isinstance(d, np.ndarray) for d in data):
                raise TypeError("Elements of the data tuple must be of type np.ndarray.")
            elif not all(isinstance(k, str) for k in key):
                raise TypeError("Elements of the key tuple must be of type str.")
        
        if key != None:
            if not isinstance(key, tuple):
                # A single key is provided; we therefore make both data and key
                # iterable for key checks below, and such that key check is
                # forced in case of state labels (type(data) != np.ndarray).
                key = [key]
                data = [data]
            
            if any(d.ndim != 2 for d in data):
                raise ValueError("All data arrays must be two dimensional.")

            for k in key:
                if k == 'A' or k == 'D':
                    if 'AS' in key or 'DS' in key:
                        raise ValueError("Mixed index and state level indices are " \
                                "not supported; use either A and D, or AS and DS.")
                    self.sl_index = 0
                else:
                    if 'A' in key or 'D' in key:
                        raise ValueError("Mixed index and state level indices are " \
                                "not supported; use either A and D, or AS and DS.")
                    elif type(label_key) != str:
                        raise TypeError("The label_key argument must be of type str and is " \
                                "mandatory for state label indices.")
                    self.sl_index = 1

                if k not in ['A', 'D', 'AS', 'DS']:
                    raise ValueError("Invalid key argument; allowed options are 'A', 'D', 'AS', and 'DS'.")

        if isinstance(data, np.ndarray):
            if not data.shape[1] == 2:
                raise ValueError("Incorrect ex data shape; expected a two column array.")
            # No energy level differences.
            self.n_a = data.shape[0]
            self.n_d = 0
            # Subtract one, since we need an index starting at zero, whereas ex
            # levels start at 1. 
            self.e = np.ascontiguousarray(data[:, 1], dtype=np.float64)
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
                        lah[i] = cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))

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
                        ildh[i] = cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))
                        clabels = np.ascontiguousarray(self.fd_states[i, :], dtype=np.int32)
                        fldh[i] = cfl.fnv_hash(&clabels[0], ll*sizeof(int)/sizeof(char))

                    self.ild = np.zeros(self.n_d, dtype=np.int32)
                    self.fld = np.zeros(self.n_d, dtype=np.int32)
                    self.ildh = np.ascontiguousarray(ildh, dtype=np.int32)
                    self.fldh = np.ascontiguousarray(fldh, dtype=np.int32)
                    
                if len(key) == 2:
                    self.e = np.ascontiguousarray(np.hstack((data[key.index('AS')][:, ll],
                        data[key.index('DS')][:, 2*ll])), dtype=np.float64)
                elif key[0] == 'AS':
                    self.e = np.ascontiguousarray(data[key.index('AS')][:, ll], dtype=np.float64)
                    self.n_d = 0
                elif key[0] == 'DS':
                    self.e = np.ascontiguousarray(data[key.index('DS')][:, 2*ll], dtype=np.float64)
                    self.n_a = 0
                    self.la = np.zeros(0)
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
                    self.e = np.ascontiguousarray(np.hstack((data[key.index('A')][:, 1],
                        data[key.index('D')][:, 2])), dtype=np.float64)
                elif key[0] == 'A':
                    self.e = np.ascontiguousarray(data[key.index('A')][:, 1], dtype=np.float64)
                    self.n_d = 0
                elif key[0] == 'D':
                    self.e = np.ascontiguousarray(data[key.index('D')][:, 2], dtype=np.float64)
                    self.n_a = 0
                    self.la = np.zeros(0)
        
        self.n_obs = self.n_a + self.n_d


cdef class EFitRunner(object):
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
    cpdef public int n_p_real
    cpdef public int n_obs
    cpdef public dict param_types
    cdef cfl.ex_data *ex_data
    cdef public ExData ex
    cdef cfl.param_type **param_array
    cdef np.ndarray p0_real
    cdef cfl.efit_data *efit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    def __init__(self, parameters, h, ex, **kwargs):
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_la
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ild
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fld
        cdef np.ndarray[int, ndim=1, mode="c"] ex_lah
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ildh
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fldh
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x

        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters
        
        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")
      
        # Create a local copy of coefficients for each parameter and an array of
        # arrays specifying the parameters specific to each Hamiltonian. 
        if h.coeff_dict == None:
            raise ValueError("Hamiltonian must have coefficients set prior to efit.")
        else:
            self.coeff = h.coeff_dict

        # Determine the type of each parameter. 
        self.param_types = {}
        self.n_p_real = 0
        for p in parameters:
            if p not in h:
                raise ValueError("Parameter %s not found in the Hamiltonian." % p)
            if not isinstance(self.coeff[p], Number):
                raise ValueError("The coefficient %s was not specified as a number the Hamiltonian." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                self.n_p_real += 2
                self.param_types[p] = "c"
            else:
                self.param_types[p] = "r"
                self.n_p_real += 1
        
        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False
        
        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        
        # Parse the energy level data, if required.
        if not isinstance(ex, ExData):
            self.ex = ExData(ex)
        else:
            self.ex = ex
        
        self.n_obs = self.ex.n_obs
        
        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of parameters, %i, exceeds "
                    "the number of observables, %i.  If you must nevertheless proceed, you can do "
                    "so at your own peril by setting the kwarg ignore_ndof=True." % (self.n_p_real, len(ex)))
        
        self.ex_data = <cfl.ex_data *>malloc(sizeof(cfl.ex_data))
        if self.ex_data == NULL:
            raise MemoryError("ex_data alloc failed")

        self.ex_data.n_obs = self.ex.n_obs
        self.ex_data.n_a = self.ex.n_a
        self.ex_data.n_d = self.ex.n_d
        ex_e = <np.ndarray[double, ndim=1, mode="c"]> self.ex.e
        self.ex_data.e = &ex_e[0]

        if self.ex.n_a:
            ex_la = <np.ndarray[int, ndim=1, mode="c"]> self.ex.la
            self.ex_data.la = &ex_la[0]
        else:
            # There are no absolute energy level observables.
            self.ex_data.la = NULL

        if self.ex.n_d:
            ex_ild = <np.ndarray[int, ndim=1, mode="c"]> self.ex.ild
            ex_fld = <np.ndarray[int, ndim=1, mode="c"]> self.ex.fld
            self.ex_data.ild = &ex_ild[0]
            self.ex_data.fld = &ex_fld[0]
        else:
            # There are no energy level difference observables.
            self.ex_data.ild = NULL
            self.ex_data.fld = NULL

        if self.ex.sl_index:
            if self.ex.n_a:
                ex_lah = <np.ndarray[int, ndim=1, mode="c"]> self.ex.lah
                self.ex_data.lah = &ex_lah[0]
            else:
                self.ex_data.lah = NULL

            if self.ex.n_d:
                ex_ildh = <np.ndarray[int, ndim=1, mode="c"]> self.ex.ildh
                ex_fldh = <np.ndarray[int, ndim=1, mode="c"]> self.ex.fldh
                self.ex_data.ildh = &ex_ildh[0]
                self.ex_data.fldh = &ex_fldh[0]
            else:
                self.ex_data.ildh = NULL
                self.ex_data.fldh = NULL
        else:
            self.ex_data.lah = NULL
            self.ex_data.ildh = NULL
            self.ex_data.fldh = NULL

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_array = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
        if param_array == NULL:
            free(self.ex_data)
            raise MemoryError("param_array alloc failed")
        
        ip_real = 0
        for i,p in enumerate(parameters):
            param_array[i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))
            
            param_array[i].type = ord(self.param_types[p])
            param_array[i].index = h.index(p)

            if self.param_types[p] == 'c':
                self.p0_real[ip_real] = np.real(self.coeff[p])
                self.p0_real[ip_real+1] = np.imag(self.coeff[p])
                ip_real += 2
            else:
                self.p0_real[ip_real] = self.coeff[p]
                ip_real += 1

        self.param_array = param_array 
        if self.ex.sl_index:
            self.efit_data = cfl.efit_data_alloc('S', <cfl.zh *>PyCapsule_GetPointer(
                h.h_cap, "pycfl.Hamiltonian"), self.ex_data, self.n_p, self.param_array);
        else:
            self.efit_data = cfl.efit_data_alloc('N', <cfl.zh *>PyCapsule_GetPointer(
                h.h_cap, "pycfl.Hamiltonian"), self.ex_data, self.n_p, self.param_array);
        self.fit_data_cap = PyCapsule_New(<void *>self.efit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.efit_obj, "pycfl.MinObjF", NULL)
        self.cov_f_cap = PyCapsule_New(<void *>&cfl.efit_cov, "pycfl.MinCovF", NULL)
        
        # Run efit_chi2 so that the initial chi^2 weighting is set.
        chi2 = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(1)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        cfl.efit_chi2(&x[0], self.efit_data, &chi2[0])

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
        cdef sigma = 0

        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real

        fmin = min_object.minimize(self, x)
        
        coeff = self.coeff.copy()
        ri = 0
        
        for p in self:
            if (self.param_types[p] == 'c'): 
                coeff[p] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p] = x[ri]
                ri += 1
        
        return(coeff, fmin)


cdef class MHFitRunner(object):
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
    cpdef public list h_list
    cdef int n_p
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public int n_obs
    cpdef public dict param_types
    cdef list h_param_list
    cdef cfl.zh **ha
    cdef np.ndarray weights
    cdef cfl.ex_data **ex_data
    cdef list ex_list
    cdef np.ndarray n_zx
    cdef cfl.param_type ***param_arrays
    cdef np.ndarray p0_real
    cdef cfl.mhfit_data *mhfit_data
    cdef np.ndarray job_a
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    def __init__(self, parameters, h_list, weights_list, ex_list, **kwargs):
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_la
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ild
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fld
        cdef np.ndarray[int, ndim=1, mode="c"] ex_lah
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ildh
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fldh
        cdef np.ndarray[double, ndim=1, mode="c"] weights
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        cdef np.ndarray[int, ndim=1, mode="c"] n_zx
        cdef np.ndarray[char, ndim=1, mode="c"] job_a

        self.n_h = len(h_list)
        self.n_p = len(parameters)
        self.h_list = h_list
        self.parameters = parameters
        
        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")
      
        # Create a local copy of coefficients for each parameter and an array of
        # arrays specifying the parameters specific to each Hamiltonian. 
        self.coeff = {}
        h_param_list = []
        self.n_zx = np.empty(self.n_h, dtype=np.int32)
        for i,h in enumerate(h_list):
            if h.coeff_dict == None:
                raise ValueError("Hamiltonian must have coefficients set prior to mhfit.")
            else:
                self.coeff.update(h.coeff_dict)
            h_param_list += [[p for p in parameters if p in h]]
            self.n_zx[i] = len(h_param_list[i])
        self.h_param_list = h_param_list

        # The number of complex valued parameters for each Hamiltonian
        n_zx = <np.ndarray[int, ndim=1, mode="c"]> self.n_zx

        # Determine the type of each parameter. 
        self.param_types = {}
        self.n_p_real = 0
        for p in parameters:
            if all((p not in h for h in h_list)):
                raise ValueError("Parameter %s not found in any Hamiltonian." % p)
            if not isinstance(self.coeff[p], Number):
                raise ValueError("The coefficient %s was not specfied as a "\
                        "number in one of the Hamiltonians." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                self.n_p_real += 2
                self.param_types[p] = "c"
            else:
                self.param_types[p] = "r"
                self.n_p_real += 1

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.

        # Parse the energy level data. 
        self.n_obs = 0
        self.ex_list = []
        if (len(ex_list) != self.n_h):
            raise ValueError("The number of Hamiltonians does not match the number of elements in ex_list.")
        for i,ex in enumerate(ex_list):
            if not isinstance(ex, ExData):
                self.ex_list += [ExData(ex)]
            else:
                self.ex_list += [ex]
            self.n_obs += self.ex_list[i].n_obs

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False
        
        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of parameters, %i, exceeds "
                    "the number of observables, %i.  If you must nevertheless proceed, you can do "
                    "so at your own peril by setting the kwarg ignore_ndof=True." % (self.n_p_real, len(ex)))

        self.ha = <cfl.zh **>malloc(self.n_h*sizeof(cfl.zh *))
        if self.ha == NULL:
            raise MemoryError("ha alloc failed")

        for i in range(self.n_h):
            self.ha[i] = <cfl.zh *>PyCapsule_GetPointer(h_list[i].h_cap, "pycfl.Hamiltonian")

        self.ex_data = <cfl.ex_data **>malloc(self.n_h*sizeof(cfl.ex_data *))
        if self.ex_data == NULL:
            free(self.ha)
            raise MemoryError("exa alloc failed")
        
        for i in range(self.n_h):
            self.ex_data[i] = <cfl.ex_data *>malloc(sizeof(cfl.ex_data))
            if self.ex_data[i] == NULL:
                for j in range(i):
                    free(self.ex_data[j])
                free(self.ex_data)
                free(self.ha)
                raise MemoryError("ex_data alloc failed")
            
            self.ex_data[i].n_obs = self.ex_list[i].n_obs
            self.ex_data[i].n_a = self.ex_list[i].n_a
            self.ex_data[i].n_d = self.ex_list[i].n_d

            ex_e = <np.ndarray[double, ndim=1, mode="c"]> self.ex_list[i].e
            self.ex_data[i].e = &ex_e[0]
            if self.ex_list[i].n_a:
                ex_la = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].la
                self.ex_data[i].la = &ex_la[0]
            else:
                # There are no absolute energy level observables.
                self.ex_data[i].la = NULL

            if self.ex_list[i].n_d:
                ex_ild = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].ild
                ex_fld = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].fld
                self.ex_data[i].ild = &ex_ild[0]
                self.ex_data[i].fld = &ex_fld[0]
            else:
                # There are no energy level difference observables.
                self.ex_data[i].ild = NULL
                self.ex_data[i].fld = NULL

            if self.ex_list[i].sl_index:
                if self.ex_list[i].n_a:
                    ex_lah = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].lah
                    self.ex_data[i].lah = &ex_lah[0]
                else:
                    self.ex_data[i].lah = NULL

                if self.ex_list[i].n_d:
                    ex_ildh = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].ildh
                    ex_fldh = <np.ndarray[int, ndim=1, mode="c"]> self.ex_list[i].fldh
                    self.ex_data[i].ildh = &ex_ildh[0]
                    self.ex_data[i].fldh = &ex_fldh[0]
                else:
                    self.ex_data[i].ildh = NULL
                    self.ex_data[i].fldh = NULL
            else:
                self.ex_data[i].lah = NULL
                self.ex_data[i].ildh = NULL
                self.ex_data[i].fldh = NULL

        self.weights = np.array(weights_list, dtype=np.float64)
        weights = <np.ndarray[double, ndim=1, mode="c"]> self.weights

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_arrays = <cfl.param_type ***>malloc(self.n_h*sizeof(cfl.param_type **))
        if param_arrays == NULL:
            for i in range(self.n_h):
                free(self.ex_data[i])
            free(self.ex_data)
            free(self.ha)
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
                raise MemoryError("param_arrays[{}] alloc failed".format(i))
       
        for hi,h in enumerate(h_list):
            for i,p in enumerate(h_param_list[hi]):
                param_arrays[hi][i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
                if param_arrays[hi][i] is NULL:
                    for hj in range(hi):
                        for j in range(self.n_p):
                            free(param_arrays[hj][j])
                    for j in range(i):
                        free(param_arrays[hi][j])
                    for hj in range(self.n_h):
                        free(param_arrays[hj])
                    free(self.param_arrays)
                    for j in range(self.n_h):
                        free(self.ex_data[j])
                    free(self.ex_data)
                    free(self.ha)
                    raise MemoryError("param_arrays[{0}][{1}] alloc failed".format(hi, i))
                
                param_arrays[hi][i].type = ord(self.param_types[p])
                param_arrays[hi][i].index = h_list[hi].index(p)
        
        # Set initial values.
        ip_real = 0
        for p in parameters:
            if self.param_types[p] == 'c':
                self.p0_real[ip_real] = np.real(self.coeff[p])
                self.p0_real[ip_real+1] = np.imag(self.coeff[p])
                ip_real += 2
            else:
                self.p0_real[ip_real] = self.coeff[p]
                ip_real += 1
        
        self.param_arrays = param_arrays
                
        self.job_a = np.empty(self.n_h, dtype=np.dtype('S'))
        for i,ex in enumerate(self.ex_list):
            if ex.sl_index:
                self.job_a[i] = 'S'
            else:
                self.job_a[i] = 'N'
        job_a = self.job_a
        self.mhfit_data = mhfit_data_alloc(&job_a[0], self.n_h, self.ha, &weights[0], self.ex_data, &n_zx[0], self.param_arrays)
        
        self.fit_data_cap = PyCapsule_New(<void *>self.mhfit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.mhfit_obj, "pycfl.MinObjF", NULL)
        self.cov_f_cap = PyCapsule_New(<void *>&cfl.mhfit_cov, "pycfl.MinCovF", NULL)
        
        # Run mhfit_chi2 so that the initial chi^2 weighting is set.
        chi2 = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(1)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        cfl.mhfit_chi2(&x[0], self.mhfit_data, &chi2[0])

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
        cdef sigma = 0

        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        
        fmin = min_object.minimize(self, x)
        
        coeff = self.coeff.copy() 
        ri = 0
        
        for p in self:
            if (self.param_types[p] == 'c'): 
                coeff[p] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p] = x[ri]
                ri += 1
        
        return(coeff, fmin)


cdef class ESHFitRunner(object):
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
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    cdef SpinHamiltonian sh
    cdef public Hamiltonian h
    cdef Hamiltonian hpro
    cpdef public dict coeff
    cdef int n_p
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public int n_obs
    cpdef public int n_ushx
    cpdef public dict param_types
    cdef cfl.ex_data *ex_data
    cdef public ExData ex
    cdef cfl.param_type **param_array
    cdef cfl.shx_data **shx_array
    cdef list shx_list
    cdef dict weights
    cdef np.ndarray p0_real
    cdef cfl.eshfit_data *eshfit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    def __init__(self, parameters, h, sh, ex, shx, weights, **kwargs):
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_la
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ild
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fld
        cdef np.ndarray[int, ndim=1, mode="c"] ex_lah
        cdef np.ndarray[int, ndim=1, mode="c"] ex_ildh
        cdef np.ndarray[int, ndim=1, mode="c"] ex_fldh
        cdef np.ndarray[double complex, ndim=1, mode="c"] shx_pa
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        
        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters
        self.sh = sh

        if not all((isinstance(p, str) for p in parameters)):
            raise TypeError("Parameters must be strings of tensor names.")

        if h.coeff_dict == None:
            raise ValueError("Hamiltonian must have coefficients set prior to eshfit.")
        else:
            self.coeff = h.coeff_dict

        if not sh.pro_data_set:
            raise ValueError("Spin Hamiltonian must have projection data set prior to eshfit.")
        else:
            self.coeff.update(sh.coeff_dict)

        # If not present, add small magnetic field to Hamiltonian to order
        # states.
        magzs = None
        if 'MAGZS' not in h.coeff_dict:
            for t in sh.tensors:
                if t.get_name() == 'MAGZ':
                    # Call to sh.set_pro_data ensures MAGZ is present. 
                    magzs = 0.0001 * t
                    magzs.name = 'MAGZS'
                    break
            
            tmp_h_coeff = h.coeff_dict
            tmp_h_coeff['MAGZS'] = 1
            h = Hamiltonian([magzs] + h.tensors)
            h.set_coeff(tmp_h_coeff)

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
            self.hpro = Hamiltonian(pro_h_tensors)
            self.hpro.set_coeff(self.h.coeff_dict)
        else:
            self.hpro = None

        # Determine the type of each parameter. 
        self.param_types = {}
        # The number of real parameters. 
        self.n_p_real = 0
        # We also record the number of parameters unique to sh.
        self.n_ushx = 0
        for i,p in enumerate(parameters):
            if all((p not in hh for hh in [h, sh])):
                raise ValueError("Parameter %s not found in any Hamiltonian." % p)
            if not isinstance(self.coeff[p], Number):
                raise ValueError("The coefficient %s was not specfied as a number the Hamiltonian." % p)
            # The parameter type is recorded such that any complex parameters
            # can be split into two real parameters.
            if isinstance(self.coeff[p], complex):
                self.n_p_real += 2
                self.param_types[p] = "c"
            elif p == 'HYP':
                self.n_p_real += 1
                self.param_types[p] = "h"
                if p not in h:
                    self.n_ushx += 1
            elif p == 'QUAD':
                self.n_p_real += 1
                self.param_types[p] = "q"
                if p not in h:
                    self.n_ushx += 1
            else:
                self.param_types[p] = "r"
                self.n_p_real += 1

        if 'ignore_ndof' not in kwargs:
            kwargs['ignore_ndof'] = False

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        
        # Parse the energy level data, if required.
        if not isinstance(ex, ExData):
            self.ex = ExData(ex)
        else:
            self.ex = ex
        self.n_obs = self.ex.n_obs

        self.n_obs += sh.nsh
        if self.n_p_real > self.n_obs and kwargs['ignore_ndof'] != True:
            raise ValueError("The total (real and imaginary) number of parameters, %i, exceeds "
                    "the number of observables, %i.  If you must nevertheless proceed, you can do "
                    "so at your own peril by setting the kwarg ignore_ndof=True." % (self.n_p_real, len(ex)))

        self.ex_data = <cfl.ex_data *>malloc(sizeof(cfl.ex_data))
        if self.ex_data == NULL:
            raise MemoryError("ex_data alloc failed")

        self.ex_data.n_obs = self.ex.n_obs
        self.ex_data.n_a = self.ex.n_a
        self.ex_data.n_d = self.ex.n_d
        ex_e = <np.ndarray[double, ndim=1, mode="c"]> self.ex.e
        self.ex_data.e = &ex_e[0]

        if self.ex.n_a:
            ex_la = <np.ndarray[int, ndim=1, mode="c"]> self.ex.la
            self.ex_data.la = &ex_la[0]
        else:
            # There are no absolute energy level observables.
            self.ex_data.la = NULL

        if self.ex.n_d:
            ex_ild = <np.ndarray[int, ndim=1, mode="c"]> self.ex.ild
            ex_fld = <np.ndarray[int, ndim=1, mode="c"]> self.ex.fld
            self.ex_data.ild = &ex_ild[0]
            self.ex_data.fld = &ex_fld[0]
        else:
            # There are no energy level difference observables.
            self.ex_data.ild = NULL
            self.ex_data.fld = NULL

        if self.ex.sl_index:
            if self.ex.n_a:
                ex_lah = <np.ndarray[int, ndim=1, mode="c"]> self.ex.lah
                self.ex_data.lah = &ex_lah[0]
            else:
                self.ex_data.lah = NULL

            if self.ex.n_d:
                ex_ildh = <np.ndarray[int, ndim=1, mode="c"]> self.ex.ildh
                ex_fldh = <np.ndarray[int, ndim=1, mode="c"]> self.ex.fldh
                self.ex_data.ildh = &ex_ildh[0]
                self.ex_data.fldh = &ex_fldh[0]
            else:
                self.ex_data.ildh = NULL
                self.ex_data.fldh = NULL
        else:
            self.ex_data.lah = NULL
            self.ex_data.ildh = NULL
            self.ex_data.fldh = NULL

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_array = <cfl.param_type **>malloc(self.n_p*sizeof(cfl.param_type *))
        if param_array == NULL:
            free(self.ex_data)
            raise MemoryError("param_array alloc failed")
        self.param_array = param_array 
       
        ip_real = 0

        for i,p in enumerate(parameters):
            param_array[i] = <cfl.param_type *> malloc(sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))
            
            param_array[i].type = ord(self.param_types[p])
            if (i < self.n_p - self.n_ushx):
                param_array[i].index = self.h.index(p)

            if self.param_types[p] == 'c':
                self.p0_real[ip_real] = np.real(self.coeff[p])
                self.p0_real[ip_real+1] = np.imag(self.coeff[p])
                ip_real += 2
            else:
                self.p0_real[ip_real] =  self.coeff[p]
                ip_real += 1
        
        # Array of experimental spin Hamiltonian data.
        self.weights = weights
        shx_array = <cfl.shx_data **>malloc(len(sh.interactions)*sizeof(cfl.shx_data *))
        if shx_array == NULL:
            for i in range(self.n_p):
                free(param_array[i])
            free(self.ex_data)
            free(param_array)
            raise MemoryError("shx_array alloc failed")
        self.shx_list = []
        self.shx_array = shx_array
        for i,inter in enumerate(sh.interactions):
            if inter not in shx:
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(shx_array)
                raise ValueError("The spin Hamiltonian experimental data dictonary "
                        "is missing data for the {} interaction.".format(inter))
            elif not isinstance(shx[inter], np.ndarray):
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(shx_array)
                raise TypeError("exp_tensor must be a np.ndarray.")
            elif shx[inter].shape == (3, 3):
                self.shx_list += [np.ascontiguousarray(shx[inter].flatten(), dtype=np.complex128)]
            elif shx[inter].shape == (9,):
                self.shx_list += [np.ascontiguousarray(shx[inter], dtype=np.complex128)]
            else:
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(shx_array)
                raise ValueError("exp_tensor must either be a (3, 3) or (9, 1) array.")
            
            shx_array[i] = <cfl.shx_data *>malloc(sizeof(cfl.shx_data))
            if shx_array[i] == NULL:
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(shx_array)
                raise MemoryError("shx_array[{}] alloc failed".format(i))
            shx_pa = <np.ndarray[double complex, ndim=1, mode="c"]> self.shx_list[i]
            shx_array[i].pa = &shx_pa[0]
            shx_array[i].chisq_weight = 1

        # Alloc data for objective functions and estimate initial chi^2 values. 
        chi2 = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(len(sh.interactions)+1)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        if (self.hpro != None):
            if self.ex.sl_index:
                self.eshfit_data = eshfit_data_alloc('S', <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
                    <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"),
                    self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.n_ushx, self.param_array)
            else:
                self.eshfit_data = eshfit_data_alloc('N', <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
                    <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"),
                    self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.n_ushx, self.param_array)
            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_obj, "pycfl.MinObjF", NULL)
            self.cov_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_cov, "pycfl.MinCovF", NULL)
            
            # Unweighted initial chi^2 estimation.
            cfl.eshfit_hpro_chi2(&x[0], self.eshfit_data, &chi2[0])

        else:
            if self.ex.sl_index:
                self.eshfit_data = eshfit_data_alloc('S', <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
                    NULL, self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.n_ushx, self.param_array)
            else:
                self.eshfit_data = eshfit_data_alloc('N', <cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
                    NULL, self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                    shx_array, self.n_p, self.n_ushx, self.param_array)
            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_obj, "pycfl.MinObjF", NULL)
            self.cov_f_cap = PyCapsule_New(<void *>&cfl.eshfit_cov, "pycfl.MinCovF", NULL)
            
            # Unweighted initial chi^2 estimation.
            cfl.eshfit_chi2(&x[0], self.eshfit_data, &chi2[0])

        self.fit_data_cap = PyCapsule_New(<void *>self.eshfit_data, "pycfl.MinData", NULL)

        # Energy levels are always weighted to unity provided a call to
        # eshfit_hpro_chi2 or eshfit_chi2 has been made. 
        if 'energy' in self.weights:
            ew_scale = 1.0/self.weights['energy']
        else:
            ew_scale = 1.0

        for i,inter in enumerate(sh.interactions):
            try:
                shwi = self.weights[inter]
            except KeyError:
                shwi = 1.0
            # FIXME: removed chi2 scaling; used to be:
            # shx_array[i].chisq_weight = shwi/chi2[i+1] * ew_scale 
            # which is broken with current cfl weighting setup... 
            shx_array[i].chisq_weight = shwi * ew_scale 
    
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
        cdef sigma = 0

        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        fmin = min_object.minimize(self, x)
        
        coeff = self.coeff.copy()
        ri = 0
        for p in self:
            if (self.param_types[p] == 'c'): 
                coeff[p] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p] = x[ri]
                ri += 1
        
        return(coeff, fmin)


cdef class CFLMin:
    r"""
    Object for initializing and configuring minimization routines to be passed
    to e_fit or esh_fit.

    Parameters
    ----------
    method : string
        The minimization routine to employ.  Available options are:

            - 'basinhopping'
            - 'nlopt_cobyla'
            - 'nlopt_bobyqa'
            - 'nlopt_sbplx'
            - 'nlopt_crs2_lm'
            - 'nlopt_esch'.

    bounds : dict, optional
        Parameter bounds.  Keys specify the tensor name (note that tensors
        created by tensor arithmetic should have their name attribute set
        explicitly), while values correspond to tuples, the first entry of which
        is the lower bound and the second entry the upper bound.  The number of
        elements in bounds must match the length of the parameters list. 
    cov : bool, optional
        Evaluate the covariance matrix for the fit; defaults to False.
    lmin : CFLMin, optional
        The local minimization routine to be used by the basinhopping algorithm;
        defaults to nlopt_bobyqa.  Implemented options fall into two categories,
        routines from gsl, and routines from nlopt.  For the former, available
        algorithms are:
        
            - 'gsl_nmsimplex2'
            - 'gsl_conjugate_fr'
            - 'gsl_conjugate_pr'
            - 'gsl_vector_bfgs2'

        For the latter, available algorithms are:

            - 'nlopt_cobyla'
            - 'nlopt_bobyqa'
            - 'nlopt_sbplx'.

    stepsize : dict, optional 
        The stepsize for parameter variation; this argument is  only referenced
        if the basinhopping algorithm is selected.  Keys specify the tensor
        name, while values correspond to the stepsize.  If adaptive stepsize is
        enabled, then this dictionary is used as the starting stepsize, and all
        step sizes are scaled by the same factor in order to achieve the target
        acceptance rate.  In other words, this kwarg is then used to set the
        relative proportion between the step sizes. 
    niter : int, optional
        The number of basinhopping iterations to complete; defaults to 100. 
    target_accept_rate : float, optional
        The target acceptance rate for basinhopping steps; used for adaptive
        stepsize tuning.  To disable adaptive stepsize, set this parameter to 0.
        The default is 0.5.
    step_adapt_int : int, optional
        The number of iterations between adaptive stepsize checks; defaults to 20.
    xtol : float, optional
        If either the global optimization or a local basinhopping minimization
        routine is from nlopt, the ``xtol`` argument can be used to set the
        relative tolerance in parameters x to be used as a stopping criteria.
        Defaults to 1e-5.

    """
    cpdef public str method
    cpdef public dict kwargs
    cdef int niter
    cdef size_t nx
    cdef double xtol
    cdef cfl.cfl_min_bounds *cfl_bounds
    cdef cfl.cfl_min_obj *min_obj
    cdef cfl.cfl_min_obj *bh_lmin_obj 

    def __cinit__(self, method, **kwargs):
        if 'cov' not in kwargs:
            kwargs['cov'] = False

        if method == 'basinhopping':
            if 'niter' in kwargs:
                self.niter = kwargs['niter']
            else:
                self.niter = 100
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
        else:
            raise NotImplementedError("Minimization method '%s' is not an existing option." % method)

        self.method = method
        self.kwargs = kwargs

    def __dealloc__(self):
        if self.cfl_bounds != NULL:
            free(self.cfl_bounds)

    @cython.boundscheck(False)
    cpdef minimize(self, fit_obj, x0):
        r"""
        Run the minimization. 

        Parameters
        ----------
        fit_obj : EFitRunner or ESHFitRunner
            The object for which to perform the fit. 
        x0 : np.ndarray
            Real valued vector.  Upon entry, these are the initial guesses for
            the parameters; if minimization is successful, x0 will be
            overwritten with the solution.
        """
        cdef np.ndarray[double, ndim=1, mode="c"] cx0
        cdef size_t cnx
        cdef double cxtol
        cdef double (*obj_f_ptr)(size_t, double *, double *, void *)
        cdef void (*cov_f_ptr)(double *, double *, cfl_min_obj *)
        cdef void *data_ptr
        cdef cfl.cfl_min_obj *min_obj
        cdef cfl.cfl_min_obj *lmin_obj
        cdef double fmin = 0
        cdef np.ndarray[double, ndim=1, mode="c"] lb 
        cdef np.ndarray[double, ndim=1, mode="c"] ub
        cdef np.ndarray[double, ndim=1, mode="c"] cstepsize
        cdef double *stepsize_ptr
        cdef float target_accept_rate
        cdef int step_adapt_int
        cdef np.ndarray[double, ndim=2, mode="c"] cov_inv
        cdef double *cov_ptr
        
        cnx = <size_t> len(x0)
        obj_f_ptr = <double (*)(size_t, double *, double *, void *)>PyCapsule_GetPointer(
                fit_obj.obj_f_cap, "pycfl.MinObjF")
        cov_f_ptr = <void (*)(double *, double *, cfl_min_obj *)>PyCapsule_GetPointer(
                fit_obj.cov_f_cap, "pycfl.MinCovF")
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

        if self.kwargs['cov']:
            self.kwargs['cov_inv'] = np.zeros([fit_obj.n_p_real, fit_obj.n_p_real])
            cov_inv = <np.ndarray[double, ndim=2, mode="c"]> self.kwargs['cov_inv']
            cov_ptr = &cov_inv[0,0]
        else:
            cov_ptr = NULL

        # Set xtol to default if not provided. 
        if 'xtol' in self.kwargs:
            cxtol = self.kwargs['xtol']
        else:
            cxtol = 1e-5

        if self.method == 'basinhopping':
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
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, gsl_nmsimplex2rand)
                elif lmin == 'gsl_nmsimplex2':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, gsl_nmsimplex2)
                elif lmin == 'gsl_conjugate_fr':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, gsl_conjugate_fr)
                elif lmin == 'gsl_conjugate_pr':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, gsl_conjugate_pr)
                elif lmin == 'gsl_vector_bfgs2':
                    lmin_obj = cfl_gsl_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, gsl_vector_bfgs2)
                elif lmin == 'nlopt_cobyla':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_cobyla,
                            cxtol, self.cfl_bounds)
                elif lmin == 'nlopt_bobyqa':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                            cxtol, self.cfl_bounds)
                elif lmin == 'nlopt_sbplx':
                    lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_sbplx,
                            cxtol, self.cfl_bounds)
                else:
                    raise ValueError("Unknown lmin argument: %s" % lmin)
            else:
                lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                        cxtol, self.cfl_bounds)
           
            min_obj = cfl_bh_min_setup(self.niter, stepsize_ptr, target_accept_rate, step_adapt_int,
                    self.cfl_bounds, lmin_obj)
            
            # Assign to self to guarantee there exists a reference to these
            # objects until the CFLMin destructor is called.
            self.nx = cnx
            self.xtol = cxtol
            self.bh_lmin_obj = lmin_obj
            self.min_obj = min_obj
        elif self.method == 'nlopt_cobyla':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_cobyla,
                    cxtol, self.cfl_bounds)
        elif self.method == 'nlopt_bobyqa':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_bobyqa,
                    cxtol, self.cfl_bounds)
        elif self.method == 'nlopt_sbplx':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_sbplx,
                    cxtol, self.cfl_bounds)
        elif self.method == 'nlopt_crs2_lm':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_crs2_lm,
                    cxtol, self.cfl_bounds)
        elif self.method == 'nlopt_esch':
            min_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_esch,
                    cxtol, self.cfl_bounds)

        cx0 = <np.ndarray[double, ndim=1, mode="c"]> x0
        with nogil:
            retval = cfl.cfl_min(&cx0[0], &fmin, cov_ptr, min_obj)

        # Assign some kwargs to self for summary printing.
        self.kwargs['retval'] = retval
        self.kwargs['n_obs'] = fit_obj.n_obs
        self.kwargs['n_param'] = fit_obj.n_p_real

        return fmin


def e_fit(parameters, h, ex, cfl_min, **kwargs):
    r"""
    Fit parameters to energy level data. 

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of
    coefficients when the are set also determines whether they are fit as real
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
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    efit = EFitRunner(parameters, h, ex, **kwargs)
    (x, fmin) = efit.fit(cfl_min)
    h.set_coeff(x)
    (w, z) = h.diag()

    # The number of degrees of freedom of the chi-squared distribution
    ndof = max(efit.n_p_real - efit.n_obs, 1)

    e_sigma = e_fit_sigma(w, efit.ex, ndof, z, h.tensors[0].states.labels)

    summary = "=============\n"
    summary+= "e_fit summary\n"
    summary+= "=============\n"
    summary += gen_pycf_summary()
    summary += efit.h.gen_summary(ex=efit.ex, sigma=e_sigma)
    summary += "\n"
    summary += gen_fit_summary(x, efit, cfl_min.method, fmin, sigma=e_sigma, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary}



def mh_fit(parameters, h_list, weights_list, ex_list, cfl_min, **kwargs):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    multiple eigenvalue vectors.  Typically, this would consist of one vector of
    energy levels at zero field without hyperfine or quadrupole interactions,
    complemented by a set of eigenvalue vectors at linearly independent magnetic
    field orientations and possibly containing hyperfine interactions.  These
    additional eigenvalues can either be measured or synthetically calculated
    for specific crystal field levels from spin Hamiltonian data.  

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
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    mhfit = MHFitRunner(parameters, h_list, weights_list, ex_list, **kwargs)
    (x, fmin) = mhfit.fit(cfl_min)

    summary = "==============\n"
    summary+= "mh_fit summary\n"
    summary+= "==============\n"
    summary += gen_pycf_summary()

    # The number of degrees of freedom of the chi-squared distribution
    ndof = max(mhfit.n_p_real - mhfit.n_obs, 1)

    for i,h in enumerate(mhfit.h_list):
        h.set_coeff(x)
        (w, z) = h.diag()

        e_sigma = e_fit_sigma(w, ex_list[i], ndof, z, h.tensors[0].states.labels)
        summary += h.gen_summary(ex=ex_list[i], sigma=e_sigma)
        summary += "\n"

    summary += gen_fit_summary(x, mhfit, cfl_min.method, fmin, sigma=e_sigma, **cfl_min.kwargs)
    
    return {'fmin': fmin, 'coeff': x, 'summary': summary}


def esh_fit(parameters, h, sh, ex, shx, weights, cfl_min, **kwargs):
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
    ignore_ndof : bool, optional
        Force minimization even if there are fewer observables than parameters;
        use at your own peril.
    """
    eshfit = ESHFitRunner(parameters, h, sh, ex, shx, weights, **kwargs)
    (x, fmin) = eshfit.fit(cfl_min)
    h.set_coeff(x)
    (w, z) = h.diag()
    
    # The number of degrees of freedom of the chi-squared distribution
    ndof = max(eshfit.n_p_real - eshfit.n_obs, 1)

    sh_param = sh.calc_param(h)
    e_sigma = e_fit_sigma(w, ex, ndof, z, h.tensors[0].states.labels)
    sh_sigma = sh_fit_sigma(sh_param, sh, shx, ndof)

    summary = "===============\n"
    summary+= "esh_fit summary\n"
    summary+= "===============\n"
    summary += gen_pycf_summary()
    summary += h.gen_summary(ex=eshfit.ex, sigma=e_sigma)
    summary += "\n"
    summary += gen_sh_summary(sh_param, sh, shx, sigma=sh_sigma)
    summary += "\n"
    summary += gen_fit_summary(x, eshfit, cfl_min.method, fmin, sigma=e_sigma+sh_sigma, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary}

