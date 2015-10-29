# filename = pycfl.pyx
#cython: c_string_encoding=ascii
#cython: embedsignature=True

#   Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
cimport cfl, cython
cimport numpy as np
import numpy as np
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
        the order used in the state tuples that make up the labels list.
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
    def __cinit__(self, label_key, labels):
        cdef size_t n
        cdef char *key
        cdef int *int_ptr
        cdef np.ndarray[int, ndim=1, mode='c'] clabels
        cdef int **l_a

        n = <size_t> len(labels)
        key = <char *> label_key
        l_a = <int **>malloc(len(labels)*cython.sizeof(int_ptr))
        if l_a == NULL:
            raise MemoryError("l_a malloc failed")

        self.labels = labels
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
    cdef public str tmp_name
    cpdef public int n
    cdef public StateLabels states
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, char *name, np.ndarray[int, ndim=1, mode='c'] row_ptr, 
            np.ndarray[int, ndim=1, mode='c'] col_in, np.ndarray[double complex, ndim=1, mode='c'] val, 
            states, object data_tuple=None):
        cdef cfl.zt *t
        cdef cfl.zt *t1
        cdef cfl.zt *t2
        self.name = <str> name
        self.states = states
        
        if (data_tuple == None):
            n = len(row_ptr)-1
            self.n = n
            t = cfl.zt_csr_alloc(name, n, &row_ptr[0], &col_in[0], &val[0], <cfl.sl *>PyCapsule_GetPointer(states.sl_cap, "pycfl.StateLabels"))
            
        elif (len(data_tuple)==3):
            # Addition or subtraction of tensors.
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t2 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[1].t_cap, "pycfl.Tensor")
            t = cfl.zt_sa(<char *>self.name, t1, t2, 1, data_tuple[2])

        else:
            # Scaling of a tensor.
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t = cfl.zt_s(<char *>self.name, t1, <double complex> data_tuple[1])

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
        t1.tmp_name = "{0}+{1}".format(t1.name, t2.name)
        d = (t1, t2, 1)
        return Tensor(<char *>t1.tmp_name, None, None, None, t1.states, data_tuple=d) 

    def __sub__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        t1.tmp_name = "{0}-{1}".format(t1.name, t2.name)
        d = (t1, t2, -1)
        return Tensor(<char *>t1.tmp_name, None, None, None, t1.states, data_tuple=d) 

    def __mul__(x, y):
        if isinstance(x, Number):
            if isinstance(y, Tensor):
                y.tmp_name = "{0:.2f}x{1}".format(x, y.name)
                d = (y, x)
                return Tensor(<char *>y.tmp_name, None, None, None, y.states, data_tuple=d)
        elif isinstance(x, Tensor):
            if isinstance(y, Number):
                x.tmp_name = "{0:.2f}x{1}".format(y, x.name)
                d = (x, y)
                return Tensor(<char *>x.tmp_name, None, None, None, x.states, data_tuple=d)
        else:
            raise TypeError("Tensors can only be multiplied by scalar numbers")


cdef class Hamiltonian:
    r"""
    The crystal field Hamiltonian class.  Creates a cfl zh object and provides
    an interface for diagonalizing zh.  Can be used to calculate:

        * energy levels given a list of :class:`Tensor`s and corresponding coefficients;
        * spin Hamiltoian parameters from crystal field parameters;
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
        cdef cfl.zt *ten_array_ptr

        n = tensors[0].n
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
        self.coeff_dict = None
        self.diag_run = 0
                
        # Create array of tensors and array of character arrays to be passed to
        # the zh_set cfl function. 
        tensor_array = <cfl.zt **>malloc(len(tensors)*cython.sizeof(ten_array_ptr))
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

    def __contains__(self, tensor):
        return tensor in self.tensors

    def __iter__(self):
        for t in self.tensors:
            yield t

    def index(self, tensor):
        try:
            return self.tensors.index(tensor)
        except ValueError:
            raise ValueError("Tensor {} is not an element of the Hamiltonian".format(tensor.name))
            
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
                self.coeff = np.append(self.coeff, coeff[t.name])
            except KeyError:
                raise KeyError("Missing coefficient for tensor: %s" % t.name)
        
        co = <np.ndarray[double complex, ndim=1, mode='c']> self.coeff
        cfl.zh_set_coeff(self.cfl_zh, &co[0])

        return None

    cpdef diag(self):
        r"""
        Diagonalize the Hamiltonian. 

        Returns
        -------
        (w, z) : tuple
            The eignvalues and eigenvectors, respectively, of the diagonalized
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

    cpdef gen_summary(self, ex=None, nstates=2, sigma=None):
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
        """
        if self.diag_run:
            return gen_e_summary(self.w, self.z, self.tensors[0].states.labels, ex, nstates, sigma)
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
    used for calculating spin Hamiltonian paremeters from crystal field
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
    cdef public dict coupling_constants

    def __init__(self, interactions, **kwargs):
        cdef int csz
        cdef int ciz
        cdef double complex *doublecomplex_ptr
        cdef char *char_ptr
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] a

        if not isinstance(interactions, list):
            interactions = [interactions]
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list: '{}'.".format(i))
        self.interactions = interactions

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

        self.inter_array = <char **>malloc(n_inter*cython.sizeof(char_ptr))
        if self.inter_array == NULL:
            raise MemoryError("inter_array malloc failed")

        self.inv_data_ptrs = <double complex **>malloc(len(interactions)*cython.sizeof(doublecomplex_ptr))
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
                dz = 2*self.Sz+1
                B_a = np.zeros([3, dz**2, 9], dtype = np.complex)
                for j in range(3):
                    B_a[j, :, :] = zeeman_sh_coeff(np.eye(3,3)[j,:], self.S_matel)
                self.inv_data += [np.asfortranarray(np.reshape(B_a, (3 * dz**2, 9)), dtype=np.complex128)]
                self.nsh += 3
                # Three g-values plus three Euler rotation parameters.
                self.nobs += 6
                self.required_tensors += ['MAGX', 'MAGY', 'MAGZ']

            if inter == 'hyperfine':
                dh = 2*self.Sz+1 + 2*self.Iz+1
                # The ordering of S_matel and I_matel is opposite to what makes
                # intuitive sense here... should probably figure this out
                # sometime.
                self.inv_data += [np.asfortranarray(hyperfine_sh_coeff(self.S_matel, self.I_matel), dtype=np.complex128)]
                self.nsh += 1
                # Three hyperfine values plus three Euler rotation parameters.
                self.nobs += 6
                self.required_tensors += ['HYP']

            if inter == 'quadrupole': 
                dq = 2*self.Iz+1
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
    
    def set_pro_data(self, tensors, coupling_constants={}):
        r"""
        Set the projection data for a specific spin Hamiltonian interaction. 

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

        t_array = <cfl.zt **>malloc(len(self.required_tensors)*cython.sizeof(cfl.zt))
        if t_array == NULL:
            raise MemoryError("t_array malloc failed")
        
        # Ensure all tensors required for projecting the interactions of this
        # spin Hamiltonian are provided. 
        cc_list = []
        for i,rt in enumerate(self.required_tensors):
            try:
                t_array[i] = <cfl.zt *>PyCapsule_GetPointer(next((t for t in tensors if t.name == rt)).t_cap, "pycfl.Tensor")
            except StopIteration:
                raise ValueError("Missing tensor %s in tensors list" % rt)
            if rt == 'HYP':
                try:
                    cc_list += [coupling_constants['HYP']]
                except KeyError:
                    raise KeyError("Missing the nuclear dipole coupling constant.")
            elif rt == 'QUAD':
                try:
                    cc_list += [coupling_constants['QUAD']]
                except KeyError:
                    raise KeyError("Missing the nuclear quadrupole coupling constant.")
            else:
                # Default to unity for Zeeman/magz. 
                cc_list += [1.0]
       
        self.coupling_constants = coupling_constants
        cc = np.array(cc_list, dtype=np.float64)
        retval = zsh_set_pro(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"), t_array, self.level, &cc[0])
        
        free(t_array)
        if retval != 1:
            raise ValueError("zsh_set_pro failed.  See the cfl error message for details.")

        self.tensors = tensors
        self.pro_data_set = 1


    def calc_param(self, h):
        r"""
        Calculate the spin Hamiltonian parameters given a complete Hamiltonian.

        Parameters
        ----------
        h : Hamiltonian
            The corresponding complete Hamiltonian. 

        Returns
        -------
        param : list
            Elements are nd.arrays corresponding to spin Hamiltonian tensors of
            interactions specified when the spin Hamiltonian object was
            instantiated. 
        """

        cdef cfl.zshp_w *shp_w
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] cz
        cdef np.ndarray[double complex, ndim=1, mode="c"] a
        
        if not self.pro_data_set:
            raise ValueError("The spin Hamiltonian interaction is missing projection data.")


        # If not present, add small magnetic field to Hamiltonian to order
        # states.
        if 'MAGZS' not in h.coeff_dict:
            for t in self.tensors:
                if t.name == 'MAGZ':
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
            if t.name not in pro_tensor_list:
                pro_h_tensors += [t]
            else:
                create_pro_h = True

        if create_pro_h:
            tmp_coeff = h.coeff_dict
            h = Hamiltonian(pro_h_tensors)
            h.set_coeff(tmp_coeff)

        (w, z) = h.diag()
        cz = <np.ndarray[double complex, ndim=2, mode="fortran"]> z
        shp_w = zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"));
        a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(9, dtype=np.complex128)
        
        result_list = []
        for i in range(len(self.interactions)):
            zshp(&a[0], &cz[0,0], i, <cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SpinHamiltonian"), shp_w);
            result_list += [np.copy(a.reshape(3,3))]

        zshp_w_free(shp_w);

        return result_list

def parse_param_helper(parameters, h, sh=None):
    r"""
    In the following, the word parameters is used to refer to tensor
    coefficients which are varied during the fitting routine.  
    
    Create a list of initial parameter values using the coefficients of the
    provided Hamiltonian, and record their type.  The type determines whether we
    fit a real or complex parameter. 
    """
    param_list = []
    param_types = []
    n_p_real = 0

    # Parameters that can be part of a spin Hamiltonian require a dedicated
    # type.
    sh_param = ['HYP', 'QUAD']  
    for i,p in enumerate(parameters):
        if p not in h:
            raise ValueError("Tensor %s in parameters not found in h." % p.name)
        try:
            if not isinstance(h.coeff_dict[p.name], Number):
                raise ValueError("Element %s in coefficients is not a number." % p.name)
        except KeyError:
            raise ValueError("Missing %s form Hamiltonian coefficients." % p.name)

        # The parameter type is recorded such that any complex parameters
        # can be split into two real parameters.
        if isinstance(h.coeff_dict[p.name], complex):
            param_types.append("c")
            n_p_real += 2
        elif p.name == 'HYP':
            param_types.append("h")
            n_p_real += 1
            sh_param.remove('HYP')
        elif p.name == 'QUAD':
            param_types.append("q")
            n_p_real += 1
            sh_param.remove('QUAD')
        else:
            param_types.append("r")
            n_p_real += 1
        
        param_list += [h.coeff_dict[p.name]]

    # If there's a spin Hamiltonian, we add the hyp and quad coupling to the
    # parameters, provided they have not been added already.  These are
    # necessarily real, and will be interpreted by cfl as such.  We also record
    # the parameters unique to the spin Hamiltonian. 
    n_ushx = 0
    ush_param = []
    if sh != None:
        for t in sh.tensors:
            if t.name == 'HYP' and t.name in sh_param:
                param_types.append("h")
                n_p_real += 1
                param_list += [sh.coupling_constants[t.name]]
                n_ushx += 1
                ush_param += [t]
            elif t.name == 'QUAD' and t.name in sh_param:
                param_types.append("q")
                n_p_real += 1
                param_list += [sh.coupling_constants[t.name]]
                n_ushx += 1
                ush_param += [t]

    return {'n_p_real': n_p_real, 'param_list' : param_list, 'param_types':
            param_types, 'ush_param': ush_param, 'n_ushx': n_ushx}


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
    ex : np.ndarray
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    """
    cdef public Hamiltonian h
    cdef int n_p
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public list param_list
    cpdef public list param_types
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef np.ndarray p0_real
    cdef cfl.efit_data *efit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    
    def __init__(self, parameters, h, ex):
        cdef cfl.param_type *param_type_ptr
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        
        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters
        
        if h.coeff_dict == None:
            raise ValueError("Hamiltonian must have coefficients set prior to diagonalization.")

        pp = parse_param_helper(parameters, h)
        self.n_p_real = pp['n_p_real']
        self.param_list = pp['param_list']
        self.param_types = pp['param_types']

        if self.n_p_real > len(ex):
            raise ValueError("The total (real and imaginary) number of parameters exceeds "
                    "the number of observables.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.

        # Prepare experimental energy level data
        self.ex_e = np.ascontiguousarray(ex[:,1], dtype=np.float64)
        # Subtract one, since we need an index starting at zero, whereas ex
        # levels start at 1. 
        self.ex_li = np.ascontiguousarray(ex[:,0]-1, dtype=np.int32)
       
        ex_e = <np.ndarray[double, ndim=1, mode="c"]> self.ex_e
        ex_li = <np.ndarray[int, ndim=1, mode="c"]> self.ex_li
        self.ex_data = <cfl.ex_data *>malloc(cython.sizeof(cfl.ex_data))
        if self.ex_data == NULL:
            raise MemoryError("ex_data alloc failed")
        self.ex_data.n = len(ex)
        self.ex_data.e = &ex_e[0]
        self.ex_data.li = &ex_li[0]

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_array = <cfl.param_type **>malloc(self.n_p*cython.sizeof(param_type_ptr))
        if param_array == NULL:
            free(self.ex_data)
            raise MemoryError("param_array alloc failed")
        
        ip_real = 0
        param_enc = {'r': 114, 'i': 105, 'c': 99}
        for i in range(self.n_p):
            param_array[i] = <cfl.param_type *> malloc(cython.sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))
            
            param_array[i].type = param_enc[self.param_types[i]]
            param_array[i].index = h.index(parameters[i])

            if self.param_types[i] == 'c':
                self.p0_real[ip_real] = np.real(self.param_list[i])
                self.p0_real[ip_real+1] = np.imag(self.param_list[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = self.param_list[i]
                ip_real += 1

        self.param_array = param_array 

        self.efit_data = cfl.efit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"), 
                self.ex_data, self.n_p, self.param_array);
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
        
        coeff = self.h.coeff_dict 
        ri = 0
        
        for i,p in enumerate(self):
            if (self.param_types[i] == 'c'): 
                coeff[p.name] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p.name] = x[ri]
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
        match the corresponing experimental energy level data.
    weights_list : list
        A list of floating point weights that determine the weighting added to
        the chi^2 contribution of each eigenvalue vector.
    bc_blockdim_list : list
        The barycenter block dimension for each corresponding h_list entry.  If
        0, no barycenter shift is applied.  For entries of value n, the
        barycenter shift for n dimensional blocks of energy levels is calculated
        and subtracted from the theoretical eigenvalues prior to the chi^2
        evaluation.  This is useful for ensuring that magnetic or hyperfine data
        available for a subset of CF levels is not dominated by a shift of the
        entire multiplet.  If non-zero, the experimental data must be in blocks
        of the specified size with no missing levels. 
    ex_list : list
        A list of 2 by n dimensional arrays, with n the number of available
        experimental energy levels for each corresponding Hamiltonian in h_list.
        The first column of each element contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    """
    cdef int n_h
    cdef int n_p
    cdef public Hamiltonian h
    cpdef public list h_list
    cdef cfl.zh **ha
    cdef np.ndarray weights
    cdef np.ndarray bc_blockdim
    cdef list ex_e_list
    cdef list ex_li_list
    cdef cfl.ex_data **ex_data
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public list param_list
    cpdef public list param_types
    cdef cfl.param_type ***param_arrays
    cdef np.ndarray p0_real
    cdef cfl.mhfit_data *mhfit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    
    def __init__(self, parameters, h_list, weights_list, bc_blockdim_list, ex_list):
        cdef cfl.zh *zh_ptr
        cdef cfl.ex_data *ex_data_ptr
        cdef cfl.param_type *param_type_ptr
        cdef cfl.param_type **param_type_ptr_ptr
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double, ndim=1, mode="c"] weights
        cdef np.ndarray[int, ndim=1, mode="c"] bc_blockdim
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x

        # Verify that the tensor order and coefficient type is the same in all
        # Hamiltonians.
        for i,t in enumerate(h_list[0]):
            for j,h in enumerate(h_list[1:]):
                try:
                    if h.index(t) != i:
                        raise ValueError("The tensor order of the %ith " \
                                "Hamiltonian does not match the tensor order "\
                                "of the 0th Hamiltonian." % j)

                    if not isinstance(h.coeff_dict[t], type(h_list[0].coeff_dict[t])):
                            raise ValueError("The coefficient type of the %ith " \
                                    "Hamiltonian does not match the corresponding "\
                                    "coefficient type of the 0th Hamiltonian" % j)
                except:
                    continue

        # Determine which Hamiltonian has the complete set of tensors.
        total_nt = 0
        pp_h_index = 0
        for i,h in enumerate(h_list):
            if h.nt > total_nt:
                total_nt = h.nt
                pp_h_index = i

        self.n_h = len(h_list)
        self.n_p = len(parameters)
        self.h = h_list[pp_h_index]
        self.h_list = h_list
        self.parameters = parameters
        
        pp = parse_param_helper(parameters, h_list[pp_h_index])
        self.n_p_real = pp['n_p_real']
        self.param_list = pp['param_list']
        self.param_types = pp['param_types']

        n_ex = ex_list[0].shape[0]
        for ex in ex_list:
            n_ex += ex.shape[0]
        if self.n_p_real > n_ex:
            raise ValueError("The total (real and imaginary) number of parameters exceeds "
                    "the number of observables.")

        self.ha = <cfl.zh **>malloc(self.n_h*cython.sizeof(zh_ptr))
        if self.ha == NULL:
            raise MemoryError("ha alloc failed")

        for i in range(self.n_h):
            self.ha[i] = <cfl.zh *>PyCapsule_GetPointer(h_list[i].h_cap, "pycfl.Hamiltonian")

        self.ex_data = <cfl.ex_data **>malloc(self.n_h*cython.sizeof(ex_data_ptr))
        if self.ex_data == NULL:
            free(self.ha)
            raise MemoryError("exa alloc failed")

        self.ex_e_list = []
        self.ex_li_list = []
        for i in range(self.n_h):
            # Prepare experimental energy level data
            self.ex_e_list += [np.ascontiguousarray(ex_list[i][:,1], dtype=np.float64)]
            # Subtract one, since we need an index starting at zero, whereas ex
            # levels start at 1. 
            self.ex_li_list += [np.ascontiguousarray(ex_list[i][:,0]-1, dtype=np.int32)]
       
            ex_e = self.ex_e_list[i]
            ex_li = self.ex_li_list[i]
            self.ex_data[i] = <cfl.ex_data *>malloc(cython.sizeof(cfl.ex_data))
            if self.ex_data[i] == NULL:
                for j in range(i):
                    free(self.ex_data[j])
                free(self.ex_data)
                free(self.ha)
                raise MemoryError("ex_data alloc failed")
            self.ex_data[i].n = len(ex)
            self.ex_data[i].e = &ex_e[0]
            self.ex_data[i].li = &ex_li[0]

        self.weights = np.array(weights_list, dtype=np.float64)
        weights = <np.ndarray[double, ndim=1, mode="c"]> self.weights
        self.bc_blockdim = np.array(bc_blockdim_list, dtype=np.int32)
        bc_blockdim = <np.ndarray[int, ndim=1, mode="c"]> self.bc_blockdim

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_arrays = <cfl.param_type ***>malloc(self.n_h*cython.sizeof(param_type_ptr_ptr))
        if param_arrays == NULL:
            for i in range(self.n_h):
                free(self.ex_data[i])
            free(self.ex_data)
            free(self.ha)
            raise MemoryError("param_arrays alloc failed")

        for i in range(self.n_h):
            param_arrays[i] = <cfl.param_type **>malloc(self.n_p*cython.sizeof(param_type_ptr))
            if param_arrays[i] == NULL:
                for j in range(i):
                    free(param_arrays[j])
                free(param_arrays)
                for j in range(self.n_h):
                    free(self.ex_data[j])
                free(self.ex_data)
                free(self.ha)
                raise MemoryError("param_arrays[{}] alloc failed".format(i))
       
        param_enc = {'r': 114, 'i': 105, 'c': 99}
        for hi,h in enumerate(h_list):
            ip_real = 0
            for i in range(self.n_p):
                param_arrays[hi][i] = <cfl.param_type *> malloc(cython.sizeof(cfl.param_type))
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
                
                param_arrays[hi][i].type = param_enc[self.param_types[i]]
                param_arrays[hi][i].index = h.index(parameters[i])

                if self.param_types[i] == 'c':
                    self.p0_real[ip_real] = np.real(self.param_list[i])
                    self.p0_real[ip_real+1] = np.imag(self.param_list[i])
                    ip_real += 2
                else:
                    self.p0_real[ip_real] = self.param_list[i]
                    ip_real += 1

        self.param_arrays = param_arrays 
        
        self.mhfit_data = mhfit_data_alloc(self.n_h, self.ha, &weights[0], &bc_blockdim[0], self.ex_data, self.n_p, self.param_arrays)

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
                for i in range(self.n_p):
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
        
        coeff = self.h.coeff_dict 
        ri = 0
        
        for i,p in enumerate(self):
            if (self.param_types[i] == 'c'): 
                coeff[p.name] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p.name] = x[ri]
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
    ex : np.ndarray
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.  Any ommited values will be set
        to unity.
    """
    cdef SpinHamiltonian sh
    cdef public Hamiltonian h
    cdef Hamiltonian hpro
    cdef int n_p
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public int n_ushx
    cpdef public list param_list
    cpdef public list param_types
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef cfl.shx_data **shx_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef list shx_list
    cdef dict weights
    cdef np.ndarray p0_real
    cdef cfl.eshfit_data *eshfit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    def __init__(self, parameters, h, sh, ex, shx, weights):
        cdef cfl.param_type *param_type_ptr
        cdef cfl.shx_data *shx_data_ptr
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double complex, ndim=1, mode="c"] shx_pa
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        
        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters
        self.sh = sh

        if h.coeff_dict == None:
            raise ValueError("Hamiltonian must have coefficients set prior to esh fit.")

        if not sh.pro_data_set:
            raise ValueError("Spin Hamiltonian must have projection data set prior to esh fit.")

        # If not present, add small magnetic field to Hamiltonian to order
        # states.
        magzs = None
        if 'MAGZS' not in h.coeff_dict:
            for t in sh.tensors:
                if t.name == 'MAGZ':
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
            if t.name not in pro_tensor_list:
                pro_h_tensors += [t]
            else:
                create_pro_h = True
        
        if create_pro_h:
            self.hpro = Hamiltonian(pro_h_tensors)
            self.hpro.set_coeff(self.h.coeff_dict)
        else:
            self.hpro = None

        if self.n_p_real > len(ex) + sh.nsh:
            raise ValueError("The total (real and imaginary) number of parameters "
                "exceeds the number of observables.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        pp = parse_param_helper(parameters, self.h, self.sh)
        self.n_p_real = pp['n_p_real']
        self.n_ushx = pp['n_ushx']
        self.n_p += self.n_ushx
        self.param_list = pp['param_list']
        self.param_types = pp['param_types']

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
       
        # Prepare experimental energy level data
        self.ex_e = np.ascontiguousarray(ex[:,1], dtype=np.float64)
        # Subtract one, since we need an index starting at zero, whereas ex
        # levels start at 1. 
        self.ex_li = np.ascontiguousarray(ex[:,0]-1, dtype=np.int32)
       
        ex_e = <np.ndarray[double, ndim=1, mode="c"]> self.ex_e
        ex_li = <np.ndarray[int, ndim=1, mode="c"]> self.ex_li
        self.ex_data = <cfl.ex_data *>malloc(cython.sizeof(cfl.ex_data))
        if self.ex_data == NULL:
            raise MemoryError("ex_data alloc failed")
        self.ex_data.n = len(ex)
        self.ex_data.e = &ex_e[0]
        self.ex_data.li = &ex_li[0]

        # Prepare array of pointers to parameter data structs.
        self.p0_real = np.ascontiguousarray(np.zeros(self.n_p_real), dtype=np.float64)
        param_array = <cfl.param_type **>malloc(self.n_p*cython.sizeof(param_type_ptr))
        if param_array == NULL:
            free(self.ex_data)
            raise MemoryError("param_array alloc failed")
        self.param_array = param_array 
       
        ip_real = 0

        param_enc = {'r': 114, 'i': 105, 'c': 99, 'h': 104, 'q': 113}
        for i in range(self.n_p):
            param_array[i] = <cfl.param_type *> malloc(cython.sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))

            param_array[i].type = param_enc[self.param_types[i]]
            try:
                param_array[i].index = self.h.index(parameters[i])
            except IndexError:
                # Spin Hamiltonian parameter; doesn't require index.
                param_array[i].index = -1

            if self.param_types[i] == 'c':
                self.p0_real[ip_real] = np.real(self.param_list[i])
                self.p0_real[ip_real+1] = np.imag(self.param_list[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] =  self.param_list[i]
                ip_real += 1
        
        # Done with the CF Hamiltonian specific parameters; update parameters to
        # include spin Hamiltonian specific interactions.
        self.parameters += pp['ush_param']

        # Array of experimental spin Hamiltonian data.
        self.weights = weights
        shx_array = <cfl.shx_data **>malloc(len(sh.interactions)*cython.sizeof(shx_data_ptr))
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
            
            shx_array[i] = <cfl.shx_data *>malloc(cython.sizeof(cfl.shx_data))
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
            self.eshfit_data = eshfit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
                <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"),
                self.ex_data, <cfl.zsh *>PyCapsule_GetPointer(sh.sh_cap, "pycfl.SpinHamiltonian"),
                shx_array, self.n_p, self.n_ushx, self.param_array)
            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_obj, "pycfl.MinObjF", NULL)
            self.cov_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_cov, "pycfl.MinCovF", NULL)
            
            # Unweighted initial chi^2 estimation.
            cfl.eshfit_hpro_chi2(&x[0], self.eshfit_data, &chi2[0])

        else:
            self.eshfit_data = eshfit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), 
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
            shx_array[i].chisq_weight = shwi/chi2[i+1] * ew_scale 
    
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
        
        coeff = self.h.coeff_dict
        ri = 0
        for i,p in enumerate(self):
            if (self.param_types[i] == 'c'): 
                coeff[p.name] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[p.name] = x[ri]
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
        created by tensor arithmethic should have their name attribute set
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
        The stepsize for parameter variation; presently only supported for the
        basinhopping algorithm.  Keys specify the tensor name, while values
        correspond to the stepsize.  If adaptive stepsize is enabled (default,
        see CFLMin doc for details), then this dictionary is used as the
        starting stepsize, and all step sizes are scaled by the same factor in
        order to achieve the target acceptance rate.  In other words, this kwarg
        is then used to set the relative proportion between the step sizes. 
    niter : int, optional
        The number of basinhopping iterations to complete.  Defaults to 100.
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
        cdef int step_adapt_int = 0
        cdef np.ndarray[double, ndim=2, mode="c"] cov_inv
        cdef double *cov_ptr
        
        cnx = <size_t> len(x0)
        obj_f_ptr = <double (*)(size_t, double *, double *, void *)>PyCapsule_GetPointer(fit_obj.obj_f_cap, "pycfl.MinObjF")
        cov_f_ptr = <void (*)(double *, double *, cfl_min_obj *)>PyCapsule_GetPointer(fit_obj.cov_f_cap, "pycfl.MinCovF")
        data_ptr = <void *>PyCapsule_GetPointer(fit_obj.fit_data_cap, "pycfl.MinData")

        # If bounds are specified, convert them to real valued lists the order of
        # which matches the order of the real valued parameter lists. 
        if 'bounds' in self.kwargs:
            lb = np.zeros(fit_obj.n_p_real)
            ub = np.zeros(fit_obj.n_p_real)
            rpi = 0

            bounds = self.kwargs['bounds']
            for i,p in enumerate(fit_obj):
                if fit_obj.param_types[i] == 'c':
                    try:
                        if not isinstance(bounds[p.name][0], complex) or \
                                not isinstance(bounds[p.name][1], complex):
                            raise ValueError("%s bounds are not complex, yet the "
                                    "corresponding coefficient in the Hamiltonian is." % p.name)
                    except KeyError:
                        raise KeyError("Missing bounds key %s." % p.name)
                    lb[rpi] = np.real(bounds[p.name][0])
                    lb[rpi+1] = np.imag(bounds[p.name][0])
                    ub[rpi] = np.real(bounds[p.name][1])
                    ub[rpi+1] = np.imag(bounds[p.name][1])
                    if np.real(fit_obj.h.coeff_dict[p.name]) < lb[rpi]:
                        raise ValueError("The real part of the %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p.name)
                    elif np.imag(fit_obj.h.coeff_dict[p.name]) < lb[rpi+1]:
                        raise ValueError("The imaginary part of the %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p.name)
                    elif np.real(fit_obj.h.coeff_dict[p.name]) > ub[rpi]:
                        raise ValueError("The real part of the %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p.name)
                    elif np.imag(fit_obj.h.coeff_dict[p.name]) > ub[rpi+1]:
                        raise ValueError("The imaginary part of the %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p.name)
                    rpi += 2
                else:
                    try:
                        lb[rpi] = np.real(bounds[p.name][0])
                        ub[rpi] = np.real(bounds[p.name][1])
                    except KeyError:
                        raise KeyError("Missing bounds key %s." % p.name)
                    try:
                        if fit_obj.h.coeff_dict[p.name] < lb[rpi]:
                            raise ValueError("The %s coefficient in the Hamiltonian is "
                                    "less than the specified lower bound." % p.name)
                        elif fit_obj.h.coeff_dict[p.name] > ub[rpi]:
                            raise ValueError("The %s coefficient in the Hamiltonian is "
                                    "greater than the specified lower bound." % p.name)
                    except KeyError:
                        pass
                    rpi += 1

            cfl_bounds = <cfl.cfl_min_bounds *>malloc(cython.sizeof(cfl.cfl_min_bounds))
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
                for i,p in enumerate(fit_obj.parameters):
                    if fit_obj.param_types[i] == 'c':
                        try:
                            if not isinstance(stepsize[p.name], complex):
                                raise ValueError("%s stepsize is not complex, yet the "
                                        "corresponding Hamiltonian coefficient is." % p.name)
                        except KeyError:
                            raise KeyError("Missing stepsize key %s." % p.name)
                        cstepsize[rpi] = np.real(stepsize[p.name])
                        cstepsize[rpi+1] = np.imag(stepsize[p.name])
                        rpi += 2
                    else:
                        try:
                            cstepsize[rpi] = np.real(stepsize[p.name])
                        except KeyError:
                            raise KeyError("Missing stepsize key %s." % p.name)
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
        self.kwargs['retval'] = retval

        return fmin


def e_fit(parameters, h, ex, cfl_min):
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
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    cfl_min : CFLMin
        The minimization object which sets the optimization algorithm and
        corresponding options.
    """
    efit = EFitRunner(parameters, h, ex)
    (x, fmin) = efit.fit(cfl_min)
    h.set_coeff(x)
    (w, z) = h.diag()

    # The number of degrees of freedom of the chi-squared distribution
    ndof = len(ex)-len(parameters)

    e_sigma = e_fit_sigma(w, ex, ndof)

    summary = "=============\n"
    summary+= "e_fit summary\n"
    summary+= "=============\n"
    summary += gen_pycf_summary()
    summary += efit.h.gen_summary(ex=ex, sigma=e_sigma)
    summary += "\n"
    summary += gen_fit_summary(x, efit, cfl_min.method, fmin, sigma=e_sigma, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary}



def mh_fit(parameters, h_list, weights_list, bc_blockdim_list, ex_list, cfl_min):
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
        match the corresponing experimental energy level data.
    weights_list : list
        A list of floating point weights that determine the weighting added to
        the chi^2 contribution of each eigenvalue vector.
    bc_blockdim_list : list
        The barycenter block dimension for each corresponding h_list entry.  If
        0, no barycenter shift is applied.  For entries of value n, the
        barycenter shift for n dimensional blocks of energy levels is calculated
        and subtracted from the theoretical eigenvalues prior to the chi^2
        evaluation.  This is useful for ensuring that magnetic or hyperfine data
        available for a subset of CF levels is not dominated by a shift of the
        entire multiplet.  If non-zero, the experimental data must be in blocks
        of the specified size with no missing levels. 
    ex_list : list
        A list of 2 by n dimensional arrays, with n the number of available
        experimental energy levels for each corresponding Hamiltonian in h_list.
        The first column of each element contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    """
    mhfit = MHFitRunner(parameters, h_list, weights_list, bc_blockdim_list, ex_list)
    (x, fmin) = mhfit.fit(cfl_min)

    summary = "==============\n"
    summary+= "mh_fit summary\n"
    summary+= "==============\n"
    summary += gen_pycf_summary()

    # The number of degrees of freedom of the chi-squared distribution
    ndof = 0
    for e in ex_list:
        ndof += len(e)
    ndof -= len(parameters)

    for i,h in enumerate(mhfit.h_list):
        h.set_coeff(x)
        (w, z) = h.diag()

        e_sigma = e_fit_sigma(w, ex_list[i], ndof)
        summary += h.gen_summary(ex=ex_list[i], sigma=e_sigma)
        summary += "\n"

    summary += gen_fit_summary(x, mhfit, cfl_min.method, fmin, sigma=e_sigma, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary}


def esh_fit(parameters, h, sh, ex, shx, weights, cfl_min):
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
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.  Any ommited values will be set
        to unity.

    cfl_min : CFLMin 
        The minimization object which sets the optimization algorithm and
        corresponding options.
    """
    eshfit = ESHFitRunner(parameters, h, sh, ex, shx, weights)
    (x, fmin) = eshfit.fit(cfl_min)
    h.set_coeff(x)
    (w, z) = h.diag()
    
    # The number of degrees of freedom of the chi-squared distribution
    ndof = len(ex) + sh.nobs - len(parameters)
    
    sh_param = sh.calc_param(h)
    e_sigma = e_fit_sigma(w, ex, ndof)
    sh_sigma = sh_fit_sigma(sh_param, sh, shx, ndof)

    summary = "===============\n"
    summary+= "esh_fit summary\n"
    summary+= "===============\n"
    summary += gen_pycf_summary()
    summary += h.gen_summary(ex=ex, sigma=e_sigma)
    summary += "\n"
    summary += gen_sh_summary(sh_param, sh, shx, sigma=sh_sigma)
    summary += "\n"
    summary += gen_fit_summary(x, eshfit, cfl_min.method, fmin, sigma=e_sigma+sh_sigma, **cfl_min.kwargs)

    return {'fmin': fmin, 'coeff': x, 'summary': summary}

