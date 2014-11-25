# filename = pycfl.pyx
#cython: c_string_encoding=ascii
#cython: embedsignature=True

#   Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
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

cimport cfl, cython
cimport numpy as np
import numpy as np
from numbers import Number
from cpython.pycapsule cimport *
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from matel import matel
from cfl_util import gen_e_summary, gen_sh_summary, gen_fit_summary

# TODO: 
#       + Apply a small magnetic field along z, to obtain state labels. Maybe
#         something to directly implement in the cfl projection interface.
#       + Add checks whether efit/eshfit data alloc functions return NULL and
#       corresponding frees.
#       + Python free bug if one does not provide the correct shx data dict
#       (change zeeman to something else). 
#       + Also, add check to ensure weighting is present for all sh terms. IF
#       not, either fail, or set to 1.
#       + set default self.coeff_dict=None, add check to parse_param, and change
#       other coeff_set checks to also use coeff_dict!=None.
#       + Add magz to SpinHamiltonian calc_coeff method.
#       + Change SpinHamiltonian set_pro_data to be a dict/list of all tensors,
#       s.t. one can consistently supply magz even in hyp or quad only cases.

cdef class StateLabels:
    r"""
    State label type for tensors and spin Hamiltonians.  State labels are
    generally not entered manually but should be generated with
    :class:`import_sljm.ImportSLJM`.


    Paramters
    ---------
    n : int
        The number of states.
    states : list
        Elements are strings corresponding to labels; all labels must be of
        equal length.
    """
    cdef:
        cfl.sl *cfl_sl
        public object sl_cap

    def __cinit__(self, labels):
        cdef char *char_ptr
        cdef char **state_labels
        cdef int label_length

        state_labels = <char **>malloc(len(labels)*cython.sizeof(char_ptr))
        if state_labels == NULL:
            raise MemoryError("state_labels array alloc failed")

        label_length = len(labels[0])
        for i,l in enumerate(labels):
            if (len(l) != label_length):
                free(state_labels)
                raise ValueError("State label '%s' is not of the same length as the first "
                    "label in states" % l)
            state_labels[i] = l

        self.cfl_sl = cfl.sl_alloc(len(labels), state_labels) 
        free(state_labels)
        if self.cfl_sl == NULL:
            raise MemoryError("cfl_sl alloc failed")
        else:
            self.sl_cap = PyCapsule_New(<void *>self.cfl_sl, "pycfl.StateLabels", NULL)

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
    def __cinit__(self, char *name, np.ndarray[double complex, ndim=2, mode='c']a, states,
            object data_tuple=None):
        cdef cfl.zt *t
        cdef cfl.zt *t1
        cdef cfl.zt *t2

        self.name = <str> name
        self.states = states

        if (data_tuple == None):
            n = a.shape[0]
            self.n = n
            t = cfl.zt_alloc(name, &a[0,0], n, <cfl.sl *>PyCapsule_GetPointer(states.sl_cap, "pycfl.StateLabels"))
            
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
        return Tensor(<char *>t1.tmp_name, np.array([[]],dtype=np.complex128), t1.states, data_tuple=d) 

    def __sub__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        t1.tmp_name = "{0}-{1}".format(t1.name, t2.name)
        d = (t1, t2, -1)
        return Tensor(<char *>t1.tmp_name, np.array([[]],dtype=np.complex128), t1.states, data_tuple=d) 

    def __mul__(x, y):
        if isinstance(x, Number):
            if isinstance(y, Tensor):
                y.tmp_name = "{0:.2f}x{1}".format(x, y.name)
                d = (y, x)
                return Tensor(<char *>y.tmp_name, np.array([[]],dtype=np.complex128), y.states, data_tuple=d)
        elif isinstance(x, Tensor):
            if isinstance(y, Number):
                x.tmp_name = "{0:.2f}x{1}".format(y, x.name)
                d = (x, y)
                return Tensor(<char *>x.tmp_name, np.array([[]],dtype=np.complex128), x.states, data_tuple=d)
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

    Hamiltonians are iterable over the tensors used to instantiate it.


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
    cdef int nt
    cdef public list tensors
    cdef public dict coeff_dict
    cdef public np.ndarray coeff
    cdef public np.ndarray w
    cdef public np.ndarray z
    cdef public object h_cap
    cdef int coeff_set
    cdef int diag_run
    def __cinit__(self, tensors):
        cdef cfl.zt *ten_array_ptr

        n = tensors[0].n
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
        self.coeff_set = 0
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
            raise ValueError("Tensor {} not an element of the Hamiltonian".format(tensor.name))
            
    cpdef public set_coeff(self, coeff):
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

        # Keep copy of dict, since fitting routines need to know the original
        # type of coeff elements to determine whether a param is real or
        # complex.
        self.coeff_dict = coeff

        self.coeff = np.array([], dtype=np.complex128)
        for t in self.tensors:
            try:
                self.coeff = np.append(self.coeff, coeff[t.name])
            except KeyError:
                raise KeyError("Missing coefficient for tensor: %s" % t.name)
        
        co = <np.ndarray[double complex, ndim=1, mode='c']> self.coeff
        cfl.zh_set_coeff(self.cfl_zh, &co[0])
        self.coeff_set = 1
        return None

    cpdef public diag(self):
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
        cdef np.ndarray[double complex, ndim=2, mode="c"] z
        
        if not self.coeff_set:
            raise ValueError("Hamiltonian must have coefficients set prior to diagonalization.")
        hd_w = cfl.zhd_w_alloc(self.cfl_zh)
        if hd_w is NULL:
            free(self.tensor_array)
            cfl.zh_free(self.cfl_zh)
            raise MemoryError("hd_w alloc failed")

        self.w = np.ascontiguousarray(np.zeros(self.n), dtype=np.float64)
        self.z = np.ascontiguousarray(np.zeros(self.n*self.n).reshape((self.n,self.n)), dtype=np.complex128)
        w = <np.ndarray[double, ndim=1, mode="c"]> self.w
        z = <np.ndarray[double complex, ndim=2, mode="c"]> self.z

        with nogil:
            cfl.zhd(&w[0], &z[0,0], h, hd_w)
        

        cfl.zhd_w_free(hd_w)

        self.diag_run = 1
        return (w, z)

    cpdef public gen_summary(self):
        r"""
        Generate an energy level summary resulting from a diagonalization. 

        Returns
        -------
        s : string
            The summary string.
        """
        cdef cfl.zh *h = self.cfl_zh

        if self.diag_run:
            labels = [] 
            for i in range(self.n):
                labels += [h.states.states[i]]
            return gen_e_summary(self.w, self.z, labels)
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
    This consists of a `2j_1+1 \times 2j_2+1` by `3 \times 3` array containing
    the matrix elements of the operators `I_a S_b`, with `a,b \in \{x, y, z\}`
    and `j_1` and `j_2` the angular momentum of the rank one tensors `I` and
    `S`, respectively.  Here the rows enumerate the `2j_1+1 \times 2j_2+1`
    different state combinations while the columns enumerate all combinations of
    `a` and `b`.  
    
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


cdef class SHTerm:
    r"""
    Data storage for a single spin Hamiltonian term; it is a wrapper for cfl sh
    objects.  As such, it is a representation for the matrix elements of a given
    interaction type. 

    Parameters
    ----------
    n : int
        The dimension of the spin Hamiltonian term. 
    interaction : string
        The type of interaction that this term is associated with; valid values
        are 'zeeman', 'hyperfine', and 'quadrupole'.

    Returns
    -------
    term : SHTerm

    """
    cpdef public int n
    cdef cfl.zsh *cfl_sh
    cpdef public str type
    cdef public object sh_cap
    cdef public Tensor tensor

    def __cinit__(self, n, interaction):
        self.n = n
        self.type = interaction
        
        self.cfl_sh = zsh_alloc(n, interaction)
        if self.cfl_sh is NULL:
            raise MemoryError("Failed to alloc zsh memory")
        else:
            self.sh_cap = PyCapsule_New(<void *>self.cfl_sh, "pycfl.SHTerm", NULL)

    def __dealloc__(self):
        if self.cfl_sh != NULL:
            cfl.zsh_free(self.cfl_sh)

    def set_pro_data(self, tensor, l):
        r""" 
        Set projection data. 

        Parameters
        ----------
        tensor : Tensor
            The projection tensor. 
        int : l 
            The starting level that the spin Hamiltonian corresponds to. 
        """
        cfl.zsh_set_pro(self.cfl_sh, <cfl.zt *>PyCapsule_GetPointer(tensor.t_cap, "pycfl.Tensor"), l)
        self.tensor = tensor

        
cdef class SHInteractionData(object):
    r"""
    Class used to store spin Hamiltonian data for a specific interaction.  For
    'hyperfine' and 'quadrupole' interactions, this corresponds to a single
    SHTerm, whereas for 'zeeman' interactions, it corresponds to three SHTerms. 

    Parameters
    ----------
    d : int
        The dimension of the spin Hamiltonian interaction term(s). 
    inter : string
        The type of interaction; valid values are 'zeeman', 'hyperfine', and
        'quadrupole'.
    coeff : np.ndarray
        The coefficient array used for the inversion of the spin Hamiltonian;
        calculated using the helper functions 'zeeman_sh_coeff',
        'hyperfine_sh_coeff', and 'quadrupole_sh_coeff'.
    
    Returns
    -------
    data : SHInteractionData

    """
    cpdef public str type
    cdef public int pro_data
    cdef public list terms
    cdef public SHTerm term
    cdef np.ndarray coeff
    cdef int level
    cdef cfl.zsh_inv_data *cfl_inv_data
    cdef public object inv_data_cap

    def __init__(self, d, inter, coeff, level):
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] a
        self.type = inter
        self.level = level
        self.pro_data = 0
        if inter == 'zeeman':
            self.terms = [SHTerm(d, 'zeeman_x'), SHTerm(d, 'zeeman_y'), SHTerm(d, 'zeeman_z')]
        else:
            self.term = SHTerm(d, inter)
        
        # Assign coeff to self to ensure there exists a reference to the coeff
        # memory for as long as this object exists. 
        self.coeff = np.asfortranarray(coeff, dtype=np.complex128)
        a = <np.ndarray[double complex, ndim=2, mode='fortran']> self.coeff
        self.cfl_inv_data = zsh_inv_data_alloc(&a[0,0], coeff.shape[0], coeff.shape[1])
        if self.cfl_inv_data == NULL:
            raise MemoryError("Failed to alloc inv_data memory")
        else:
            self.inv_data_cap = PyCapsule_New(<void *>self.cfl_inv_data, "pycfl.InvData", NULL)

    def __dealloc__(self):
        if self.cfl_inv_data != NULL:
            cfl.zsh_inv_data_free(self.cfl_inv_data)

    def set_pro_data(self, tensor):
        if self.type == 'zeeman':
            for i,t in enumerate(self.terms):
                t.set_pro_data(tensor[i], self.level)
        else:
            self.term.set_pro_data(tensor, self.level)
        self.pro_data = 1


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
    cdef public list interactions 
    cdef public list inter_data
    cpdef public int level
    cdef public int nsh
    cdef public int dsh
    cdef public int nobs
    cdef int nzeeman
    cpdef public float S_spin
    cpdef public list S_matel
    cpdef public float I_spin
    cpdef public list I_matel
    def __init__(self, interactions, **kwargs):
        if not isinstance(interactions, list):
            interactions = [interactions]
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list: '{}'.".format(i))
        self.interactions = interactions

        if 'level' not in kwargs:
            raise KeyError("Missing keyword argument 'level'.")
        self.level = kwargs['level']
           
        # Calculate matrix elements for the specified interactions.
        j_l = ['jx', 'jy', 'jz']
        if 'zeeman' in interactions or 'hyperfine' in interactions:
            try: 
                S_spin = kwargs['S']
            except KeyError:
                raise KeyError("Missing keyword argument S.")
            # Calculate the matrix elements of spin operator.
            S_matel = [None]*3
            for i in range(3):
                S_matel[i] = matel(j_l[i], S_spin)
            self.S_spin = S_spin
            self.S_matel = S_matel
        else:
            S_matel = None

        if 'hyperfine' in interactions or 'quadrupole' in interactions:
            try:
                I_spin = kwargs['I']
            except KeyError:
                raise KeyError("Missing keyword argument I.")
            # Calculate the matrix elements of nuclear spin operator.
            I_matel = [None]*3
            for i in range(3):
                I_matel[i] = matel(j_l[i], I_spin)

            self.I_spin = I_spin
            self.I_matel = I_matel
        else:
            I_matel = None

        # Determine spin Hamiltonian dimension.
        if 'zeeman' in interactions:
            if I_matel == None:
                # Only the zeeman interaction.
                dsh = 2*S_spin+1
            else:
                # Both the zeeman and quadrupole interactions.
                dsh = (2*S_spin+1) * (2*I_spin+1)
        elif S_matel == None:
            # Only the quadrupole interactions.
            dsh = 2*I_spin+1 
        else:
            # Contains hyperfine interactions.
            dsh = (2*S_spin+1) * (2*I_spin+1)
        
        self.dsh = dsh
        # Calculate the coefficient arrays and alloc spin Hamiltonian
        # interactions.
        self.inter_data = []
        self.nsh = 0
        self.nobs = 0
        if 'zeeman' in interactions:
            # Coefficient arrays are calculated for three B fields in \hat{x},
            # \hat{y}, and \hat{z} directions, respectively. 
            dz = 2*S_spin+1
            B_a = np.zeros([3, dz**2, 9], dtype = np.complex)
            for i in range(3):
                B_a[i, :, :] = zeeman_sh_coeff(np.eye(3,3)[i,:], S_matel)
            zeeman = SHInteractionData(dz, 'zeeman', np.reshape(B_a, (3 * dz**2, 9)), self.level)
            self.inter_data += [zeeman]
            self.nsh += 3
            # Three g-values plus three Euler rotation parameters.
            self.nobs += 6

        if 'hyperfine' in interactions:
            dh = 2*S_spin+1 + 2*I_spin+1
            hyperfine = SHInteractionData(dh, 'hyperfine', hyperfine_sh_coeff(I_matel, S_matel), self.level)
            self.inter_data += [hyperfine]
            self.nsh += 1
            # Three hyperfine values plus three Euler rotation parameters.
            self.nobs += 6

        if 'quadrupole' in interactions: 
            dq = 2*I_spin+1
            quadrupole = SHInteractionData(dq, 'quadrupole', quadrupole_sh_coeff(I_matel), self.level)
            self.inter_data += [quadrupole]
            self.nsh += 1
            # Two quadrupole values plus three Euler rotation parameters.
            self.nobs += 5

    
    def set_pro_data(self, interaction, tensor):
        r"""
        Set the projection data for a specific spin Hamiltonian interaction. 

        Parameters
        ----------
        interaction : string
            Valid options are 'zeeman', 'hyperfine', and 'quadrupole'. 
        tensor : list or Tensor
            For Zeeman interactions a list of three tensors corresponding to
            `\hat{x}`, `\hat{y}`, and `\hat{z}` interactions must be specified. 

        """
        for i in self.inter_data:
            if i.type == interaction:
                if interaction == 'zeeman':
                    if not isinstance(tensor, list):
                        raise ValueError("For Zeeman interactions tensor must be a list.")
                i.set_pro_data(tensor)
                return

        raise ValueError("This spin Hamiltonian object was not instantiated with {} "
            "interaction support.".format(interaction))

    
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
        cdef cfl.zshp_w shp_w
        cdef list shp_work_list = []
        cdef list shi_work_list = []
        cdef list result_list = []
        cdef np.ndarray[double complex, ndim=1, mode="c"] a
        cdef np.ndarray[double complex, ndim=1, mode="c"] cz
        cdef int cj
        cdef int z_num
        #FIXME: add check to make sure proj data has been added. 
        c_sh_tensors = []
        self.nzeeman = -1
        for i,inter in enumerate(self.inter_data):
            # Alloc projection and inversion workspace.
            if inter.type == 'zeeman':
                for t in inter.terms: 
                    shp_work_list += [PyCapsule_New(<void *>cfl.zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(
                        t.sh_cap, "pycfl.SHTerm")), "pycfl.SHCalcParamProWork", NULL)]
            else:
                shp_work_list += [PyCapsule_New(<void *>cfl.zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(
                    inter.term.sh_cap, "pycfl.SHTerm")), "pycfl.SHCalcParamProWork", NULL)]
            shi_work_list += [PyCapsule_New(<void *>cfl.zshi_w_alloc(<cfl.zsh_inv_data *>PyCapsule_GetPointer(
                inter.inv_data_cap, "pycfl.InvData")), "pycfl.SHCalcParamInvWork", NULL)]

            # Determine whether the complete Hamiltonian contains any
            # interactions that are also part of the spin Hamiltonian.
            # Furthermore, we record the location of the Zeeman tensor, if it
            # exists. 
            if not inter.pro_data:
                raise ValueError("The spin Hamiltonian interaction {} is missing projection data.".format(i.type))
            if inter.type == 'zeeman':
                for t in inter.terms:
                    if t.tensor in h:
                        c_sh_tensors += [t.tensor]
                self.nzeeman = i
            else:
                if inter.term.tensor in h:
                    c_sh_tensors += [inter.term.tensor]

        # Get Zeeman tensor.
        for i in self.inter_data:
            if i.type == 'zeeman':
                small_magz =  0.0001 * i.terms[2].tensor
                small_magz.name = 'MAGZ_small'
        
        tmp_h_coeff = h.coeff_dict
        tmp_h_coeff['MAGZ_small'] = 1
        h = Hamiltonian([small_magz] + h.tensors)
        h.set_coeff(tmp_h_coeff)
        
        # If required, replace h with a dedicated projection Hamiltonian.
        pro_tensors = []
        if len(c_sh_tensors) != 0:
            for i,t in enumerate(h):
                if t not in c_sh_tensors:
                    pro_tensors += [t]
            tmp_coeff = h.coeff_dict
            h = Hamiltonian(pro_tensors)
            h.set_coeff(tmp_coeff)

        # Diagonalize the complete Hamiltonian, then determine the sh terms and
        # finally do the inversion for each interaction of sh.
        (w, z) = h.diag()

        cz = <np.ndarray[double complex, ndim=1, mode="c"]> z.flatten()
        for i,inter in enumerate(self.inter_data):
            if inter.type == 'zeeman':
                # Since Zeeman interactions require three sh terms for inversion
                # we create a results array (a) big enough to hold the matrix
                # elements of three sh terms; then we fill a in three blocks.
                z_num = inter.terms[0].n**2
                a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(z_num*3, dtype=np.complex128)
                for j,t in enumerate(inter.terms):
                    cj = j
                    cfl.zshp(&a[cj*z_num], &cz[0], <cfl.zsh *>PyCapsule_GetPointer(t.sh_cap, "pycfl.SHTerm"),
                            <cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
            else:
                a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(inter.term.n**2, dtype=np.complex128)
                cfl.zshp(&a[0], &cz[0], <cfl.zsh *>PyCapsule_GetPointer(inter.term.sh_cap, "pycfl.SHTerm"), 
                        <cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
            # Do the inversion; we can directly pass on 'a' even in the Zeeman
            # case.
            cfl.zshi(&a[0], <cfl.zshi_w *>PyCapsule_GetPointer(shi_work_list[i], "pycfl.SHCalcParamInvWork"))
            result_list += [a[0:9].reshape(3,3)]

        for i in range(len(shp_work_list)):
            cfl.zshp_w_free(<cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
        for i in range(len(shi_work_list)):
            cfl.zshi_w_free(<cfl.zshi_w *>PyCapsule_GetPointer(shi_work_list[i], "pycfl.SHCalcParamInvWork"))
        
        return result_list


def parse_param_helper(parameters, h):
    r"""
    In the following, the word parameters in only used to refer to tensor
    coefficients which are varied during the fitting routine.  
    
    Ceate a list of initial parameter values using the coefficients of the
    provided Hamiltonian, and record their type.  The type determines whether we
    fit a real or complex parameter. 
    """
    param_list = []
    param_types = []
    n_p_real = 0

    for i,p in enumerate(parameters):
        if p not in h:
            raise ValueError("Tensor %s in parameters not found in h." % p.name)

        if not isinstance(h.coeff_dict[p.name], Number):
            raise ValueError("Element %s in coefficients is not a number." % p.name)
        # The parameter type is recorded such that any complex parameters
        # can be split into two real parameters.
        if isinstance(h.coeff_dict[p.name], complex):
            param_types.append('c')
            n_p_real += 2
        else:
            param_types.append('r')
            n_p_real += 1
        
        param_list += [h.coeff_dict[p.name]]

    return {'n_p_real': n_p_real, 'param_list' : param_list, 
            'param_types': param_types}


cdef class EFitRunner(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    energy level data. 

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
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
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
    cdef np.ndarray coeff
    cdef cfl.efit_data *efit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    
    def __init__(self, parameters, h, ex):
        cdef cfl.param_type *param_type_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        
        self.h = h
        self.n_p = len(parameters)
        self.parameters = parameters

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
        self.coeff = np.ascontiguousarray(h.coeff, dtype=np.complex128)
        coeff = <np.ndarray[double, ndim=1, mode="c"]> self.coeff     

        # Prepare experimental energy level data
        self.ex_e = np.ascontiguousarray(ex[:,1], dtype=np.float64)
        self.ex_li = np.ascontiguousarray(ex[:,0], dtype=np.int32)
       
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
        for i in range(self.n_p):
            param_array[i] = <cfl.param_type *> malloc(cython.sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i-1):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))
            
            param_array[i].type = cfl.atoi(self.param_types[i])
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
                &coeff[0], self.ex_data, self.n_p, self.param_array);
        self.fit_data_cap = PyCapsule_New(<void *>self.efit_data, "pycfl.MinData", NULL)
        self.obj_f_cap = PyCapsule_New(<void *>&cfl.efit_obj, "pycfl.MinObjF", NULL)
        self.cov_f_cap = PyCapsule_New(<void *>&cfl.efit_cov, "pycfl.MinCovF", NULL)
        
        # Run eshfit_chi2 so that the initial chi^2 weighting is set.
        chi2 = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(1)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        cfl.efit_chi2(&x[0], self.efit_data, &chi2[0])

    def __dealloc__(self):
        if self.ex_data != NULL:
            free(self.ex_data)
        if self.param_array != NULL:
            for i in range(self.n_p):
                if self.param_array[i] != NULL:
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
        
        coeff = self.coeff 
        ri = 0
        
        for i,p in enumerate(self):
            if (self.param_types[i] == 'c'): 
                coeff[self.h.index(p)] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[self.h.index(p)] = x[ri]
                ri += 1
        
        return(coeff, fmin)



cdef class ESHFitRunner(object):
    r"""
    Class used to store data required by, and to run, a crystal field fit using
    energy level and spin Hamiltonian data.

    The Hamiltonian must have coefficients set with set_coeff, since these are
    used as initial estimates for the parameters to-be-fit.  The type of
    coefficients when the are set also determines whether they are fit as real
    or complex parameters. 

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor. 
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    ex : np.ndarray
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.
    """
    cdef SpinHamiltonian sh
    cdef public Hamiltonian h
    cdef Hamiltonian hpro
    cdef int n_p
    cdef public list parameters
    cpdef public int n_p_real
    cpdef public list param_list
    cpdef public list param_types
    cdef int nzeeman
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef cfl.zsh **sh_array
    cdef cfl.shx_data **shx_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef list shx_list
    cdef dict weights
    cdef np.ndarray p0_real
    cdef np.ndarray coeff
    cdef cfl.eshfit_data *eshfit_data
    cpdef public object obj_f_cap
    cpdef public object cov_f_cap
    cpdef public object fit_data_cap
    # Set level when SH is instantiated. Tensors can either be set manually with
    # current set_pro interface, or, will also be set automatically by eshfit.
    # This means one only has to specify them once for multiple spin
    # Hamiltonians, and also provides a convenient place to specify MAGZ under
    # all circumstances. 
    def __init__(self, parameters, sh_tensors, h, sh, ex, shx, weights):
        cdef cfl.param_type *param_type_ptr
        cdef cfl.zsh *zsh_array_ptr
        cdef cfl.shx_data *shx_data_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double complex, ndim=1, mode="c"] shx_pa
        cdef np.ndarray[double, ndim=1, mode="c"] chi2
        cdef np.ndarray[double, ndim=1, mode="c"] x
        
        self.n_p = len(parameters)
        self.parameters = parameters
        self.sh = sh

        sh_tensor_dict = {}
        for t in sh_tensors:
            sh_tensor_dict[t.name] = t
        
        # Determine whether the complete Hamiltonian contains any interactions
        # that are also part of the spin Hamiltonian.  Furthermore, we set the
        # spin Hamiltonian projection tensors and we record the index of the
        # Zeeman tensor, if it exists. 
        c_sh_tensors = []
        pro_data_tensors = []
        self.nzeeman = -1
        for i,inter in enumerate(sh.inter_data):
            if inter.type == 'zeeman':
                try:
                    inter.set_pro_data([sh_tensor_dict['MAGX'], sh_tensor_dict['MAGY'], sh_tensor_dict['MAGZ']])
                except KeyError:
                    raise ValueError("Missing a Zeeman tensor from the sh_tensors_list.")
                for t in inter.terms:
                    if t.tensor in h:
                        c_sh_tensors += [t.tensor]
                self.nzeeman = i
            else:
                if inter.type == 'hyperfine':
                    try:
                        inter.set_pro_data(sh_tensor_dict['AHYP'])
                    except KeyError:
                        raise ValueError("Missing hyperfine tensor from the sh_tensors_list.")
                elif inter.type == 'quadrupole':
                    try:
                        inter.set_pro_data(sh_tensor_dict['EQHYP'])
                    except KeyError:
                        raise ValueError("Missing quadrupole tensor from the sh_tensors_list.")
                if inter.term.tensor in h:
                    c_sh_tensors += [inter.term.tensor]
       
        # Add small magnetic field along the \hat{z} direction to corretly order
        # S=1/2 and S=-1/2 states.  Since we have already determined
        # c_sh_tensors (list of tensors in the complete h that are also present
        # in sh), the small MAGZ term will always be added to hpro even if the
        # zeeman term is present.
        try:
            magz_small = 0.0001 * sh_tensor_dict['MAGZ']
            magz_small.name = 'MAGZ_small'
        except KeyError:
            raise ValueError("Missing 'MAGZ' from the sh_tensors list; 'MAGZ' is always "
                    "required, since it is used to distinguish S=+1/2 and S=-1/2 states.")
        
        if 'MAGZ_small' not in h.coeff_dict:
            tmp_coeff = h.coeff_dict
            tmp_coeff['MAGZ_small'] = 1
            self.h = Hamiltonian([magz_small] + h.tensors)
            self.h.set_coeff(tmp_coeff)
        else:
            self.h = h

        # Generate a dedicated projection Hamiltonian, if required. 
        pro_tensors = []
        if len(c_sh_tensors) != 0:
            for i,t in enumerate(self.h):
                if t not in c_sh_tensors:
                    pro_tensors += [t]
            self.hpro = Hamiltonian(pro_tensors)
            self.hpro.set_coeff(self.h.coeff_dict)
        else:
            self.hpro = None

        if self.n_p_real > len(ex) + sh.nsh:
            raise ValueError("The total (real and imaginary) number of parameters "
                "exceeds the number of observables.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        pp = parse_param_helper(parameters, self.h)
        self.n_p_real = pp['n_p_real']
        self.param_list = pp['param_list']
        self.param_types = pp['param_types']

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        self.coeff = np.ascontiguousarray(self.h.coeff, dtype=np.complex128)
        coeff = <np.ndarray[double, ndim=1, mode="c"]> self.coeff
       
        # Prepare experimental energy level data
        self.ex_e = np.ascontiguousarray(ex[:,1], dtype=np.float64)
        self.ex_li = np.ascontiguousarray(ex[:,0], dtype=np.int32)
       
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
        for i in range(self.n_p):
            param_array[i] = <cfl.param_type *> malloc(cython.sizeof(cfl.param_type))
            if param_array[i] is NULL:
                for j in range(i-1):
                    free(param_array[j])
                free(self.ex_data)
                free(self.param_array)
                raise MemoryError("param_array[{}] alloc failed".format(i))

            param_array[i].type = cfl.atoi(self.param_types[i])
            param_array[i].index = self.h.index(parameters[i])

            if self.param_types[i] == 'c':
                self.p0_real[ip_real] = np.real(self.param_list[i])
                self.p0_real[ip_real+1] = np.imag(self.param_list[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] =  self.param_list[i]
                ip_real += 1

        # Array of spin Hamiltonians and experimental spin Hamiltonian data.
        sh_array = <cfl.zsh **>malloc(sh.nsh*cython.sizeof(zsh_array_ptr))
        if (sh_array == NULL):
            for i in range(self.n_p):
                free(param_array[i])
            free(self.ex_data)
            free(param_array)
            raise MemoryError("sh_array alloc failed")
        self.sh_array = sh_array
        self.weights = weights
        shx_array = <cfl.shx_data **>malloc(len(sh.inter_data)*cython.sizeof(shx_data_ptr))
        if shx_array == NULL:
            for i in range(self.n_p):
                free(param_array[i])
            free(self.ex_data)
            free(param_array)
            free(sh_array)
            raise MemoryError("shx_array alloc failed")
        self.shx_list = []
        self.shx_array = shx_array
        j = 0
        for i,inter in enumerate(sh.inter_data):
            if inter.type not in shx:
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(sh_array)
                free(shx_array)
                raise ValueError("The spin Hamiltonian experimental data dictonary "
                        "is missing data for the {} interaction.".format(inter.type))
            elif not isinstance(shx[inter.type], np.ndarray):
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(sh_array)
                free(shx_array)
                raise TypeError("exp_tensor must be a np.ndarray.")
            elif shx[inter.type].shape == (3, 3):
                self.shx_list += [np.ascontiguousarray(shx[inter.type].flatten(), dtype=np.complex128)]
            elif shx[inter.type].shape == (9,):
                self.shx_list += [np.ascontiguousarray(shx[inter.type], dtype=np.complex128)]
            else:
                for j in range(i):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(sh_array)
                free(shx_array)
                raise ValueError("exp_tensor must either be a (3, 3) or (9, 1) array.")
            if inter.type == 'zeeman':
                for j,t in enumerate(inter.terms):
                    sh_array[i+j] = <cfl.zsh *>PyCapsule_GetPointer(t.sh_cap, "pycfl.SHTerm")
            else:
                sh_array[i+j] = <cfl.zsh *>PyCapsule_GetPointer(inter.term.sh_cap, "pycfl.SHTerm")
            
            shx_array[i] = <cfl.shx_data *>malloc(cython.sizeof(cfl.shx_data))
            if shx_array[i] == NULL:
                for j in range(i-1):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(sh_array)
                free(shx_array)
                raise MemoryError("shx_array[{}] alloc failed".format(i))
            shx_pa = <np.ndarray[double complex, ndim=1, mode="c"]> self.shx_list[i]
            shx_array[i].pa = &shx_pa[0]
            shx_array[i].inv_data = <cfl.zsh_inv_data *>PyCapsule_GetPointer(inter.inv_data_cap, "pycfl.InvData")
            shx_array[i].chisq_weight = 1

        # Alloc data for objective functions and estimate initial chi^2 values. 
        chi2 = <np.ndarray[double, ndim=1, mode="c"]> np.zeros(len(sh.interactions)+1)
        x = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real


        if (self.hpro != None):
            self.eshfit_data = cfl.eshfit_data_alloc(sh_array, sh.nsh, self.nzeeman,
                    <cfl.zh*>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"),
                    <cfl.zh *>PyCapsule_GetPointer(self.hpro.h_cap, "pycfl.Hamiltonian"), &coeff[0],
                    self.ex_data, shx_array, self.n_p, self.param_array)
            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_obj, "pycfl.MinObjF", NULL)
            self.cov_f_cap = PyCapsule_New(<void *>&cfl.eshfit_hpro_cov, "pycfl.MinCovF", NULL)
            
            # Unweighted initial chi^2 estimation.
            cfl.eshfit_hpro_chi2(&x[0], self.eshfit_data, &chi2[0])

        else:
            self.eshfit_data = cfl.eshfit_data_alloc(sh_array, sh.nsh, self.nzeeman,
                    <cfl.zh*>PyCapsule_GetPointer(self.h.h_cap, "pycfl.Hamiltonian"), NULL, &coeff[0], 
                    self.ex_data, shx_array, self.n_p, self.param_array)
            self.obj_f_cap = PyCapsule_New(<void *>&cfl.eshfit_obj, "pycfl.MinObjF", NULL)
            self.cov_f_cap = PyCapsule_New(<void *>&cfl.eshfit_cov, "pycfl.MinCovF", NULL)
            
            # Unweighted initial chi^2 estimation.
            cfl.eshfit_chi2(&x[0], self.eshfit_data, &chi2[0])

        self.fit_data_cap = PyCapsule_New(<void *>self.eshfit_data, "pycfl.MinData", NULL)

        self.weights[inter.type]

        # Energy levels are always weighted to unity provided a call to
        # eshfit_hpro_chi2 or eshfit_chi2 has been made. 
        if 'e' in self.weights:
            ew_scale = 1/self.weights['e']
        else:
            ew_scale = 1

        for i,inter in enumerate(sh.inter_data):
            shx_array[i].chisq_weight = self.weights[inter.type]/chi2[i+1] * ew_scale 

    def __dealloc__(self):
        if self.ex_data != NULL:
            free(self.ex_data)
        if self.param_array != NULL:
            for i in range(self.n_p):
                if self.param_array[i] != NULL:
                    free(self.param_array[i])
            free(self.param_array)
        if self.sh_array != NULL:
            free(self.sh_array)
        if self.shx_array != NULL:
            for i in range(len(self.sh.inter_data)):
                if self.shx_array[i] != NULL:
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
        
        coeff = self.coeff 
        ri = 0
        for i,p in enumerate(self):
            if (self.param_types[i] == 'c'): 
                coeff[self.h.index(p)] = np.complex(x[ri], x[ri+1])
                ri += 2
            else:
                coeff[self.h.index(p)] = x[ri]
                ri += 1


        return(coeff, fmin)

cdef class CFLMin:
    r"""
    Object for initializing and configuring minimization routines to be passed
    to e_fit or esh_fit.

    Parameters
    ----------
    method : string
        The minimization routine to employ.
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

    cov : bool, optional
        Evaluate the covariance matrix for the fit; defaults to False.
    bounds : dict, optional
        Parameter bounds.  Keys specify the tensor name (note that tensors
        created by tensor arithmethic should have their name attribute set
        explicitly), while values correspond to tuples, the first entry of which
        is the lower bound and the second entry the upper bound.  The number of
        elements in bounds must match the length of the parameters list. 
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
        cdef int naccept
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
            if len(fit_obj.param_list) != len(self.kwargs['bounds']):
                raise ValueError("The number of provided bounds does not match the "
                        "number of provided parameters.")
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
                    if fit_obj.h.coeff_dict[p.name] < lb[rpi]:
                        raise ValueError("The %s coefficient in the Hamiltonian is "
                                "less than the specified lower bound." % p.name)
                    elif fit_obj.h.coeff_dict[p.name] > ub[rpi]:
                        raise ValueError("The %s coefficient in the Hamiltonian is "
                                "greater than the specified lower bound." % p.name)
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
                if len(fit_obj.param_list) != len(self.kwargs['stepsize']):
                    raise ValueError("The of elements of stepsize does not match the "
                            "number of provided parameters.")
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
        elif self.method == 'nlopt_crs2_lm':
            lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_crs2_lm,
                    cxtol, self.cfl_bounds)
        elif self.method == 'nlopt_esch':
            lmin_obj = cfl_nlopt_min_setup(obj_f_ptr, cov_f_ptr, cnx, data_ptr, nlopt_esch,
                    cxtol, self.cfl_bounds)

        cx0 = <np.ndarray[double, ndim=1, mode="c"]> x0
        
        with nogil:
            naccept = cfl.cfl_min(&cx0[0], &fmin, cov_ptr, min_obj)
        
        self.kwargs['naccept'] = naccept

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
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    cfl_min : CFLMin
        The minimization object which sets the optimization algorithm and
        corresponding options.
    """
    efit = EFitRunner(parameters, h, ex)
    (x, fmin) = efit.fit(cfl_min)
    h.coeff = x
    (w, z) = h.diag()

    # The number of degrees of freedom of the chi-squared distribution
    ndof = len(ex)-len(parameters)

    # Generate labels and run gen_e_summary directly.  We do this rather than
    # call h.gen_summary, since public methods can't have optional arguments in
    # cython, so one can't include experimental data there.
    cdef cfl.zh *cflh = <cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian")
    labels = [] 
    for i in range(h.n):
        labels += [cflh.states.states[i]]

    summary = "=============\n"
    summary+= "e_fit summary\n"
    summary+= "=============\n\n"
    summary += gen_e_summary(w, z, labels, ex, ndof=ndof)
    summary += "\n"
    summary += gen_fit_summary(x, efit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'coeff': x, 'summary': summary}


def esh_fit(parameters, sh_tensors, h, sh, ex, shx, weights, cfl_min):
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
    sh_tensors : list
        A list of tensor objects for which to project spin Hamiltonian terms. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit. 
    ex : np.ndarray
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    weights : dict
        Set the weighting for `\chi^2` contributions of terms to be fit.  Valid
        keys are 'energy', 'zeeman', 'hyperfine', and 'quadrupole';
        corresponding values should be floats.
    cfl_min : CFLMin 
        The minimization object which sets the optimization algorithm and
        corresponding options.
    """
    eshfit = ESHFitRunner(parameters, sh_tensors, h, sh, ex, shx, weights)
    (x, fmin) = eshfit.fit(cfl_min)
    eshfit.h.coeff = x
    (w, z) = eshfit.h.diag()
    
    # The number of degrees of freedom of the chi-squared distribution
    ndof = len(ex) + sh.nobs - len(parameters)

    # Generate labels and run gen_e_summary directly.  We do this rather than
    # call h.gen_summary, since public methods can't have optional arguments in
    # cython, so one can't include experimental data there.
    cdef cfl.zh *cflh = <cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian")
    labels = [] 
    for i in range(h.n):
        labels += [cflh.states.states[i]]

    summary = "===============\n"
    summary+= "esh_fit summary\n"
    summary+= "===============\n\n"
    summary += gen_e_summary(w, z, labels, ex, ndof=ndof)
    summary += "\n"
    summary += gen_sh_summary(sh.calc_param(h), sh, shx, ndof=ndof)
    summary += "\n"
    summary += gen_fit_summary(x, eshfit, cfl_min.method, fmin, **cfl_min.kwargs)

    return {'coeff': x, 'summary': summary}

