# filename = pycfl.pyx
#cython: c_string_encoding=ascii

# Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
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



cimport cfl, cython
cimport numpy as np
import numpy as np
from numbers import Number
from cpython.pycapsule cimport *
from python_ref cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from matel import matel


# TODO: 
#       + Implement direct calls to zshi and zshp
#       + Apply a small magnetic field along z, to obtain state labels. Maybe
#         something to directly implement in the cfl projection interface. 

cdef class Tensor:
    r"""
    The Tensor class provides an interface for the creation of cfl zt objects.
    They are employed for the creation of both complete Hamiltonians and the
    projection of spin Hamiltonian interactions from complete Hamiltonians.
    Objects of type Tensor support standard arithmetic operations and can be
    added, subtracted, and scaled to yield new Tensor objects.

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
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, char *name, np.ndarray[double complex, ndim=2, mode='c']a,
            object data_tuple=None):
        cdef cfl.zt *t
        cdef cfl.zt *t1
        cdef cfl.zt *t2

        self.name = <str> name

        if (data_tuple == None):
            n = a.shape[0]
            self.n = n
            t = cfl.zt_alloc(name, &a[0,0], n)
            
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
        return Tensor(<char *>t1.tmp_name, np.array([[]],dtype=np.complex128), data_tuple=d) 

    def __sub__(t1, t2):
        if not (isinstance(t1, Tensor) and isinstance(t2, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        t1.tmp_name = "{0}-{1}".format(t1.name, t2.name)
        d = (t1, t2, -1)
        return Tensor(<char *>t1.tmp_name, np.array([[]],dtype=np.complex128), data_tuple=d) 

    def __mul__(x, y):
        if isinstance(x, Number):
            if isinstance(y, Tensor):
                y.tmp_name = "{0:.2f}x{1}".format(x, y)
                d = (y, x)
                return Tensor(<char *>y.tmp_name, np.array([[]],dtype=np.complex128), data_tuple=d)
        elif isinstance(x, Tensor):
            if isinstance(y, Number):
                x.tmp_name = "{0:.2f}x{1}".format(y, x)
                d = (x, y)
                return Tensor(<char *>x.tmp_name, np.array([[]],dtype=np.complex128), data_tuple=d)
        else:
            raise TypeError("Tensors can only be multiplied by scalar numbers")


cdef class Hamiltonian:
    r"""
    The crystal field Hamiltonian class.  Creates a cfl zh object and provides
    an interface for diagonalizing zh.

    Parameters
    ----------
    tensors : list
        A list with components of type Tensor; this specifies the type of
        interactions modeled by the Hamiltonian.
    states : list
        A list of strings, which specify the state labels of the Hamiltonian.
        This may be depreciated in favour of Tensor state labeling.

    Returns
    -------
    h : Hamiltonian

    """
    cdef cfl.zh *cfl_zh
    cdef cfl.zt **tensor_array
    cdef char **state_labels
    cdef public list states
    cdef int n
    cdef int nt
    cdef list tensors
    cdef public np.ndarray coeff
    cdef public np.ndarray w
    cdef public np.ndarray z
    cdef public object h_cap
    def __cinit__(self, tensors, states):
        cdef char *char_ptr
        cdef cfl.zt *ten_array_ptr

        n = tensors[0].n
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
        self.states = states
                
        # Create array of tensors and array of character arrays to be passed to
        # the zh_set cfl function. 
        tensor_array = <cfl.zt **>malloc(len(tensors)*cython.sizeof(ten_array_ptr))
        if tensor_array is NULL:
            raise MemoryError("tensor_array alloc failed")
        state_labels = <char **>malloc(len(states)*cython.sizeof(char_ptr))
        if state_labels is NULL:
            free(tensor_array)
            raise MemoryError("state_labels alloc failed")
        
        self.state_labels = state_labels
        self.tensor_array = tensor_array

        for i,s in enumerate(states):
            state_labels[i] = s

        for i,t in enumerate(tensors):
            tensor_array[i] = <cfl.zt *> PyCapsule_GetPointer(t.t_cap, "pycfl.Tensor")

        # Allocate storage for zh. 
        self.cfl_zh = cfl.zh_alloc(n, self.nt, state_labels, tensor_array)
        if self.cfl_zh is NULL:
            free(tensor_array)
            free(state_labels)
            raise MemoryError("cfl_zh alloc failed")
        else:
            self.h_cap = PyCapsule_New(<void *>self.cfl_zh, "pycfl.Hamiltonian", NULL)

    def __dealloc__(self):
        if self.cfl_zh is not NULL:
            cfl.zh_free(self.cfl_zh)

        if self.state_labels is not NULL:
            free(self.state_labels)

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
        coeff : np.ndarray
            The order of elements in the array must match the order in which
            tensors were specfied when the Hamiltonian object was instantiated.
            dtype should be np.complex128.

        """
        cdef np.ndarray[double complex, ndim=1, mode='c'] co

        self.coeff = coeff
        co = <np.ndarray[double complex, ndim=1, mode='c']> self.coeff
        cfl.zh_set_coeff(self.cfl_zh, &co[0])
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
        
        hd_w = cfl.zhd_w_alloc(self.cfl_zh)
        if hd_w is NULL:
            free(self.tensor_array)
            free(self.state_labels)
            cfl.zh_free(self.cfl_zh)
            raise MemoryError("hd_w alloc failed")

        self.w = np.ascontiguousarray(np.zeros(self.n), dtype=np.float64)
        self.z = np.ascontiguousarray(np.zeros(self.n*self.n).reshape((self.n,self.n)), dtype=np.complex128)
        w = <np.ndarray[double, ndim=1, mode="c"]> self.w
        z = <np.ndarray[double complex, ndim=2, mode="c"]> self.z

        with nogil:
            cfl.zhd(&w[0], &z[0,0], h, hd_w)

        cfl.zhd_w_free(hd_w)
        return (w, z)


cpdef zeeman_sh_coeff(v, t):
    r"""
    Generate the Zeeman interaction spin Hamiltonian `coefficient array`.  This
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
    Generate the hyperfine interaction spin Hamiltonian `coefficient array`.
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
    Generate the quadrupole interaction spin Hamiltonian `coefficient array`.
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
    Data storage for a single spin Hamiltonian term. 

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
        cfl.zsh_set_pro(self.cfl_sh, <cfl.zt *>PyCapsule_GetPointer(tensor.t_cap, "pycfl.Tensor"), l)
        self.tensor = tensor

        
cdef class SHTermData(object):
    r"""
    Class used to store spin Hamiltonian data for a single interaction. 

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
    term_data : SHTermData

    """
    cpdef public str type
    cdef public int pro_data
    cdef public int exp_data
    cdef public list terms
    cdef public SHTerm term
    cdef np.ndarray coeff
    cdef cfl.zsh_inv_data *cfl_inv_data
    cdef public object inv_data_cap
    cdef public np.ndarray exp_tensor
    cdef public float chisq_weight

    def __init__(self, d, inter, coeff):
        cdef np.ndarray[double complex, ndim=2, mode="fortran"] a
        self.type = inter
        self.pro_data = 0
        self.exp_data = 0
        if inter == 'zeeman':
            self.terms = [SHTerm(d, 'zeeman_x'), SHTerm(d, 'zeeman_y'), SHTerm(d, 'zeeman_z')]
        else:
            self.term = SHTerm(d, inter)
        
        # Assign coeff to self, to ensure there exists a reference to the coeff
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

    def set_pro_data(self, tensor, l):
        if self.type == 'zeeman':
            for i,t in enumerate(self.terms):
                t.set_pro_data(tensor[i], l)
        else:
            self.term.set_pro_data(tensor, l)
        self.pro_data = 1

    def set_exp_data(self, exp_tensor, chisq_weight):
        self.exp_tensor = exp_tensor
        self.chisq_weight = chisq_weight
        self.exp_data = 1


cdef class SpinHamiltonian:
    r""" 
    Container for holding data about the spin Hamiltonian. 

    Parameters
    ----------
    interactions : list
        Elements are strings which specify the interactions of the spin
        Hamiltonian.  Possible values are: 'zeeman', 'hyperfine', and
        'quadrupole'.  
    B : numpy.ndarray 
        A `3` by `1` vector containing values for the magnetic field strengths
        `B_x`, `B_y` and `B_z`; if ``terms`` contains 'zeeman' this keyword
        argument must be specified.  
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
    cpdef public np.ndarray B
    cpdef public float S_spin
    cpdef public list S_matel
    cpdef public float I_spin
    cpdef public list I_matel
    cdef public list interactions 
    cdef public int nsh
    cdef public int dsh
    cdef int nzeeman
    def __init__(self, interactions, **kwargs):
        if not isinstance(interactions, list):
            interactions = [interactions]
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list: '{}'.".format(i))

        if 'zeeman' in interactions:
            try:
                B = kwargs['B']
                self.B = B
            except KeyError:
                raise ValueError("Missing keyword argument B.")
        else:
            B = None
           
        # Calculate matrix elements for the specified interactions.
        j_l = ['jx', 'jy', 'jz']
        if 'zeeman' in interactions or 'hyperfine' in interactions:
            try: 
                S_spin = kwargs['S']
            except KeyError:
                raise ValueError("Missing keyword argument S.")
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
                raise ValueError("Missing keyword argument I.")
            # Calculate the matrix elements of nuclear spin operator.
            I_matel = [None]*3
            for i in range(3):
                I_matel[i] = matel(j_l[i], I)

            self.I_spin = I_spin
            self.I_matel = I_matel
        else:
            I_matel = None

        # Determine spin Hamiltonian dimension.
        if B != None:
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
        # Calculate the cofficient arrays and alloc spin Hamiltonian
        # interactions.
        self.interactions = []
        self.nsh = 0
        if 'zeeman' in interactions:
            # Coefficient arrays are calculated for three B fields; the user
            # specified B-direction is ignored for inversion. 
            dz = 2*S_spin+1
            B_a = np.zeros([3, dz**2, 9], dtype = np.complex)
            for i in range(3):
                B_a[i, :, :] = zeeman_sh_coeff(np.eye(3,3)[i,:], S_matel)
            zeeman = SHTermData(dz, 'zeeman', np.reshape(B_a, (3 * dz**2, 9)))
            self.interactions += [zeeman]
            self.nsh += 3

        if 'hyperfine' in interactions:
            dh = 2*S_spin+1 + 2*I_spin+1
            hyperfine = SHTermData(dh, 'hyperfine', hyperfine_sh_coeff(I_matel, S_matel))
            self.interactions += [hyperfine]
            self.nsh += 1

        if 'quadrupole' in interactions: 
            dq = 2*I_spin+1
            quadrupole = SHTermData(dq, 'quadrupole', quadrupole_sh_coeff(I_matel))
            self.interactions += [quadrupole]
            self.nsh += 1

    
    def set_pro_data(self, interaction, tensor, level):
        r"""
        Set the projection data for a specific spin Hamiltonian interaction. 

        Parameters
        ----------
        interaction : string
            Valid options are 'zeeman', 'hyperfine', and 'quadrupole'. 
        tensor : list or Tensor
            For Zeeman interactions a list of three tensors corresponding to
            `\hat{x}`, `\hat{y}`, and `\hat{z}` interactions must be specified. 
        level : int
            The level of the complete Hamiltonian for which to project the spin
            Hamiltonian.
        """
        for i in self.interactions:
            if i.type == interaction:
                if interaction == 'zeeman':
                    if not isinstance(tensor, list):
                        raise ValueError("For Zeeman interactions tensor must be a list.")
                i.set_pro_data(tensor, level)
                return

        raise ValueError("This spin Hamiltonian object was not instantiated with {} "
            "interaction support.".format(interaction))

    
    def set_exp_data(self, interaction, exp_tensor, chisq_weight):
        r"""
        Set the inversion data for a specific spin Hamiltonian interaction. 

        Parameters
        ----------
        interaction : string
            Valid options are 'zeeman', 'hyperfine', and 'quadrupole'. 
        exp_tensor : np.ndarray
            One by nine dimensional array of dtype=np.complex128 specifying the
            experimental spin Hamiltonian tensor values.
        chisq_weight : float
            The weighting applied to the chi_square fit for this interaction.
        """
        for i in self.interactions:
            if i.type == interaction:
                if not isinstance(exp_tensor, np.ndarray):
                    raise TypeError("exp_tensor must be a np.ndarray.")
                if exp_tensor.shape == (3, 3):
                    i.set_exp_data(np.ascontiguousarray(exp_tensor.flatten(), dtype=np.complex128), chisq_weight)
                elif exp_tensor.shape == (9,):
                    i.set_exp_data(np.ascontiguousarray(exp_tensor, dtype=np.complex128), chisq_weight)
                else: 
                    raise ValueError("exp_tensor must either be a (3, 3) or (9, 1) array.")
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
            Elements are nd.arrays corresponding spin Hamiltonian tensors of
            interactions specified when the spin Hamiltonian object was
            instantiated. 
        """
        cdef cfl.zshp_w shp_w
        cdef list shp_work_list = []
        cdef list shi_work_list = []
        cdef list result_list = []
        cdef np.ndarray[double complex, ndim=1, mode="c"] a
        cdef np.ndarray[double complex, ndim=2, mode="c"] cz
        cdef int cj
        cdef int z_num
        
        # Alloc projection and inversion workspace.
        for inter in self.interactions:
            if inter.type == 'zeeman':
                for t in inter.terms: 
                    shp_work_list += [PyCapsule_New(<void *>cfl.zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(
                        t.sh_cap, "pycfl.SHTerm")), "pycfl.SHCalcParamProWork", NULL)]
            else:
                shp_work_list += [PyCapsule_New(<void *>cfl.zshp_w_alloc(<cfl.zsh *>PyCapsule_GetPointer(
                    inter.term.sh_cap, "pycfl.SHTerm")), "pycfl.SHCalcParamProWork", NULL)]
            shi_work_list += [PyCapsule_New(<void *>cfl.zshi_w_alloc(<cfl.zsh_inv_data *>PyCapsule_GetPointer(
                inter.inv_data_cap, "pycfl.InvData")), "pycfl.SHCalcParamInvWork", NULL)]

        # Determine whether there are any second order tensors; that is, whether
        # the complete Hamiltonian contains any interactions that are also part
        # of the spin Hamiltonian.  Furthermore, we record the location of the
        # Zeeman tensor. 
        so_tensors = []
        self.nzeeman = -1
        for i,inter in enumerate(self.interactions):
            if not inter.pro_data:
                raise ValueError("The spin Hamiltonian interaction {} is missing projection data.".format(i.type))
            if inter.type == 'zeeman':
                for t in inter.terms:
                    if t.tensor in h:
                        so_tensors += [t.tensor]
                self.nzeeman = i
            else:
                if inter.term.tensor in h:
                    so_tensors += [inter.term.tensor]
           
        # If required, replace the h with the first order Hamiltonian
        fo_tensors = []
        if len(so_tensors) != 0:
            for i,t in enumerate(h):
                if t not in so_tensors:
                    fo_tensors += [t]
            h = Hamiltonian(fo_tensors, h.states)

        # Diagonalize the complete Hamiltonian, then determine the sh terms and
        # finally do the inversion for each interaction of sh.
        (w, z) = h.diag()
        cz = <np.ndarray[double complex, ndim=2, mode="c"]> z
        for i,inter in enumerate(self.interactions):
            if inter.type == 'zeeman':
                # Since Zeeman interactions require three sh terms for inversion
                # we create a results array (a) big enough to hold the matrix
                # elements of three sh terms; then we fill a in three blocks.
                z_num = inter.terms[0].n**2
                a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(z_num*3, dtype=np.complex128)
                for j,t in enumerate(inter.terms):
                    cj = j
                    cfl.zshp(&a[cj*z_num], &cz[0,0], <cfl.zsh *>PyCapsule_GetPointer(t.sh_cap, "pycfl.SHTerm"),
                            <cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
            else:
                a = <np.ndarray[double complex, ndim=1, mode="c"]> np.zeros(inter.term.n**2, dtype=np.complex128)
                cfl.zshp(&a[0], &cz[0,0], <cfl.zsh *>PyCapsule_GetPointer(inter.term.sh_cap, "pycfl.SHTerm"), 
                        <cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
            # Do the inversion; we can directly pass on 'a' even in the Zeeman
            # case, since the coefficient matrix in zsh_inv_data is of
            # appropriate dimension to solve using three Zeeman terms.
            cfl.zshi(&a[0], <cfl.zshi_w *>PyCapsule_GetPointer(shi_work_list[i], "pycfl.SHCalcParamInvWork"))
            result_list += [a[0:9].reshape(3,3)]

        for i in range(len(shp_work_list)):
            cfl.zshp_w_free(<cfl.zshp_w *>PyCapsule_GetPointer(shp_work_list[i], "pycfl.SHCalcParamProWork"))
        for i in range(len(shi_work_list)):
            cfl.zshi_w_free(<cfl.zshi_w *>PyCapsule_GetPointer(shi_work_list[i], "pycfl.SHCalcParamInvWork"))
        
        return result_list



cdef class EFitRunner(object):
    """
    Class used to store data required by, and to run, a crystal field fit using
    energy level data. 

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    coeff_list : list
        Coefficients for tensors.  Values in ``coeff_list`` that correspond to
        parameters to be fit, that is, that are specified in ``parameters``, are
        used as initial values in the fitting process.  Furthermore, the type of
        elements in ``coeff_list`` determines whether only the real, or both the
        real and the imaginary, components of the corresponding prefactors are
        varied by the fitting routine.
    ex : np.ndarray
        2 by n dimensional array, with n the number available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    """
    cdef Hamiltonian h
    cdef int n_p
    cdef int n_p_real
    cdef list param_types
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef np.ndarray p0_real
    cdef np.ndarray coeff
    cdef cfl.efit_data *efit_data

    def __init__(self, parameters, h, coeff_list, ex):
        cdef cfl.param_type *param_type_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li

        self.h = h
        self.n_p = len(parameters)

        param_indices = []
        param_types = []
        n_p_real = 0
        for i,p in enumerate(parameters):
            # Ensure h contains all the listed tensors
            try:
                pi = h.index(p)
            except ValueError:
                raise ValueError("Tensor {} in param_list not found in h".format(p.name))
            # Work out which tensor each element in param_list corresponds to,
            # determine whether it is real or complex, and split any complex
            # parameters to create a purley real initial value array.
            if not isinstance(coeff_list[pi], Number):
                raise ValueError("Element {} in coefficients is not a number.".format(coeff_list[pi]))
            if isinstance(coeff_list[pi], complex):
                param_types.append('c')
                n_p_real += 2
            else:
                param_types.append('r')
                n_p_real += 1
            param_indices.append(pi)
        
        self.n_p_real = n_p_real
        self.param_types = param_types
        if n_p_real > len(ex):
            raise ValueError("The total (real and imaginary) number of parameters exceeds "
                    "the number of observables. Don't do that.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        self.coeff = np.ascontiguousarray(np.array(coeff_list), dtype=np.complex128)
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
        self.p0_real = np.ascontiguousarray(np.zeros(n_p_real), dtype=np.float64)
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

            param_array[i].type = cfl.atoi(param_types[i])
            param_array[i].index = param_indices[i]

            if param_types[i] == 'c':
                self.p0_real[ip_real] = np.real(coeff_list[i])
                self.p0_real[ip_real+1] = np.imag(coeff_list[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = coeff_list[i]
                ip_real += 1

        self.param_array = param_array 

        self.efit_data = cfl.efit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"),
                &coeff[0], self.ex_data, self.n_p, self.param_array);

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
    
    def bh_fit(self, niter):
        cdef np.ndarray[double, ndim=1, mode="c"] x0
        cdef int cniter
        
        x0 = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        cniter = niter
        with nogil:
            bh_e_fit(&x0[0], self.n_p_real, self.efit_data, cniter, NULL, gsl_vector_bfgs2)

        coeff = np.zeros(self.n_p, dtype=np.complex128)
        ri = 0
        for i in range(self.n_p):
            if (self.param_types[i] == 'c'): 
                coeff[i] = np.complex(x0[ri], x0[ri+1])
                ri += 2
            else:
                coeff[i] = x0[ri]
                ri += 1

        return(coeff)




cdef class ESHFitRunner(object):
    """
    Class used to store data required by, and to run, a crystal field fit using
    energy level and spin Hamiltonian data. 

    Parameters
    ----------
    parameters : list
        A list of tensor objects for which to vary the prefactor. 
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    coeff_list : list
        Coefficients for tensors.  Values in ``coeff_list`` that correspond to
        parameters to be fit, that is, that are specified in ``parameters``, are
        used as initial values in the fitting process.  Furthermore, the type of
        elements in ``coeff_list`` determines whether only the real, or both the
        real and the imaginary, components of the corresponding prefactors are
        varied by the fitting routine.
    ex : np.ndarray
        2 by n dimensional array, with n the number available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    """
    cdef SpinHamiltonian sh
    cdef Hamiltonian h
    cdef Hamiltonian hfo
    cdef int n_p
    cdef int n_p_real
    cdef list param_types
    cdef int nzeeman
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef cfl.zsh **sh_array
    cdef cfl.shx_data **shx_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef np.ndarray p0_real
    cdef np.ndarray coeff
    cdef cfl.eshfit_data *eshfit_data

    def __init__(self, parameters, sh, h, coeff_list, ex):
        cdef cfl.param_type *param_type_ptr
        cdef cfl.zsh *zsh_array_ptr
        cdef cfl.shx_data *shx_data_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        cdef np.ndarray[double complex, ndim=1, mode="c"] shx_pa
        
        self.sh = sh
        self.h = h
        self.n_p = len(parameters)

        param_indices = []
        param_types = []
        n_p_real = 0
        for i,p in enumerate(parameters):
            # Ensure h contains all the listed tensors
            try:
                pi = h.index(p)
            except ValueError:
                raise ValueError("Tensor {} in param_list not found in h".format(p.name))
            # Work out which tensor each element in param_list corresponds to,
            # determine whether it is real or complex, and split any complex
            # parameters to create a purley real initial value array.
            if not isinstance(coeff_list[pi], Number):
                raise ValueError("Element {} in coefficients is not a number.".format(coeff_list[pi]))
            if isinstance(coeff_list[pi], complex):
                param_types.append('c')
                n_p_real += 2
            else:
                param_types.append('r')
                n_p_real += 1
            param_indices.append(pi)
        
        self.n_p_real = n_p_real
        self.param_types = param_types
        
        # Determine whether there are any second order tensors; that is, whether
        # the complete Hamiltonian contains any interactions that are also part
        # of the spin Hamiltonian.  Furthermore, we record the location of the
        # Zeeman tensor. 
        so_tensors = []
        self.nzeeman = -1
        for i,inter in enumerate(sh.interactions):
            if not inter.pro_data:
                raise ValueError("The spin Hamiltonian interaction {} is missing projection data.".format(i.type))
            if inter.type == 'zeeman':
                for t in inter.terms:
                    if t.tensor in h:
                        so_tensors += [t.tensor]
                self.nzeeman = i
            else:
                if inter.term.tensor in h:
                    so_tensors += [inter.term.tensor]
           
        # If required, generate the first order Hamiltonian. 
        fo_tensors = []
        if len(so_tensors) != 0:
            for i,t in enumerate(h):
                if t not in so_tensors:
                    fo_tensors += [t]
            self.hfo = Hamiltonian(fo_tensors, h.states)
        else:
            self.hfo = None

        if n_p_real > len(ex) + sh.nsh:
            raise ValueError("The total (real and imaginary) number of parameters "
                "exceeds the number of observables. Don't do that.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        self.coeff = np.ascontiguousarray(np.array(coeff_list), dtype=np.complex128)
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
        self.p0_real = np.ascontiguousarray(np.zeros(n_p_real), dtype=np.float64)
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

            param_array[i].type = cfl.atoi(param_types[i])
            param_array[i].index = param_indices[i]

            if param_types[i] == 'c':
                self.p0_real[ip_real] = np.real(coeff_list[i])
                self.p0_real[ip_real+1] = np.imag(coeff_list[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = coeff_list[i]
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
        shx_array = <cfl.shx_data **>malloc(len(sh.interactions)*cython.sizeof(shx_data_ptr))
        if shx_array == NULL:
            for i in range(self.n_p):
                free(param_array[i])
            free(self.ex_data)
            free(param_array)
            free(sh_array)
            raise MemoryError("shx_array alloc failed")
        self.shx_array = shx_array
        j = 0
        for i,inter in enumerate(sh.interactions):
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
            if not inter.exp_data:
                for j in range(i-1):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(param_array)
                free(sh_array)
                free(shx_array)
                raise ValueError("The spin Hamiltonian interaction {} is missing experimental data.".format(i.type))
            shx_pa = <np.ndarray[double complex, ndim=1, mode="c"]> inter.exp_tensor
            shx_array[i].pa = &shx_pa[0]
            shx_array[i].chisq_weight = inter.chisq_weight
            shx_array[i].inv_data = <cfl.zsh_inv_data *>PyCapsule_GetPointer(inter.inv_data_cap, "pycfl.InvData")
        
        if (self.hfo != None):
            self.eshfit_data = cfl.eshfit_data_alloc(sh_array, sh.nsh, self.nzeeman,
                    <cfl.zh*>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"),
                    <cfl.zh *>PyCapsule_GetPointer(self.hfo.h_cap, "pycfl.Hamiltonian"), &coeff[0],
                    self.ex_data, shx_array, self.n_p, self.param_array)
        else:
            self.eshfit_data = cfl.eshfit_data_alloc(sh_array, sh.nsh, self.nzeeman,
                    <cfl.zh*>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"), NULL, &coeff[0], 
                    self.ex_data, shx_array, self.n_p, self.param_array)
 
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
            for i in range(len(self.sh.interactions)):
                if self.shx_array[i] != NULL:
                    free(self.shx_array[i])
            free(self.shx_array)
        if self.eshfit_data != NULL:
            cfl.eshfit_data_free(self.eshfit_data)
     
    def bh_fit(self, niter):
        cdef np.ndarray[double, ndim=1, mode="c"] x0
        cdef int cniter
        
        x0 = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        cniter = niter
        with nogil:
            bh_esh_fit(&x0[0], self.n_p_real, self.eshfit_data, cniter, NULL, gsl_vector_bfgs2)

        coeff = np.zeros(self.n_p, dtype=np.complex128)
        ri = 0
        for i in range(self.n_p):
            if (self.param_types[i] == 'c'): 
                coeff[i] = np.complex(x0[ri], x0[ri+1])
                ri += 2
            else:
                coeff[i] = x0[ri]
                ri += 1

        return(coeff)


def e_fit(parameters, h, coeff, ex, niter):
    r"""
    Fit parameters to energy level data. 

    Parameters
    ----------
    param_list : list
        A list of tensor objects for which to vary the prefactor. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    coeff : list
        Coefficients for tensors.  Values in coeff which correspond to
        parameters to be fit, that is, they are specified in param_list, are
        used as initial values in the fitting process.  Furthermore, the type of
        elements in coeff determines whether only the real, or both the real and
        the imaginary, components of the corresponding prefactors are varied by
        the fitting routine. 
    ex : np.ndarray
        2 by n dimensional array, with n the number available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    niter : int
        The number of basinhopping iterations to complete. 
        
    stepsize and bounds not implemented for the moment. Leave stepsize to auto
    tune and no bounds... perhaps implement by accepting a minimization object
    instead, which has options relevant to that routine already set.  

    """
    efit = EFitRunner(parameters, h, coeff, ex)
    coeff = efit.bh_fit(niter)

    return coeff


def esh_fit(parameters, sh, h, coeff, ex, niter):
    r"""
    Fit parameters to energy level data. 
 
    Parameters
    ----------
    param_list : list
        A list of tensor objects for which to vary the prefactor. 
    sh : SpinHamiltonian
        The spin Hamiltonian object to be fit. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    coeff : list
        Coefficients for tensors.  Values in coeff which correspond to
        parameters to be fit, that is, that are specified in param_list, are
        used as initial values in the fitting process.  Furthermore, the type of
        elements in coeff determines whether only the real, or both the real and
        the imaginary, components of the corresponding prefactors are varied by
        the fitting routine.
    ex : np.ndarray
        2 by n dimensional array, with n the number available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    niter : int
        The number of basinhopping iterations to complete. 
        
    """
    eshfit = ESHFitRunner(parameters, sh, h, coeff, ex)
    coeff = eshfit.bh_fit(niter)
 
    return coeff

   







