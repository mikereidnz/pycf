# filename = pycfl.pyx
#cython: c_string_encoding=ascii

cimport cfl, cython
cimport numpy as np
import numpy as np
from numbers import Number
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free
from matel import matel

cdef class Tensor:
    cdef object t_cap
    cpdef public str name
    cpdef public str tmp_name
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

    property t_cap:
        def __get__(self):
            return self.t_cap

    property dim:
        def __get__(self):
            return self.n


cdef class Hamiltonian:
    cdef cfl.zh *cfl_zh
    cdef cfl.zt **tensor_array
    cdef char **state_labels
    cdef int n
    cdef int nt
    cdef list tensors
    cpdef public np.ndarray co
    cpdef public np.ndarray w
    cpdef public np.ndarray z
    cpdef public object h_cap
    def __cinit__(self, tensors, states):
        cdef char *char_ptr
        cdef cfl.zt *ten_array_ptr

        n = tensors[0].dim
        self.n = n
        self.nt = len(tensors)
        self.tensors = tensors
                
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

    def index(self, tensor):
        try:
            return self.tensors.index(tensor)
        except ValueError:
            raise ValueError("Tensor {} not an element of the Hamiltonian".format(tensor.name))
            
    cpdef public set_coeff(self, coeff):
        cdef np.ndarray[double complex, ndim=1, mode='c'] co

        self.co = coeff
        co = <np.ndarray[double complex, ndim=1, mode='c']> self.co
        cfl.zh_set_coeff(self.cfl_zh, &co[0])
        return None

    cpdef public diag(self):
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
    cpdef public str inter
    cpdef public int n
    cdef cfl.zsh *cfl_sh

    def __cinit__(self, n, interaction):

        self.n = n
        self.inter = interaction
        self.cfl_sh = zsh_alloc(n, interaction)

        if self.cfl_sh is NULL:
            raise MemoryError("Failed to alloc zsh memory")

    def __dealloc__(self):
        if self.cfl_sh != NULL:
            cfl.zsh_free(self.cfl_sh)

    def set_pro_data(self, object tensor, l):
        cfl.zsh_set_pro(self.cfl_sh, <cfl.zt *>PyCapsule_GetPointer(tensor, "pycfl.Tensor"), l)

        
cdef class SHTermData(object):
    """
    Class used to store term references together with the corresponding inversion data.
    """
    cdef cfl.zsh_inv_data *cfl_inv_data
    cdef np.ndarray coeff
    def __init__(self, n, inter, coeff):
        cdef np.ndarray[double complex, ndim=2, mode="c"] a

        self.inter = inter
        if inter == 'zeeman':
            self.terms = [SHTerm(n, 'zeeman_x'), SHTerm(n, 'zeeman_y'), SHTerm(n, 'zeeman_z')]
        else:
            self.terms[SHTerm(n, inter)]
        
        # Assign coeff to self, to ensure there exists a reference to the coeff
        # memory for as long as this object exists. 
        self.coeff = coeff
        a = <np.ndarray[double complex, ndim=2, mode='c']> self.coeff
        self.cfl_inv_data = zsh_inv_data_alloc(&a[0,0], coeff.shape[0], coeff.shape[1])
        if self.cfl_inv_data == NULL:
            raise MemoryError("Failed to alloc inv_data memory")

    def __dealloc__(self):
        if self.cfl_inv_data != NULL:
            cfl.zsh_inv_data_free(self.cfl_inv_data)


class SpinHamiltonian:
    r""" 
    Container for holding data about the spin Hamiltonian. 

    Parameters
    ----------
    terms : list
        Elements are strings which specify the interaction terms.  Possible
        values are: 'zeeman', 'hyperfine', and 'quadrupole'.  
    B : numpy.ndarray 
        A `3` by `1` vector containing values for the magnetic field strengths
        `B_x`, `B_y` and `B_z`; if ``terms`` contains 'zeeman' this keyword
        argument must be specified.  
    S : float
        The spin projection `S_z`; if ``terms`` contains 'zeeman' or 'hyperfine'
        this keyword argument must be specified.
    I : float
        The nuclear spin projection `I_z`; if ``terms`` contains 'hyperfine' or
        'quadrupole' this keyword argument must be specified.

    Returns
    -------
    object : SpinHamiltonian
    """
    def __init__(self, terms, **kwargs):
        for t in terms:
            if not any(t in term for term in ['zeeman', 'hyperfine', 'quadrupole']):
                raise ValueError("Invalid element in terms list.")

        if 'zeeman' in terms:
            try:
                B = kwargs['B']
                self.B = B
            except KeyError:
                raise ValueError("Missing keyword argument B.")
        else:
            B = None
           
        # Calculate matrix elements for the specified terms.
        j_l = ['jx', 'jy', 'jz']
        if 'zeeman' in terms or 'hyperfine' in terms:
            try: 
                S = kwargs['S']
            except KeyError:
                raise ValueError("Missing keyword argument S.")
            # Calculate the matrix elements of spin operator.
            S_m = [None]*3
            for i in range(3):
                S_m[i] = matel(j_l[i], S)
            self.S = S
            self.S_m = S_m
        else:
            S_m = None

        if 'hyperfine' in terms or 'quadrupole' in terms:
            try:
                I = kwargs['I']
            except KeyError:
                raise ValueError("Missing keyword argument I.")
            # Calculate the matrix elements of nuclear spin operator.
            I_m = [None]*3
            for i in range(3):
                I_m[i] = matel(j_l[i], I)

            self.I = I
            self.I_m = I_m
        else:
            I_m = None

        # Determine spin Hamiltonian dimension.
        if B != None:
            if I_m == None:
                # Only the zeeman term.
                nsh = 2*S+1
            else:
                # Both the zeeman and quadrupole terms.
                nsh = (2*S+1) * (2*I+1)
        elif S_m == None:
            # Only the quadrupole term.
            nsh = 2*I+1 
        else:
            # Contains hyperfine term.
            nsh = (2*S+1) * (2*I+1)
        
        self.nsh = nsh

        self.coeff = {}
        self.term_data = []
        # Calculate the cofficient arrays and alloc spin Hamiltonian terms.
        if 'zeeman' in terms:
            # Coefficient arrays are calculated for three B fields; the user
            # specified B-direction is ignored for inversion. 
            nz = 2*S+1
            B_a = np.zeros([3, nz**2, 9], dtype = np.complex)
            for i in range(3):
                B_a[i, :, :] = zeeman_sh_coeff(np.eye(3,3)[i,:], S_m)
            self.zeeman = SHTermData(nz, 'zeeman')
            self.zeeman.set_coeff(np.reshape(B_a, (3 * nz**2, 9)))

        if 'hyperfine' in terms:
            nh = 2*S+1 + 2*I+1
            self.hyperfine = SHTermData(nh, 'hyperfine')
            self.hyperfine.set_coeff(hyperfine_sh_coeff(I_m, S_m))

        if 'quadrupole' in terms: 
            nq = 2*I+1
            self.quadrupole = SHTermData(nq, 'quadrupole')
            self.quadrupole.set_coeff(quadrupole_sh_coeff(I_m))



cdef class EFitRunner(object):
    """
    Class used to store data required by, and to run, a crystal field fit using energy level data. 
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
    cdef np.ndarray h_coeff
    cdef cfl.efit_data *efit_data

    def __init__(self, h, parameters, co, ex):
        cdef cfl.param_type *param_type_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] h_coeff
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
            # Workout which tensor each element in param_list corresponds to,
            # determine whether it is real or complex, and split any complex
            # parameters to create a purley real initial value array.
            if not isinstance(co[pi], Number):
                raise ValueError("Element {} in coefficients is not a number.".format(co[pi]))
            if isinstance(co[pi], complex):
                param_types.append('c')
                n_p_real += 2
            else:
                param_types.append('r')
                n_p_real += 1
            param_indices.append(pi)
        
        self.n_p_real = n_p_real
        self.param_types = param_types
        if n_p_real > len(ex):
            raise ValueError("The total (real and imaginary) number of parameters exceeds the number of observables. Don't do that.")

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
       
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
        self.h_coeff = np.ascontiguousarray(np.array(co), dtype=np.complex128)
        h_coeff = <np.ndarray[double, ndim=1, mode="c"]> self.h_coeff
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
                self.p0_real[ip_real] = np.real(co[i])
                self.p0_real[ip_real+1] = np.imag(co[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = co[i]
                ip_real += 1

        self.param_array = param_array 
        self.efit_data = cfl.efit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"), &h_coeff[0],
                self.ex_data, self.n_p, self.param_array);

    
    def bh_fit(self, niter):
        cdef np.ndarray[double, ndim=1, mode="c"] x0
        
        x0 = <np.ndarray[double, ndim=1, mode="c"]> self.p0_real
        with nogil:
            bh_e_fit(&x0[0], self.n_p_real, self.efit_data, niter, NULL, gsl_vector_bfgs2)

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


def e_fit(h, parameters, co, ex, niter):
    r"""
    Fit parameters to energy level data. 

    Parameters
    ----------
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    param_list : list
        A list of tensor objects for which to vary the prefactor. 
    co : list
        Coefficients for tensors of h.  Values in co which correspond to
        parameters to be fit, that is, they are specified in param_list, are
        used as initial values in the fitting process.  Furthermore, the type of
        elements in co determines whether only the real, or both the real and
        the imaginary, components of the corresponding prefactors are varied by
        the fitting routine. 
    ex : np.ndarray
        2 by n dimensional array, with n the dimension of h. The first column
        contains energy level indices and the second column contains
        corresponding experimental energy level values. 
    niter : int
        The number of basinhopping iterations to complete. 
        
    stepsize and bounds not implemented for the moment. Leave stepsize to auto
    tune and no bounds... perhaps implement by accepting a minimization object
    instead, which has options relevant to that routine already set.  

    """
    efit = EFitRunner(h, parameters, co, ex)
    coeff = efit.bh_fit(niter)

    return coeff


    







