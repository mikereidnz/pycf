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

        self.co = coeff
        co = <np.ndarray[double complex, ndim=1, mode='c']> self.co
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
        self.tensor = tensor

        
cdef class SHTermData(object):
    """
    Class used to store term references together with the corresponding inversion data.
    """
    cdef cfl.zsh_inv_data *cfl_inv_data
    cdef np.ndarray coeff
    def __init__(self, n, inter, coeff):
        cdef np.ndarray[double complex, ndim=2, mode="c"] a
        
        self.pro_data = False
        self.exp_data = False
        self.inter = inter
        if inter == 'zeeman':
            self.terms = [SHTerm(n, 'zeeman_x'), SHTerm(n, 'zeeman_y'), SHTerm(n, 'zeeman_z')]
        else:
            self.term = SHTerm(n, inter)
        
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

    def set_pro_data(self, tensor, l):
        if self.inter == 'zeeman':
            for i,t in enumerate(self.terms):
                t.set_pro_data(tensor[i], l)
        else:
            t.set_pro_data(tensor, l)
        self.pro_data = True

    def set_exp_data(self, exp_tensor, chisq_weight):
        self.exp_tensor = exp_tensor
        self.chisq_weight = chisq_weight
        self.exp_data = True


class SpinHamiltonian:
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
    def __init__(self, interactions, **kwargs):
        for i in interactions:
            if i not in ['zeeman', 'hyperfine', 'quadrupole']:
                raise ValueError("Invalid element in interactions list.")

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

        if 'hyperfine' in interactions or 'quadrupole' in interactions:
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
                # Only the zeeman interaction.
                nsh = 2*S+1
            else:
                # Both the zeeman and quadrupole interactions.
                nsh = (2*S+1) * (2*I+1)
        elif S_m == None:
            # Only the quadrupole interactions.
            nsh = 2*I+1 
        else:
            # Contains hyperfine interactions.
            nsh = (2*S+1) * (2*I+1)

        # Calculate the cofficient arrays and alloc spin Hamiltonian
        # interactions.
        self.interactions = []
        if 'zeeman' in interactions:
            # Coefficient arrays are calculated for three B fields; the user
            # specified B-direction is ignored for inversion. 
            nz = 2*S+1
            B_a = np.zeros([3, nz**2, 9], dtype = np.complex)
            for i in range(3):
                B_a[i, :, :] = zeeman_sh_coeff(np.eye(3,3)[i,:], S_m)
            zeeman = SHTermData(nz, 'zeeman')
            zeeman.set_coeff(np.reshape(B_a, (3 * nz**2, 9)))
            self.interactions += [zeeman]

        if 'hyperfine' in interactions:
            nh = 2*S+1 + 2*I+1
            hyperfine = SHTermData(nh, 'hyperfine')
            hyperfine.set_coeff(hyperfine_sh_coeff(I_m, S_m))
            self.interactions += [hyperfine]

        if 'quadrupole' in interactions: 
            nq = 2*I+1
            quadrupole = SHTermData(nq, 'quadrupole')
            quadrupole.set_coeff(quadrupole_sh_coeff(I_m))
            self.interactions += [quadrupole]

        self.nsh = nsh

    
    def set_pro_data(self, interaction, tensor, level):
        r"""
        Set the projection data interactions of spin Hamiltonian. 

        Parameters
        ----------
        interaction : string
            Valid options are 'zeeman', 'hyperfine', and 'quadrupole'. 
        tensor : list or Tensor
            For Zeeman interactions a list of three tensors corresponding to
            `\hat{x}`, `\hat{y}`, and `\hat{z}` interactions must be specified. 
        level : int
            The level of the complete Hamiltonian for which to project the spin Hamiltonian.

        """
        for i in self.interactions:
            if i.inter == interaction:
                if interaction == 'zeeman':
                    if not isinstance(tensor, list):
                        raise ValueError("For Zeeman interactions tensor must be a list.")
                i.set_pro_data(tensor, level)
                return

        raise ValueError("This spin Hamiltonian object was not instantiated with {} interaction support.".format(interaction))

    
    def set_exp_data(self, interaction, exp_tensor, chisq_weight):
        for i in self.interactions:
            if i.inter == interaction:
                i.set_exp_data(exp_tensor, chisq_weight)
                return
            
        raise ValueError("This spin Hamiltonian object was not instantiated with {} interaction support.".format(interaction))




cdef class EFitRunner(object):
    """
    Class used to store data required by, and to run, a crystal field fit using
    energy level data. 
    """
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

    def __init__(self, parameters, h, co, ex):
        cdef cfl.param_type *param_type_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] h_coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li

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
            raise ValueError("The total (real and imaginary) number of parameters exceeds "
                    "the number of observables. Don't do that.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        self.h_coeff = np.ascontiguousarray(np.array(co), dtype=np.complex128)
        h_coeff = <np.ndarray[double, ndim=1, mode="c"]> self.h_coeff     

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
                self.p0_real[ip_real] = np.real(co[i])
                self.p0_real[ip_real+1] = np.imag(co[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = co[i]
                ip_real += 1

        self.param_array = param_array 

        self.efit_data = cfl.efit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"),
                &h_coeff[0], self.ex_data, self.n_p, self.param_array);

    
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


cdef class ESHFitRunner(object):
    """
    Class used to store data required by, and to run, a crystal field fit using
    energy level and spin Hamiltonian data. 
    """
    cdef Hamiltonian hfo
    cdef int n_p
    cdef int n_p_real
    cdef list param_types
    cdef cfl.ex_data *ex_data
    cdef cfl.param_type **param_array
    cdef cfl.shx_data **shx_array
    cdef np.ndarray ex_e
    cdef np.ndarray ex_li
    cdef np.ndarray p0_real
    cdef np.ndarray h_coeff
    cdef np.ndarray hfo_coeff
    cdef cfl.eshfit_data *eshfit_data

    def __init__(self, parameters, sh, h, h_co, ex, shx):
        cdef cfl.param_type *param_type_ptr
        cdef cfl.shx_data *shx_data_ptr
        cdef np.ndarray[double complex, ndim=1, mode="c"] h_coeff
        cdef np.ndarray[double, ndim=1, mode="c"] ex_e
        cdef np.ndarray[int, ndim=1, mode="c"] ex_li
        
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
            if not isinstance(h_co[pi], Number):
                raise ValueError("Element {} in coefficients is not a number.".format(h_co[pi]))
            if isinstance(h_co[pi], complex):
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
        # of the spin Hamiltonian. 
        so_tensors = []
        for i in sh.interactions:
            if not i.pro_data:
                raise ValueError("The spin Hamiltonian interaction {} is missing projection data.".format(i.inter))
            if i.inter == 'zeeman':
                for t in i.terms:
                    if t.name in h:
                        so_tensors += [t.tensor]
            else:
                if i.term.name in h:
                    so_tensors += [i.term.tensor]
            
        if n_p_real > len(ex):
            raise ValueError("The total (real and imaginary) number of parameters "
                "exceeds the number of observables. Don't do that.")
        elif ex.shape[1] != 2:
            raise ValueError("Incorrect ex shape; expected a two column array.")

        # Generate the first order Hamiltonian, if required.
        fo_tensors = []
        hfo_co = []
        if len(so_tensors) != 0:
            for i,t in enumerate(h):
                if t not in so_tensors:
                    fo_tensors += [t]
                    hfo_co += [h_co[i]]
            self.hfo = Hamiltonian(fo_tensors, h.states)
            self.hfo.set_coeff(np.array(hfo_co, dtype=np.complex128))
        else:
            self.hfo = NULL

        # We assign pointers to self to make sure a reference exists for as long
        # as the object, and consequently prevent the GC from freeing the
        # pointers until after __dealloc__ is called.
        self.h_coeff = np.ascontiguousarray(np.array(h_co), dtype=np.complex128)
        h_coeff = <np.ndarray[double, ndim=1, mode="c"]> self.h_coeff
        self.hfo_coeff = np.ascontiguousarray(np.array(hfo_co), dtype=np.complex128)
        hfo_coeff = <np.ndarray[double, ndim=1, mode="c"]> self.hfo_coeff
       
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
                self.p0_real[ip_real] = np.real(co[i])
                self.p0_real[ip_real+1] = np.imag(co[i])
                ip_real += 2
            else:
                self.p0_real[ip_real] = co[i]
                ip_real += 1

        self.param_array = param_array 

        # Experimental spin Hamiltonian data. 
        shx_array = <cfl.shx_data **>malloc(n_shx*cython.sizeof(shx_data_ptr))
        if shx_array == NULL:
            for i in range(self.n_p):
                free(param_arrai[i])
            free(self.ex_data)
            free(self.param_array)
            raise MemoryError("shx_array alloc failed")
        for i,inter in enumerate(sh.interactions):
            shx_array[i] = <cfl.shx_data *>malloc(cython.sizeof(cfl.shx_data))
            if shx_array[i] == NULL:
                for j in range(i-1):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(self.param_array)
                free(self.shx_array)
                raise MemoryError("shx_array[{}] alloc failed".format(i))
            if not i.exp_data:
                for j in range(i-1):
                    free(shx_array[j])
                for j in range(self.n_p):
                    free(param_array[i])
                free(self.ex_data)
                free(self.param_array)
                free(self.shx_array)
                raise ValueError("The spin Hamiltonian interaction {} is missing experimental data.".format(i.inter))
            # FIXME: inter.pa and inv_data have to be proper cdef pointers
            shx_array[i].pa = inter.pa
            shx_array[i].chisq_weight = inter.chisq_weight
            shx_array[i].inv_data = inter.inv_data

        self.eshfit_data = cfl.eshfit_data_alloc(<cfl.zh *>PyCapsule_GetPointer(h.h_cap, "pycfl.Hamiltonian"),
                &h_coeff[0], self.ex_data, self.n_p, self.param_array);

    
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



def e_fit(parameters, h, co, ex, niter):
    r"""
    Fit parameters to energy level data. 

    Parameters
    ----------
    param_list : list
        A list of tensor objects for which to vary the prefactor. 
    h : Hamiltonian
        The Hamiltonian for which to fit the energy levels. 
    co : list
        Coefficients for tensors.  Values in co which correspond to parameters
        to be fit, that is, they are specified in param_list, are used as
        initial values in the fitting process.  Furthermore, the type of
        elements in co determines whether only the real, or both the real and
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
    efit = EFitRunner(parameters, h, co, ex)
    coeff = efit.bh_fit(niter)

    return coeff


def esh_fit(parameters, sh, h, co, ex, shx, niter):
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
    co : list
        Coefficients for tensors.  Values in co which correspond to parameters
        to be fit, that is, that are specified in param_list, are used as
        initial values in the fitting process.  Furthermore, the type of
        elements in co determines whether only the real, or both the real and
        the imaginary, components of the corresponding prefactors are varied by
        the fitting routine.
    ex : np.ndarray
        2 by n dimensional array, with n the number available experimental
        energy levels. The first column contains energy level indices and the
        second column contains corresponding experimental energy level values. 
    shx : dict
        Experimental spin Hamiltonian parameter data. 
    niter : int
        The number of basinhopping iterations to complete. 
        
    """
    eshfit = ESHFitRunner(parameters, sh, h, co, ex, shx)
    coeff = eshfit.bh_fit(niter)
 
    return coeff

   







