# filename = pycfl.pyx
#cython: c_string_encoding=ascii

cimport cfl, cython
cimport numpy as np
import numpy as np
from numbers import Number
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free


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
            self.n = data_tuple[0].n
            t1 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[0].t_cap, "pycfl.Tensor")
            t2 = <cfl.zt *>PyCapsule_GetPointer(data_tuple[1].t_cap, "pycfl.Tensor")
            t = cfl.zt_sa(<char *>self.name, t1, t2, 1, data_tuple[2])

        elif (len(data_tuple)==2):
            
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
    cdef cfl.zhd_w *cfl_zhd_w
    cdef cfl.zt **tensor_array
    cdef cfl.zt *tp
    cdef char **state_labels
    cdef int n
    cdef int nt
    cpdef public np.ndarray w
    cpdef public np.ndarray z
    def __cinit__(self, tensors, states):
        cdef char *char_ptr
        cdef cfl.zt *ten_array_ptr
        cdef np.ndarray[double, ndim=1, mode="c"] w
        cdef np.ndarray[double complex, ndim=2, mode="c"] z

        n = tensors[0].dim
        self.n = n
        self.nt = len(tensors)
        self.w = np.ascontiguousarray(np.zeros(n), dtype=np.float64)
        self.z = np.ascontiguousarray(np.zeros(n*n).reshape((n,n)), dtype=np.complex128)

        w = <np.ndarray[double, ndim=1, mode="c"]> self.w
        z = <np.ndarray[double complex, ndim=2, mode="c"]> self.z
                
        # Create array of tensors and array of character arrays to be passed to
        # the zh_set cfl function. 
        tensor_array = <cfl.zt **>malloc(len(tensors)*cython.sizeof(ten_array_ptr))
        if tensor_array is NULL:
            raise MemoryError("tensor_array alloc failed")
        state_labels = <char **>malloc(len(states)*cython.sizeof(char_ptr))
        if state_labels is NULL:
            raise MemoryError("state_labels alloc failed")
        
        self.state_labels = state_labels
        self.tensor_array = tensor_array

        for i,s in enumerate(states):
            state_labels[i] = s

        for i,t in enumerate(tensors):
            tensor_array[i] = <cfl.zt *> PyCapsule_GetPointer(t.t_cap, "pycfl.Tensor")

        # Allocate storage for zh. 
        self.cfl_zh = cfl.zh_alloc(n, self.nt, state_labels, tensor_array,
                &w[0], &z[0,0])
        if self.cfl_zh is NULL:
            raise MemoryError("cfl_zh alloc failed")

        self.cfl_zhd_w = cfl.zhd_w_alloc(self.cfl_zh)
        if self.cfl_zhd_w is NULL:
            raise MemoryError("cfl_zhd_w alloc failed")

    def __dealloc__(self):
        if self.cfl_zh is not NULL:
            cfl.zh_free(self.cfl_zh)

        if self.cfl_zhd_w is not NULL:
            cfl.zhd_w_free(self.cfl_zhd_w)

        if self.state_labels is not NULL:
            free(self.state_labels)

        if self.tensor_array is not NULL:
            free(self.tensor_array)
        
    cpdef public set_coeff(self, np.ndarray[double complex, ndim=1, mode='c'] coeff):
        cfl.zh_set_coeff(self.cfl_zh, &coeff[0])
        return None

    cpdef public diag(self):
        cdef cfl.zh *h = self.cfl_zh
        cdef cfl.zhd_w *hd_w = self.cfl_zhd_w

        with nogil:
            cfl.zhd(h, hd_w)
        return (self.w, self.z)


cdef class SHTerm:
    cpdef public str inter
    cpdef public int n
    cdef object sh_cap

    def __cinit__(self, n, interaction):
        cdef cfl.zsh *sh

        self.n = n
        self.iner = interaction
        sh = zsh_alloc(n)

        if sh is NULL:
            self.t_cap = None
            raise MemoryError("Failed to alloc zsh memory")
        else:
            self.sh_cap = PyCapsule_New(<void *>sh, "pycfl.SHTerm", NULL)

    def __dealloc__(self):
        if self.t_cap is not None:
            cfl.zsh_free(<cfl.zsh *>PyCapsule_GetPointer(self.sh_cap, "pycfl.SHTerm"))


cpdef zeeman_coeff_array(v, t):
    r"""
    Generate the Zeeman interaction `coefficient array`.  This consists of a
    `2j+1 \times 2j+1` by `3 \times 3` array containing the matrix elements of
    the terms `B_a S_b`, with `a,b \in \{x, y, z\}` and `j` the angular momentum
    of the rank one tensor `S`.  Here the rows enumerate the `2j+1 \times 2j+1`
    different state combinations while the columns enumerate all combinations of
    `a` and `b`.

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


cpdef hyperfine_coeff_array(t1, t2):
    r"""
    Generate the hyperfine interaction `coefficient array`.  This consists of a
    `2j_1+1 \times 2j_2+1` by `3 \times 3` array containing the matrix elements
    of the operators `I_a S_b`, with `a,b \in \{x, y, z\}` and `j_1` and `j_2`
    the angular momentum of the rank one tensors `I` and `S`, respectively.
    Here the rows enumerate the `2j_1+1 \times 2j_2+1` different state
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
        A `2j_1+1 \times 2j_2+1` by `3 \times 3` array.
    """

    t1l = len(t1[0])
    t2l = len(t2[0])
    l = len(t1)

    cdef __ci(t1i, t2i):
        """
        Calculate the row/column index for the coefficient array.
        """
        # The t1 upper bound is the t1 length.
        return(t1i + t1l * t2i)

    a = np.zeros([t1l * t2l, t1l * t2l, l, l], dtype = np.complex)

    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(l):
                        for j in range(l):
                            a[__ci(t1r, t2r), __ci(t1c, t2c), i, j] = \
                                t1[i][t1r, t1c] * t2[j][t2r, t2c]

    return(np.reshape(a, (t1l*t2l*t1l*t2l, l*l)))


cpdef quadrupole_coeff_array(t):
    r""" 
    Generate the quadrupole interaction `coefficient array`.  This consists of a
    `2j+1 \times 2j+1` by `3 \times 3` array containing the matrix elements of
    the operators `I_a I_b`, with `a,b \in \{x, y, z\}` and `j` the angular
    momentum of the rank one tensor `I`.  Here the rows enumerate the `2j+1
    \times 2j+1` different state combinations while the columns enumerate all
    combinations of `a` and `b`.  

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
    cpdef public np.ndarray B
    cpdef public float S
    cpdef public float I
    cpdef public list S_m
    cpdef public list I_m
    cpdef public int nsh
    cpdef public dict coeff_a
    cpdef public list terms
    def __cinit__(self, terms, **kwargs):
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

        self.coeff_a = {}
        self.terms = []
        # Calculate the cofficient arrays and alloc spin Hamiltonian terms.
        if 'zeeman' in terms:
            # Coefficient arrays are calculated for three B fields; the user
            # specified B-direction is ignored for inversion. 
            nz = 2*S+1
            B_a = np.zeros([3, nz**2, 9], dtype = np.complex)
            for i in range(3):
                B_a[i, :, :] = zeeman_coeff_array(np.eye(3,3)[i,:], S_m)
            self.coeff_a['z'] = np.reshape(B_a, (3 * nz**2, 9))
            self.terms += [SHTerm(nz, 'zeeman')]

        if 'hyperfine' in terms:
            nh = 2*S+1 + 2*I+1
            self.coeff_a['h'] = hyperfine_coeff_array(I_m, S_m)
            self.terms += [SHTerm(nh, 'hyperfine')]

        if 'quadrupole' in terms: 
            nq = 2*I+1
            self.coeff_a['q'] = quadrupole_coeff_array(I_m)
            self.terms += [SHTerm(nq, 'quadrupole')]

    
