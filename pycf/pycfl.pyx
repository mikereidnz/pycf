# filename = pycfl.pyx
#cython: c_string_encoding=utf-8

cimport cfl, cython
cimport numpy as np
import numpy as np
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free


cdef class Tensor:
    cdef object t_cap
    cpdef public str name
    cpdef public int n
    cpdef public str name_add
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, char *name, np.ndarray[double complex, ndim=2, mode='c'] a, object t_cap=None, int n_res=0):
        cdef cfl.zt *cfl_zt

        if(t_cap != None):
            # The cfl.zt object already exists. 
            self.name = <str> name
            self.n = n_res
            self.t_cap = t_cap
        else:
            self.name = <str> name
            n = a.shape[0]
            self.n = n
            cfl_zt = cfl.zt_alloc(name, &a[0,0], n)
            if cfl_zt is NULL:
                self.t_cap = None
                raise MemoryError("Cannot alloc zt memory")
            else:
                self.t_cap = PyCapsule_New(<void *>cfl_zt, "pycfl.Tensor", NULL)

    def __dealloc__(self):
        if self.t_cap is not None:
            cfl.zt_free(<cfl.zt *>PyCapsule_GetPointer(self.t_cap, "pycfl.Tensor"))
    
    def __add__(self, t):
        cdef cfl.zt *t1
        cdef cfl.zt *t2
        cdef cfl.zt *t_add
        cdef object t_cap

        if not (isinstance(self, Tensor) or isinstance(t, Tensor)):
            raise TypeError("Only objects of type Tensor can be added to Tensors")
        
        self.name_add = self.name + t.name
        t1 = <cfl.zt *>PyCapsule_GetPointer(self.t_cap, "pycfl.Tensor")
        t2 = <cfl.zt *>PyCapsule_GetPointer(t.t_cap, "pycfl.Tensor")

        t_add = zt_sa(<char *>self.name_add, t1, t2, 1, 1)
        if t_add is NULL:
            raise MemoryError("Cannot alloc memory for t_add")

        t_cap = PyCapsule_New(<void *>t_add, "pycfl.Tensor", NULL)

        return Tensor(<char *>self.name_add, np.array([[]],dtype=np.complex128), t_cap, self.n) 

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

