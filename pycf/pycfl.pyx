# filename = pycfl.pyx

cimport cfl, cython
cimport numpy as np
import numpy as np
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free

cdef class Tensor:
    cdef cfl.ztensor *cfl_ztensor
    cpdef public int n
    cdef char *name

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, char *name, np.ndarray[double complex, ndim=2, mode='c'] a):
        n = a.shape[0]
        self.n = n
        self.name = name
        self.cfl_ztensor = cfl.ztensor_alloc(name, &a[0,0], n)
        if self.cfl_ztensor is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.cfl_ztensor is not NULL:
            cfl.ztensor_free(self.cfl_ztensor)

    property cfl_ztensor_ptr:
        def __get__(self):
            return PyCapsule_New(<void *>self.cfl_ztensor,"pycfl.Tensor",NULL)

    property dim:
        def __get__(self):
            return self.n
    
cdef class Hamiltonian:
    cdef cfl.zh *cfl_zh
    cdef cfl.zhd_w *cfl_zhd_w
    cdef cfl.ztensor **tensor_array
    cdef cfl.ztensor *tp
    cdef char **state_labels
    cdef int n
    cdef int nt
    cdef np.ndarray w
    cdef np.ndarray z
    def __cinit__(self, tensors, states):
        cdef char *char_ptr
        cdef cfl.ztensor *ten_array_ptr
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
        tensor_array = <cfl.ztensor **>malloc(len(tensors)*cython.sizeof(ten_array_ptr))
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
            tensor_array[i] = <cfl.ztensor *> PyCapsule_GetPointer(t.cfl_ztensor_ptr, "pycfl.Tensor")

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

