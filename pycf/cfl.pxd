# file: ccfl.pxd

cdef extern from "../../cfl/include/cfl_crs.h":
    ctypedef struct crs_zhm:
        pass

cdef extern from "../../cfl/include/cfl_h.h":
    ctypedef struct ztensor:
        pass
    ctypedef struct zh:
        pass
    ctypedef struct zhd_w:
        pass
    
    ztensor *ztensor_alloc(char *name, double complex *a, int n)
    void ztensor_free(ztensor *zt)
    zh *zh_alloc(int n, int nt, char **s, ztensor **t, double *w, double complex *z)
    void zh_free(zh *h) 
    void zh_set_coeff(zh *h, double complex *coeff)
    zhd_w *zhd_w_alloc(zh *h)
    void zhd_w_free(zhd_w *hd_w)
    void zhd(zh *h, zhd_w *hd_w) nogil
   
