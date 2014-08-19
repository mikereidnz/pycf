# file: ccfl.pxd

cdef extern from "complex.h":
    double complex conj(double complex z)
    double complex cexp(double complex z)
    double complex I


cdef extern from "../../cfl/include/cfl_crs.h":
    ctypedef struct crs_zhm:
        pass


cdef extern from "../../cfl/include/cfl_h.h":
    ctypedef struct zt:
        pass
    ctypedef struct zh:
        pass
    ctypedef struct zhd_w:
        pass
    
    zt *zt_alloc(char *name, double complex *a, int n)
    void zt_free(zt *zt)
    zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2)
    zt *zt_s(char *name, zt *t, double complex s)
    zh *zh_alloc(int n, int nt, char **s, zt **t, double *w, double complex *z)
    void zh_free(zh *h) 
    void zh_set_coeff(zh *h, double complex *coeff)
    zhd_w *zhd_w_alloc(zh *h)
    void zhd_w_free(zhd_w *hd_w)
    void zhd(zh *h, zhd_w *hd_w) nogil


cdef extern from "../../cfl/include/cfl_sh.h":
    ctypedef struct zsh:
        pass
    ctypedef struct zshp_w:
        pass
    ctypedef struct zshi_w:
        pass
    
    zsh *zsh_alloc(size_t n)
    void zsh_free(zsh *sh)
    zshp_w *zshp_w_alloc(zt *t)
    void zshp(zh *h, zsh *sh, zshp_w *shp_w, int l)
    zshi_w *zshi_w_alloc(double complex *a, double complex *b, size_t m, size_t n)
    void zshi_w_free(zshi_w *w)
    void zshi(double complex *b, zshi_w *w)
