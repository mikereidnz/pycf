# file: ccfl.pxd
#cython: c_string_encoding=ascii

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

    ctypedef struct zsh_pro_data:
        pass

    ctypedef struct zsh_inv_data:
        double complex *a
        size_t m
        size_t n
    
    zsh *zsh_alloc(size_t n, char *type)
    void zsh_free(zsh *sh)
    zshp_w *zshp_w_alloc(zsh *sh)
    void zsh_set_pro(zsh *sh, zt *t, int l)
    zsh_inv_data *zsh_inv_data_alloc(double complex *a, size_t m, size_t n)
    void zsh_inv_data_free(zsh_inv_data *d)
    void zshp(double complex *a, double complex *hz, zsh *sh, zshp_w *shp_w)
    zshi_w *zshi_w_alloc(zsh_inv_data *d)
    void zshi_w_free(zshi_w *w)
    void zsh_set_inv(zsh *sh, double complex *a, size_t m, size_t n)
    void zshi(double complex *a, zshi_w *w)


cdef extern from "../../cfl/include/basinhopping.h":
    ctypedef struct bh_bounds:
        double *l
        double *u

    ctypedef enum bh_lmin:
        gsl_nmsimplex2rand = 0
        gsl_nmsimplex2 = 1
        gsl_conjugate_fr = 2
        gsl_conjugate_pr = 3
        gsl_vector_bfgs2 = 4

    int bh_fit(double (*obj_f)(size_t n, double *x, double *grad, void *data),
            double *x0, size_t nx, void *data, size_t niter, bh_bounds *bounds,
            bh_lmin lmintype)
    void bh_set_step(bh_work *w, double *stepsize, float target_accept_rate,
            size_t interval, float factor)


cdef extern from "../../cfl/include/h_fit.h":
    ctypedef struct param_type:
        int type
        size_t index

    ctypedef struct shx_data:
        double complex *pa
        float chisq_weight
        zsh_inv_data *inv_data
        size_t l

    efit_data *efit_data_alloc(zh *h, double complex *h_co, ex_data *ex, size_t
            n_zx, param_type **p)
    void efit_data_free(efit_data *data)
    eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, size_t nzeeman, zh *h,
            zh *hfo, double complex *h_co, double complex *hfo_co, ex_data *ex,
            shx_data **shx, size_t n_zx, size_t n_fozx, param_type **p)
    void eshfit_data_free(eshfit_data *data)
    int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds
            *bounds, bh_lmin lmintype) nogil
    int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds
            *bounds, bh_lmin lmintype) nogil
