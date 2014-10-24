# file: ccfl.pxd
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



cdef extern from "complex.h":
    double complexconj(double complexz)
    double complexcexp(double complexz)
    double complexI


cdef extern from "stdlib.h":
    int atoi(const char *str)


cdef extern from "../../cfl/include/cfl_crs.h":
    ctypedef struct crs_zhm:
        pass


cdef extern from "../../cfl/include/cfl_tensor.h":
    ctypedef struct sl:
        size_t n
        char **states
        long hash

    ctypedef struct zt:
        pass
    
    sl *sl_alloc(size_t n, char **states)
    void sl_free(sl *l)
    zt *zt_alloc(char *name, double complex *a, size_t n, sl *states)
    void zt_free(zt *t)
    zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2)
    zt *zt_s(char *name, zt *t, double complex s)


cdef extern from "../../cfl/include/cfl_h.h":
    ctypedef struct zh:
        int n
        int nt
        sl *states
        zt **t
        double complex *coeff
        double complex *ap
        
    ctypedef struct zhd_w:
        pass
    
    zh *zh_alloc(int n, int nt, zt **t) 
    void zh_free(zh *h)
    void zh_set_coeff(zh *h, double complex *coeff)
    zhd_w *zhd_w_alloc(zh *h)
    void zhd_w_free(zhd_w *hd_w)
    void zhd(double *w, double complex *z, zh *h, zhd_w *hd_w) nogil
    void h_getlabels(zh *h, char **states)


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
    void zshp_w_free(zshp_w *shp_w)
    zsh_inv_data *zsh_inv_data_alloc(double complex *a, size_t m, size_t n)
    void zsh_inv_data_free(zsh_inv_data *d)
    zshi_w *zshi_w_alloc(zsh_inv_data *d)
    void zshi_w_free(zshi_w *w)
    void zsh_set_pro(zsh *sh, zt *t, int l)
    void zsh_set_inv(zsh *sh, double complex *a, size_t m, size_t n) 
    void zshp(double complex *a, double complex *hz, zsh *sh, zshp_w *shp_w)
    void zshi(double complex *a, zshi_w *w)


cdef extern from "../../cfl/include/cfl_min.h":
    ctypedef struct cfl_min_bounds:
        double *l
        double *u

    ctypedef enum gsl_min_alg:
        gsl_nmsimplex2rand = 0
        gsl_nmsimplex2 = 1
        gsl_conjugate_fr = 2
        gsl_conjugate_pr = 3,
        gsl_vector_bfgs2 = 4

    ctypedef enum nlopt_min_alg:
        nlopt_cobyla = 1
        nlopt_bobyqa = 2 
        nlopt_sbplx = 3
        nlopt_crs2_lm = 4
        nlopt_esch = 5

    ctypedef struct cfl_min_obj:
        pass

    cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad, void *data), size_t n, void *data, nlopt_min_alg algorithm, double xtol, cfl_min_bounds *bounds)
    cfl_min_obj *cfl_gsl_min_setup(double (*obj_f)(size_t n, double *x, double *grad, void *data), size_t n, void *data, gsl_min_alg algorithm)
    int cfl_min(double *x0, double *fmin, cfl_min_obj *obj) nogil
    void cfl_min_free(cfl_min_obj *obj)


cdef extern from "../../cfl/include/basinhopping.h":
    cfl_min_obj *cfl_bh_min_setup(size_t niter, cfl_min_bounds *bounds, cfl_min_obj *lmin)


cdef extern from "../../cfl/include/cfl_h_fit.h":
    ctypedef struct param_type:
        int type
        size_t index

    ctypedef struct ex_data:
        size_t n
        double *e
        int *li

    ctypedef struct shx_data:
        double complex *pa
        float chisq_weight
        zsh_inv_data *inv_data

    ctypedef struct efit_data:
        pass

    ctypedef struct eshfit_data:
        pass

    efit_data *efit_data_alloc(zh *h, double complex *coeff, ex_data *ex, size_t
            n_zx, param_type **p)
    void efit_data_free(efit_data *data)
    eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, size_t nzeeman, zh *h,
            zh *hfo, double complex *coeff, ex_data *ex, shx_data **shx, size_t
            n_zx, param_type **p)
    void eshfit_data_free(eshfit_data *data)
    int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds, cfl_min_obj *min_obj)
    int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds, cfl_min_obj *min_obj)
    double efit_obj(size_t n, double *x, double *grad, void *data) nogil
    double eshfit_obj(size_t n, double *x, double *grad, void *data) nogil
    double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) nogil
    void eshfit_chi2(size_t n, double *x, double *grad, void *data, double *chi2) nogil
    void eshfit_hpro_chi2(size_t n, double *x, double *grad, void *data, double *chi2) nogil
