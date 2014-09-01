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
    double complex conj(double complex z)
    double complex cexp(double complex z)
    double complex I

cdef extern from "stdlib.h":
    int atoi(const char *str)

cdef extern from "../../cfl/include/cfl_crs.h":
    ctypedef struct crs_zhm:
        pass


cdef extern from "../../cfl/include/cfl_tensor.h":
    ctypedef struct zt:
        pass

    zt *zt_alloc(char *name, double complex *a, size_t n)
    void zt_free(zt *zt)
    zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2)
    zt *zt_s(char *name, zt *t, double complex s)


cdef extern from "../../cfl/include/cfl_h.h":
    ctypedef struct zh:
        pass

    ctypedef struct zhd_w:
        pass
    
    zh *zh_alloc(int n, int nt, char **s, zt **t) 
    void zh_free(zh *h)
    void zh_set_coeff(zh *h, double complex *coeff)
    zhd_w *zhd_w_alloc(zh *h)
    void zhd_w_free(zhd_w *hd_w)
    void zhd(double *w, double complex *z, zh *h, zhd_w *hd_w) nogil


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

    ctypedef struct bh_work:
        pass

    int bh_fit(double (*obj_f)(size_t n, double *x, double *grad, void *data),
            double *x0, size_t nx, void *data, size_t niter, bh_bounds *bounds,
            bh_lmin lmintype)
    void bh_set_step(bh_work *w, double *stepsize, float target_accept_rate,
            size_t interval, float factor)


cdef extern from "../../cfl/include/h_fit.h":
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
    int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds
            *bounds, bh_lmin lmintype) nogil
    int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds
            *bounds, bh_lmin lmintype) nogil
