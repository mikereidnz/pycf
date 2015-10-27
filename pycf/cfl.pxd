# file: ccfl.pxd
#cython: c_string_encoding=ascii

#   Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.


cdef extern from "complex.h":
    double complexconj(double complexz)
    double complexcexp(double complexz)
    double complexI


cdef extern from "../../cfl/include/cfl_csr.h":
    ctypedef struct csr_zhm:
        pass


cdef extern from "../../cfl/include/cfl_tensor.h":
    ctypedef struct sl:
        size_t n
        char *key
        char **labels
        long hash

    ctypedef struct zt:
        pass

    sl *sl_alloc(int n, char *key, int **labels)
    void sl_free(sl *l)
    zt *zt_alloc(char *name, double complex *a, int n, sl *slabels)
    zt *zt_csr_alloc(char *name, int n, int *row_ptr, int *col_in, double complex *val, sl *slabels)
    void zt_free(zt *t)
    zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2)
    zt *zt_s(char *name, zt *t, double complex s)


cdef extern from "../../cfl/include/cfl_h.h":
    ctypedef struct zh:
        int n
        int nt
        sl *slabels
        zt **t
        double complex *coeff
        double complex *ap
        
    ctypedef struct zhd_w:
        pass
   
    zh *zh_alloc(int n, int nt, zt **t) 
    void zh_free(zh *h)
    void zh_set_coeff(zh *h, double complex *coeff)
    zhd_w *zhd_w_alloc(char job, zh *h)
    void zhd_w_free(zhd_w *hd_w)
    void zhd(char job, double *w, double complex *z, zh *h, zhd_w *hd_w) nogil


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
        double complex *b
        size_t m

    ctypedef struct zshp_p_w:
        pass

    zsh *zsh_alloc(char **inter, size_t ninter, int sz, int iz, double complex **a)
    void zsh_free(zsh *sh)
    int zsh_set_pro(zsh *sh, zt **t, size_t l)
    void zsh_set_inv(zsh *sh, double complex *b, char *inter)
    zshp_p_w *zshp_p_w_alloc(zsh *sh)
    void zshp_p_w_free(zshp_p_w *shp_p_w)
    void zshp_gen_sort(double complex *hz, int pro_i, zsh *sh, zshp_p_w *shp_p_w)
    void zshp_parse(double complex *a, zsh *sh, int pro_i, zshp_p_w *shp_p_w)
    void zshp_p(double complex *hz, zsh *sh, int pro_i, zshp_p_w *shp_p_w)
    zshi_w *zshi_w_alloc(zsh_inv_data *d)
    void zshi_w_free(zshi_w *w)
    void zshi(double complex *a, zshi_w *w)
    zshp_w *zshp_w_alloc(zsh *sh)
    void zshp_w_free(zshp_w *w)
    void zshp(double complex *a, double complex *hz, int int_i, zsh *sh, zshp_w *w) nogil


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

    cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad, void *data), void (*cov_f)(double *x0, double *cov_inv, cfl_min_obj *obj), size_t n, void *data, nlopt_min_alg algorithm, double xtol, cfl_min_bounds *bounds)
    cfl_min_obj *cfl_gsl_min_setup(double (*obj_f)(size_t n, double *x, double *grad, void *data), void (*cov_f)(double *x0, double *cov_inv, cfl_min_obj *obj), size_t n, void *data, gsl_min_alg algorithm)
    int cfl_min(double *x0, double *fmin, double *cov_inv, cfl_min_obj *obj) nogil
    void cfl_min_free(cfl_min_obj *obj)


cdef extern from "../../cfl/include/basinhopping.h":
    cfl_min_obj *cfl_bh_min_setup(size_t niter, double *stepsize, float target_accept_rate, int step_adapt_int, cfl_min_bounds *bounds, cfl_min_obj *lmin)


cdef extern from "../../cfl/include/cfl_h_fit.h":
    ctypedef struct param_type:
        char type
        size_t index

    ctypedef struct ex_data:
        size_t n
        double *e
        int *li

    ctypedef struct shx_data:
        double complex *pa
        float chisq_weight

    ctypedef struct efit_data:
        pass

    ctypedef struct mhfit_data:
        pass

    ctypedef struct eshfit_data:
        pass

    efit_data *efit_data_alloc(zh *h, ex_data *ex, size_t n_zx, param_type **p)
    void efit_data_free(efit_data *data)
    mhfit_data *mhfit_data_alloc(int n, zh **ha, double *weights, int *bc_blockdim, ex_data **exa, size_t n_zx, param_type ***p)
    void mhfit_data_free(mhfit_data *data)
    eshfit_data *eshfit_data_alloc(zh *h, zh *hpro, ex_data *ex, zsh *sh, shx_data **shx, size_t n_zx, param_type **p)
    void eshfit_data_free(eshfit_data *data)
    int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds, cfl_min_obj *min_obj)
    int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds, cfl_min_obj *min_obj) 
    double efit_obj(size_t n, double *x, double *grad, void *data) nogil
    double mhfit_obj(size_t n, double *x, double *grad, void *data) nogil
    double eshfit_obj(size_t n, double *x, double *grad, void *data) nogil 
    double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) nogil 
    void efit_chi2(double *x, void *data, double *chi2) nogil
    void mhfit_chi2(double *x, void *data, double *chi2) nogil 
    void eshfit_chi2(double *x, void *data, double *chi2) nogil
    void eshfit_hpro_chi2(double *x, void *data, double *chi2) nogil 
    void efit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) nogil 
    void mhfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) nogil
    void eshfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) nogil
    void eshfit_hpro_cov(double *x0, double *cov_inv, cfl_min_obj *obj) nogil
