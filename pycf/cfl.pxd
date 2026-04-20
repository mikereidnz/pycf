# file: cfl.pxd
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
        int n
        char *key
        int **labels
        unsigned int *lh
        unsigned int th

    ctypedef struct zt:
        pass

    sl *sl_alloc(int n, char *key, int **labels)
    void sl_free(sl *l)
    zt *zt_alloc(char *name, double complex *a, int n, sl *slabels)
    zt *zt_csr_alloc(char *name, int n, int *row_ptr, int *col_in, double complex *val, sl *slabels)
    void zt_free(zt *t)
    void zt_get_matel(zt *t, double complex *a)
    zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2)
    zt *zt_s(char *name, zt *t, double complex s)
    unsigned int fnv_hash(void *buf, int len)


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

    ctypedef struct svd_sym_w:
        pass

    zsh *zsh_alloc(char **inter, size_t ninter, int sz, int iz, int kramers, double complex **a)
    void zsh_free(zsh *sh)
    int zsh_set_pro(zsh *sh, zt **t, int l, double *coupling) 
    void zsh_set_inv(zsh *sh, double complex *b, char *inter)
    zshp_p_w *zshp_p_w_alloc(zsh *sh)
    void zshp_p_w_free(zshp_p_w *shp_p_w)
    svd_sym_w *svd_sym_w_alloc()
    void svd_sym_w_free(svd_sym_w *w)
    void svd_sym(double *a, svd_sym_w *w)
    void zshi_w_free(zshi_w *w)
    zshp_w *zshp_w_alloc(char job, zsh *sh)
    void zshp_w_free(zshp_w *w)
    void zshp(double *a, double complex *b, double complex *hz, int int_i, zsh *sh, zshp_w *w) nogil


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

    cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad, void *data), 
            size_t n, void *data, nlopt_min_alg algorithm, double xtol, double maxtime, cfl_min_bounds *bounds)
    cfl_min_obj *cfl_gsl_min_setup(double (*obj_f)(size_t n, double *x, double *grad, void *data), 
            size_t n, void *data, gsl_min_alg algorithm)
    cfl_min_obj *cfl_gsl_nls_setup(void (*f)(double *x, void *data, double *y), int n, int p, 
            void *data, double *wts, double xtol, double gtol, double ftol, double *covar, int niter)
    cfl_min_obj *cfl_siman_min_setup(double (*f)(size_t n, double *x, double *grad, void *data), 
            size_t n, void *data, int niter, cfl_min_bounds *bounds, double *stepsize, double Tstart, 
            double Tmin, double muT, double k, double *chi2accept, double *xaccept, double maxtime)
    int cfl_min(double *x0, double *fmin, cfl_min_obj *obj) nogil
    void cfl_min_free(cfl_min_obj *obj)


cdef extern from "../../cfl/include/basinhopping.h":
    cfl_min_obj *cfl_bh_min_setup(size_t niter, double *stepsize, float target_accept_rate, 
            int step_adapt_int, cfl_min_bounds *bounds, cfl_min_obj *lmin)


cdef extern from "../../cfl/include/cfl_h_fit.h":
    ctypedef struct param_type:
        char type
        int xi
        int ci

    ctypedef struct ex_data:
        int n_obs
        int n_a
        int n_d
        double *e
        double *w
        int *la
        int *ild
        int *fld
        int *lah
        int *ildh
        int *fldh

    ctypedef struct shx_data:
        double *pa
        double chisq_weight

    ctypedef struct efit_data:
        pass

    ctypedef struct mhfit_data:
        pass

    ctypedef struct eshfit_data:
        pass

    ctypedef struct meshfit_data:
        pass

    efit_data *efit_data_alloc(char job, zh *h, ex_data *ex, int n_zx, param_type **p)
    void efit_data_free(efit_data *data)
    mhfit_data *mhfit_data_alloc(char *job, int n, zh **ha, ex_data **exa, int *n_zx, param_type ***p)
    void mhfit_data_free(mhfit_data *data)
    eshfit_data *eshfit_data_alloc(char job, char inv_job, zh *h, zh *hpro, ex_data *ex, zsh *sh, 
            shx_data **shx, int n_zx, param_type **p)
    void eshfit_data_free(eshfit_data *data)
    meshfit_data *meshfit_data_alloc(int n, eshfit_data **eshfit_d)
    void meshfit_data_free(meshfit_data *data)
    double efit_obj(size_t n, double *x, double *grad, void *data) nogil
    double mhfit_obj(size_t n, double *x, double *grad, void *data) nogil
    double eshfit_obj(size_t n, double *x, double *grad, void *data) nogil 
    double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) nogil 
    double meshfit_obj(size_t n, double *x, double *grad, void *data) nogil
    void efit_chi2(double *x, void *data, double *chi2) nogil
    void mhfit_chi2(double *x, void *data, double *chi2) nogil 
    void eshfit_chi2(double *x, void *data, double *chi2) nogil
    void eshfit_hpro_chi2(double *x, void *data, double *chi2) nogil 
    void meshfit_chi2(double *x, void *data, double *chi2) nogil
    void efit_nls(double *x, void *data, double *y) 
    void mhfit_nls(double *x, void *data, double *y)

cdef extern from "../../cfl/include/cfl_zefoz.h":
    ctypedef struct zefoz_d:
        pass

    ctypedef struct zefoz_a:
        double *B
        double *v
        int ctr
        int size

    zefoz_d *zefoz_d_alloc(zh *h, int *zi)
    void zefoz_d_free(zefoz_d *data)
    zefoz_a *zefoz_a_alloc(int init_size)
    void zefoz_a_free(zefoz_a *za)
    void zefoz_search(double *Bx, double *By, double *Bz, int nx, int ny, int nz, int k, int l, double xtol, double complex **m, zefoz_a *za, zefoz_d *data) nogil

