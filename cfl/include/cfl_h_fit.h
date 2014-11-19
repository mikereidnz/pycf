/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "cfl_h.h"
#include "cfl_sh.h"
#include "cfl_min.h"

#ifndef _CFL_H_FIT_H_ 
#define _CFL_H_FIT_H_

/* Parameter type, used for conversion of real parameters returned by
 * optimization routines into complex parameters for Hamiltonian
 * diagonalization. */
typedef struct {
  /* Indicator whether real, purely imaginary, or complex. */
  int type;
  /* Complex (resultant) parameter array index. */
  size_t index;
} param_type;

/* Experimental energy level data. */
typedef struct {
  /* Number of experimental energy levels. */
  size_t n;
  /* Array of experimental energy level data. */
  double *e;
  /* Index array specifying for which levels we have data. */
  int *li;
} ex_data;

/* Experimental spin Hamiltonian data. */
typedef struct {
  /* Array of nine experimental spin Hamiltonian parameter values. */
  complex double *pa;
  /* chi^2 weighting. */
  double chisq_weight;
  /* Pointer to spin Hamiltonian inversion data. */
  zsh_inv_data *inv_data; 
} shx_data;

/* Data for covariance matrix estimation. */
typedef struct {
  /* Index of parameter with repsect to differentiate. */
  size_t par_index;
  /* Index of current observable being differentiated w.r.t. parameters. */
  size_t obs_index;
  /* Storage for real-valued parameter list.  Note: par_index element will be
   * modified upon exit. */
  double *df_x;
  /* Pointer to data for minimization objective function. */
  void *obj_f_data;
  /* The index for the current spin Hamiltonian; required for cases containing
   * Zeeman terms, which require three spin Hamiltonians per inversion. */
  size_t sh_index;
} cov_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Eigenvector array. */
  complex double *evect;
  /* Eigenvalue array. */
  double *eval;
  /* Experimental energy level data */
  ex_data *ex;
  /* The number of parameters after conversion to complex type. */
  size_t n_zx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* Complete cofficient array to be passed to the diagonalization. */
  complex double *coeff;
  /* chi^2 weighting for energy levels. */
  double echisq_weight;
} efit_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the complete Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Pointer to workspace for projection Hamiltonian diagonalization. */
  zhd_w *hprod_w;
  /* Complete Hamiltonian eigenvector array. */
  complex double *h_evect;
  /* Complete Hamiltonian eigenvalue array. */
  double *h_eval;
  /* Pointer to the projection Hamiltonian. */
  zh *hpro;
  /* Projection Hamiltonian eigenvector array. */
  complex double *hpro_evect;
  /* Projection Hamiltonian eigenvalue array. */
  double *hpro_eval;
  /* Array of pointers to spin Hamiltonians. */
  zsh **sh_a;
  /* Number of spin Hamiltonians. */
  size_t nsh;
  /* The index of the first Zeeman term. */
  size_t nzeeman;
  /* Number of spin Hamiltonian inversions (depends on term types). */
  size_t ninv;
  /* Array of pointers to spin Hamiltonian projection workspaces. */
  zshp_w **shp_w_array;
  /* Array of pointers spin Hamiltonian inversion workspaces. */
  zshi_w **shi_w_array;
  /* Array of pointers to store inverted spin Hamiltonian parameters. */
  complex double **sh_pa;
  /* Experimental energy level data */
  ex_data *ex;
  /* Array of pointers to spin Hamiltonian experimental data. */
  shx_data **shx;
  /* The number of parameters after conversion to complex type. */
  size_t n_zx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* Complete cofficient array to be passed to the diagonalization. */
  complex double *coeff;
  /* chi^2 weighting for energy levels. */
  double echisq_weight;
} eshfit_data;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
efit_data *efit_data_alloc(zh *h, complex double *coeff, ex_data *ex, size_t
    n_zx, param_type **p);
void efit_data_free(efit_data *data);
eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, size_t nzeeman, zh *h, zh
    *hpro, complex double *coeff, ex_data *ex, shx_data **shx, size_t n_zx,
    param_type **p);
void eshfit_data_free(eshfit_data *data);
int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds,
    cfl_min_obj *min_obj);
int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds
    *bounds, cfl_min_obj *min_obj); 
double efit_obj(size_t n, double *x, double *grad, void *data);
double eshfit_obj(size_t n, double *x, double *grad, void *data);
double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data);
void efit_chi2( double *x, void *data, double *chi2);
void eshfit_chi2(double *x, void *data, double *chi2);
void eshfit_hpro_chi2(double *x, void *data, double *chi2);
void efit_cov(double *x0, double *cov, cfl_min_obj *obj);
void eshfit_cov(double *x0, double *cov, cfl_min_obj *obj);
void eshfit_hpro_cov(double *x0, double *cov, cfl_min_obj *obj); 
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_FIT_H_ */
