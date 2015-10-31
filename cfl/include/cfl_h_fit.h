/*
    Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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
  /* Indicator whether parameter is real, purely imaginary, or complex. */
  char type;
  /* Complex (resultant) parameter array index. */
  size_t index;
} param_type;

/* Experimental energy level data. */
typedef struct {
  /* Number of experimental energy levels. */
  size_t n;
  /* Array of experimental energy level data. */
  double *e;
  /* Index array specifying for which levels we have data, starting at zero. */
  int *li;
} ex_data;

/* Experimental spin Hamiltonian data. */
typedef struct {
  /* Array of nine experimental spin Hamiltonian parameter values. */
  complex double *pa;
  /* chi^2 weighting. */
  double chisq_weight;
} shx_data;

/* Data for covariance matrix estimation. */
typedef struct {
  /* Index of parameter with repsect to which we differentiate. */
  size_t par_index;
  /* Index of current observable being differentiated w.r.t. parameters. */
  size_t obs_index;
  /* Storage for real-valued parameter list.  Note: par_index element will be
   * modified upon exit. */
  double *df_x;
  /* Pointer to data for minimization objective function. */
  void *obj_f_data;
  /* Array that maps obs_index, minus the number of energy level observables, to
   * the correct spin Hamlitonian interaction. */
  size_t *shi_index;
  /* Array that maps obs_index, minus the number of energy level observables, to
   * the element of the spin Hamiltonian specified by shi_index. */
  size_t *shel_index;
} cov_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Eigenvalue array. */
  double *eval;
  /* Experimental energy level data. */
  ex_data *ex;
  /* The number of parameters after conversion to complex type. */
  int n_zx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* chi^2 weighting for energy levels. */
  double echisq_weight;
} efit_data;

/* Data for multi-eigenvalue vector fit. */
typedef struct {
  /* The number of eigenvalue vectors. */
  int n;
  /* Array of length n containing pointers to Hamiltonians for each eigenvalue
   * vector. */
  zh **ha;
  /* Array of length n with each entry specifying the chi^2 weighting of the
   * corresponding ha entry. */
  double *weights;
  /* The barycenter block dimension for each corresponding ha entry. */
  int *bc_blockdim;
  /* Array of pointers to experimental energy level data. */
  ex_data **exa;
  /* The number of Hamiltonian diagonalization workspaces. */
  int nhd_w;
  /* Index specifying which hd_w/eval entry corresponds to which mh_d entry. */
  int *hi;
  /* Array of pointers to Hamiltonian diagonalization workspaces. */
  zhd_w **hd_w;
  /* Array of eigenvalue arrays. */
  double **h_eval;
  /* The number of parameters after conversion to complex type for each
   * Hamiltonian. */
  int *n_zx;
  /* Array of length n to arrays of pointers to parameter type structs. */
  param_type ***p;
  /* chi^2 weighting for first energy level vector. */
  double echisq_weight;
} mhfit_data;

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
  /* Pointer to the spin Hamiltonian. */
  zsh *sh;
  /* Pointer to spin Hamiltonian parameter projection workspace. */
  zshp_w *shp_w;
  /* Array of pointers to store inverted spin Hamiltonian parameters. */
  complex double **sh_pa;
  /* Experimental energy level data. */
  ex_data *ex;
  /* Array of pointers to spin Hamiltonian experimental data. */
  shx_data **shx;
  /* The number of parameters after conversion to complex type. */
  int n_zx;
  /* The number of parameters that are unique to sh; that is, not in coeff. */
  int n_ushx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* chi^2 weighting for energy levels. */
  double echisq_weight;
} eshfit_data;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
efit_data *efit_data_alloc(zh *h, ex_data *ex, int n_zx, param_type **p);
void efit_data_free(efit_data *data);
mhfit_data *mhfit_data_alloc(int n, zh **ha, double *weights, 
    int *bc_blockdim, ex_data **exa, int *n_zx, param_type ***p); 
void mhfit_data_free(mhfit_data *data);
eshfit_data *eshfit_data_alloc(zh *h, zh *hpro, ex_data *ex, zsh *sh, 
    shx_data **shx, int n_zx, int n_ushx, param_type **p); 
void eshfit_data_free(eshfit_data *data);
int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds,
    cfl_min_obj *min_obj);
int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds
    *bounds, cfl_min_obj *min_obj); 
double efit_obj(size_t n, double *x, double *grad, void *data);
double mhfit_obj(size_t n, double *x, double *grad, void *data);
double eshfit_obj(size_t n, double *x, double *grad, void *data);
double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data);
void efit_chi2(double *x, void *data, double *chi2);
void mhfit_chi2(double *x, void *data, double *chi2);
void eshfit_chi2(double *x, void *data, double *chi2);
void eshfit_hpro_chi2(double *x, void *data, double *chi2);
void efit_cov(double *x0, double *cov_inv, cfl_min_obj *obj);
void mhfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj);
void eshfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj);
void eshfit_hpro_cov(double *x0, double *cov_inv, cfl_min_obj *obj); 
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_FIT_H_ */
