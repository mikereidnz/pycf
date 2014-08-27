/*
 * Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include <cfl_h.h>
#include <cfl_sh.h>
#include <basinhopping.h>

#ifndef _H_FIT_H_ 
#define _H_FIT_H_

/* Parameter type, used for conversion of complex parameters to real parameters
 * to-be-varied. */
typedef struct {
  /* Indicator whether real, purely imaginary, or complex. */
  int type;
  /* Index of parameter. */
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
  /* Chi^2 weighting. */
  float chisq_weight;
  /* Pointer to spin Hamiltonian inversion data. */
  zsh_inv_data *inv_data; 
  /* Level index of complete Hamiltonian which this spin Hamiltonian corresponds
   * to. */
  size_t l;
} shx_data;


/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Eigenvector array. */
  double complex *evect;
  /* Eigenvalue array. */
  double *eval;
  /* Experimental energy level data */
  ex_data *ex;
  /* The number of parameters once converted to complex type. */
  size_t n_zx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* Complete cofficient array to be passed to the diagonalization. */
  double complex *h_co;
} efit_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the complete Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Pointer to workspace for first order Hamiltonian diagonalization. */
  zhd_w *hfod_w;
  /* Complete Hamiltonian eigenvector array. */
  double complex *h_evect;
  /* Complete Hamiltonian eigenvalue array. */
  double *h_eval;
  /* Pointer to the first order Hamiltonian. */
  zh *hfo;
  /* First order Hamiltonian eigenvector array. */
  double complex *hfo_evect;
  /* First order Hamiltonian eigenvalue array. */
  double *hfo_eval;
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
  double complex **sh_pa;
  /* Experimental energy level data */
  ex_data *ex;
  /* Array of pointers to spin Hamiltonian experimental data. */
  shx_data **shx;
  /* The number of parameters once converted to complex type. */
  size_t n_zx;
  /* The number of parameters of the first order Hamiltonian in complex type. */
  size_t n_fozx;
  /* Array of pointers to parameter type structs. */
  param_type **p;
  /* Complete cofficient array to be passed to the H diagonalization. */
  double complex *h_co;
  /* Complete cofficient array to be passed to the first order H
   * diagonalization. */
  double complex *hfo_co;
} eshfit_data;

typedef enum {
  gsl_nmsimplex2rand = 0,
  gsl_nmsimplex2 = 1,
  gsl_conjugate_fr = 2,
  gsl_conjugate_pr = 3,
  gsl_vector_bfgs2 = 4, 
} bh_lmin;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
efit_data *efit_data_alloc(zh *h, double complex *h_co, ex_data *ex, size_t n_zx,
    param_type **p);
void efit_data_free(efit_data *data);
eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, size_t nzeeman, zh *h, zh
    *hfo, double complex *h_co, double complex *hfo_co, ex_data *ex, shx_data
    **shx, size_t n_zx, size_t n_fozx, param_type **p);
void eshfit_data_free(eshfit_data *data);
int bh_e_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds *bounds,
    bh_lmin lmintype);
int bh_esh_fit(double *x0, size_t nx, void *data, size_t niter, bh_bounds
    *bounds, bh_lmin lmintype);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _H_FIT_H_ */
