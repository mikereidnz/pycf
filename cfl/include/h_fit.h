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

#ifndef _H_MIN_H_ 
#define _H_MIN_H_

/* Parameter type, used for conversion of complex parameters to real parameters
 * to-be-varied. */
typedef struct {
  /* Indicator whether real, purely imaginary, or complex. */
  char type;
  /* Index of parameter. */
  size_t index;
} param_type;

/* Experimental spin Hamiltonian data. */
typedef struct {
  /* Array of nine experimental spin Hamiltonian parameter values. */
  complex double *pa;
  /* Chi^2 weighting. */
  float chisq_weight;
  /* Pointer to inversion data. */
  zsh_inv_data *inv_data; 
} shx_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Pointer to eigenvector array. */
  double complex *evect;
  /* Pointer to eigenvalue array. */
  double *eval;
  /* Pointer to array of experimental energy level data. */
  double *ex;
  /* The number of parameters once converted to complex type. */
  size_t n_zx;
  /* Pointer to array of parameter type structs. */
  param_type *p;
  /* Complete cofficient array to be passed to the diagonalization. */
  double complex *coeff;
} efit_data;

/* Data for Hamiltonian fitting objective function. */
typedef struct {
  /* Pointer to the complete Hamiltonian. */
  zh *h;
  /* Pointer to workspace for Hamiltonian diagonalization. */
  zhd_w *hd_w;
  /* Pointer to workspace for first order Hamiltonian diagonalization. */
  zhd_w *hfod_w;
  /* Pointer to the complete Hamiltonian eigenvector array. */
  double complex *h_evect;
  /* Pointer to the complete Hamiltonian eigenvalue array. */
  double *h_eval;
  /* Pointer to the first order Hamiltonian. */
  zh *hfo;
  /* Pointer to the first order Hamiltonian eigenvector array. */
  double complex *hfo_evect;
  /* Pointer to the first order Hamiltonian eigenvalue array. */
  double *hfo_eval;
  /* Pointer to array of pointers to spin Hamiltonians. */
  zsh **sh_array;
  /* Number of spin Hamiltonians. */
  size_t nsh;
  /* The index of the first Zeeman term. */
  size_t nzeeman;
  /* Number of spin Hamiltonian inversions (depends on term types). */
  size_t ninv;
  /* Pointer to array of pointers to spin Hamiltonian projection workspaces. */
  zshp_w **shp_w_array;
  /* Pointer to array of pointers spin Hamiltonian inversion workspaces. */
  zshi_w **shi_w_array;
  /* Pointer to array of pointers to store inverted spin Hamiltonian parameters. */
  double complex **sh_pa;
  /* Pointer to array of experimental energy level data. */
  double *ex;
  /* Pointer to array of pointers to spin Hamiltonian experimental data. */
  shx_data **shx;
  /* The number of parameters once converted to complex type. */
  size_t n_zx;
  /* The number of parameters of the first order Hamiltonian in complex type. */
  size_t n_fozx;
  /* Pointer to array of parameter type structs. */
  param_type **p;
  /* Complete cofficient array to be passed to the H diagonalization. */
  double complex *h_coeff;
  /* Complete cofficient array to be passed to the first order H
   * diagonalization. */
  double complex *hfo_coeff;
} eshfit_data;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
efit_data *efit_data_alloc(zh *h, double *ex, size_t n_zx, param_type **p);
void efit_data_free(efit_data *data);
eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, zh *h, zh *hfo, double *ex, shx_data **shx, size_t n_zx, size_t n_fozx; param_type **p);
void eshfit_data_free(eshfit_data *data);
double efit_obj(size_t n, double *x, double *grad, void *data);
void eshfit_obj(size_t n, double *x, double *grad, void *data);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _H_MIN_H_ */
