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

/**
 * @file    cfl_h_diag.h
 * @brief   Diagonalization, and associated, routines for crystal-field and spin
 *          Hamiltonians.
 */

#ifndef _CFL_H_DIAG_H_
#define _CFL_H_DIAG_H_

#include <lapacke.h>
#include <complex.h>

/*
 * @brief The Hamiltonian structure.
 */
typedef struct {
  size_t dim;
  char[] states;
  char[] tensors;
  double *matel;
  double *coeff;
  double *evals;
  double *evec;
} hamiltonian;

/*
 * @brief The spin Hamiltonian structure.
 * @note  This may require amendment if spin Hamiltonian inversion support is
 *        added.
 */
typedef struct {
  size_t dim;
  char[] states;
  char[] tensors;
  double *matel;
} spinh;

/*
 * @brief Workspace definition for the Hamiltonian diagonalization.
 */
typedef struct {
  double *sum_w;
  double *eig_w;
} hdiag_work;

/*
 * @brief Workspace definition for the projection from a complete Hamiltonian to
 *        the spin Hamiltonian. 
 */
typedef struct {
  double *h_prime;
  hdiag_work *hdiag_w;
} spinh_proj_work;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

hamiltonian *h_alloc(int dim);
void h_init(hamiltonian *h, char *states, char *tensors, double *matel, double *coeff, double *evals, double *evec);
void h_free(hamiltonian *h);

spinh *spinh_alloc(int dim);
void spinh_init(spinh *sh, char *states, char *tensors, double *matel);
void spinh_free(spinh *sh);

hdiag_work *hdiag_alloc(hamiltonian *h);
void h_diag(hamiltonian *h, hdiag_work *w);
void hdiag_free(hdiag_work *w);

spinh_proj_work *spinh_proj_alloc(hamiltonian *h);
void spinh_proj(hamiltonian *h, char state_range, spinh_proj_work *w);
void spinh_proj_free(spinh_proj_work *w);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_DIAG_H_ */
