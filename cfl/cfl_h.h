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

/*
 * @file    cfl_h_diag.h
 * @brief   Diagonalization, and associated, routines for crystal-field and spin
 *          Hamiltonians.
 */

#ifndef _CFL_H_DIAG_H_
#define _CFL_H_DIAG_H_

#include <lapacke.h>
#include <complex.h>

/*
 * @brief A dense matrix for double valued complex entries. 
 */
typedef struct {
  /* Number of rows. */
  size_t m;
  /* Number of columns. */
  size_t n;
  /* The leading dimension of the matrix X. */
  size_t ldX;
  /* Pointer to the matrix X stored in column major form. */
  double complex *X;
} zmatrix;

/*
 * @brief Compressed row storage matrix for complex valued sparse matrices.
 */
typedef struct {
  /* Number of rows. */
  size_t m;
  /* Number of columns. */
  size_t n;
  /* Number of non-zero entries. */
  int nnz;
  /* Pointer to start of values memory. */
  double complex *val;
  /* Pointer to start of column index memory. */
  int *col_in;
  /* Pointer to start of row pointer memory. */
  int *row_ptr;
} crs_zmatrix;

/* 
 * @brief The tensor structure for complex valued matrix elements.
 */
typedef struct {
  /* Pointer to tensor name character array. */
  char *name;
  /* Pointer to the matrix elements stored in CSR form. */
  crs_zm *matel;
} ztensor; 

/*
 * @brief The Hamiltonian structure.
 */
typedef struct {
  /* Dimension of the Hamiltonian. */
  size_t n;
  /* Number of tensors. */
  size_t nt;
  /* State labels corresponding to eigenvalues. */
  char **states;
  /* Complex valued tensors of the Hamiltonian. */
  ztensor *t;
  /* Tensor coefficients. */
  double complex *coeff;
  /* Matrix of the complete Hamiltonian in row major form. */
  double complex *a;
  /* Eigenvalues. */
  double complex *evals;
  /* Eigenvector matrix stored in row major form. */
  double complex *evec;
} zh;

/*
 * @brief The spin Hamiltonian structure.
 */
typedef struct {
  /* Dimension of the Hamiltonian. */
  size_t n;
  /* State labels corresponding to eigenvalues. */
  char **states;
  /* Complex valued tensors of the Hamiltonian. */
  ztensor *t;
} zspinh;

/*
 * @brief Workspace definition for the Hamiltonian diagonalization.
 */
typedef struct {
  double complex *sum_w;
  double complex *eig_w;
} zhdiag_work;

/*
 * @brief Workspace definition for the projection from a complete Hamiltonian to
 *        the spin Hamiltonian. 
 */
typedef struct {
  double *h_prime;
  hdiag_work *hdiag_w;
} zspinh_proj_work;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

crs_zmatrix *crs_zmatrix_alloc(double complex a[], size_t m, size_t n);
void crs_zmatrix_free(crs_zmatrix *crs_m);

zh *zh_alloc(size_t n, char **s);
void zh_init(zh* h, ztensor *t, complex double *coeff);
void zh_free(zh *h); 

zspinh *zspinh_alloc(size_t n, char **s);
void zspinh_init(zspinh *sh, ztensor *t);
void spinh_free(spinh *sh);

zhdiag_work *hdiag_work_alloc(zh *h);
void h_diag(hamiltonian *h, hdiag_work *w);
void hdiag_free(hdiag_work *w);

spinh_proj_work *spinh_proj_alloc(hamiltonian *h);
void spinh_proj(hamiltonian *h, char state_range, spinh_proj_work *w);
void spinh_proj_free(spinh_proj_work *w);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_DIAG_H_ */
