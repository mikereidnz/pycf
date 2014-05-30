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
 * @file    cfl_h.h
 * @brief   Diagonalization, and associated, routines for crystal-field and spin
 *          Hamiltonians.
 */

#ifndef _CFL_H_H_
#define _CFL_H_H_

#include <complex.h>
#include <cfl_crs.h>


/* 
 * @brief The tensor structure for complex valued matrix elements.
 */
typedef struct {
  /* Pointer to tensor name character array. */
  char *name;
  /* Dimension of the matrix elements. */
  int n;
  /* Pointer to the matrix elements stored in CSR form. */
  crs_zhm *matel;
} ztensor; 

/*
 * @brief The Hamiltonian structure.
 */
typedef struct {
  /* Dimension of the Hamiltonian. */
  int n;
  /* Number of tensors. */
  int nt;
  /* State labels corresponding to eigenvalues. */
  char **states;
  /* Pointer to array of pointers to complex valued tensors. */
  ztensor **t;
  /* Tensor coefficients. */
  double complex *coeff;
  /* Matrix of the complete Hamiltonian in row major form. */
  double complex *ap;
  /* Pointer to eigenvalue array. */
  double *w;
  /* Pointer to eigenvector matrix stored in col major form. */
  double complex *z;
} zh;


/*
 * @brief Work space type declaration for Hamiltonian diagonalization. 
 */
typedef struct {
  /* Workspace for summing the tensors for currently set coefficents. */
  crs_zhm **coeff_w;
  /* Length of coeff_w array. */
  int lcoeff_w;
  /* Workspace for LAPACKE_zhpevd. */
  double complex *work;
  /* Dimensions of LAPACKE_zhpevd work. */
  int lwork;
  /* LAPACKE_zhpevd RWORK. */
  double *rwork;
  /* Dimensions of LAPACKE_zhpevd rwork. */
  int lrwork;
  /* LAPACKE_zhpevd IWORK. */
  int *iwork;
  /* Dimensions of LAPACKE_zhpevd iwork. */
  int liwork;
} zhd_w;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

ztensor *ztensor_alloc(char *name, double complex *a, size_t n);
void ztensor_free(ztensor *zt);
zh *zh_alloc(int n, int nt, char **s, ztensor **t, double *w, double complex *z); 
void zh_free(zh *h);
void zh_set_coeff(zh *h, double complex *coeff);
zhd_w *zhd_w_alloc(zh *h);
void zhd_w_free(zhd_w *hd_w);
void zhd(zh *h, zhd_w *hd_w);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_H_ */
