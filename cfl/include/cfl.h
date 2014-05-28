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
 * @file    cfl_crs.h
 * @brief   Compressed row storage routines used by CFL. 
 */

#ifndef _CFL_CRS_H_
#define _CFL_CRS_H_

#include <complex.h>

/*
 * @brief Compressed row storage matrix for complex valued Hermitian sparse
 *        matrices.
 */
typedef struct {
  /* Number of rows and columns. */
  int n;
  /* Number of non-zero entries. */
  int nnz;
  /* Pointer to data array of length nnz. */
  double complex *val;
  /* Pointer to column index array of length nnz. */
  int *col_in;
  /* Pointer to row pointer array of length n+1, with the last element
   * corresponding to nnz for use in row-by-row comparisons. */
  int *row_ptr;
} crs_zhm;


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

#endif /* _CFL_CRS_H_ */
