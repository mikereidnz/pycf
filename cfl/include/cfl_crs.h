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

/*===========================================================================*/
/* Data structures and types.                                                */
/*===========================================================================*/

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

/*===========================================================================*/
/* External declarations.                                                    */
/*===========================================================================*/

#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

crs_zhm *crs_zhm_alloc(double complex a[], int n);
void crs_zhm_free(crs_zhm *crs_m); 
void crs_zhm2zhpa(crs_zhm *crs_m, double complex *ap);
crs_zhm *crs_zhsam_alloc(crs_zhm *a, crs_zhm *b);
void crs_zhsam(crs_zhm *a, crs_zhm *b, crs_zhm *c, double complex alpha, double
    complex beta);
crs_zhm *crs_zhsm_alloc(crs_zhm *crs_m);
void crs_zhsm(crs_zhm *crs_m, crs_zhm *crs_sm, double complex s);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_CRS_H_ */
