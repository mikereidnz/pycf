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

#ifndef _CFL_CRS_H_
#define _CFL_CRS_H_

#include <complex.h>

/* Compressed row storage matrix for complex valued Hermitian sparse matrices. */
typedef struct {
  /* Number of rows and columns. */
  int n;
  /* Number of non-zero entries. */
  int nnz;
  /* Pointer to data array of length nnz. */
  complex double *val;
  /* Pointer to column index array of length nnz. */
  int *col_in;
  /* Pointer to row pointer array of length n+1, with the last element
   * corresponding to nnz for use in row-by-row comparisons. */
  int *row_ptr;
} zhcrs;

/* Compressed row storage matrix for complex valued sparse matrices. */
typedef struct {
  /* Number of rows and columns. */
  int n;
  /* Number of non-zero entries. */
  int nnz;
  /* Pointer to data array of length nnz. */
  complex double *val;
  /* Pointer to column index array of length nnz. */
  int *col_in;
  /* Pointer to row pointer array of length n+1, with the last element
   * corresponding to nnz for use in row-by-row comparisons. */
  int *row_ptr;
} zcrs;

#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zhcrs *zhcrs_alloc(complex double a[], int n);
void zhcrs_free(zhcrs *m);
zcrs *zhcrs2zcrs_alloc(zhcrs *hcrs_m);
void zhcrs2zcrs(zhcrs *hcrs_m, zcrs *crs_m);
void zcrs_free(zcrs *m); 
void zhcrs2zhpa(zhcrs *hcrs_m, complex double *ap);
void zhcrs2zha(zhcrs *hcrs_m, complex double *a);
void zcrs2zha(zcrs *crs_m, complex double *a);
zhcrs *zhcrssam_alloc(zhcrs *a, zhcrs *b);
void zhcrssam(zhcrs *a, zhcrs *b, zhcrs *c, complex double alpha, double
    complex beta);
zhcrs *zhcrssm_alloc(zhcrs *hcrs_m);
void zhcrssm(zhcrs *hcrs_m, zhcrs *hcrs_sm, complex double s);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_CRS_H_ */
