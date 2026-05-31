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
} zhcsr;

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
} zcsr;

/* Data for summing and scaling an array of hcrs_m matrices. */
typedef struct {
  /* The number of hcsr_m matrices to sum. */
  int n;
  /* Pointer to the resultant hcsr_m matrix. */
  zhcsr *hcsr_m;
  /* Array of mappings from each hcsr_m->val entry to the corresponding val
   * entry of the resultant matrix. */
  int **map;
} zhcsrsama_data;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

zhcsr *zhcsr_gen(complex double *a, int n);
zhcsr *zhcsr_alloc(int n, int *row_ptr, int *col_in, complex double *val);
void zhcsr_free(zhcsr *m);
zcsr *zhcsr2zcsr_alloc(zhcsr *hcsr_m);
void zhcsr2zcsr(zhcsr *hcsr_m, zcsr *csr_m);
void zcsr_free(zcsr *m);
void zhcsr2zhpa(zhcsr *hcsr_m, complex double *ap);
void zhcsr2zha(zhcsr *hcsr_m, complex double *a);
void zcsr2zha(zcsr *csr_m, complex double *a);
zhcsr *zhcsrsam_alloc(zhcsr *a, zhcsr *b);
void zhcsrsam(zhcsr *a, zhcsr *b, zhcsr *c, complex double alpha, double
    complex beta);
zhcsrsama_data *zhcsrsama_alloc(int n, zhcsr **csr_ma);
void zhcsrsama_free(zhcsrsama_data *data);
void zhcsrsama(zhcsr **csr_ma, complex double *ca, zhcsrsama_data *data);
zhcsr *zhcsrsm_alloc(zhcsr *hcsr_m);
void zhcsrsm(zhcsr *hcsr_m, zhcsr *hcsr_sm, complex double s);
void ivperm(int n, int *ix, int *perm);
zcsr *zcsr_row_perm_alloc(zcsr *m, int *p);
void zcsr_row_perm(zcsr *m, zcsr *pm, int *p);
zcsr *zcsr_col_perm_alloc(zcsr *m, int *p, int *pj);
void zcsr_col_perm(zcsr *m, zcsr *pm, int *p, int *pj);
int zcsr_cc(zcsr *m, int *labels);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_CRS_H_ */
