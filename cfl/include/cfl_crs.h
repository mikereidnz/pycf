/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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
 * Compressed row storage matrix for complex valued Hermitian sparse matrices.
 */
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
} crs_zhm;

/*===========================================================================*/
/* External declarations.                                                    */
/*===========================================================================*/

#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

crs_zhm *crs_zhm_alloc(complex double a[], int n);
void crs_zhm_free(crs_zhm *crs_m); 
void crs_zhm2zhpa(crs_zhm *crs_m, complex double *ap);
void crs_zhm2zha(crs_zhm *crs_m, complex double *a);
crs_zhm *crs_zhsam_alloc(crs_zhm *a, crs_zhm *b);
void crs_zhsam(crs_zhm *a, crs_zhm *b, crs_zhm *c, complex double alpha, double
    complex beta);
crs_zhm *crs_zhsm_alloc(crs_zhm *crs_m);
void crs_zhsm(crs_zhm *crs_m, crs_zhm *crs_sm, complex double s);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_CRS_H_ */
