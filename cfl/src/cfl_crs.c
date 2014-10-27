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
 * @file    cfl_crs.c
 * @brief   Compressed row storage routines used by CFL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cfl_error.h"
#include "cfl_crs.h"

/*===========================================================================*/
/* Exported functions.                                                       */
/*===========================================================================*/

/*
 * Allocate storage and fill in values of a Hermitian sparse matrix in
 * upper-triangular compressed row storage format for double valued complex
 * entries.
 *
 * Parameters
 * ----------
 * a    An n by n dense matrix stored as a one dimensional array.
 * n    The number of columns and rows of a. 
 */
crs_zhm *crs_zhm_alloc(complex double a[], int n) {
  int i,j;
  int vi = 0;
  int ri = 0;
  int nnz = 0;
  int zrow = 1;
  crs_zhm *crs_m;
  complex double *val;
  int *col_in;
  int *row_ptr;
  /* Determining the number of non-zero entries in the upper-triangular portion
   * of a. Since we only check columns j >= i, all inspected elements may turn
   * out to be zero for large i; consequently, we add one to nnz in such cases
   * to avoid the row from being dropped. */
   for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      if (cabs(a[i*n+j]) != 0) {
        nnz++;
        zrow = 0;
      }
      else if (j==(n-1) && zrow) {
        nnz++;
      }
    }
    zrow = 1;
  }

  crs_m = (crs_zhm *) malloc(sizeof(crs_zhm));
  if (crs_m == 0) {
    CFL_ERROR_NULL("malloc failed for crs_m");
  }
  val = (complex double *) calloc(nnz,sizeof(complex double));
  if (val == 0) {
    free(crs_m);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_in = (int *) calloc(nnz,sizeof(int));
  if (col_in == 0) {
    free(crs_m);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  row_ptr = (int *) calloc((n+1),sizeof(int));
  if (row_ptr == 0) {
    free(crs_m);
    free(val);
    free(col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* col_in is an array containing the column index of all non-zero entries of
   * crs_m.  row_ptr is an array containing the entry number, that is the number
   * of non-zero entries that precede it, for each first non-zero entry of a
   * given row.  Some rows only contain elements in the lower-triangular part
   * and, since we're only checking the upper-triangular half, we include a
   * single zero entry in the last column to prevent such rows from being
   * dropped. */
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      if (cabs(a[i*n + j]) != 0) {
        val[vi] = a[i*n + j];
        col_in[vi] = j;
        if (ri == i) {
          row_ptr[ri] = vi;
          ri++;
        }
        vi++;
      }
      else if(j==(n-1) && ri==i) {
        val[vi] = 0;
        col_in[vi] = j;
        row_ptr[ri] = vi;
        ri++;
        vi++;
      }
    }
  }
  /* CRS by convention sets the n+1 value of row_ptr to nnz, since this allows
   * for convenient looping through values row by row. */
  row_ptr[n] = nnz;

  crs_m->n = n;
  crs_m->nnz = nnz;
  crs_m->val = val;
  crs_m->col_in = col_in;
  crs_m->row_ptr = row_ptr;
  return crs_m;
}

void crs_zhm_free(crs_zhm *crs_m) {
  free(crs_m->val);
  free(crs_m->col_in);
  free(crs_m->row_ptr);
  free(crs_m);
}

/*
 * Convert a Hermitian CRS matrix to a Hermitian dense matrix AP in packed
 * storage form.  Provided the input array for the CRS matrix creation was in
 * column-major form then AP will correspond to the lower-triangular portion of
 * A, packed columnwise with index such that AP(i + j*(2*n-(j+1))/2) = A(i,j)
 * for 0<=i<=j.  
 *       
 * Parameters
 * ----------
 * crs_m    Pointer to the sparse matrix in CRS form of dimension n.
 * a        Pointer to complex double valued array of length n*(n+1)/2. 
 */
void crs_zhm2zhpa(crs_zhm *crs_m, complex double *ap) {
  int i, j;
  int vi = 0;
  int n = crs_m->n;

  /* The readout is in row-major form since it allows for an iteration over a
   * contiguous block of memory to recover the original ordering of elements
   * prior to them being arranged in compressed row storage.  Provided the CRS
   * input arrays were correctly arranged in column-major form, the resulting
   * packed matrix AP will also be in column-major form and can be passed to
   * LAPACK without transposing. */
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      /* Ensure we're matching column indices on the current row. */
      if (vi == crs_m->row_ptr[i+1]) {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
      else if (crs_m->col_in[vi] == j) {
        ap[j+i*(2*n-(i+1))/2] = crs_m->val[vi];
        vi++;
      }
      else {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
    }
  }
}

/*
 * Convert a Hermitian CRS matrix to a Hermitian dense matrix A. 
 *
 * Parameters
 * ----------
 * crs_m    Pointer to the sparse matrix in CRS form.  
 * a        Pointer to allocated block of sufficient size to store n*n complex
 *          double values.
 */
void crs_zhm2zha(crs_zhm *crs_m, complex double *a) {
  int i, j;
  int vi = 0;
  int n = crs_m->n;

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      if (i>j) {
        a[i*n+j] = conj(a[j*n+i]);
      }
      /* Ensure we're matching column indices on the current row. */
      else if (vi == crs_m->row_ptr[i+1]) {
        a[i*n+j] = 0;
      }
      else if (crs_m->col_in[vi] == j) {
        a[i*n+j] = crs_m->val[vi];
        vi++;
      }
      else {
        a[i*n+j] = 0;
      }
    }
  }
}

/*
 * Given three sparse matrices of the same shape in Hermitian CRS form, A, B,
 * and C, this function calculates the number of non-zero elements of C, the
 * row_ptr of C, and allocates storage for a crs_zhm object. 
 *
 * Parameters
 * ----------
 * a    Pointer to the sparse Hermitian matrix A, in CRS form.
 * b    Pointer to the sparse Hermitian matrix B, in CRS form.
 * m    The number of columns of A, B, and C.
 * n    The number of rows of A, B, and C. 
 */
crs_zhm *crs_zhsam_alloc(crs_zhm *a, crs_zhm *b) {
  int i,j,k;
  int nnz;
  complex double *val;
  int *col_in;
  int *row_ptr;
  crs_zhm *crs_m;
  int n;
  int match = 0;
  
  if (a->n != b->n) {
    CFL_ERROR_NULL("matrix dimensions don't match");
  }
  else
    n = a->n;

  row_ptr = (int *) calloc((n+1),sizeof(int));
  if (row_ptr == 0) {
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Determine the number of non-zero elements and the row pointer of C.  The
   * row pointer of the first row is always zero, hence we do not need to worry
   * whether a or b has the first entry. */
  for (i=0; i<n; i++) {
    row_ptr[i] = a->row_ptr[i]+b->row_ptr[i]-match;
    for (j=a->row_ptr[i]; j<a->row_ptr[i+1]; j++) {
      for (k=b->row_ptr[i]; k<b->row_ptr[i+1]; k++) {
        if (a->col_in[j]==b->col_in[k]) {
          match++;
          break;
        }
      }
    }
  }

  nnz = a->nnz + b->nnz - match;
  row_ptr[n] = nnz;

  crs_m = (crs_zhm *) malloc(sizeof(crs_zhm));
  if (crs_m == 0) {
    free(row_ptr);
    CFL_ERROR_NULL("malloc failed for crs_m");
  }
  val = (complex double *) calloc(nnz,sizeof(complex double));
  if (val == 0) {
    free(row_ptr);
    free(crs_m);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_in = (int *) calloc(nnz,sizeof(int));
  if (col_in == 0) {
    free(row_ptr);
    free(crs_m);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }

  crs_m->n = n;
  crs_m->nnz = nnz;
  crs_m->val = val;
  crs_m->col_in = col_in;
  crs_m->row_ptr = row_ptr;

  return crs_m;
}

  
/*
 * Add and scale matrices in Hermitian CRS form; that is, given CRS matrices A,
 * B, and C, in addition to scalers alpha and beta, then this function
 * calculates C where C = alpha * A + beta * B.
 *
 * Parmeters
 * ---------
 * a      Hermitian CRS matrix A of dimension n by n.
 * b      Hermitian CRS matrix B of dimension n by n.
 * c      Hermitian CRS matrix C of dimension n by n.
 * alpha  Double complex valued scaler alpha.
 * beta   Double complex valued scalar beta.
 */
void crs_zhsam(crs_zhm *a, crs_zhm *b, crs_zhm *c, complex double alpha, double
    complex beta) {
  int i, j;
  int ai = 0;
  int bi = 0;

  /* The first two cases correspond to no further elements for either b or a on
   * the current row, respectively.  The next two cases correspond to further
   * elements for both a and b on the current row, yet one has a lower column
   * index and hence comes first.  Finally, the only option that remains is that
   * the column indices of both a and b match for the current row, hence we have
   * a matching entry. */
  for (i=0; i<c->n; i++) {
    for (j=c->row_ptr[i]; j<c->row_ptr[i+1]; j++) {
      if (bi == b->row_ptr[i+1]) {
        c->val[j] = alpha*a->val[ai];
        c->col_in[j] = a->col_in[ai];
        ai++;
      }
      else if (ai == a->row_ptr[i+1]) {
        c->val[j] = beta*b->val[bi];
        c->col_in[j] = b->col_in[bi];
        bi++;
      }
      else if (a->col_in[ai] < b->col_in[bi]) {
        c->val[j] = alpha*a->val[ai];
        c->col_in[j] = a->col_in[ai];
        ai++;
      }
      else if (b->col_in[bi] < a->col_in[ai]) {
        c->val[j] = beta*b->val[bi];
        c->col_in[j] = b->col_in[bi];
        bi++;
      }
      else {
        c->val[j] = alpha*a->val[ai] + beta*b->val[bi];
        c->col_in[j] = a->col_in[ai];
        ai++;
        bi++;
      }
    }
  }
}

/*
 * Allocate storage for multiplication of Hermitian CRS matrix by a double
 * complex scalar.
 *
 * Parameters
 * ----------
 * crs_m    Pointer to CRS matrix to be scaled. 
 */
crs_zhm *crs_zhsm_alloc(crs_zhm *crs_m) {
  int i;
  crs_zhm *crs_sm;
  complex double *val;
  int *col_in;
  int *row_ptr;

  crs_sm = (crs_zhm *) malloc(sizeof(crs_zhm));
  if (crs_sm == 0) {
    CFL_ERROR_NULL("malloc failed for crs_sm");
  }
  val = (complex double *) calloc(crs_m->nnz,sizeof(complex double));
  if (val == 0) {
    free(crs_sm);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  col_in = (int *) calloc(crs_m->nnz,sizeof(int));
  if (col_in == 0) {
    free(crs_sm);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  row_ptr = (int *) calloc((crs_m->n+1),sizeof(int));
  if (row_ptr == 0) {
    free(crs_sm);
    free(val);
    free(col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Identical row and column pointers. */
  for (i=0; i<crs_m->nnz; i++) 
    col_in[i] = crs_m->col_in[i];
  for (i=0; i<crs_m->n+1; i++)
    row_ptr[i] = crs_m->row_ptr[i];

  crs_sm->n = crs_m->n;
  crs_sm->nnz = crs_m->nnz;
  crs_sm->val = val;
  crs_sm->col_in = col_in;
  crs_sm->row_ptr = row_ptr;

  return crs_sm;
}

/*
 * Multiply a matrix in Hermitian CRS form by a complex double scalar. 
 *
 * Parameters
 * ----------
 * crs_m    Pointer to a CRS matrix of dimension n by n.
 * crs_sm   Pointer to a CRS matrix to which the result will be written; must
 *          have the same n, nnz, col_in, and row_ptr values as crs_m. 
 * s        Double complex valued scalar whereby to multiply crs_m.
 */
void crs_zhsm(crs_zhm *crs_m, crs_zhm *crs_sm, complex double s) {
  int i;

  for (i=0; i<crs_m->nnz; i++) {
    crs_sm->val[i] = s * crs_m->val[i];
  }
}

