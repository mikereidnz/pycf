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

/* Compressed row storage (CRS) routines. For a description of CRS, see [1]. 
 *
 * The row and column permutation algorithms, and the associated ivperm and
 * zvperm, were adapted from SPARSKIT2/FORMATS/unary.f, originally written by Y.
 * Saad.  See [2] for the original implementations. 
 *
 * [1] http://netlib.org/linalg/html_templates/node91.html
 * [2] https://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit2/sparsekit2.html
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cfl_error.h"
#include "cfl_crs.h"

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
zhcrs *zhcrs_alloc(complex double a[], int n) {
  int i,j;
  int vi = 0;
  int ri = 0;
  int nnz = 0;
  int zrow = 1;
  zhcrs *m;
  complex double *val;
  int *col_in;
  int *row_ptr;
  /* Determining the number of non-zero entries in the upper-triangular portion
   * of a. Since we only check columns j >= i, all inspected elements may turn
   * out to be zero; consequently, we add one to nnz in such cases to avoid the
   * row from being dropped. */
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

  m = (zhcrs *) malloc(sizeof(zhcrs));
  if (m == 0) {
    CFL_ERROR_NULL("malloc failed for m");
  }
  val = (complex double *) calloc(nnz,sizeof(complex double));
  if (val == 0) {
    free(m);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_in = (int *) calloc(nnz,sizeof(int));
  if (col_in == 0) {
    free(m);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  row_ptr = (int *) calloc((n+1),sizeof(int));
  if (row_ptr == 0) {
    free(m);
    free(val);
    free(col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* col_in is an array containing the column index of all non-zero entries of
   * m.  row_ptr is an array containing the entry number, that is the number
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

  m->n = n;
  m->nnz = nnz;
  m->val = val;
  m->col_in = col_in;
  m->row_ptr = row_ptr;
  return m;
}

void zhcrs_free(zhcrs *m) {
  free(m->val);
  free(m->col_in);
  free(m->row_ptr);
  free(m);
}

/*
 * Convert a matrix in Hermitian CRS form to plain CRS form.  This function
 * allocates the CRS matrix with appropriate sparsity pattern.
 *
 * Parameters
 * ----------
 * hcrs_m   Pointer to the sparse matrix in Hermitian CRS form.  
 */
zcrs *zhcrs2zcrs_alloc(zhcrs *hcrs_m) {
  int i,j,k;
  int n, nnz, nnzd, nnzz, vi;
  complex double *val;
  int *col_in;
  int *row_ptr;
  zcrs *crs_m;
  
  n = hcrs_m->n;
    
  /* Determine the number of non-zero diagonal elements. */
  nnzd = 0;
  nnzz = 0;
  for (i=0; i<n; i++) {
    if (hcrs_m->col_in[hcrs_m->row_ptr[i]] == i) {
      nnzd++;
    }
    if (hcrs_m->val[hcrs_m->row_ptr[i]] == 0) {
      /* Record the number of placeholder "non-zero" zeros required for
       * Hermitian CRS (zhcrs_alloc for details).*/
      nnzz++;
    }
  }
  nnz = hcrs_m->nnz*2-nnzd-nnzz;
  
  row_ptr = (int *) calloc((n+1),sizeof(int));
  if (row_ptr == 0) {
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }
  
  col_in = (int *) calloc(nnz, sizeof(int));
  if (col_in == 0) {
    free(row_ptr);
    CFL_ERROR_NULL("calloc failed for col_ind");
  }

  vi = 0;
  for (i=0; i<n; i++) {
    row_ptr[i] = vi; 

    /* Fill lower-triangular values (excluding diagonal). */
    for (j=0; j<i; j++) {
      for (k=hcrs_m->row_ptr[j]; k<hcrs_m->row_ptr[j+1]; k++) {
        if (hcrs_m->col_in[k] == i) {
          col_in[vi] = j;
          vi++;
          break;
        }
        else if (hcrs_m->col_in[k] > i) {
          break;
        }
      }
    }

    /* Fill the upper-triangular values; these match the original matrix. */
    for (j=hcrs_m->row_ptr[i]; j<hcrs_m->row_ptr[i+1]; j++) {
      if (hcrs_m->val[j] != 0) {
        /* Ensure all placeholder "non-zero" zeros are removed, since we don't
         * require them if we store the lower diagonal. */
        col_in[vi] = hcrs_m->col_in[j];
        vi++;
      }
    }

  }
  row_ptr[n] = nnz;

  crs_m = (zcrs *) malloc(sizeof(zcrs));
  if (crs_m == 0) {
    free(row_ptr);
    free(col_in);
    CFL_ERROR_NULL("malloc failed for crs_m");
  }
  val = (complex double *) calloc(sizeof(complex double), nnz);
  if (val == 0) {
    free(row_ptr);
    free(col_in);
    free(crs_m);
    CFL_ERROR_NULL("calloc failed for val");
  }

  crs_m->n = n;
  crs_m->nnz = nnz;
  crs_m->val = val;
  crs_m->col_in = col_in;
  crs_m->row_ptr = row_ptr;

  return crs_m;
}


void zcrs_free(zcrs *m) {
  free(m->val);
  free(m->col_in);
  free(m->row_ptr);
  free(m);
}


/*
 * Convert a matrix in Hermitian CRS form to plain CRS form.  This function
 * performs the conversion for a zcrs matrix previously created with
 * zhcrs2zcrs_alloc.  
 *
 * Parameters
 * ----------
 * hcrs_m   Pointer to the sparse matrix in Hermitian CRS form. 
 * crs_m    Pointer to the CRS matrix with correct sparsity pattern.
 */
void zhcrs2zcrs(zhcrs *hcrs_m, zcrs *crs_m) {
  int i, j;
  int hvi, row;

  row = 0;
  for (i=0; i<crs_m->nnz; i++) {
    if (crs_m->col_in[i] < row) {
      /* The lower-triangular part; we need to seek the val index of hcrs_m that
       * corresponds to the current column. */
      for(j=hcrs_m->row_ptr[crs_m->col_in[i]]; j<hcrs_m->row_ptr[crs_m->col_in[i]+1]; j++) {
        if (hcrs_m->col_in[j] == row) {
          crs_m->val[i] = conj(hcrs_m->val[j]);
          break;
        }
      }
    }
    else {
      /* Process the upper-triangular portion of the current row.  We have to
       * update hvi for the current row in case the hermitian value index is out
       * of sync due to "non-zero zeros" that have been dropped from
       * crs_m->row_ptr. */
      hvi = hcrs_m->row_ptr[row]; 
      for (; i<crs_m->row_ptr[row+1]; i++) {
        crs_m->val[i] = hcrs_m->val[hvi];
        hvi++;
      }
      row++;
      i--;
    }
  }
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
 * hcrs_m   Pointer to the sparse matrix in Hermitian CRS form of dimension n.
 * a        Pointer to complex double valued array of length n*(n+1)/2. 
 */
void zhcrs2zhpa(zhcrs *hcrs_m, complex double *ap) {
  int i, j;
  int vi = 0;
  int n = hcrs_m->n;

  /* The readout is in row-major form since it allows for an iteration over a
   * contiguous block of memory to recover the original ordering of elements
   * prior to them being arranged in compressed row storage.  Provided the CRS
   * input arrays were correctly arranged in column-major form, the resulting
   * packed matrix AP will also be in column-major form and can be passed to
   * LAPACK without transposing. */
  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      /* Ensure we're matching column indices on the current row. */
      if (vi == hcrs_m->row_ptr[i+1]) {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
      else if (hcrs_m->col_in[vi] == j) {
        ap[j+i*(2*n-(i+1))/2] = hcrs_m->val[vi];
        vi++;
      }
      else {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
    }
  }
}

/*
 * Convert a Hermitian CRS matrix to a dense matrix A. 
 *
 * Parameters
 * ----------
 * hcrs_m   Pointer to the sparse matrix in Hermitian CRS form.  
 * a        Pointer to allocated block of sufficient size to store n*n complex
 *          double values.
 */
void zhcrs2zha(zhcrs *hcrs_m, complex double *a) {
  int i, j;
  int vi = 0;
  int n = hcrs_m->n;

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      if (i>j) {
        a[i*n+j] = conj(a[j*n+i]);
      }
      /* Ensure we're matching column indices on the current row. */
      else if (vi == hcrs_m->row_ptr[i+1]) {
        a[i*n+j] = 0;
      }
      else if (hcrs_m->col_in[vi] == j) {
        a[i*n+j] = hcrs_m->val[vi];
        vi++;
      }
      else {
        a[i*n+j] = 0;
      }
    }
  }
}

/*
 * Convert a CRS matrix to a dense matrix A. 
 *
 * Parameters
 * ----------
 * crs_m    Pointer to the sparse matrix in CRS form.  
 * a        Pointer to allocated block of sufficient size to store n*n complex
 *          double values.
 */
void zcrs2zha(zcrs *crs_m, complex double *a) {
  int i, j;
  int vi = 0;
  int n = crs_m->n;

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      /* Ensure we're matching column indices on the current row. */
      if (vi == crs_m->row_ptr[i+1]) {
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
 * row_ptr of C, and allocates storage for a zhcrs object. 
 *
 * Parameters
 * ----------
 * a    Pointer to the sparse Hermitian matrix A, in CRS form.
 * b    Pointer to the sparse Hermitian matrix B, in CRS form.
 * m    The number of columns of A, B, and C.
 * n    The number of rows of A, B, and C. 
 */
zhcrs *zhcrssam_alloc(zhcrs *a, zhcrs *b) {
  int i,j,k;
  int nnz;
  complex double *val;
  int *col_in;
  int *row_ptr;
  zhcrs *hcrs_m;
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

  hcrs_m = (zhcrs *) malloc(sizeof(zhcrs));
  if (hcrs_m == 0) {
    free(row_ptr);
    CFL_ERROR_NULL("malloc failed for hcrs_m");
  }
  val = (complex double *) calloc(nnz,sizeof(complex double));
  if (val == 0) {
    free(row_ptr);
    free(hcrs_m);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_in = (int *) calloc(nnz,sizeof(int));
  if (col_in == 0) {
    free(row_ptr);
    free(hcrs_m);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }

  hcrs_m->n = n;
  hcrs_m->nnz = nnz;
  hcrs_m->val = val;
  hcrs_m->col_in = col_in;
  hcrs_m->row_ptr = row_ptr;

  return hcrs_m;
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
void zhcrssam(zhcrs *a, zhcrs *b, zhcrs *c, complex double alpha, double
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
 * hcrs_m    Pointer to CRS matrix to be scaled. 
 */
zhcrs *zhcrssm_alloc(zhcrs *hcrs_m) {
  int i;
  zhcrs *hcrs_sm;
  complex double *val;
  int *col_in;
  int *row_ptr;

  hcrs_sm = (zhcrs *) malloc(sizeof(zhcrs));
  if (hcrs_sm == 0) {
    CFL_ERROR_NULL("malloc failed for hcrs_sm");
  }
  val = (complex double *) calloc(hcrs_m->nnz,sizeof(complex double));
  if (val == 0) {
    free(hcrs_sm);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  col_in = (int *) calloc(hcrs_m->nnz,sizeof(int));
  if (col_in == 0) {
    free(hcrs_sm);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  row_ptr = (int *) calloc((hcrs_m->n+1),sizeof(int));
  if (row_ptr == 0) {
    free(hcrs_sm);
    free(val);
    free(col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Identical row and column pointers. */
  for (i=0; i<hcrs_m->nnz; i++) 
    col_in[i] = hcrs_m->col_in[i];
  for (i=0; i<hcrs_m->n+1; i++)
    row_ptr[i] = hcrs_m->row_ptr[i];

  hcrs_sm->n = hcrs_m->n;
  hcrs_sm->nnz = hcrs_m->nnz;
  hcrs_sm->val = val;
  hcrs_sm->col_in = col_in;
  hcrs_sm->row_ptr = row_ptr;

  return hcrs_sm;
}

/*
 * Multiply a matrix in Hermitian CRS form by a complex double scalar. 
 *
 * Parameters
 * ----------
 * hcrs_m     Pointer to a CRS matrix of dimension n by n.
 * hcrs_sm    Pointer to a CRS matrix to which the result will be written; must
 *            have the same n, nnz, col_in, and row_ptr values as hcrs_m. 
 * s          Double complex valued scalar whereby to multiply hcrs_m.
 */
void zhcrssm(zhcrs *hcrs_m, zhcrs *hcrs_sm, complex double s) {
  int i;

  for (i=0; i<hcrs_m->nnz; i++) {
    hcrs_sm->val[i] = s * hcrs_m->val[i];
  }
}


/* Perform an inline permutation of an integer valued array ix, according to 
 * ix(perm(j)) :=  ix(j), j=1,2,.., n. */
void ivperm(int n, int *ix, int *perm) {
  int ii, j, k, init, next; 
  int tmp, tmp1;

  k=-1; 
  init=-1;

  while (k < n) {
    init++;

    /* Test for end and whether the current value has been permuted; that is,
     * whether the current perm value is negative. */
    if (init >= n)
      break;
    else if (perm[init] < 0)
      continue;
    tmp = ix[init];
    ii = perm[init];
    perm[init] -= n;

    for (;;) {
      k++;
      /* Save the chased element. */
      tmp1 = ix[ii];
      ix[ii] = tmp;
      next = perm[ii];
      /* Test for end. */
      if (next < 0)
        break;
      else if (k >= n)
        break;
      /* tmp1 value also requires permutation. */
      tmp = tmp1;
      perm[ii] -= n;
      ii = next;
    }
  }
  /* Restore positive valued permutation vector. */
  for (j=0; j<n; j++) {
    perm[j] += n;
  }
}

/* Perform a permutation of a complex double valued array zx, according
 * to zxo(perm(j)) :=  zx(j), j=1,2,.., n.  This is a minimally modified of
 * ivperm to not-inplace and complex double. */
inline void zvperm(int n, complex double *zx, complex double *zxo, int *perm) {
  int ii, j, k, init, next; 
  complex double tmp, tmp1;

  k=-1; 
  init=-1;

  while (k < n) {
    init++;

    /* Test for end and whether the current value has been permuted; that is,
     * whether the current perm value is negative. */
    if (init >= n)
      break;
    else if (perm[init] < 0)
      continue;
    tmp = zx[init];
    ii = perm[init];
    perm[init] -= n;

    for (;;) {
      k++;
      /* Save the chased element. */
      tmp1 = zx[ii];
      zxo[ii] = tmp;
      next = perm[ii];
      /* Test for end. */
      if (next < 0)
        break;
      else if (k >= n)
        break;
      /* tmp1 value also requires permutation. */
      tmp = tmp1;
      perm[ii] -= n;
      ii = next;
    }
  }
  /* Restore positive valued permutation vector. */
  for (j=0; j<n; j++) {
    perm[j] += n;
  } 
}

/* 
 * Allocate CRS matrix with row permuted sparsity pattern. Call prior to
 * zcrs_row_perm, which copies the permuted values.
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  p       The permutation array. In the returned output matrix row i is
 *          swapped with row p(i).
 */
zcrs *zcrs_row_perm_alloc(zcrs *m, int *p) {
  int i, j, k, pk;
  zcrs *pm;

  pm = (zcrs *) malloc(sizeof(zcrs));
  if (pm == 0) {
    CFL_ERROR_NULL("malloc failed for pm");
  }
  pm->val = (complex double *) calloc(m->nnz,sizeof(complex double));
  if (pm->val == 0) {
    free(pm);
    CFL_ERROR_NULL("calloc failed for val");
  }
  pm->col_in = (int *) calloc(m->nnz,sizeof(int));
  if (pm->col_in == 0) {
    free(pm);
    free(pm->val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  pm->row_ptr = (int *) calloc((m->n+1),sizeof(int));
  if (pm->row_ptr == 0) {
    free(pm);
    free(pm->val);
    free(pm->col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Determine the number of elements per row. */
  for (j=0; j<m->n; j++) {
    i = p[j];
    pm->row_ptr[i+1] = m->row_ptr[j+1] - m->row_ptr[j];
  }
  
  /* Calculate the permuted row_ptr. */
  pm->row_ptr[0] = 0;
  for (j=0; j<m->n; j++) {
    pm->row_ptr[j+1] = pm->row_ptr[j+1] + pm->row_ptr[j];
  }

  /* Assign the permuted column indices. */
  for (i=0; i<m->n; i++) {
    pk = pm->row_ptr[p[i]];
    for (k=m->row_ptr[i]; k<m->row_ptr[i+1]; k++) {
      pm->col_in[pk] = m->col_in[k];
      pk++;
    }
  }

  pm->n = m->n;
  pm->nnz = m->nnz;

  return pm;
}


/* 
 * CRS matrix row permutation.  Requires an input CRS matrix and an output CRS
 * matrix, where the latter is assumed to have an appropriatly permuted sparsity
 * pattern.  This can be generated with zcrs_row_perm_alloc. 
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  pm      The output matrix which must have a permuted sparsity pattern prior
 *          to entry.
 *  p       The permutation array. Row i is swapped with row p(i).
 */
void zcrs_row_perm(zcrs *m, zcrs *pm, int *p) {
  int i, k, pk; 

  for (i=0; i<m->n; i++) {
    pk = pm->row_ptr[p[i]];
    for (k=m->row_ptr[i]; k<m->row_ptr[i+1]; k++) {
      pm->val[pk] = m->val[k];
      pk++;
    }
  }

}


/* 
 * Allocate CRS matrix with column permuted sparsity pattern. Call prior to
 * zcrs_col_perm, which copies the permuted values.
 *
 * Parameters
 * ----------
 *  m       The matrix to permute. 
 *  p       Array of length n, with n the number of rows of m, with entries
 *          specifying the permuted column indices.
 *  pj      Array of length (n + 1), will be overwritten with the permutation
 *          that should be applied to the value array to achieve the specified
 *          column permutation.  This is achieved with a call to zcrs_col_perm. 
 */
zcrs *zcrs_col_perm_alloc(zcrs *m, int *p, int *pj) {
  int i, j, k, pk, nnz, next, irow;
  int *iwork;
  zcrs *pm;

  nnz = m->nnz;;
  pm = (zcrs *) malloc(sizeof(zcrs));
  if (pm == 0) {
    CFL_ERROR_NULL("malloc failed for pm");
  }
  pm->val = (complex double *) calloc(m->nnz,sizeof(complex double));
  if (pm->val == 0) {
    free(pm);
    CFL_ERROR_NULL("calloc failed for val");
  }
  pm->col_in = (int *) calloc(m->nnz,sizeof(int));
  if (pm->col_in == 0) {
    free(pm);
    free(pm->val);
    CFL_ERROR_NULL("calloc failed for iwork");
  }
  pm->row_ptr = (int *) calloc((m->n+1),sizeof(int));
  if (pm->row_ptr == 0) {
    free(pm);
    free(pm->val);
    free(pm->col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }
  iwork = (int *) calloc(m->nnz,sizeof(int));
  if (iwork == 0) {
    free(pm);
    free(pm->val);
    free(pm->col_in);
    free(pm->row_ptr);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  /* Permute the column indices. */
  for (k=0; k<nnz; k++) {
    pm->col_in[k] = p[m->col_in[k]];
  }
  
  /* Now we sort the resulting matrix by increasing column order. */

  /* Compute the column pointers of the matrix; first count the number of
   * elements per column, then add them. */
  for (j=0; j<m->n; j++) {
    pj[j+1] = 0;
  }
  for (i=0; i<m->n; i++) {
    for (k=m->row_ptr[i]; k<m->row_ptr[i+1]; k++) {
      j = pm->col_in[k];
      pj[j+1] += 1;
    }
  }
  pj[0] = 0;
  for (i=0; i<m->n; i++) {
    pj[i+1] = pj[i] + pj[i+1];
  }

  /* pj starts off as the CCS col_ptr, but as we step through we increment
   * entries to step through all non-zero elements of each column. */
  for (i=0; i<m->n; i++) {
    for (k=m->row_ptr[i]; k<m->row_ptr[i+1]; k++) {
      /* j = the unsorted index of the kth permuted column. */
      j = pm->col_in[k];
      /* next = the index of the next element of the jth column. */
      next = pj[j];
      /* iwork = the sorted index of the next element. */
      iwork[next] = k;
      pj[j] += 1; 
    }
  }

  /* Record which row each nz element is in. */
  for (i=0; i<m->n; i++) {
    for (k=m->row_ptr[i]; k<m->row_ptr[i+1]; k++) {
      pj[k] = i;
    }
  }

  for (k=0; k<nnz; k++) {
    /* The permuted k index. */
    pk = iwork[k];
    /* The row index of the current nz element. */
    irow = pj[pk];
    /* row_ptr gives, for the current row, the first nz element. */
    next = m->row_ptr[irow];
    /* The current nz element should be permuted to the next position in row; we
     * keep track of this with pj. */
    pj[pk] = next;
    m->row_ptr[irow] += 1;
  }

  /* Reshift the row pointers of the original matrix. */
  for (i=m->n-1; i>=0; i--) {
    m->row_ptr[i+1] = m->row_ptr[i];
  }
  m->row_ptr[0] = 0;

  free(iwork);
  /* Permute col_in of the new matrix. */
  ivperm(nnz, pm->col_in, pj); 

  pm->n = m->n;
  pm->nnz = nnz;
  memcpy(pm->row_ptr, m->row_ptr, (m->n+1)*sizeof(int));

  return pm;
}

/* 
 * CRS matrix column permutation.  Requires an input CRS matrix and an output CRS
 * matrix, where the latter is assumed to have an appropriatly permuted sparsity
 * pattern.  This can be generated with zcrs_col_perm_alloc. 
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  pm      The output matrix which must have a permuted sparsity pattern prior
 *          to entry.
 *  pj      The permutation index used to permute the val array of m.  Values
 *          can be generated with zcrs_col_alloc.
 */
void zcrs_col_perm(zcrs *m, zcrs *pm, int *p, int *pj) {
  int i, j, pk;

  zvperm(m->nnz, m->val, pm->val, pj);
}
