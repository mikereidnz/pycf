/*
   Copyright (C) 2014-2016 Sebastian Horvath (sebastian.horvath@gmail.com)

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

/* Compressed sparse row (CSR) storage routines (also referred to as compressed
 * row storage, or CRS). For a description of CSR, see [1].
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
#include "cfl_csr.h"

/*
 * Allocate storage, generate the sparsity pattern, and fill in the values of a
 * Hermitian sparse matrix in upper-triangular compressed sparse row format.
 * Only the upper diagonal of the array a is inspected.
 *
 * Parameters
 * ----------
 * a    An n by n dense matrix stored as a one dimensional array.
 * n    The number of columns and rows of a.
 */
zhcsr *zhcsr_gen(complex double *a, int n) {
  int i,j, vi, nnz;
  complex double *val;
  int *col_in;
  int *row_ptr;

  zhcsr *m;

  nnz = 0;    /* The number of non-zero elements. */
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (cabs(a[i*n+j]) != 0) {
        nnz++;
      }
    }
  }

  m = (zhcsr *) malloc(sizeof(zhcsr));
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
   * m. row_ptr[i] stores the start offset of row i, including empty rows. */
  vi = 0;   /* Value index for the Hermitian CSR matrix. */
  for (i = 0; i < n; i++) {
    /* Fix: record the row start directly instead of using 0 as an "unset"
     * sentinel, which breaks for row 0 and leading zero rows. */
    row_ptr[i] = vi;
    for (j = i; j < n; j++) {
      if (cabs(a[i*n + j]) != 0) {
        val[vi] = a[i*n + j];
        col_in[vi] = j;
        vi++;
      }
    }
  }
  /* Fix: terminate row_ptr with the number of stored upper-triangular values. */
  row_ptr[n] = vi;

  m->n = n;
  m->nnz = nnz;
  m->val = val;
  m->col_in = col_in;
  m->row_ptr = row_ptr;

  return m;
}


/* Allocate storage and generate Hermitian CSR matrix from non-Hermitian CSR
 * matrix.  Only upper diagonal values are added.
 *
 * Parameters
 * ----------
 *  n         The matrix dimension.
 *  row_ptr   The row pointer array.
 *  col_in    The column index array.
 *  val       Array containing the values of the non-zero elements.
 */
zhcsr *zhcsr_alloc(int n, int *row_ptr, int *col_in, complex double *val) {
  int i, k, vi, nnz;
  zhcsr *m;

  /* The number of non-zero values for the Hermitian CSR matrix. */
  nnz = 0;
  for (i = 0; i < n; i++) {
    /* Check whether there's a non-zero element in the upper diagonal. */
    for (k = row_ptr[i]; k < row_ptr[i+1]; k++) {
      if (col_in[k] >= i) {
        nnz++;
      }
    }
  }

  m = (zhcsr *) malloc(sizeof(zhcsr));
  if (m == 0) {
    CFL_ERROR_NULL("malloc failed for m");
  }
  m->val = (complex double *) calloc(nnz, sizeof(complex double));
  if (m->val == 0) {
    free(m);
    CFL_ERROR_NULL("calloc failed for m->val");
  }
  m->col_in = (int *) malloc(nnz*sizeof(int));
  if (m->col_in == 0) {
    free(m->val);
    free(m);
    CFL_ERROR_NULL("malloc failed for m->col_in");
  }
  m->row_ptr = (int *) calloc((n+1),sizeof(int));
  if (m->row_ptr == 0) {
    free(m->val);
    free(m->col_in);
    free(m);
    CFL_ERROR_NULL("calloc failed for m->row_ptr");
  }

  vi = 0;     /* Value index for the Hermitian CSR matrix. */
  for (i = 0; i < n; i++) {
    /* Fix: store each row start explicitly so empty rows do not rely on
     * 0 acting as both a valid offset and a sentinel value. */
    m->row_ptr[i] = vi;
    for (k = row_ptr[i]; k < row_ptr[i+1]; k++) {
      if (col_in[k] >= i) {
        m->col_in[vi] = col_in[k];
        m->val[vi] = val[k];
        vi++;
      }
    }
  }
  /* Fix: terminate row_ptr with the number of stored upper-triangular values. */
  m->row_ptr[n] = vi;
  m->n = n;
  m->nnz = nnz;

  return m;
}


void zhcsr_free(zhcsr *m) {
  free(m->val);
  free(m->col_in);
  free(m->row_ptr);
  free(m);
}

/*
 * Convert a matrix in Hermitian CSR form to plain CSR form.  This function
 * allocates the CSR matrix with appropriate sparsity pattern.
 *
 * Parameters
 * ----------
 * hcsr_m   Pointer to the sparse matrix in Hermitian CSR form.
 */
zcsr *zhcsr2zcsr_alloc(zhcsr *hcsr_m) {
  int i,j,k;
  int n, nnz, vi;
  complex double *val;
  int *col_in;
  int *row_ptr;
  zcsr *csr_m;

  n = hcsr_m->n;

  /* Determine the number of non-zero diagonal elements. */
  nnz = 0;
  for (i = 0; i < n; i++) {
    if (hcsr_m->row_ptr[i] < hcsr_m->row_ptr[i+1] &&
        hcsr_m->col_in[hcsr_m->row_ptr[i]] == i) {
      nnz++;
    }
  }
  nnz = hcsr_m->nnz*2-nnz;

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
  for (i = 0; i < n; i++) {
    row_ptr[i] = vi;
    /* The lower-triangular part; we need to seek through the column index of
     * hcsr_m that corresponds to the current row. */
    for (j = 0; j < i; j++) {
      for (k = hcsr_m->row_ptr[j]; k < hcsr_m->row_ptr[j+1]; k++) {
        if (hcsr_m->col_in[k] == i) {
          col_in[vi] = j;
          vi++;
          break;
        }
        else if (hcsr_m->col_in[k] > i) {
          break;
        }
      }
    }

    /* Fill the upper-triangular values; these match the original matrix. */
    for (j = hcsr_m->row_ptr[i]; j < hcsr_m->row_ptr[i+1]; j++) {
      col_in[vi] = hcsr_m->col_in[j];
      vi++;
    }
  }
  row_ptr[n] = nnz;

  csr_m = (zcsr *) malloc(sizeof(zcsr));
  if (csr_m == 0) {
    free(row_ptr);
    free(col_in);
    CFL_ERROR_NULL("malloc failed for csr_m");
  }
  val = (complex double *) calloc(nnz, sizeof(complex double));
  if (val == 0) {
    free(row_ptr);
    free(col_in);
    free(csr_m);
    CFL_ERROR_NULL("calloc failed for val");
  }

  csr_m->n = n;
  csr_m->nnz = nnz;
  csr_m->val = val;
  csr_m->col_in = col_in;
  csr_m->row_ptr = row_ptr;

  return csr_m;
}


void zcsr_free(zcsr *m) {
  free(m->val);
  free(m->col_in);
  free(m->row_ptr);
  free(m);
}


/*
 * Convert a matrix in Hermitian CSR form to plain CSR form.  This function
 * performs the conversion for a zcsr matrix previously created with
 * zhcsr2zcsr_alloc.
 *
 * Parameters
 * ----------
 * hcsr_m   Pointer to the sparse matrix in Hermitian CSR form.
 * csr_m    Pointer to the CSR matrix with correct sparsity pattern.
 */
void zhcsr2zcsr(zhcsr *hcsr_m, zcsr *csr_m) {
  int i, j;
  int row, utvi;

  row = 0;  /* Index of the current row. */
  utvi = 0;  /* Value index of the upper-triangular portion. */
  for (i = 0; i < csr_m->nnz; i++) {
    if (csr_m->col_in[i] < row) {
      /* The lower-triangular part; we need to seek through the column index of
       * hcsr_m that corresponds to the current row. */
      for(j = hcsr_m->row_ptr[csr_m->col_in[i]];
          j < hcsr_m->row_ptr[csr_m->col_in[i]+1]; j++) {
        if (hcsr_m->col_in[j] == row) {
          csr_m->val[i] = conj(hcsr_m->val[j]);
          break;
        }
      }
    }
    else {
      /* Process the upper-triangular portion of the current row. */
      for (; i < csr_m->row_ptr[row+1]; i++) {
        csr_m->val[i] = hcsr_m->val[utvi];
        utvi++;
      }
      row++;
      i--;
    }
  }
}

/*
 * Convert a Hermitian CSR matrix to a Hermitian dense matrix AP in packed
 * storage form.  Provided the input array for the CSR matrix creation was in
 * column-major form then AP will correspond to the lower-triangular portion of
 * A, packed column wise with index such that AP(i + j*(2*n-(j+1))/2) = A(i,j)
 * for 0<=i<=j.
 *
 * Parameters
 * ----------
 * hcsr_m   Pointer to the sparse matrix in Hermitian CSR form of dimension n.
 * a        Pointer to complex double valued array of length n*(n+1)/2.
 */
void zhcsr2zhpa(zhcsr *hcsr_m, complex double *ap) {
  int i, j;
  int vi = 0;
  int n = hcsr_m->n;

  /* The readout is in row-major form since it allows for an iteration over a
   * contiguous block of memory to recover the original ordering of elements
   * prior to them being arranged in compressed row storage.  Provided the CSR
   * input arrays were correctly arranged in column-major form, the resulting
   * packed matrix AP will also be in column-major form and can be passed to
   * LAPACK without transposing. */
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      /* Ensure we're matching column indices on the current row. */
      if (vi == hcsr_m->row_ptr[i+1]) {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
      else if (hcsr_m->col_in[vi] == j) {
        ap[j+i*(2*n-(i+1))/2] = hcsr_m->val[vi];
        vi++;
      }
      else {
        ap[j+i*(2*n-(i+1))/2] = 0;
      }
    }
  }
}

/*
 * Convert a Hermitian CSR matrix to a dense matrix A (row-major).
 *
 * Each entry stored in the CSR is written verbatim into the dense block.
 * In our zhcsr layout only the upper triangle (j >= i) is ever stored
 * (see zhcsr_alloc, ~line 125, which discards strictly-lower entries),
 * so the resulting dense matrix has a zeroed lower triangle.
 *
 * Note (history, 2026-04): an earlier version of this routine ran a
 * second pass that Hermitian-completed the lower triangle from
 * conj(upper). That made sense for the assembled Hamiltonian (which is
 * Hermitian by construction) but silently fabricated a fake conjugate
 * lower triangle for *individual* tensors T_kq, which are not
 * themselves Hermitian. The completion was therefore visible — and
 * incorrect — through tensor.get_matel() (the Cython introspection API)
 * and through the spin-Hamiltonian projection in cfl_sh.c, which had
 * to work around it (see inten.vtrans for one such workaround). The
 * second pass has been removed.
 *
 * The Hamiltonian assembly path does NOT use this function: it fills
 * its dense block via zhcsr2zcsr (above), which performs its own
 * Hermitian completion (line 312) on the *summed* CSR. That re-derivation
 * is correct because the assembled H is Hermitian, regardless of how
 * individual tensors were stored.
 *
 * Callers that need a fully Hermitian dense block from an individual
 * tensor (e.g. for cblas_zhemm) must perform the lower-triangle fill
 * themselves; see cfl_sh.c around line 364 for an example.
 *
 * Parameters
 * ----------
 * hcsr_m   Pointer to the sparse matrix in Hermitian CSR form.
 * a        Pointer to allocated block of sufficient size to store n*n complex
 *          double values.
 */
void zhcsr2zha(zhcsr *hcsr_m, complex double *a) {
  int i, k;
  int n = hcsr_m->n;

  memset(a, 0, (size_t)n * (size_t)n * sizeof(complex double));
  for (i = 0; i < n; i++) {
    for (k = hcsr_m->row_ptr[i]; k < hcsr_m->row_ptr[i+1]; k++) {
      a[i*n + hcsr_m->col_in[k]] = hcsr_m->val[k];
    }
  }
}

/*
 * Convert a CSR matrix to a dense matrix A.
 *
 * Parameters
 * ----------
 * csr_m    Pointer to the sparse matrix in CSR form.
 * a        Pointer to allocated block of sufficient size to store n*n complex
 *          double values.
 */
void zcsr2zha(zcsr *csr_m, complex double *a) {
  int i, j;
  int vi = 0;
  int n = csr_m->n;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      /* Ensure we're matching column indices on the current row. */
      if (vi == csr_m->row_ptr[i+1]) {
        a[i*n+j] = 0;
      }
      else if (csr_m->col_in[vi] == j) {
        a[i*n+j] = csr_m->val[vi];
        vi++;
      }
      else {
        a[i*n+j] = 0;
      }
    }
  }
}

/*
 * Given three sparse matrices of the same shape in Hermitian CSR form, A, B,
 * and C as well as scalars alpha and beta, then, for C = alpha * A + beta * B,
 * this function calculates the number of non-zero elements of C, the row_ptr of
 * C, and allocates storage for a zhcsr object.
 *
 * Parameters
 * ----------
 * a    Pointer to the sparse Hermitian matrix A, in CSR form.
 * b    Pointer to the sparse Hermitian matrix B, in CSR form.
 * m    The number of columns of A, B, and C.
 * n    The number of rows of A, B, and C.
 */
zhcsr *zhcsrsam_alloc(zhcsr *a, zhcsr *b) {
  int i, j, k, ai, bi;
  int n, nnz, match;
  complex double *val;
  int *col_in;
  int *row_ptr;
  zhcsr *hcsr_m;


  if (a->n != b->n) {
    CFL_ERROR_NULL("matrix dimensions don't match");
  }
  else {
    n = a->n;
  }
  row_ptr = (int *) malloc((n+1)*sizeof(int));
  if (row_ptr == 0) {
    CFL_ERROR_NULL("malloc failed for row_ptr");
  }

  /* Match always calculates the i+1 number of matching col_in entries of a and
   * b, which is subtracted from the c row_ptr during the next iteration.*/
  match=0;
  row_ptr[0] = 0;
  for (i = 0; i < n; i++) {
    row_ptr[i] = a->row_ptr[i] + b->row_ptr[i] - match;
    for (j = a->row_ptr[i]; j < a->row_ptr[i+1]; j++) {
      for (k = b->row_ptr[i]; k < b->row_ptr[i+1]; k++) {
        if (a->col_in[j] == b->col_in[k]) {
          match++;
          break;
        }
      }
    }
  }

  nnz = a->nnz + b->nnz - match;
  row_ptr[n] = nnz;

  col_in = (int *) calloc(nnz,sizeof(int));
  if (col_in == 0) {
    free(row_ptr);
    /* Fix: hcsr_m has not been allocated on this error path. */
    CFL_ERROR_NULL("calloc failed for col_in");
  }

  /* The first two cases correspond to no further elements for either b or a on
   * the current row, respectively.  The next two cases correspond to further
   * elements for both a and b on the current row, yet one has a lower column
   * index and hence comes first.  Finally, the only option that remains is that
   * the column indices of both a and b match for the current row, hence we have
   * a matching entry. */
  ai = 0;
  bi = 0;
  for (i = 0; i < n; i++) {
    for (j = row_ptr[i]; j < row_ptr[i+1]; j++) {
      if (bi == b->row_ptr[i+1]) {
        col_in[j] = a->col_in[ai];
        ai++;
      }
      else if (ai == a->row_ptr[i+1]) {
        col_in[j] = b->col_in[bi];
        bi++;
      }
      else if (a->col_in[ai] < b->col_in[bi]) {
        col_in[j] = a->col_in[ai];
        ai++;
      }
      else if (b->col_in[bi] < a->col_in[ai]) {
        col_in[j] = b->col_in[bi];
        bi++;
      }
      else {
        col_in[j] = a->col_in[ai];
        ai++;
        bi++;
      }
    }
  }

  hcsr_m = (zhcsr *) malloc(sizeof(zhcsr));
  if (hcsr_m == 0) {
    free(row_ptr);
    free(col_in);
    CFL_ERROR_NULL("malloc failed for hcsr_m");
  }
  val = (complex double *) calloc(nnz, sizeof(complex double));
  if (val == 0) {
    free(row_ptr);
    free(col_in);
    free(hcsr_m);
    CFL_ERROR_NULL("calloc failed for val");
  }

  hcsr_m->n = n;
  hcsr_m->nnz = nnz;
  hcsr_m->val = val;
  hcsr_m->col_in = col_in;
  hcsr_m->row_ptr = row_ptr;

  return hcsr_m;
}


/*
 * Add and scale matrices in Hermitian CSR form; that is, given CSR matrices A,
 * B, and C, in addition to scalers alpha and beta, then this function
 * calculates C where C = alpha * A + beta * B.
 *
 * Parameters
 * ----------
 * a      Hermitian CSR matrix A of dimension n by n.
 * b      Hermitian CSR matrix B of dimension n by n.
 * c      Hermitian CSR matrix C of dimension n by n.
 * alpha  Double complex valued scaler alpha.
 * beta   Double complex valued scalar beta.
 */
void zhcsrsam(zhcsr *a, zhcsr *b, zhcsr *c, complex double alpha, double
    complex beta) {
  int i, j, ai, bi;

  /* The first two cases correspond to no further elements for either b or a on
   * the current row, respectively.  The next two cases correspond to further
   * elements for a and b on the current row, respectively.
   */
  ai = 0;
  bi = 0;
  for (i = 0; i < c->n; i++) {
    for (j = c->row_ptr[i]; j < c->row_ptr[i+1]; j++) {
      if (bi == b->row_ptr[i+1]) {
        c->val[j] = alpha * a->val[ai];
        ai++;
      }
      else if (ai == a->row_ptr[i+1]) {
        c->val[j] = beta * b->val[bi];
        bi++;
      } else {
        c->val[j] = 0;
        if (a->col_in[ai] == c->col_in[j]) {
          c->val[j] += alpha * a->val[ai];
          ai++;
        }
        if (b->col_in[bi] == c->col_in[j]) {
          c->val[j] += beta * b->val[bi];
          bi++;
        }
      }
    }
  }
}

/* Allocate storage and determine intermediate sparsity patterns for scaling
 * and adding an array of zhcsr matrices.
 *
 * Parameters
 * ----------
 * n          The number of zhcsr matrices.
 * csr_ma     Array of zhcsr matrices.
 */
zhcsrsama_data *zhcsrsama_alloc(int n, zhcsr **csr_ma) {
  int i, j, k, jj, vi;
  int match, col_found, nnz;
  size_t nnz_sz;
  int *row_ptr, *col_in;
  complex double *val;
  zhcsr *hcsr_m;
  zhcsrsama_data *data;

  row_ptr = (int *) calloc((csr_ma[0]->n+1), sizeof(int));
  if (row_ptr == 0) {
    CFL_ERROR_NULL("malloc failed for row_ptr");
  }
  data = (zhcsrsama_data *) malloc(sizeof(zhcsrsama_data));
  if (data == 0) {
    free(row_ptr);
    CFL_ERROR_NULL("malloc failed for data");
  }
  data->map = (int **) malloc(n*sizeof(int *));
  if (data->map == 0) {
    free(data);
    free(row_ptr);
    CFL_ERROR_NULL("malloc failed for data->map");
  }
  for (i=0; i<n; i++) {
    data->map[i] = (int *) malloc(csr_ma[i]->nnz*sizeof(int));
    if (data->map[i] == 0) {
      for (j=0; j<i; j++) {
        free(data->map[j]);
      }
      free(data->map);
      free(data);
      free(row_ptr);
      CFL_ERROR_NULL("malloc failed for data->map[i]");
    }
  }

  col_found = 0;
  match = 0;
  row_ptr[0] = 0;
  for (i=0; i<csr_ma[0]->n; i++) {
    for (k=0; k<n; k++) {
      row_ptr[i] += csr_ma[k]->row_ptr[i];
    }
    /* Subtract the total number of previous matching elements, since we only
     * want the number of unique entries up to this row for the resultant
     * matrix.*/
    row_ptr[i] -= match;
    for (j=0; j<csr_ma[0]->n; j++) {
      for (k=0; k<n; k++) {
        for (jj=csr_ma[k]->row_ptr[i]; jj<csr_ma[k]->row_ptr[i+1]; jj++) {
          if (csr_ma[k]->col_in[jj] == j) {
            /* Check whether a previous matrix had an element in this column, in
             * which case this constitutes a matching entry. */
            if (col_found) {
              match++;
            }
            else {
              col_found = 1;
            }
            break;
          }
        }
      }
      col_found = 0;
    }
  }

  nnz = 0;
  for (i=0; i<n; i++) {
    nnz += csr_ma[i]->nnz;
  }
  nnz -= match;
  if (nnz < 0) {
    for (i=0; i<n; i++) {
      free(data->map[i]);
    }
    free(data->map);
    free(data);
    free(row_ptr);
    CFL_ERROR_NULL("negative nnz computed while merging sparse matrices");
  }
  nnz_sz = (size_t) nnz;
  row_ptr[csr_ma[0]->n] = nnz;

  col_in = (int *) calloc(nnz_sz, sizeof(int));
  if (col_in == 0) {
    for (i=0; i<n; i++) {
      free(data->map[i]);
    }
    free(data->map);
    free(data);
    free(row_ptr);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  hcsr_m = (zhcsr *) malloc(sizeof(zhcsr));
  if (hcsr_m == 0) {
    for (i=0; i<n; i++) {
      free(data->map[i]);
    }
    free(data->map);
    free(data);
    free(col_in);
    free(row_ptr);
    CFL_ERROR_NULL("malloc failed for hcsr_m");
  }
  val = (complex double *) calloc(nnz_sz, sizeof(complex double));
  /* This check most probably intended to test val == 0 rather than col_in == 0. */
  if (val == 0) {
    for (i=0; i<n; i++) {
      free(data->map[i]);
    }
    free(data->map);
    free(data);
    free(col_in);
    free(row_ptr);
    free(hcsr_m);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_found = 0;  /* Flag for whether current column has been found before. */
  vi = 0;         /* Index keeping track of the next val entry. */
  for (i=0; i<csr_ma[0]->n; i++) {
    for (j=0; j<csr_ma[0]->n; j++) {
      for (k=0; k<n; k++) {
        for (jj=csr_ma[k]->row_ptr[i]; jj<csr_ma[k]->row_ptr[i+1]; jj++) {
          if (csr_ma[k]->col_in[jj] == j) {
            data->map[k][jj] = vi;
            if (!col_found) {
              col_in[vi] = j;
              col_found = 1;
            }
          }
        }
      }
      if (col_found) {
        vi++;
      }
      col_found = 0;
    }
  }

  hcsr_m->n = csr_ma[0]->n;
  hcsr_m->nnz = nnz;
  hcsr_m->val = val;
  hcsr_m->col_in = col_in;
  hcsr_m->row_ptr = row_ptr;
  data->n = n;
  data->hcsr_m = hcsr_m;

  return data;
}

void zhcsrsama_free(zhcsrsama_data *data) {
  int i;

  if (data == 0) {
    return;
  }

  zhcsr_free(data->hcsr_m);
  for (i=0; i<data->n; i++) {
    free(data->map[i]);
  }
  free(data->map);
  free(data);
}


/* Scale and add array of hermitian csr matrices.  This function requires an
 * hcsr_m matrix, and associated val index mapping, allocated with the
 * zhcsrsama_alloc function.
 *
 * Parameters
 * ----------
 * csr_ma     Array of zhcsr matrices.
 * ca         Array of coefficients used to scale each corresponding zhcsr
 *            matrix.
 * data       Data type for this scale and addition previously alloced with
 *            zhcsrsama_alloc; contains all the required index mapping as well
 *            as a pointer to the hcsr matrix.
 */
void zhcsrsama(zhcsr **csr_ma, complex double *ca, zhcsrsama_data *data) {
  int i, j;

  memset(data->hcsr_m->val, 0, data->hcsr_m->nnz*sizeof(complex double));
  for (i=0; i<data->n; i++) {
#pragma omp parallel for private(j) schedule(dynamic)
    for (j=0; j<csr_ma[i]->nnz; j++) {
      data->hcsr_m->val[data->map[i][j]] += ca[i]*csr_ma[i]->val[j];
    }
  }
}


/*
 * Allocate storage for multiplication of Hermitian CSR matrix by a double
 * complex scalar.
 *
 * Parameters
 * ----------
 * hcsr_m    Pointer to CSR matrix to be scaled.
 */
zhcsr *zhcsrsm_alloc(zhcsr *hcsr_m) {
  int i;
  zhcsr *hcsr_sm;
  complex double *val;
  int *col_in;
  int *row_ptr;

  hcsr_sm = (zhcsr *) malloc(sizeof(zhcsr));
  if (hcsr_sm == 0) {
    CFL_ERROR_NULL("malloc failed for hcsr_sm");
  }
  val = (complex double *) calloc(hcsr_m->nnz,sizeof(complex double));
  if (val == 0) {
    free(hcsr_sm);
    CFL_ERROR_NULL("calloc failed for val");
  }
  col_in = (int *) calloc(hcsr_m->nnz,sizeof(int));
  if (col_in == 0) {
    free(hcsr_sm);
    free(val);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  row_ptr = (int *) calloc((hcsr_m->n+1),sizeof(int));
  if (row_ptr == 0) {
    free(hcsr_sm);
    free(val);
    free(col_in);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Identical row and column pointers. */
  for (i = 0; i < hcsr_m->nnz; i++)
    col_in[i] = hcsr_m->col_in[i];
  for (i = 0; i < hcsr_m->n+1; i++)
    row_ptr[i] = hcsr_m->row_ptr[i];

  hcsr_sm->n = hcsr_m->n;
  hcsr_sm->nnz = hcsr_m->nnz;
  hcsr_sm->val = val;
  hcsr_sm->col_in = col_in;
  hcsr_sm->row_ptr = row_ptr;

  return hcsr_sm;
}

/*
 * Multiply a matrix in Hermitian CSR form by a complex double scalar.
 *
 * Parameters
 * ----------
 * hcsr_m     Pointer to a CSR matrix of dimension n by n.
 * hcsr_sm    Pointer to a CSR matrix to which the result will be written; must
 *            have the same n, nnz, col_in, and row_ptr values as hcsr_m.
 * s          Double complex valued scalar whereby to multiply hcsr_m.
 */
void zhcsrsm(zhcsr *hcsr_m, zhcsr *hcsr_sm, complex double s) {
  int i;

  for (i = 0; i < hcsr_m->nnz; i++) {
    hcsr_sm->val[i] = s * hcsr_m->val[i];
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
  for (j = 0; j < n; j++) {
    perm[j] += n;
  }
}


/*
 * Allocate CSR matrix with row permuted sparsity pattern. Call prior to
 * zcsr_row_perm, which copies the permuted values.
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  p       The permutation array. In the returned output matrix row i is
 *          swapped with row p(i).
 */
zcsr *zcsr_row_perm_alloc(zcsr *m, int *p) {
  int i, j, k, pk;
  zcsr *pm;

  pm = (zcsr *) malloc(sizeof(zcsr));
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
    /* Fix: free owned buffers before freeing the parent struct. */
    free(pm->val);
    free(pm);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  pm->row_ptr = (int *) calloc((m->n+1),sizeof(int));
  if (pm->row_ptr == 0) {
    /* Fix: free owned buffers before freeing the parent struct. */
    free(pm->val);
    free(pm->col_in);
    free(pm);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }

  /* Determine the number of elements per row. */
  for (j = 0; j < m->n; j++) {
    i = p[j];
    pm->row_ptr[i+1] = m->row_ptr[j+1] - m->row_ptr[j];
  }

  /* Calculate the permuted row_ptr. */
  pm->row_ptr[0] = 0;
  for (j = 0; j < m->n; j++) {
    pm->row_ptr[j+1] = pm->row_ptr[j+1] + pm->row_ptr[j];
  }

  /* Assign the permuted column indices. */
  for (i = 0; i < m->n; i++) {
    pk = pm->row_ptr[p[i]];
    for (k = m->row_ptr[i]; k < m->row_ptr[i+1]; k++) {
      pm->col_in[pk] = m->col_in[k];
      pk++;
    }
  }

  pm->n = m->n;
  pm->nnz = m->nnz;

  return pm;
}


/*
 * CSR matrix row permutation.  Requires an input CSR matrix and an output CSR
 * matrix, where the latter is assumed to have an appropriately permuted
 * sparsity pattern.  This can be generated with zcsr_row_perm_alloc.
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  pm      The output matrix which must have a permuted sparsity pattern prior
 *          to entry.
 *  p       The permutation array. Row i is swapped with row p(i).
 */
void zcsr_row_perm(zcsr *m, zcsr *pm, int *p) {
  int i, k, pk;

#pragma omp parallel for private(i,pk,k) schedule(dynamic)
  for (i = 0; i < m->n; i++) {
    pk = pm->row_ptr[p[i]];
    for (k = m->row_ptr[i]; k < m->row_ptr[i+1]; k++) {
      pm->val[pk] = m->val[k];
      pk++;
    }
  }
}


/*
 * Allocate CSR matrix with column permuted sparsity pattern. Call prior to
 * zcsr_col_perm, which copies the permuted values.
 *
 * Parameters
 * ----------
 *  m       The matrix to permute.
 *  p       Array of length n, with n the number of rows of m, with entries
 *          specifying the permuted column indices.
 *  pj      Array of length (n + 1), will be overwritten with the permutation
 *          that should be applied to the value array to achieve the specified
 *          column permutation.  This is achieved with a call to zcsr_col_perm.
 */
zcsr *zcsr_col_perm_alloc(zcsr *m, int *p, int *pj) {
  int i, j, k, pk, nnz, next, irow;
  int *iwork;
  zcsr *pm;

  nnz = m->nnz;;
  pm = (zcsr *) malloc(sizeof(zcsr));
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
    /* Free members before the struct to avoid use-after-free. */
    free(pm->val);
    free(pm);
    CFL_ERROR_NULL("calloc failed for col_in");
  }
  pm->row_ptr = (int *) calloc((m->n+1),sizeof(int));
  if (pm->row_ptr == 0) {
    free(pm->val);
    free(pm->col_in);
    free(pm);
    CFL_ERROR_NULL("calloc failed for row_ptr");
  }
  iwork = (int *) calloc(m->nnz,sizeof(int));
  if (iwork == 0) {
    free(pm->val);
    free(pm->col_in);
    free(pm->row_ptr);
    free(pm);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  /* Permute the column indices. */
  for (k = 0; k < nnz; k++) {
    int col_idx = m->col_in[k];
    /* Validate column index before permutation */
    if (col_idx < 0 || col_idx >= m->n) {
      CFL_ERROR_NULL("invalid column index in input matrix");
    }
    int perm_idx = p[col_idx];
    /* Validate permuted index is within bounds */
    if (perm_idx < 0 || perm_idx >= m->n) {
      CFL_ERROR_NULL("permutation array produced out-of-bounds column index");
    }
    pm->col_in[k] = perm_idx;
  }

  /* Now we sort the resulting matrix by increasing column order. */

  /* Compute the column pointers of the matrix; first count the number of
   * elements per column, then add them. */
  for (j = 0; j < m->n; j++) {
    pj[j+1] = 0;
  }
  for (i = 0; i < m->n; i++) {
    for (k = m->row_ptr[i]; k < m->row_ptr[i+1]; k++) {
      j = pm->col_in[k];
      /* j is already validated above; this bounds check is redundant but explicit */
      if (j < 0 || j >= m->n) {
        CFL_ERROR_NULL("internal error: invalid column index in permuted matrix");
      }
      pj[j+1] += 1;
    }
  }
  pj[0] = 0;
  for (i = 0; i < m->n; i++) {
    pj[i+1] = pj[i] + pj[i+1];
  }

  /* pj starts off as the CSC col_ptr, but as we step through we increment
   * entries to step through all non-zero elements of each column. */
  for (i = 0; i < m->n; i++) {
    for (k = m->row_ptr[i]; k < m->row_ptr[i+1]; k++) {
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
  for (i = 0; i < m->n; i++) {
    for (k = m->row_ptr[i]; k < m->row_ptr[i+1]; k++) {
      pj[k] = i;
    }
  }

  for (k = 0; k < nnz; k++) {
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
  for (i = m->n-1; i >= 0; i--) {
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
 * CSR matrix column permutation.  Requires an input CSR matrix and an output CSR
 * matrix, where the latter is assumed to have an appropriately permuted
 * sparsity pattern.  This can be generated with zcsr_col_perm_alloc.
 *
 * Parameters
 * ----------
 *  m       Matrix to permute.
 *  pm      The output matrix which must have a permuted sparsity pattern prior
 *          to entry.
 *  pj      The permutation index used to permute the val array of m.  Values
 *          can be generated with zcsr_col_alloc.
 */
void zcsr_col_perm(zcsr *m, zcsr *pm, int *p, int *pj) {
  int i, j, pk;

#pragma omp parallel for private(i) schedule(dynamic)
  for (i = 0; i < m->nnz; i++) {
    pm->val[pj[i]] = m->val[i];
  }
}

/*
 * Find the connected components of a symmetric csr matrix.  Algorithm follows
 * Scipy _connected_components_undirected implementation.
 *
 * Parameters
 * ----------
 *  m       The matrix for which to find the connected components.
 *  labels  Array of length n which will be overwritten with the connected
 *          component index of each row.
 */
int zcsr_cc(zcsr *m, int *labels) {
  int i, ii, j, k, label, s_top, *s;

  /* Initialize to -1, designating an unvisited vertex. */
  for (i = 0; i < m->n; i++) {
    labels[i] = -1;
  }
  /* Alias s and labels pointers, since labels are only set after a specific
   * node has been popped from the stack. */
  s = labels;

  label = 0;
  /* Top of stack set to -2 indicates the end of the current connected
   * component. */
  for (i = 0; i < m->n; i++) {
    if (labels[i] == -1) {
      s_top = i;
      s[i] = -2;

      while (s_top != -2) {
        ii = s_top;
        s_top = s[ii];

        labels[ii] = label;

        for (k = m->row_ptr[ii]; k < m->row_ptr[ii+1]; k++) {
          j = m->col_in[k];
          if (s[j] == -1) {
            s[j] = s_top;
            s_top = j;
          }
        }
      }
      label++;
    }
  }

  return label;
}
