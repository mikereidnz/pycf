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
 * Diagonalization, and associated, routines for crystal-field and spin
 * Hamiltonians.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <lapacke.h>

#include "cfl_config.h"
#include "cfl_error.h"
#include "cfl_tensor.h"
#include "cfl_h.h"

/*
 * Allocate storage for complex valued Hamiltonians.
 *
 * Parameters
 * ----------
 *  n     The dimension of the Hamiltonian.
 *  nt    The number of tensors. 
 *  t     Pointer to array of zts.
 */
zh *zh_alloc(int n, int nt, zt **t) {
  zh *h;
  int i;
  complex double *ap;

  h = (zh *) malloc(sizeof(zh));
  if (h == 0) {
    CFL_ERROR_NULL("malloc failed for h");
  }

  /* Ensure all tensors have matching state labels. */
  for (i=1; i<nt; i++) {
    if (t[0]->states->hash != t[i]->states->hash) {
      CFL_ERROR_NULL("Tensors have mismatching state labels")
    }
  }
  h->states = t[0]->states;

  ap = (complex double *) calloc(n*(n+1)/2,sizeof(complex double));
  if (ap == 0) {
    free(h);
    CFL_ERROR_NULL("calloc failed for ap");
  }

  h->n = n;
  h->nt = nt;
  h->t = t;
  h->ap = ap;

  return h;
}

void zh_free(zh *h) {
  free(h->ap);
  free(h);
}

/*
 * Set the coefficient array pointer; a wrapper for Cython. 
 *
 * Parameters
 * ----------
 *  coeff     Pointer to the coefficient array.  
 */
void zh_set_coeff(zh *h, complex double *coeff) {
  h->coeff = coeff;
}

/*
 * Allocate storage for the Hamiltonian diagonalization. 
 *
 * Parameters
 * ----------
 * h    The Hamiltonian to be diagonalized.
 */
zhd_w *zhd_w_alloc(zh *h) {
  zhd_w *hd_w;

  hd_w = (zhd_w *) malloc(sizeof(zhd_w));
  if (hd_w == 0) {
    CFL_ERROR_NULL("malloc failed for hd_w");
  }


  /* hpevd workspace query. */
  complex double *work, wquery;
  double *rwork, rwquery;
  int *iwork, iwquery, lwork, lrwork, liwork, info;
  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, NULL,
      NULL, h->n, &wquery, -1, &rwquery, -1, &iwquery, -1);
  if (info != 0) {
    free(hd_w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }
  lwork = (int)wquery;
  lrwork = (int)rwquery;
  liwork = (int)iwquery;

  work = calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  rwork = calloc(lrwork,sizeof(double));
  if (rwork == 0) {
    free(hd_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for rwork");
  }
  iwork = calloc(liwork,sizeof(int));
  if (iwork == 0) {
    free(hd_w);
    free(work);
    free(rwork);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  hd_w->work = work;
  hd_w->lwork = lwork;
  hd_w->rwork = rwork;
  hd_w->lrwork = lrwork;
  hd_w->iwork = iwork;
  hd_w->liwork = liwork;

  /* Allocation for matrix element scaling an addition. */
  int i, j;
  crs_zhm **coeff_w;

  if (h->nt>1) {
    coeff_w = (crs_zhm **) malloc((h->nt-1)*sizeof(crs_zhm *));
    hd_w->lcoeff_w = h->nt-1;
  }
  else {
    coeff_w = (crs_zhm **) malloc((h->nt)*sizeof(crs_zhm *));
    hd_w->lcoeff_w = h->nt;
  }
  if (coeff_w == 0) {
    free(hd_w);
    free(work);
    free(rwork);
    CFL_ERROR_NULL("malloc failed for coeff_w");
  }

  /* Allocation for summing matrix elements of tensors.  The zhsam function
   * calculates C for C = alpha A + beta C, for A, B, and C CRS matrices and
   * alpha and beta complex scalars.  The first two matrix elements are summed
   * directly with respective coefficients set for alpha and beta.  Further
   * matrix elements are then iteratively added to the previous result.  Since
   * crs_zhsam_alloc also calculates the row_ptr array and number of non-zero
   * elements of C, we have to run through the actual additions in order to
   * determine these values for each of the intermediate sums.  Finally, in case
   * there is only a single tensor, we use the scaling function crs_zhsm for
   * which we still have to allocate separate memory. 
   */
  if (h->nt>1) {
    coeff_w[0] = crs_zhsam_alloc((h->t[0])->matel, (h->t[1])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w);
      free(work);
      free(rwork);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
    crs_zhsam((h->t[0])->matel, (h->t[1])->matel, coeff_w[0], 1, 1);
    for (i=1; i<h->nt-1; i++) {
      coeff_w[i] = crs_zhsam_alloc(coeff_w[i-1], (h->t[i+1])->matel);
      if (coeff_w[i] == 0) {
        free(hd_w);
        free(work);
        free(rwork);
        free(coeff_w);
        for (j=0; j<i; j++) {
          crs_zhm_free(coeff_w[j]);
        }
        CFL_ERROR_NULL("alloc failed for coeff_w");
      }
      crs_zhsam(coeff_w[i-1], (h->t[i+1])->matel, coeff_w[i], 1, 1);
    }
  }
  else {
    coeff_w[0] = crs_zhsm_alloc((h->t[0])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w);
      free(work);
      free(rwork);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
  }

  hd_w->coeff_w = coeff_w;

  return hd_w;
}

void zhd_w_free(zhd_w *hd_w) {
  int i;

  for (i=0; i<hd_w->lcoeff_w; i++) {
    crs_zhm_free(hd_w->coeff_w[i]);
  }
  free(hd_w->coeff_w);
  free(hd_w->work);
  free(hd_w->rwork);
  free(hd_w->iwork);
  free(hd_w);
}

/*
 * Calculate the eigenvalues and corresponding eigenvectors of a Hamiltonian. 
 * 
 * Parameters
 * ----------
 *  w       Pointer to double valued array of length n to which eigenvalues
 *          will be written.  
 *  z       Pointer to complex double valued array of length n^2 to which the
 *          eigenvectors will be written.
 *  h       The Hamiltonian. 
 *  hd_w    The work space for diagonalization; allocated using zhd_w_alloc.
 */
void zhd(double *w, complex double *z, zh *h, zhd_w *hd_w) {
  int i;
  char lapack_err[] = "LAPACKE_zhpevd failed with error code: 0";

  /* Multiply the tensor matrix elements by coefficients and sum them.  The
   * result is stored in hd_w->coeff_w[i], where i is the number of tensors -1.
   */
  if (h->nt>1) {
    crs_zhsam((h->t[0])->matel, (h->t[1])->matel, hd_w->coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i=1; i<hd_w->lcoeff_w; i++) {
      crs_zhsam(hd_w->coeff_w[i-1], (h->t[i+1])->matel, hd_w->coeff_w[i], 1,
          h->coeff[i+1]);
    }
  }
  else
    crs_zhsm((h->t[0])->matel, hd_w->coeff_w[0], h->coeff[0]);

  /* Convert the Hamiltonian from CRS to dense lower-triangular packed storage
   * for diagonalization. */
  crs_zhm2zhpa(hd_w->coeff_w[hd_w->lcoeff_w-1], h->ap);

  int info;
  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, w, z,
      h->n, hd_w->work, hd_w->lwork, hd_w->rwork, hd_w->lrwork, hd_w->iwork,
      hd_w->liwork);

  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zhpevd failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
}
