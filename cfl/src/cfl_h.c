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

/*
 * Overview
 * ========
 *
 * Diagonalization, and associated, routines for crystal-field and spin
 * Hamiltonians.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "cfl_config.h"

#if USE_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif /* USE_MKL */

#include "rcm.h"
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
  complex double *a;

  h = (zh *) malloc(sizeof(zh));
  if (h == 0) {
    CFL_ERROR_NULL("malloc failed for h");
  }

  /* Ensure all tensors have matching state labels. */
  for (i=1; i<nt; i++) {
    if (t[0]->slabels->hash != t[i]->slabels->hash) {
      CFL_ERROR_NULL("Tensors have mismatching state labels")
    }
  }
  h->slabels = t[0]->slabels;

  a = (complex double *) calloc(n*n,sizeof(complex double));
  if (a == 0) {
    free(h);
    CFL_ERROR_NULL("calloc failed for a");
  }

  h->n = n;
  h->nt = nt;
  h->t = t;
  h->a = a;
  
  return h;
}

void zh_free(zh *h) {
  free(h->a);
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
  int n = h->n, lda = h->n, ldz = h->n, il, iu, info;
  double vl, vu;
  int lwork, lrwork, liwork;
  complex double *work, wquery;
  double *rwork, rwquery;
  int *iwork, iwquery;
  zcrs *zcrs_h;
  signed char *node_mask;
  int *node_deg, *tmp_perm;

  zhd_w *hd_w;

  hd_w = (zhd_w *) malloc(sizeof(zhd_w));
  if (hd_w == 0) {
    CFL_ERROR_NULL("malloc failed for hd_w");
  }

  hd_w->isuppz = (int *) malloc(2*n*sizeof(int));
  if (hd_w->isuppz == 0) {
    free(hd_w);
    CFL_ERROR_NULL("malloc failed for hd_w->isuppz");
  }

  /* Set the absolute error tolerance to which each eigenvalue/eigenvector is
   * required to the safe minimum machine precision. */
  hd_w->abstol = LAPACKE_dlamch('S');

  /* zheevr workspace query. */
  lwork = -1;
  lrwork = -1;
  liwork = -1;

  info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'V', 'A', 'U', n, h->a, lda, vl, vu, il, iu,
      hd_w->abstol, &(hd_w->m), NULL, NULL, ldz, hd_w->isuppz, &wquery, lwork, &rwquery, lrwork,
      &iwquery, liwork);
  if (info != 0) {
    free(hd_w->isuppz);
    free(hd_w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (int) creal(wquery);
  lrwork = (int) rwquery;
  liwork = iwquery;

  work = (complex double *) calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(hd_w->isuppz);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  rwork = (double *) calloc(lrwork,sizeof(double));
  if (rwork == 0) {
    free(hd_w->isuppz);
    free(hd_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for rwork");
  }
  iwork = (int *) calloc(liwork,sizeof(int));
  if (iwork == 0) {
    free(hd_w->isuppz);
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
  zhcrs **coeff_w;

  if (h->nt>1) {
    coeff_w = (zhcrs **) malloc((h->nt-1)*sizeof(zhcrs *));
    hd_w->lcoeff_w = h->nt-1;
  }
  else {
    coeff_w = (zhcrs **) malloc((h->nt)*sizeof(zhcrs *));
    hd_w->lcoeff_w = h->nt;
  }
  if (coeff_w == 0) {
    free(hd_w->isuppz);
    free(hd_w);
    free(work);
    free(rwork);
    free(iwork);
    CFL_ERROR_NULL("malloc failed for coeff_w");
  }

  /* Allocation for summing matrix elements of tensors.  The zhsam function
   * calculates C for C = alpha A + beta C, for A, B, and C CRS matrices and
   * alpha and beta complex scalars.  The first two matrix elements are summed
   * directly with respective coefficients set for alpha and beta.  Further
   * matrix elements are then iteratively added to the previous result.  Since
   * zhcrssam_alloc also calculates the row_ptr array and number of non-zero
   * elements of C, we have to run through the actual additions in order to
   * determine these values for each of the intermediate sums.  Finally, in case
   * there is only a single tensor, we use the scaling function zhcrssm for
   * which we still have to allocate separate memory. 
   */
  if (h->nt>1) {
    coeff_w[0] = zhcrssam_alloc((h->t[0])->matel, (h->t[1])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w->isuppz);
      free(hd_w);
      free(work);
      free(rwork);
      free(iwork);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
    zhcrssam((h->t[0])->matel, (h->t[1])->matel, coeff_w[0], 1, 1);
    for (i=1; i<h->nt-1; i++) {
      coeff_w[i] = zhcrssam_alloc(coeff_w[i-1], (h->t[i+1])->matel);
      if (coeff_w[i] == 0) {
        free(hd_w->isuppz);
        free(hd_w);
        free(work);
        free(rwork);
        free(iwork);
        free(coeff_w);
        for (j=0; j<i; j++) {
          zhcrs_free(coeff_w[j]);
        }
        CFL_ERROR_NULL("alloc failed for coeff_w");
      }
      zhcrssam(coeff_w[i-1], (h->t[i+1])->matel, coeff_w[i], 1, 1);
    }
  }
  else {
    coeff_w[0] = zhcrssm_alloc((h->t[0])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w->isuppz);
      free(hd_w);
      free(work);
      free(rwork);
      free(iwork);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
  }

  printf("alloc nnz = %i\n", coeff_w[hd_w->lcoeff_w-1]->nnz);

  /* Generate the reverse Cuthill-McKee ordering of the complete Hamiltonian. */
  zcrs_h = zhcrs2zcrs_alloc(coeff_w[hd_w->lcoeff_w-1]);
  if (zcrs_h == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("alloc failed for zcrs_h");
  }
  hd_w->rcm_perm = (int *) calloc(zcrs_h->n, sizeof(int));
  if (hd_w->rcm_perm == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for rcm_perm");
  }
  node_mask = (signed char *) calloc(zcrs_h->n, sizeof(signed char));
  if (node_mask == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for node_mask");
  }
  node_deg = (int *) calloc(zcrs_h->n, sizeof(int));
  if (node_deg == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    free(node_mask);
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for node_deg");
  }

  tmp_perm = (int *) calloc(zcrs_h->n, sizeof(int));
  if (tmp_perm == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    free(node_mask);
    free(node_deg);
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for tmp_perm");
  }
  zhcrs2zcrs(coeff_w[hd_w->lcoeff_w-1], zcrs_h);
  
  RCM_FUNC(genrcmi)(zcrs_h->n, 0, zcrs_h->row_ptr, zcrs_h->col_in,
      tmp_perm, node_mask, node_deg);
  printf("rcm perm:\n");
  for (i=0; i<h->n; i++) {
    printf("%i ", tmp_perm[i]);
  }
  printf("\n");
  /* Change permutation index to coordinate form. */
  for (i=0; i<zcrs_h->n; i++) {
    hd_w->rcm_perm[tmp_perm[i]] = i;
  }

  hd_w->zcrs_h = zcrs_h;
  hd_w->coeff_w = coeff_w;

  free(node_mask);
  free(node_deg);
  free(tmp_perm);
  
  hd_w->rcm_pj = (int *) calloc(hd_w->zcrs_h->nnz+1, sizeof(int));
  if (hd_w->rcm_pj == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    free(hd_w->coeff_w);
    free(hd_w->isuppz);
    free(hd_w->work);
    free(hd_w->rwork);
    free(hd_w->iwork);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for rcm_pj");
  }

  hd_w->rcm_rp_h = (zcrs *) zcrs_row_perm_alloc(hd_w->zcrs_h, hd_w->rcm_perm);
  hd_w->rcm_cp_h = (zcrs *) zcrs_col_perm_alloc(hd_w->zcrs_h, hd_w->rcm_perm, hd_w->rcm_pj);



  //if (hd_w->rcm_cp_h == 0) {
  //  for (i=0; i<hd_w->lcoeff_w; i++) {
  //    zhcrs_free(hd_w->coeff_w[i]);
  //  }
  //  zcrs_free(zcrs_h);
  //  free(hd_w->rcm_perm);
  //  free(hd_w->rcm_pj);
  //  free(hd_w->coeff_w);
  //  free(hd_w->isuppz);
  //  free(hd_w->work);
  //  free(hd_w->rwork);
  //  free(hd_w->iwork);
  //  free(hd_w);
  //  CFL_ERROR_NULL("zcrs_col_perm_alloc failed for rcm_cp_h");
  //}

  //if (hd_w->rcm_rp_h == 0) {
  //  for (i=0; i<hd_w->lcoeff_w; i++) {
  //    zhcrs_free(hd_w->coeff_w[i]);
  //  }
  //  zcrs_free(zcrs_h);
  //  free(hd_w->rcm_perm);
  //  zcrs_free(hd_w->rcm_cp_h);
  //  free(hd_w->rcm_pj);
  //  free(hd_w->coeff_w);
  //  free(hd_w->isuppz);
  //  free(hd_w->work);
  //  free(hd_w->rwork);
  //  free(hd_w->iwork);
  //  free(hd_w);
  //  CFL_ERROR_NULL("zcrs_row_perm_alloc failed for rcm_rp_h");
  //}

  return hd_w;
}

void zhd_w_free(zhd_w *hd_w) {
  int i;

  for (i=0; i<hd_w->lcoeff_w; i++) {
    zhcrs_free(hd_w->coeff_w[i]);
  }
  zcrs_free(hd_w->zcrs_h);
  free(hd_w->rcm_perm);
  zcrs_free(hd_w->rcm_rp_h);
  free(hd_w->rcm_pj);
  zcrs_free(hd_w->rcm_cp_h);
  free(hd_w->coeff_w);
  free(hd_w->isuppz);
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
  int n = h->n, lda = h->n, ldz = h->n, il, iu, info;
  double vl, vu;
  char lapack_err[] = "LAPACKE_zhpeevr failed with error code: 0";

  /* Multiply the tensor matrix elements by coefficients and sum them.  The
   * result is stored in hd_w->coeff_w[i], where i is the number of tensors -1.
   */
  if (h->nt>1) {
    zhcrssam((h->t[0])->matel, (h->t[1])->matel, hd_w->coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i=1; i<hd_w->lcoeff_w; i++) {
      zhcrssam(hd_w->coeff_w[i-1], (h->t[i+1])->matel, hd_w->coeff_w[i], 1,
          h->coeff[i+1]);
    }
  }
  else
    zhcrssm((h->t[0])->matel, hd_w->coeff_w[0], h->coeff[0]);

  /* Convert the Hamiltonian from Hermitian CRS to standard CRS, then apply RCM
   * permutation, and finally convert to dense storage. */
  zhcrs2zcrs(hd_w->coeff_w[hd_w->lcoeff_w-1], hd_w->zcrs_h);

  zcrs2zha(hd_w->zcrs_h, h->a);
  int j;
  printf("pre sort\n");
  for (i=0; i<h->n; i++) {
    for (j=0; j<h->n; j++) {
      printf("%f, ", creal(h->a[i*h->n+j]));
    }
    printf("\n");
  }
  printf("\n");

  for (i=0; i<h->n+1; i++) {
    printf("%i ", hd_w->zcrs_h->row_ptr[i]);
  }
  printf("\n");
  //zcrs_row_perm(hd_w->zcrs_h, hd_w->rcm_rp_h, hd_w->rcm_perm);
  zcrs_col_perm(hd_w->zcrs_h, hd_w->rcm_cp_h, hd_w->rcm_perm, hd_w->rcm_pj);

  zcrs2zha(hd_w->rcm_cp_h, h->a);
  printf("post sort\n");
  for (i=0; i<h->n; i++) {
    for (j=0; j<h->n; j++) {
      printf("%f, ", creal(h->a[i*h->n+j]));
    }
    printf("\n");
  }
  printf("\n");
  printf("post sort\n");
  for (i=0; i<h->n+1; i++) {
    printf("%i ", hd_w->rcm_rp_h->row_ptr[i]);
  }
  printf("\n");

  info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'V', 'A', 'U', n, h->a, lda, vl,
      vu, il, iu, hd_w->abstol, &(hd_w->m), w, z, ldz, hd_w->isuppz,
      hd_w->work, hd_w->lwork, hd_w->rwork, hd_w->lrwork, hd_w->iwork,
      hd_w->liwork);

  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zheevr failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
}
