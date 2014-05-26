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
 * @file    cfl_h.c
 * @brief   Diagonalization, and associated, routines for crystal-field and spin
 *          Hamiltonians.
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <cfl_crs.h>
#include <cfl_h.h>


/*
 * @brief Allocate storage for complex valued Hamiltonians.
 *
 * @param[d]    The dimension of the Hamiltonian.
 * @param[s]    Pointer to character arrays containing state labels.
 */
zh *zh_alloc(int n, char **s) {
  zh *h;
  double complex *ap;
  double *w;
  double complex *z;

  h = (zh *) malloc(sizeof(zh));
  ap = (double complex *) calloc(n*(n+1)/2,sizeof(double complex));
  w = (double *) calloc(n,sizeof(double));
  z = (double complex *) calloc(n*n,sizeof(double complex));

  if (h == 0) {
    printf("Memory allocation failed for h\n");
  }
  else if (ap == 0) {
    printf("Memory allocation failed for ap\n");
  }
  else if (w == 0) {
    printf("Memory allocation failed for w\n");
  }
  else if (z == 0) {
    printf("Memory allocation failed for z\n");
  }

  h->n = n;
  h->states = s;
  h->ap = ap;
  h->w = w;
  h->z = z;

  return h;
}

/*
 * @brief Free storage of a complex valued Hamiltonian.
 *
 * @params[m]   Pointer to the Hamiltonian to be freed. 
 */
void zh_free(zh *h) {
  free(h->ap);
  free(h->w);
  free(h->z);
  free(h);
}

/*
 * @brief Allocate storage for the Hamiltonian diagonalization. 
 *
 * @param[h]    The Hamiltonian to be diagonalized.
 */
zhd_w *zhd_w_alloc(zh *h) {
  zhd_w *hd_w;

  hd_w = (zhd_w *) malloc(sizeof(zhd_w));
  if (hd_w == 0) {
    printf("Error in hdiag_work_alloc; memory allocation failed for hdiag_work\n");
  }

  /* Allocation for tensor matrix element scaling by coefficients and summation.
  */
  int i;
  double complex alpha = 1+I;
  double complex beta = 1+I;
  crs_zhm **coeff_w;

  coeff_w = (crs_zhm **) malloc((h->nt-1)*sizeof(crs_zhm *));
  if (coeff_w == 0) {
    printf("Error in hdiag_work_alloc; memory allocation failed for coeff_w\n");
  }

  /* Allocation for summing matrix elements of tensors.  The zhsam function
   * calculates C for C = alpha A + beta C, for A, B, and C CRS matrices and
   * alpha and beta complex scalars.  The first two matrix elements are summed
   * directly with respective coefficients set for alpha and beta.  Further
   * matrix elements are then itteratively added to the previous result.  Since
   * crs_zhsam_alloc also calculates the row_ptr array and number of non-zero
   * elements of C, we have to run through the actual additions in order to
   * determine the these values for each of the intermediate sums.  Finally, in
   * case there is only a single tensor, we use the scaling function crs_zhsm
   * for which we still have to allocate separate memory. 
   */
  if (h->nt>1) {
    coeff_w[0] = crs_zhsam_alloc(h->t[0].matel, h->t[1].matel);
    if (coeff_w[0] == 0) {
      printf("Error in hdiag_work_alloc; memory allocation failed for coeff_w[0]\n");
    }
    crs_zhsam(h->t[0].matel, h->t[1].matel, coeff_w[0], alpha, beta);
    for (i=1; i<h->nt-1; i++) {
      coeff_w[i] = crs_zhsam_alloc(coeff_w[i-1], h->t[i+1].matel);
      crs_zhsam(coeff_w[i-1], h->t[i+1].matel, coeff_w[i], alpha, beta);
      if (coeff_w[i] == 0) {
        printf("Error in hdiag_work_alloc; memory allocation failed for coeff_w[%i]\n", i);
      }
    }
  }
  else {
    coeff_w[0] = crs_zhsm_alloc(h->t[0].matel);
    if (coeff_w[0] == 0) {
      printf("Error in hdiag_work_alloc; memory allocation failed for coeff_w[0]\n");
    }
  }

  hd_w->coeff_w = coeff_w;
  hd_w->lcoeff_w = h->nt-1;

  /* LAPACK eigenvalue workspace query. */
  lapack_complex_double *work, wquery;
  double *rwork, rwquery;
  lapack_int *iwork, iwquery, lwork, lrwork, liwork, info;

  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, h->w,
      h->z, h->n, &wquery, -1, &rwquery, -1, &iwquery, -1);

  if (info != 0) {
    printf("Error in hdiag_work_alloc; LAPACKE workspace query failed\n");
  }

  lwork = (lapack_int)wquery;
  lrwork = (lapack_int)rwquery;
  liwork = (lapack_int)iwquery;

  work = calloc(lwork,sizeof(lapack_complex_double));
  rwork = calloc(lrwork,sizeof(lapack_int));
  iwork = calloc(liwork,sizeof(lapack_int));
  if (work == 0) {
    printf("Error in hdiag_work_alloc; memory allocation failed for work\n");
  }
  else if (rwork == 0) {
    printf("Error in hdiag_work_alloc; memory allocation failed for rwork\n");
  }
  else if (iwork == 0) {
    printf("Error in hdiag_work_alloc; memory allocation failed for iwork\n");
  }

  hd_w->work = work;
  hd_w->lwork = lwork;
  hd_w->rwork = rwork;
  hd_w->lrwork = lrwork;
  hd_w->iwork = iwork;
  hd_w->liwork = liwork;

  return hd_w;
}


/*
 * @brief Free Hamiltonian diagonalization workspace storage.
 *
 * @param[hd_w]    Pointer to Hamiltonian diagonalization workspace.
 */
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
 * @brief Calculate the eigenvalues and corresponding eigenvectors of a
 *        Hamiltonian. 
 * 
 * @param[h]        The Hamiltonian. 
 * @param[hd_w]     The work space for diagonalization; allocated using
 *                  zhd_w_alloc.
 */
void zhd(zh *h, zhd_w *hd_w) {
  int i;
  double complex alpha = 1+I;

  /* Multiply the tensor matrix elements by coefficients and sum them.  The
   * result is stored in hd_w->coeff_w[i], where is the number of tensors -1. */
  if (h->nt>1) {
    crs_zhsam(h->t[0].matel, h->t[1].matel, hd_w->coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i=1; i<hd_w->lcoeff_w; i++) {
      crs_zhsam(hd_w->coeff_w[i-1], h->t[i+1].matel, hd_w->coeff_w[i], alpha,
          h->coeff[i+1]);
    }
  }
  else
    crs_zhsm(h->t[0].matel, hd_w->coeff_w[0], h->coeff[0]);

  /* Convert the Hamiltonian from CRS to dense lower-triangular packed storage
   * for diagonalization. */
  crs_zhm2zhpa(hd_w->coeff_w[hd_w->lcoeff_w-1], h->ap);
  
  lapack_int info; 
  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, h->w,
      h->z, h->n, hd_w->work, hd_w->lwork, hd_w->rwork, hd_w->lrwork,
      hd_w->iwork, hd_w->liwork);

  if (info != 0) {
    printf("LAPACKE_zhpevd failed with error %i\n", info);
  }
}
