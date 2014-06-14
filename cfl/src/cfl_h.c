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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <lapacke.h>
#include <cfl_error.h>
#include <cfl_crs.h>
#include <cfl_h.h>

/*
 * @brief Allocate storage for complex valued tensors. 
 *
 * @param[name]   A unique identifier of the tensor. 
 * @param[a]      Pointer to array containing the matrix elements. 
 * @param[n]      The dimension of the matrix elemet matrix.
 */
zt *zt_alloc(char *name, double complex *a, size_t n) {
  zt *t;
  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  crs_zhm *ma = crs_zhm_alloc(a, n);
  if (ma == 0) {
    free(t);
    CFL_ERROR_NULL("alloc failed for crs_zhm");
  }

  t->name = name;
  t->n = n;
  t->matel = ma;

  return t;
}

/*
 * @brief Free storage allocated for a zt.
 *
 * @param[t]     Pointer to the zt struct.
 */
void zt_free(zt *t) {
  crs_zhm_free(t->matel);
  free(t);
}


/*
 * @brief Add and scale the matrix elements of two tensors, write the result to
 *        a newly allocated tensor, and return a pointer to it. 
 *
 * @param[name] Name of the resulting third tensor. 
 * @param[t1]   Pointer to the first tensor struct. 
 * @param[t2]   Pointer to the second tensor struct.
 * @param[s1]   A complex valued scale factor for the first tensor.
 * @param[s2]   A complex valued scale factor for the second tensor.
 */
zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2) {
  zt *t;

  if (t1->n != t2->n) {
    CFL_ERROR_VOID("tensor dimensions do not match");
  }

  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  t->matel = crs_zhsam_alloc(t1->matel, t2->matel);
  if (t == 0) {
    free(t);
    CFL_ERROR_NULL("failed to alloc t");
  }
  crs_zhsam(t1->matel, t2->matel, t->matel, s1, s2);

  t->name = name;
  t->n = t1->n;

  return t;
}

/*
 * @brief Allocate storage for a new tensor, and write to it the scaled matrix
 *        elements of the provided tensor.  
 *
 * @param[name] The name of the new tensor. 
 * @param[t]    Pointer to the input tensor.
 * @param[s]    A complex valued scale factor.
 */
zt *zt_s(char *name, zt *t, double complex s) {
  zt *ts;

  ts = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  ts->matel = crs_zhsm_alloc(t->matel);
  if (ts == 0) {
    free(ts);
    CFL_ERROR_NULL("alloc failed for ts");
  }
  crs_zhsm(t->matel, ts->matel, s);

  ts->name = name;
  ts->n = t->n;

  return ts;
}

/*
 * @brief Allocate storage for complex valued Hamiltonians.
 *
 * @param[n]    The dimension of the Hamiltonian.
 * @param[nt]   The number of tensors. 
 * @param[s]    Pointer to character arrays containing state labels.
 * @param[t]   Pointer to array of zts.
 * @param[w]    Pointer to double valued array of length n to which eigenvalues
 *              will be written.  
 * @param[z]    Pointer to double complex valued array of length n^2 to which
 *              the eigenvectors will be written.
 */
zh *zh_alloc(int n, int nt, char **s, zt **t, double *w, double complex *z) {
  zh *h;
  double complex *ap;

  h = (zh *) malloc(sizeof(zh));
  if (h == 0) {
    CFL_ERROR_NULL("malloc failed for h");
  }
  ap = (double complex *) calloc(n*(n+1)/2,sizeof(double complex));
  if (ap == 0) {
    free(h);
    CFL_ERROR_NULL("calloc failed for ap");
  }

  h->n = n;
  h->nt = nt;
  h->states = s;
  h->t = t;
  h->ap = ap;
  h->w = w;
  h->z = z;

  return h;
}

/*
 * @brief Free storage of a complex valued Hamiltonian.
 *
 * @params[m]     Pointer to the Hamiltonian to be freed. 
 */
void zh_free(zh *h) {
  free(h->ap);
  free(h);
}

/*
 * @brief Set the coefficient array pointer; a wrapper for Cython. 
 *
 * @param[coeff]    Pointer to the coefficient array.  
 */
void zh_set_coeff(zh *h, double complex *coeff) {
  h->coeff = coeff;
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
    CFL_ERROR_NULL("malloc failed for zhd_w");
  }

  /* LAPACK eigenvalue workspace query. */
  lapack_complex_double *work, wquery;
  double *rwork, rwquery;
  lapack_int *iwork, iwquery, lwork, lrwork, liwork, info;

  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, h->w,
      h->z, h->n, &wquery, -1, &rwquery, -1, &iwquery, -1);

  if (info != 0) {
    free(hd_w);
    CFL_ERROR_VOID("LAPACKE workspace query failed");
  }

  lwork = (lapack_int)wquery;
  lrwork = (lapack_int)rwquery;
  liwork = (lapack_int)iwquery;

  work = calloc(lwork,sizeof(lapack_complex_double));
  if (work == 0) {
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  rwork = calloc(lrwork,sizeof(lapack_int));
  if (rwork == 0) {
    free(hd_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for rwork");
  }
  iwork = calloc(liwork,sizeof(lapack_int));
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
  double complex alpha = 1+I;
  double complex beta = 1+I;
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
   * determine the these values for each of the intermediate sums.  Finally, in
   * case there is only a single tensor, we use the scaling function crs_zhsm
   * for which we still have to allocate separate memory. 
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
    crs_zhsam((h->t[0])->matel, (h->t[1])->matel, coeff_w[0], alpha, beta);
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
      crs_zhsam(coeff_w[i-1], (h->t[i+1])->matel, coeff_w[i], alpha, beta);
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


/*
 * @brief Free Hamiltonian digitalization workspace storage.
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
  char lapack_err[] = "LAPACKE_zhpevd failed with error code: 0";

  /* Multiply the tensor matrix elements by coefficients and sum them.  The
   * result is stored in hd_w->coeff_w[i], where is the number of tensors -1. */
  if (h->nt>1) {
    crs_zhsam((h->t[0])->matel, (h->t[1])->matel, hd_w->coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i=1; i<hd_w->lcoeff_w; i++) {
      crs_zhsam(hd_w->coeff_w[i-1], (h->t[i+1])->matel, hd_w->coeff_w[i], alpha,
          h->coeff[i+1]);
    }
  }
  else
    crs_zhsm((h->t[0])->matel, hd_w->coeff_w[0], h->coeff[0]);

  /* Convert the Hamiltonian from CRS to dense lower-triangular packed storage
   * for diagonalization. */
  crs_zhm2zhpa(hd_w->coeff_w[hd_w->lcoeff_w-1], h->ap);

  lapack_int info; 
  info = LAPACKE_zhpevd_work(LAPACK_COL_MAJOR, 'V', 'L', h->n, h->ap, h->w,
      h->z, h->n, hd_w->work, hd_w->lwork, hd_w->rwork, hd_w->lrwork,
      hd_w->iwork, hd_w->liwork);

  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zhpevd failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
}
