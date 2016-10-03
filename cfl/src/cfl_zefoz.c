 /*
    Copyright (C) 2016 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#include "cfl_config.h"

#if USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <gsl/gsl_cblas.h>
#include <lapacke.h>
#endif /* USE_MKL */

#include "cfl_error.h"
#include "cfl_h.h"
#include "cfl_zefoz.h"

/*
 * Take the inner product on a Hilbert space for bra and ket vectors and a given
 * operator. 
 *
 * Parameters
 * ----------
 *  n       The length of each state vector. 
 *  bra     Array of the bra vector. 
 *  op      Array of the operator matrix elements. 
 *  ket     Array of the ket vector.
 *  zw      Workspace for multiplication -- of length n.
 */
inline double inprod(int n, double complex *bra, double complex *op, double complex
    *ket, double complex *zw) {
  double complex one, zero, dotc;
    
  one = 1;
  zero = 0;

  cblas_zhemv(CblasColMajor, CblasUpper, n, &one, op, n, ket, 1, &zero, zw, 1);
  cblas_zdotc_sub(n, bra, 1, zw, 1, &dotc);

  return cabs(dotc);  
}

/*
 * Allocate workspace for a ZEFOZ point search. 
 *
 * Parameters
 * ----------
 *  h       Reference to the crystal field Hamiltonian. 
 *  zi      Indices of the Zeeman tensors in coeff array; must be in ordered by
 *          x, y, and z indices.
 */
zefoz_w *zefoz_alloc(zh *h, int *zi) {
  zefoz_w *work;
  int info;
  double wquery;

  work = (zefoz_w *) malloc(sizeof(zefoz_w));
  if (work == 0) {
    CFL_ERROR_NULL("malloc failed for work");
  }

  work->w = (double *) calloc(h->n, sizeof(double));
  if (work->w == 0) {
    free(work);
    CFL_ERROR_NULL("calloc failed for work->w");
  }
  work->z = (double complex *) calloc(h->n*h->n, sizeof(double complex));
  if (work->z == 0) {
    free(work->w);
    free(work);
    CFL_ERROR_NULL("calloc failed for work->z");
  }

  work->hd_w = (zhd_w *) zhd_w_alloc('V', h);
  if (work->hd_w == 0) {
    free(work->w);
    free(work->z);
    free(work);
    CFL_ERROR_NULL("zhd_w_alloc failed for work->hd_w");
  }

  work->inprod_w = (double complex *) calloc(h->n, sizeof(double complex));
  if (work->inprod_w == 0) {
    free(work->w);
    free(work->z);
    free(work->hd_w);
    free(work);
  }
  
  memset(work->ipiv, 0, 3*sizeof(int));
  memset(work->v, 0, 3*sizeof(double));
  memset(work->C, 0, 9*sizeof(double));

  info = LAPACKE_dgetri_work(LAPACK_COL_MAJOR, 3, work->C, 3, work->ipiv,
      &wquery, -1);
  if (info != 0) {
    free(work->w);
    free(work->z);
    free(work->hd_w);
    free(work->inprod_w);
    free(work);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  /* Since LWORK >= N*NB, where NB is the optimal blocksize returned by ILAENV,
   * this workspace will always be big enough for the Jacobian/gradient vector
   * dot product. */
  work->dlwork = (int)wquery;
  work->dwork = (double *) calloc(work->dlwork, sizeof(double));
  if (work->dwork == 0) {
    free(work->w);
    free(work->z);
    free(work->hd_w);
    free(work->inprod_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for dwork");
  }

  work->h = h;
  work->zi = zi;

  return work;
}

void zefoz_free(zefoz_w *work) {
  free(work->w);
  free(work->z);
  zhd_w_free(work->hd_w);
  free(work->inprod_w);
  free(work->dwork);
  free(work);
}

/* Find the first derivative from perturbation theory, following PRB 74, 195101.
 *
 * Parameters
 * ----------
 *  k         The level for which to determine the derivative.
 *  mi        Zeeman operator matrix elements along direction for which to
 *            differentiate.
 *  zefoz_w   The ZEFOZ search workspace.
 */
inline double d1(int k, double complex *mi, zefoz_w *work) {
  int n;
  double s;
  double complex *phi;

  n = work->h->n;
  phi = work->z;
  s = inprod(n, &(phi[n*k]), mi, &(phi[n*k]), work->inprod_w);
  
  return s;
}

/* Find the second derivative from perturbation theory, following PRB 74,
 * 195101.
 *
 * Parameters
 * ----------
 *  k         The level for which to determine the derivative.
 *  mi        Zeeman operator matrix elements along direction for the first
 *            derivative.
 *  mj        Zeeman operator matrix elements along direction for the second
 *            derivative.
 *  zefoz_w   The ZEFOZ search workspace.
 */
inline double d2(int k, double complex *mi, double complex *mj, zefoz_w *work) {
  int l, n; 
  double s, *omega;
  double complex *phi;

  n = work->h->n;
  omega = work->w;
  phi = work->z;

  s = 0;
  for (l=0; l<n; l++) {
    if (l != k) {
      s += inprod(n, &(phi[n*k]), mi, &(phi[n*l]), work->inprod_w) * inprod(n,
          &(phi[n*l]), mj, &(phi[n*k]), work->inprod_w)/(omega[k] - omega[l]);
    }
  }

  return s;
}


/* Calculate the Zeeman gradient vector and overwrites value in the workspace
 * struct.
 *
 * Parameters
 * ----------
 *  k     Index of the kth level.
 *  l     Index of the ith level.
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  work  Workspace for the ZEFOZ search.
 */
inline void v_eval(int k, int l, double complex **m, zefoz_w *work) {
  int i;

  for (i=0; i<3; i++) {
    work->v[i] = d1(k, m[i], work) - d1(l, m[i], work);
  }
}

/* Calculate the Zeeman curvature tensor (Jacobian) and overwrites value in the
 * workspace struct. 
 *
 * Parameters
 * ----------
 *  k     Index of the kth level.
 *  l     Index of the ith level.
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  work  Workspace for the ZEFOZ search.
 */
inline void C_eval(int k, int l, double complex **m, zefoz_w *work) {
  int i, j;

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      work->C[i*3+j] = d2(k, m[i], m[j], work) - d2(l, m[i], m[j], work);
    }
  }
}


/* Iteration of ZEFOZ search. Updates value of B by diagonalizing the CF
 * Hamiltonian, computing the gradient vector and curvature tensor, and then
 * applying Newton's method.
 *
 * Parameters
 * ----------
 *  k     Index of the kth level.
 *  l     Index of the ith level.
 *  B     Array of length three containing the previous x, y, and z field
 *        strengths of the previous iteration. 
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  work  Workspace for the ZEFOZ search.
 */
void zefoz_iter(int k, int l, double *B, double complex **m, zefoz_w *work) {
  int i, info, n;
  char lapack_err[] = "LAPACKE failed with error code: 0";

  for (i=0; i<3; i++) {
    work->h->coeff[work->zi[i]] = B[i];
  }
  zhd('V', work->w, work->z, work->h, work->hd_w);

  v_eval(k, l, m, work);
  C_eval(k, l, m, work);
  
  n = work->h->n;

  /* Invert the Jacobian. */
  info = LAPACKE_dgetrf_work(LAPACK_COL_MAJOR, 3, 3, work->C, 3, work->ipiv);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
  info = LAPACKE_dgetri_work(LAPACK_COL_MAJOR, 3, work->C, 3, work->ipiv,
      work->dwork, work->dlwork);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
  
  /* Calculate C^-1 v. */
  cblas_dgemv(CblasColMajor, CblasNoTrans, 3, 3, 1, work->C, 3, work->v, 1,
      0, work->dwork, 1);

  for (i=0; i<3; i++) {
    B[i] -= 2*work->dwork[i];
  }
}
