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

  //cblas_zgemv(CblasColMajor, 'N', n, n, &one, op, n, ket, 1, &zero, zw, 1);
  cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &one, op, n, ket, 1, &zero, zw, 1);
  //cblas_zhemv(CblasRowMajor, CblasUpper, n, &one, op, n, ket, 1, &zero, zw, 1);
  cblas_zdotc_sub(n, bra, 1, zw, 1, &dotc);

  return cabs(dotc);  
}


/* Allocate storage for zefoz point array, which is used to record both the
 * magnetic field and gradient vector for located zefoz points. Resizes
 * dynamically.
 *
 * Parameters
 * ----------
 *  init_size     The number of points for which to initially alloc space.
 */
zefoz_a *zefoz_a_alloc(int init_size) {
  zefoz_a *za;

  za = (zefoz_a *) malloc(sizeof(zefoz_a));
  if (za == 0) {
    CFL_ERROR_NULL("malloc failed for za");
  }
  
  za->B = (double *) calloc(init_size*3, sizeof(double)); 
  if (za->B == 0) {
    free(za);
    CFL_ERROR_NULL("malloc failed for za->B");
  }

  za->v = (double *) calloc(init_size*3, sizeof(double)); 
  if (za->v == 0) {
    free(za->B);
    free(za);
    CFL_ERROR_NULL("malloc failed for za->v");
  }
  za->ctr = 0;
  za->size = init_size;

  return za;
}

/* Insert zefoz point into zefoz array. Will double the size whenever the
 * previously alloced storage runs out. 
 *
 * Parameters
 * ----------
 *  za    Pointer to the zefoz array. 
 *  B     Pointer the magnetic field strength array.
 *  v     Pointer to the gradient vector. 
 */
void zefoz_a_insert(zefoz_a *za, double *B, double *v) {
  int i, j;
  if (za->ctr == za->size) {
    za->size *= 2;
    za->B = (double *) realloc(za->B, za->size*3*sizeof(double));
    if (za->B == 0) {
      CFL_ERROR_VOID("realloc failed for za->B");
    }
    za->v = (double *) realloc(za->v, za->size*3*sizeof(double));
    if (za->v == 0) {
      CFL_ERROR_VOID("realloc failed for za->v");
    }
  }

  memcpy(&(za->B[3*(za->ctr)]), B, 3*sizeof(double));
  memcpy(&(za->v[3*(za->ctr)]), v, 3*sizeof(double));

  za->ctr++;
}

void zefoz_a_free(zefoz_a *za) {
  free(za->B);
  free(za->v);
  free(za);
}


/*
 * Allocate data for a ZEFOZ point search.  The data and workspace is separated
 * in this struct such that an instance can be alloced for each thread, such
 * that concurrent field values can be tested.  The only restriction is the
 * pointer to the Hamiltonian, which is probably shared between multiple
 * threads.  Consequently, care must be taken when setting h->coeff from
 * multiple threads. 
 *
 * Parameters
 * ----------
 *  h       Reference to the crystal field Hamiltonian. 
 *  zi      Indices of the Zeeman tensors in coeff array; must be in order x, y,
 *          and z indices.
 */
zd_inst *zd_inst_alloc(zh *h, int *zi) {
  zd_inst *data;
  int info;
  double wquery;

  data = (zd_inst *) malloc(sizeof(zd_inst));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for data");
  }

  data->w = (double *) calloc(h->n, sizeof(double));
  if (data->w == 0) {
    free(data);
    CFL_ERROR_NULL("calloc failed for data->w");
  }
  data->z = (double complex *) calloc(h->n*h->n, sizeof(double complex));
  if (data->z == 0) {
    free(data->w);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->z");
  }

  data->hd_w = (zhd_w *) zhd_w_alloc('V', h);
  if (data->hd_w == 0) {
    free(data->w);
    free(data->z);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }

  data->inprod_w = (double complex *) calloc(h->n, sizeof(double complex));
  if (data->inprod_w == 0) {
    free(data->w);
    free(data->z);
    free(data->hd_w);
    free(data);
  }
  
  memset(data->ipiv, 0, 3*sizeof(int));
  memset(data->v, 0, 3*sizeof(double));
  memset(data->C, 0, 9*sizeof(double));

  info = LAPACKE_dgetri_work(LAPACK_COL_MAJOR, 3, data->C, 3, data->ipiv,
      &wquery, -1);
  if (info != 0) {
    free(data->w);
    free(data->z);
    free(data->hd_w);
    free(data->inprod_w);
    free(data);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  /* Since LWORK >= N*NB, where NB is the optimal blocksize returned by ILAENV,
   * this workspace will always be big enough for the Jacobian/gradient vector
   * dot product. */
  data->dlwork = (int)wquery;
  data->dwork = (double *) calloc(data->dlwork, sizeof(double));
  if (data->dwork == 0) {
    free(data->w);
    free(data->z);
    free(data->hd_w);
    free(data->inprod_w);
    free(data);
    CFL_ERROR_NULL("calloc failed for dwork");
  }

  data->h = h;
  data->zi = zi;

  return data;
}

void zd_inst_free(zd_inst *data) {
  free(data->w);
  free(data->z);
  zhd_w_free(data->hd_w);
  free(data->inprod_w);
  free(data->dwork);
  free(data);
}


/*
 * Allocate data for a ZEFOZ point search.  
 *
 * Parameters
 * ----------
 *  h       Reference to the crystal field Hamiltonian. 
 *  zi      Indices of the Zeeman tensors in coeff array; must be in order x, y,
 *          and z indices.
 */
zefoz_d *zefoz_d_alloc(zh *h, int *zi) {
  int i, j, ninst;
  zefoz_d *data;

#ifdef _OPENMP
  ninst = omp_get_num_procs();
#else
  ninst = 1;
#endif /* _OPENMP */

  data = (zefoz_d *) malloc(sizeof(zefoz_d));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for data");
  }
  
  data->d_inst = (zd_inst **) malloc(sizeof(zd_inst *));
  if (data->d_inst == 0) {
    free(data);
    CFL_ERROR_NULL("malloc failed for data->d_inst");
  }
  for (i=0; i<ninst; i++) {
    data->d_inst[i] = zd_inst_alloc(h, zi);
    if (data->d_inst[i] == 0) {
      for (j=0; j<i; j++) {
        zd_inst_free(data->d_inst[j]);
        free(data->d_inst);
        free(data);
        CFL_ERROR_NULL("malloc failed for data->d_inst[i]");
      }
    }
  }

  data->ninst = ninst;

  return data;
}

void zefoz_d_free(zefoz_d *data) {
  int i;

  for (i=0; i<data->ninst; i++) {
    zd_inst_free(data->d_inst[i]);
  }
  free(data->d_inst);
  free(data);
}


/* Find the first derivative from perturbation theory, following PRB 74, 195101.
 *
 * Parameters
 * ----------
 *  k         The level for which to determine the derivative.
 *  mi        Zeeman operator matrix elements along direction for which to
 *            differentiate.
 *  data      The ZEFOZ search data.
 */
inline double d1(int k, double complex *mi, zd_inst *data) {
  int n;
  double s;
  double complex *phi;

  n = data->h->n;
  phi = data->z;
  s = inprod(n, &(phi[n*k]), mi, &(phi[n*k]), data->inprod_w);
  
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
 *  data      The ZEFOZ search data.
 */
inline double d2(int k, double complex *mi, double complex *mj, zd_inst *data) {
  int l, n; 
  double s, *omega;
  double complex *phi;

  n = data->h->n;
  omega = data->w;
  phi = data->z;

  s = 0;
  for (l=0; l<n; l++) {
    if (l != k) {
      s += inprod(n, &(phi[n*k]), mi, &(phi[n*l]), data->inprod_w) * inprod(n,
          &(phi[n*l]), mj, &(phi[n*k]), data->inprod_w)/(omega[k] - omega[l]);
    }
  }

  return s;
}


/* Calculate the Zeeman gradient vector and overwrites value in the data struct.
 *
 * Parameters
 * ----------
 *  k     Index of the kth level.
 *  l     Index of the ith level.
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  data  Data for the ZEFOZ search.
 */
inline void v_eval(int k, int l, double complex **m, zd_inst *data) {
  int i;

  for (i=0; i<3; i++) {
    data->v[i] = d1(k, m[i], data) - d1(l, m[i], data);
  }
}

/* Calculate the Zeeman curvature tensor (Jacobian) and overwrites value in the
 * data struct. 
 *
 * Parameters
 * ----------
 *  k     Index of the kth level.
 *  l     Index of the ith level.
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  data  Data for the ZEFOZ search.
 */
inline void C_eval(int k, int l, double complex **m, zd_inst *data) {
  int i, j;

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      data->C[i*3+j] = d2(k, m[i], m[j], data) - d2(l, m[i], m[j], data);
    }
  }
}


/* Iteration of ZEFOZ search. Updates value of B by diagonalizing the CF
 * Hamiltonian, computing the gradient vector and curvature tensor, and then
 * applying Newton's method.
 *
 * Parameters
 * ----------
 *  k     Index of one of the two levels between which the ZEFOZ search is to be
 *        performed.
 *  l     The index of the other level for the ZEFOZ search.
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.
 *  data  Data for the ZEFOZ search.
 */
inline void zefoz_iter(int k, int l, double complex **m, zd_inst *data) {
  int i, info, n;
  char lapack_err[64] = "LAPACKE failed with error code: 0";

  for (i=0; i<3; i++) {
    data->h->coeff[data->zi[i]] = data->B[i];
  }
  zhd('V', data->w, data->z, data->h, data->hd_w);

  v_eval(k, l, m, data);
  C_eval(k, l, m, data);
  
  n = data->h->n;
  
  /* Invert the Jacobian. */
  info = LAPACKE_dgetrf_work(LAPACK_COL_MAJOR, 3, 3, data->C, 3, data->ipiv);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
  info = LAPACKE_dgetri_work(LAPACK_COL_MAJOR, 3, data->C, 3, data->ipiv,
      data->dwork, data->dlwork);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
  
  /* Calculate C^-1 v. */
  cblas_dgemv(CblasColMajor, CblasNoTrans, 3, 3, 1, data->C, 3, data->v, 1,
      0, data->dwork, 1);

  for (i=0; i<3; i++) {
    data->B[i] -= 2*data->dwork[i];
  }
}


/* Check whether the provided magnetic field is near a ZEFOZ point, in which
 * case the ZEFOZ field values, along with the gradient vector, are inserted
 * into the zefoz array. This function calls zefoz_iter until either
 * CFL_ZEFOZ_MAX_ITER is exceeded or consecutive evaluations lead to a total
 * magnetic field difference less than xtol. 
 *
 * Parameters
 * ----------
 *  za    Zefoz array into which any located points are inserted. 
 *  xtol  If the total difference between the three field components of
 *        consecutive iterations is less than this value, then the field value
 *        is returned as a ZEFOZ point.
 *  k     Index of one of the two levels between which the ZEFOZ search is to be
 *        performed.
 *  l     The index of the other level for the ZEFOZ search. 
 *  m     Array of pointers to Zeeman operator matrix element arrays, in the
 *        order x, y, and z.  Should be in row major form, to make right-hand
 *        multiplication by a vector fast.
 *  data  Data for the ZEFOZ search.
 */
void zefoz_check(zefoz_a *za, double xtol, int k, int l, double complex **m,
    zd_inst *data) {
  int i;
  double s, tmp[3];
  
  for (i=0; i<CFL_ZEFOZ_MAX_ITER; i++) {
    memcpy(tmp, data->B, 3*sizeof(double));
  
    zefoz_iter(k, l, m, data);
    s = fabs(data->B[0]-tmp[0]);
    s += fabs(data->B[1]-tmp[1]);
    s += fabs(data->B[2]-tmp[2]);
    if (s > CFL_ZEFOZ_MAX_DIFF) {
      break;
    }
    else if (s<xtol) {
      #pragma omp critical 
      {
      zefoz_a_insert(za, data->B, data->v);
      }
      break;
    }
  }
}


/* Search for ZEFOZ points given magnetic field starting values, number of
 * search points, and increment size.  Located ZEFOZ points are written into the
 * za array.
 *
 * Parameters
 * ----------
 *  Bx      Array of field strengths to traverse along x. 
 *  By      Array of field strengths to traverse along y. 
 *  Bz      Array of field strengths to traverse along z
 *  nx      Number of field strengths along x. 
 *  ny      Number of field strengths along y. 
 *  nz      Number of field strengths along z. 
 *  k       Index of one of the two levels between which the ZEFOZ search is to
 *          be performed.
 *  l       The index of the other level for the ZEFOZ search. 
 *  xtol    If the total difference between the three field components of
 *          consecutive iterations is less than this value, then the field value
 *          is returned as a ZEFOZ point.
 *  m       Array of pointers to Zeeman operator matrix element arrays, in the
 *          order x, y, and z.
 *  za      Zefoz array into which any located points are inserted. 
 *  data    Zefoz search data and workspace struct.
 */
void zefoz_search(double *Bx, double *By, double *Bz, int nx, int ny, int nz,
    int k, int l, double xtol, double complex **m, zefoz_a *za, zefoz_d *data) {
  int x, y, z, tn;
  double B[3];

#ifdef _OPENMP
#pragma omp parallel for private(x, y, z, B) schedule(dynamic)
  for (x=0; x<nx; x++) {
    tn = omp_get_thread_num();
    B[0] = Bx[x];
    for (y=0; y<ny; y++) {
      B[1] = By[y];
      for (z=0; z<nz; z++) {
        B[2] = Bz[z];
        memcpy(data->d_inst[0]->B, B, 3*sizeof(double));
        zefoz_check(za, xtol, k, l, m, data->d_inst[tn]);
      }
    }
  }
#else
  for (x=0; x<nx; x++) {
    B[0] = Bx[x];
    for (y=0; y<ny; y++) {
      B[1] = By[y];
      for (z=0; z<nz; z++) {
        B[2] = Bz[z];
        memcpy(data->d_inst[0]->B, B, 3*sizeof(double));
        zefoz_check(za, xtol, k, l, m, data->d_inst[tn]);
      }
    }
  }
#endif /* _OPENMP */
}
