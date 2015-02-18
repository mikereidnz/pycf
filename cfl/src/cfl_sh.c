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
 * Overview
 * ========
 *
 * Spin Hamiltonian data structure used for spin Hamiltonian projection from a
 * complete Hamiltonian (see cfl_h.c) and inversion of spin Hamiltonians to
 * obtain the spin Hamiltonian parameter matrices. 
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "cfl_config.h"

#if USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <gsl/gsl_cblas.h>
#include <lapacke.h>
#endif /* USE_MKL */

#include "cfl_error.h"
#include "cfl_tensor.h"
#include "cfl_sh.h"


/*
 * Allocate storage for a spin Hamiltonian term. 
 *
 * Parameters
 * ----------
 * n  The dimensions of the spin Hamiltonian term.
 */
zsh *zsh_alloc(size_t n, char *type) {
  zsh *sh;

  sh = (zsh *) malloc(sizeof(zsh));
  if (sh == 0) {
    CFL_ERROR_NULL("malloc failed for sh");
  }

#if 0
  st = (label_t *) malloc(sizeof(label_t;));
  if (st == 0) {
    free(sh);
    CFL_ERROR_NULL("malloc failed for st");
  }


  s_size = sizeof(states[0][0]);
  for (i=0; i<n; i++) {
    st[i] = (char *) malloc(s_size);
    if (st[i] == 0) {
      for (j=0; j<i; j++) {
        free(st[j]);
      }
      free(st);
      free(sh);
      CFL_ERROR_NULL("malloc failed for st[i]");
    }
    strcpy(st[i], states[i]);
  }
#endif

  sh->pro_data = (zsh_pro_data *) malloc(sizeof(zsh_pro_data));
  if (sh->pro_data == 0) {
    free(sh);
    CFL_ERROR_NULL("malloc failed for sh->pro_data");
  }
  /* Since the size of pro_data->td is not known until zsh_set_pro is called we
   * cannot alloc space for it here.  Instead, we leave this to zsh_set_pro and
   * use the set_flag to notify zsh_free whether space has been alloced. */
  sh->pro_data->set_flag = 0;

  sh->n = n;
  sh->type = type;

  return sh;
}

void zsh_free(zsh *sh) {
  if (sh->pro_data->set_flag) {
    free(sh->pro_data->td);
  }
  free(sh->pro_data);
  free(sh);
}

/* Set the projection data for a spin Hamiltonian; a wrapper function for
 * cython. 
 *
 * Parameters
 * ----------
 *  sh      Pointer to the spin Hamlitonian for which to set pro_data.
 *  t       Pointer to tensor of the complete Hamiltonian for which to project
 *          out the spin Hamiltonian.  
 *  l       Integer specifying the initial level for which to project the spin
 *          Hamiltonian. 
 */
void zsh_set_pro(zsh *sh, zt *t, int l) {
  if (!sh->pro_data->set_flag) {
    sh->pro_data->td = (complex double *) malloc(t->n*t->n*sizeof(double
          complex));
    if (sh->pro_data->td == 0) {
      CFL_ERROR_VOID("malloc failed for pro_data->td");
    }

    sh->pro_data->set_flag = 1;
  }
  /* Convert to dense storage, as required by the blas zhemm and ztrmm functions
   * in zshp. */
  crs_zhm2zha(t->matel, sh->pro_data->td);
  sh->pro_data->tn = t->n;
  sh->pro_data->l = l;
}

/* Alloc spin Hamiltonian inversion data; a wrapper for cython. 
 *
 * Parameters
 * ----------
 *  a   The inversion coefficient matrix A in A x = b. 
 *  m   The number of rows of A and length of b. 
 *  n   The number of columns of B, and the length of x.
 */
zsh_inv_data *zsh_inv_data_alloc(complex double *a, size_t m, size_t n) {
  zsh_inv_data *d;

  d = (zsh_inv_data *) malloc(sizeof(zsh_inv_data));
  if (d == 0) {
    CFL_ERROR_NULL("malloc failed for d");
  }
  d->a = a;
  d->m = m;
  d->n = n;

  return d;
}

void zsh_inv_data_free(zsh_inv_data *d) {
  free(d);
}

/* Alloc workspace for the spin Hamiltonian projection. */
zshp_w *zshp_w_alloc(zsh *sh) {
  zshp_w *shp_w;
  size_t n = sh->pro_data->tn;
  complex double *a; 
  complex double *b;

  shp_w = (zshp_w *) malloc(sizeof(zshp_w));
  if (shp_w == 0) {
    CFL_ERROR_NULL("malloc failed for shp_w");
  }

  a = (complex double *) calloc(n*n, sizeof(complex double));
  if (a == 0) {
    free(shp_w);
    CFL_ERROR_NULL("calloc failed for a");
  }

  b = (complex double *) calloc(n*n,sizeof(complex double));
  if (b == 0) {
    free(shp_w);
    free(a);
    CFL_ERROR_NULL("calloc failed for b");
  }

  shp_w->a = a;
  shp_w->b = b;
  shp_w->nc = n;

  return shp_w;
}

void zshp_w_free(zshp_w *shp_w) {
  free(shp_w->a);
  free(shp_w->b);
  free(shp_w);
}

/*
 * Project out the spin Hamiltonian given an effective Hamiltonian and tensor. 
 *
 * Parameters
 * ----------
 *  a     Array of length nsh*nsh, with nsh the dimension of the spin
 *        Hamiltonian; will be overwritten with the result upon exit.  
 *  hz    Pointer to array containing the eigenvectors that diagonalize the
 *        Hamiltonian containing free-ion and crystal-field interactions.
 *  sh    The spin Hamiltonian object.
 *  shp_w The projection workspace, allocated with zshp_w_alloc.
 */
void zshp(complex double *a, complex double *hz, zsh *sh, zshp_w *shp_w) {
  int i, j;
  lapack_complex_double one, zero;
  one = 1;
  zero = 0;
  lapack_int n = shp_w->nc;
  int nsh = sh->n;
  
  /* The projection is a similarity transformation of the form V^dag H V, where
   * V is the eigenvector matrix of a Hamiltonian containing free-ion and
   * crystal-field interactions.  H are the matrix elements to project, i.e.,
   * Zeeman, hyperfine or quadrupole interaction elements. */
  /* Calculate H V. */
  cblas_zhemm(CblasColMajor, CblasLeft, CblasUpper, n, n, &one,
      sh->pro_data->td, n, hz, n, &zero, shp_w->a, n);
  /* Calculate V^dag (HV). */
  cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, n, &one,
      hz, n, shp_w->a, n, &zero, shp_w->b, n);
  size_t l = sh->pro_data->l;
  
  for (i=0; i<nsh; i++) {
    for (j=0; j<nsh; j++) {
      a[i*nsh+j] = shp_w->b[(i+l)*n+j+l];
    }
  }
}

/*
 * Allocate workspace for the spin Hamiltonian inversion function, which
 * solves the over-determined system Ax=b.
 *
 * Parameters
 * ----------
 *  data    Pointer to a data struct for complex valued spin Hamiltonian
 *          inversion data.
 */
zshi_w *zshi_w_alloc(zsh_inv_data *d) {
  zshi_w *w;
  complex double *a;

  w = (zshi_w *) malloc(sizeof(zshi_w));
  if (w == 0) {
    CFL_ERROR_NULL("malloc faild for w");
  }

  /* LAPACK workspace query for least-squares eqn solver. */
  lapack_complex_double *work, wquery;
  lapack_int lwork, info;

  info = LAPACKE_zgels_work(LAPACK_COL_MAJOR, 'N', d->m, d->n, 1, d->a, d->m, NULL,
      d->m, &wquery, -1);

  if (info != 0) {
    free(w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (lapack_int)wquery;
  work = (complex double *) calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  
  /* Storage for the inversion coefficient matrix; since this is overwritten by
   * zgels we must make a copy of the inversion matrix d->a to allow for
   * repeated evaluations. */
  a = (complex double *) calloc(d->m*d->n,sizeof(complex double));
  if (work == 0) {
    free(w);
    free(work);
    CFL_ERROR_NULL("calloc failed for work");
  }

  w->lwork = lwork,
  w->work = work;
  w->a = a;
  w->a_size = d->m*d->n*sizeof(double complex);
  w->data = d;

  return w;
}

/* 
 * Free the spin Hamiltonian inversion workspace. 
 */
void zshi_w_free(zshi_w *w) {
  free(w->work);
  free(w->a);
  free(w);
}

/*
 * Invert a spin Hamiltonian to obtain the parameter tensor.  The
 * inversion consists of solving the over-determined system Ax=b.  
 *
 * Parameters
 * ----------
 * a  The vector of the system to be solved; more specifically, the spin
 *    Hamiltonian elements of the term to be inverted stored in an array.  This
 *    will be overwritten with the solution upon exit, which will correspond to
 *    the spin Hamiltonian parameter tensor.
 * w  The workspace allocated with zshi_w_alloc. 
 */
void zshi(complex double *a, zshi_w *w) {
  lapack_int info; 
  char lapack_err[] = "LAPACKE_zgels failed with error code: 0";
  
  /* Store a copy of the inversion matrix. */
  memcpy((void *)w->a, (void *)w->data->a, w->a_size);
   
  info = LAPACKE_zgels_work(LAPACK_COL_MAJOR, 'N', w->data->m, w->data->n, 1,
      w->a, w->data->m, a, w->data->m, w->work, w->lwork);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zgels failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
}
