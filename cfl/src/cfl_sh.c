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


/*
 * Overview
 * ========
 *
 * Spin Hamiltonian data structure used for spin Hamiltonian projection from a
 * complete Hamiltonian (see cfl_h.c) and inversion of spin Hamiltonians to
 * obtain the spin Hamiltonian parameter matrices. 
 *
 * TODO: explain call sequence of gen sort, proj, and inv. 
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <errno.h>

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


/* Allocate spin Hamiltonian storage.
 *
 * inter    Array of strings specifying the interactions described by the spin
 *          Hamiltonian.  Valid options are "zeeman", "hyperfine", and
 *          "quadrupole", and each option must only be specified once.  
 * ninter   The number of interactions specified in inter.
 * sz       The spin projection S_z * 2; must be non-zero if inter contains
 *          "zeeman" or "hyperfine".  The factor of 2 ensures we're dealing with
 *          integer values.
 * iz       The nuclear spin projection I_z * 2; must be non-zero if inter
 *          contains "hyperfine" or "quadrupole". The factor of 2 ensures we're
 *          dealing with integer values.  Set to zero for Iz = 0 isotopes.
 * a        An array of length ninter with entries corresponding to the
 *          inversion coefficient matrices with order matching that of inter.
 *          The coefficent matrix for a given interaction is A in Ax = b, where
 *          b is a column vector containing the matrix elements of the spin
 *          Hamiltonian for this interaction.  Consequently, it is shdim*shdim
 *          by 1, where shdim is the dimension of the spin Hamiltonian for the
 *          specific interaction.  Additonally, x is the spin Hamiltonian
 *          parameter matrix stacked into a 9 by 1 column.  For the zeeman
 *          inversion array, A must consists of three inversion arrays,
 *          concatenated into one large array, for magnetic fields along x, y,
 *          and z directions, in that order.
 */
zsh *zsh_alloc(char **inter, size_t ninter, int sz, int iz, complex double **a) {
  int i, j, m;
  zsh *sh;
  zsh_inv_data **inv_data;

  sh = (zsh *) malloc(sizeof(zsh));
  if (sh == 0) {
    CFL_ERROR_NULL("malloc failed for sh");
  }

  inv_data = (zsh_inv_data **) malloc(sizeof(zsh_inv_data *)*ninter);
  if (inv_data == 0) {
    free(sh);
    CFL_ERROR_NULL("malloc failed for inv_data array");
  }
  
  for (i = 0; i < ninter; i++) {
    inv_data[i] = (zsh_inv_data *) malloc(sizeof(zsh_inv_data));
    if (inv_data[i] == 0) {
      for (j = 0; j < i; j++) {
        free((inv_data[j])->b);
        free(inv_data[j]);
      }
      free(inv_data);
      free(sh);
      CFL_ERROR_NULL("malloc failed for inv_data");
    }
    /* Since sz and iz correspond to 2*S_z and 2*I_z we don't have to multiply
     * the spin projection by 2 to calculate the number of states.  The factor
     * of 3 for zeeman is required since we form a column of 3 Zeeman spin
     * Hamiltonian states stacked on top of each other. */
    if (!strcmp("zeeman", inter[i])) {
      m = (sz+1)*(sz+1)*3;
    }
    else if (!strcmp("hyperfine", inter[i])) {
      m = (sz+1)*(iz+1)*(sz+1)*(iz+1); 
    }
    else if (!strcmp("quadrupole", inter[i])) {
      m = (iz+1)*(iz+1);
    }
    else {
      CFL_ERROR_NULL("inter array contained invalid interaction type");
    }
    (inv_data[i])->a = a[i];
    (inv_data[i])->b = (complex double *) calloc(m, sizeof(complex double));
    if ((inv_data[i])->b == 0) {
      for (j = 0; j < i; j++) {
        free((inv_data[j])->b);
        free(inv_data[j]);
      }
      free(inv_data[i]);
      free(inv_data);
      free(sh);
      CFL_ERROR_NULL("malloc failed for inv_data[i]->b");
    }
    (inv_data[i])->m = m;
  }

  sh->dim = (sz+1)*(iz+1);
  sh->inter = inter;
  sh->ninter = ninter; 
  sh->sz = sz;
  sh->iz = iz;
  sh->inv_data = inv_data;
  sh->ntensors = 0;
  sh->pro_data = NULL;

  return sh;
}

void zsh_free(zsh *sh) {
  int i;

  if (sh->ntensors != 0) {
    for (i = 0; i < sh->ntensors; i++) {
      free((sh->pro_data[i])->pt);
      free(sh->pro_data[i]);
    }
    free(sh->pro_data);
  }
  for (i = 0; i < sh->ninter; i++) {
    free((sh->inv_data[i])->b);
    free(sh->inv_data[i]);
  }
  free(sh->inv_data);
  free(sh);
}

/* Copy the matrix elements of a spin Hamiltonian interaction to a spin
 * Hamiltonian object.  This can be used for inverting spin Hamiltonians without
 * projecting the matrix elements from a crystal field Hamiltonian.
 *
 * Parameters
 * ----------
 *  sh      Pointer to the spin Hamlitonian for which to set matrix elements.
 *  b       Pointer to array of matrix elements; these will be copied to storage
 *          already allocated with zsh_alloc.
 *  inter   The string identifing the interaction for which to copy matrix
 *          elements. 
 */
void zsh_set_inv(zsh *sh, complex double *b, char *inter) {
  int i;

  for (i = 0; i < sh->ninter; i++) {
    if (!strcmp(sh->inter[i], inter)) {
      memcpy((sh->inv_data[i])->b, b, (sh->inv_data[i])->m*sizeof(complex double));
    }
  }
}


/* Set the projection data for a spin Hamiltonian.  The tensor matrix elements
 * are copied to dense storage, so the **t memory can be freed after calling
 * this function.
 *
 * The return value, upon success, is 1, otherwise, the return value is EINVAL,
 * or ENOMEM. 
 *
 * Parameters
 * ----------
 *  sh        Pointer to the spin Hamlitonian for which to set pro_data.
 *  t         Pointer to array of tensors for which to project to spin
 *            Hamiltonian space.  The order of tensors in t must match the order
 *            of interactions used to alloc sh; for "zeeman" interactions three
 *            tensors are expected, in the order "magx", "magy", and "magz".  
 *  l         Integer specifying the initial level for which to project the spin
 *            Hamiltonian; zero based indexing.
 *  coupling  Array containing coupling constants for hyperfine and quadrupole
 *            interactions.  Elements must only be set if the interaction is
 *            present, and the order in which they are set must match the order
 *            in which the corresponding tensor is provided in t.
 */
int zsh_set_pro(zsh *sh, zt **t, int l, double *coupling) {
  int i, j, ntensors, zc, zf, cc;
  long thash;

  /* Check for zeeman interaction, in which case we expect tensors for three
   * magnetic field directions. */
  ntensors = sh->ninter;
  for (i = 0; i < sh->ninter; i++) {
    if (!strcmp("zeeman", sh->inter[i])) {
      ntensors += 2;
    }
  }

  /* Verify that nuclear spin labels are present if and only if the spin
   * Hamiltonian has a non-zero nuclear spin. */
  if (sh->iz != 0) {
    if (strchr(t[0]->slabels->key, 'I') == NULL) {
      CFL_ERROR_VAL("Tensors passed to zsh_set_pro do not contain nuclear spin "
          "matrix elements, yet zsh_alloc was called with iz != 0.", EINVAL);
    }
  }
  else {
    if (strchr(t[0]->slabels->key, 'I') != NULL) {
      CFL_ERROR_VAL("Tensors passed to zsh_set_pro contain nuclear spin matrix "
          "elements, yet zsh_alloc was called with iz == 0.", EINVAL);
    }
  }

  sh->pro_data = (zsh_pro_data **) malloc(ntensors*sizeof(zsh_pro_data *));
  if (sh->pro_data == 0) {
    CFL_ERROR_VAL("malloc failed for pro_data", ENOMEM);
  }

  zf = 0;       /* Flag set to true when Zeeman term is encountered. */
  zc = 0;       /* Number of pro_data allocs since zf = 1. */
  cc = 0;       /* Coupling constant counter. */
  thash = (t[0])->slabels->th;
  for (i = 0; i < ntensors; i++) {
    sh->pro_data[i] = (zsh_pro_data *) malloc(sizeof(zsh_pro_data));
    if (sh->pro_data[i] == 0) {
      for (j = 0; j < i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data);
      CFL_ERROR_VAL("malloc failed for pro_data[i]", ENOMEM);
    }
    (sh->pro_data[i])->pt = (complex double *)
      calloc((t[i])->n*(t[i])->n, sizeof(complex double));
    if ((sh->pro_data[i])->pt == 0) {
      for (j = 0; j < i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data[i]);
      free(sh->pro_data);
      CFL_ERROR_VAL("malloc failed for pro_data[i]->pt", ENOMEM);
    }
    else if (thash != (t[i])->slabels->th) {
      for (j = 0; j = i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data);
      CFL_ERROR_VAL("Tensor state labels passed to zsh_set_pro don't match",
          EINVAL);
    }

    /* Convert tensor matrix elements to dense storage, as required by the blas
     * zhemm and ztrmm functions in zshp_p. */
    zhcsr2zha((t[i])->matel, (sh->pro_data[i])->pt);
    
    /* Record the size of each spin Hamiltonian interaction term; for zeeman
     * interactions we need to record the same size for three tensors. */
    if (zf && zc < 2) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1);
      zc++;
    }
    else if (!strcmp("zeeman", sh->inter[i-zc])) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1);
      zf = 1;
    }
    else if (!strcmp("hyperfine", sh->inter[i-zc])) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1)*(sh->iz+1);
      sh->pd_map[0] = i;
      /* Set the coupling coefficient. */
      sh->pro_data[i]->coupling = coupling[cc];
      cc++;
    }
    else if (!strcmp("quadrupole", sh->inter[i-zc])) {
      (sh->pro_data[i])->shi_dim = (sh->iz+1);
      sh->pd_map[1] = i;
      /* Set the coupling coefficient. */
      sh->pro_data[i]->coupling = coupling[cc];
      cc++;
    }
  }

  sh->ntensors = ntensors;
  sh->l = l;
  /* We have verified that all tensors have matching state labels. */
  sh->pt_slabels = t[0]->slabels;
  sh->pt_dim = t[0]->n;

  return 1;
}


/* Alloc workspace for the spin Hamiltonian projection. 
 *
 * Parameters
 * ----------
 *  sh    The spin Hamiltonian object for which to alloc projection workspace. 
 */
zshp_p_w *zshp_p_w_alloc(zsh *sh) {
  zshp_p_w *shp_p_w;
  int i, j;
  size_t n = sh->pt_dim;
  complex double *a; 
  complex double *b;

  if (sh->pro_data == NULL) {
    CFL_ERROR_NULL("sh does not contain projection data; run zsh_set_pro prior "
        "to zshp_p_w_alloc or zshp_w_alloc");
  }

  shp_p_w = (zshp_p_w *) malloc(sizeof(zshp_p_w));
  if (shp_p_w == 0) {
    CFL_ERROR_NULL("malloc failed for shp_p_w");
  }

  a = (complex double *) calloc(n*sh->dim, sizeof(complex double));
  if (a == 0) {
    free(shp_p_w);
    CFL_ERROR_NULL("calloc failed for a");
  }

  b = (complex double *) calloc(sh->dim*sh->dim,sizeof(complex double));
  if (b == 0) {
    free(shp_p_w);
    free(a);
    CFL_ERROR_NULL("calloc failed for b");
  }

  shp_p_w->a = a;
  shp_p_w->b = b;

  return shp_p_w;
}

void zshp_p_w_free(zshp_p_w *shp_p_w) {
  int i;

  free(shp_p_w->a);
  free(shp_p_w->b);
  free(shp_p_w);
}


/*
 * Read the projected spin Hamiltonian matrix elements.  
 *
 * Parameters
 * ----------
 *  a           Array of length shi_dim*shi_dim, with shi_dim the dimension of
 *              the spin Hamiltonian inversion term; this will be overwritten
 *              with the result upon exit.  
 *  sh          The spin Hamiltonian.
 *  pro_i       The index for which tensor to project out the spin Hamiltonian
 *              matrix elements.
 *  zf          Flag whether we're dealing with a Zeeman term.
 *  shp_p_w     Projection workspace. 
 */
inline void zshp_parse(complex double *a, zsh *sh, int pro_i, int zf, zshp_p_w
    *shp_p_w) {
  int i, j, ii, jj, shi_dim, sh_dim;
  zsh_pro_data *pd;

  pd = sh->pro_data[pro_i];

  shi_dim = sh->pro_data[pro_i]->shi_dim;
  sh_dim = sh->dim;
  /* We read out the shi_dim*shi_dim block corresponding to the spin Hamiltonian
   * matrix elements specific to the interaction type.  The ordering after proj
   * is with S state labels 'slow' and I state labels 'fast'.  Consequently, for
   * Zeeman we have to add an offset of 2 Iz (sh->iz is multplied by 2 by
   * default, extra +1 comes from incrementing index) to get matrix elements of
   * Sz \pm 1 with matching Iz.*/
  if (zf != 0) {
    for (i = 0; i < shi_dim; i++) {
      for (j = 0; j < shi_dim; j++) {
        ii = (i % 2 == 0) ? i : i+sh->iz;
        jj = (j % 2 == 0) ? j : j+sh->iz;
        a[i*shi_dim+j] = shp_p_w->b[ii*sh_dim + jj];
      }
    }
  }
  else {
    for (i = 0; i < shi_dim; i++) {
      for (j = 0; j < shi_dim; j++) {
        a[i*shi_dim+j] = shp_p_w->b[i*sh_dim + j]*pd->coupling;
      }
    }
  }
}


/*
 * Project out the spin Hamiltonian given an effective Hamiltonian and tensor. 
 *
 * Parameters
 * ----------
 *  hz      Pointer to array containing the eigenvectors that diagonalize the
 *          Hamiltonian containing free-ion and crystal-field interactions.
 *  sh      The spin Hamiltonian.
 *  pro_i   The index for which tensor to project the spin Hamiltonian matrix
 *          elements.
 *  shp_p_w The projection workspace, allocated with zshp_p_w_alloc.
 */
inline void zshp_p(complex double *hz, zsh *sh, int pro_i, zshp_p_w *shp_p_w) {
  int d;
  complex double one, zero;
  zsh_pro_data *pd;

  pd = sh->pro_data[pro_i];
  d = sh->pt_dim;
  one = 1;
  zero = 0;

  /* The projection is a similarity transformation of the form V^dag H V, where
   * V is the eigenvector matrix of a Hamiltonian containing free-ion and
   * crystal-field interactions.  H are the matrix elements to project, i.e.,
   * Zeeman, hyperfine or quadrupole interaction elements.
   *
   * We only calculate the submatrix that corresponds to the spin Hamiltonian
   * (denoted with entries x):
   *     v^dag       H        V
   *  (        )(--------)(      ||)    (        )
   *  (        )(--------)(      ||) =  (        )
   *  (--------)(--------)(      ||)    (      xx)
   *  (--------)(--------)(      ||)    (      xx)
   *            `------------------'
   *                      HV
   *                  (      ||)
   *               =  (      ||)
   *                  (      ||)
   *                  (      ||)
   */
  cblas_zhemm(CblasColMajor, CblasLeft, CblasUpper, d, sh->dim, &one, pd->pt, d,
      &hz[(sh->l)*d], d, &zero, shp_p_w->a, d);
  cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, sh->dim, sh->dim, d,
      &one, &hz[(sh->l)*d], d, shp_p_w->a, d, &zero, shp_p_w->b, sh->dim);
}


/* 
 * Allocate workspace for symmeterizing spin Hamiltonian parameter tensors using
 * a singular value decomposition. 
 */
svd_sym_w *svd_sym_w_alloc(void) {
  int lwork, info;
  double *s, *u, *vt, wquery;
  svd_sym_w *w;

  w = (svd_sym_w *) malloc(sizeof(svd_sym_w));
  if (w == 0) {
    CFL_ERROR_NULL("malloc failed for w");
  }

  info = LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'A', 'A', 3, 3, NULL, 3, NULL,
      NULL, 3, NULL, 3, &wquery, -1);
  if (info != 0) {
    free(w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (int)wquery;
  w->work = (double *) calloc(lwork,sizeof(double));
  if (w->work == 0) {
    free(w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  memset(w->s, 0, 9*sizeof(double));
  memset(w->u, 0, 9*sizeof(double));
  memset(w->vt, 0, 9*sizeof(double));

  w->lwork = lwork;

  return w;
}


/* 
 * Free the spin Hamiltonian tensor symmeterization workspace. 
 */
void svd_sym_w_free(svd_sym_w *w) {
  free(w->work);
  free(w);
}

/* 
 * Perform a symmeterization of an spin Hamiltonian parameter tensor using a
 * singular value decomposition.
 *
 * Parameters
 * ----------
 *  a     The parameter tensor array.
 *  w     The symmeterization workspace.
 */
void svd_sym(double *a, svd_sym_w *w) {
  int info;
  char lapack_err[] = "LAPACKE_zgesvd failed with error code: 0";

  memcpy(w->tmp_a, a, 9*sizeof(double));
  info = LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'A', 'A', 3, 3, a, 3, w->s, w->u,
      3, w->vt, 3, w->work, w->lwork);
  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zgesvd failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }

  /* w->work is at least 5*MIN(M,N), which in our case is 15.  Therefore, we can
   * use w->work as a matrix multiplication workspace. */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1, w->tmp_a, 3,
      w->u, 3, 0, w->work, 3);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 3, 1, w->work,
      3, w->vt, 3, 0, a, 3);  

}


/*
 * Allocate workspace for the spin Hamiltonian inversion function, which
 * solves the over-determined system Ax=b.
 *
 * Parameters
 * ----------
 *  job     Specify whether to allocate space for a singular value decomposition
 *          that symmeterizes the resulting spin Hamiltonian parameter tensor.
 *          Set to 'S' to enable SVD symmeterization, and to 'N' otherwise.
 *  data    Pointer to a data struct for complex valued spin Hamiltonian
 *          inversion data.
 */
zshi_w *zshi_w_alloc(char job, zsh_inv_data *d) {
  int ldb, lwork, info;
  zshi_w *w;
  complex double *a, *work, wquery;

  w = (zshi_w *) malloc(sizeof(zshi_w));
  if (w == 0) {
    CFL_ERROR_NULL("malloc faild for w");
  }

  /* LAPACK workspace query for least-squares eqn solver. */
  ldb = (d->m > 9 ? d->m: 9);
  info = LAPACKE_zgels_work(LAPACK_COL_MAJOR, 'N', d->m, 9, 1, d->a, d->m, NULL,
      ldb, &wquery, -1);

  if (info != 0) {
    free(w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (int)wquery;
  work = (complex double *) calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(w);
    CFL_ERROR_NULL("calloc failed for work");
  }

  /* Storage for the inversion coefficient matrix; since this is overwritten by
   * zgels we must make a copy of the inversion matrix d->a to allow for
   * repeated evaluations. */
  a = (complex double *) calloc(d->m*9,sizeof(complex double));
  if (work == 0) {
    free(w);
    free(work);
    CFL_ERROR_NULL("calloc failed for work");
  }

  if (job == 'S') {
    w->job = 'S';
    w->svd_w = (svd_sym_w *) svd_sym_w_alloc();
    if (w->svd_w == 0) {
      free(w);
      free(work);
      free(a);
      CFL_ERROR_NULL("svd_sym_w_alloc failed");
    }
  }
  else {
    w->job = 'N';
    w->svd_w = NULL;
  }

  w->job = job;
  w->lwork = lwork,
    w->work = work;
  w->a = a;
  w->a_size = d->m*9*sizeof(double complex);
  w->data = d;
  w->ldb = ldb;

  return w;
}

/* 
 * Free the spin Hamiltonian inversion workspace. 
 */
void zshi_w_free(zshi_w *w) {
  if (w->job == 'S') {
    svd_sym_w_free(w->svd_w);
  }
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
 *  a   An array of length 9 which will be overwritten with the spin Hamiltonian
 *      parameter matrix of the interaction up on exit.
 *  w   The workspace allocated with zshi_w_alloc. 
 */
inline void zshi(double *a, zshi_w *w) {
  int i, info; 
  char lapack_err[] = "LAPACKE_zgels failed with error code: 0";

  /* Store a copy of the inversion matrix. */
  memcpy((void *)w->a, (void *)w->data->a, w->a_size);

  info = LAPACKE_zgels_work(LAPACK_COL_MAJOR, 'N', w->data->m, 9, 1, w->a,
      w->data->m, w->data->b, w->ldb, w->work, w->lwork);

  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zgels failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
  for (i = 0; i < 9; i++) {
    a[i] = creal(w->data->b[i]);
  }

  if (w->job == 'S') {
    svd_sym(a, w->svd_w);
  }
}

/*
 * Alloc storage for a crystal field Hamiltonian to spin Hamiltonian parameter
 * projection. 
 *
 * Parameters
 * ----------
 *  job   Specify whether to allocate space for a singular value decomposition
 *        that symmeterizes the resulting spin Hamiltonian parameter tensor. Set
 *        to 'S' to enable SVD symmeterization, and to 'N' otherwise. 
 *  sh    The spin Hamiltonian object.
 */
zshp_w *zshp_w_alloc(char job, zsh *sh) {
  int i, j;
  zshp_w *w;

  if (sh->sz != 1) {
    CFL_ERROR_NULL("projecting spin hamiltonians with "
        "sz != 1 is presently not implemented")
  }

  w = (zshp_w *) malloc(sizeof(zshp_w));
  if (w == 0) {
    CFL_ERROR_NULL("malloc faild for w");
  }

  w->shp_p_w = (zshp_p_w *) zshp_p_w_alloc(sh);
  if (w->shp_p_w == 0) {
    free(w);
    CFL_ERROR_NULL("malloc failed for zshp_p_w");
  }

  w->shi_w = (zshi_w **) malloc(sh->ninter*sizeof(zshi_w *));
  if (w->shi_w == 0) {
    free(w->shp_p_w);
    free(w);
    CFL_ERROR_NULL("malloc failed for zshi_w");
  }

  /* Alloc inversion workspace. */
  for (i = 0; i < sh->ninter; i++) {
    /* Disable SVD for quadrupole, irrespective of what the job flag specifies,
     * since there's no S matrix elements. */
    if (!strcmp("quadrupole", sh->inter[i])) {
      job = 'N';
    }
    w->shi_w[i] = zshi_w_alloc(job, sh->inv_data[i]);
    if (w->shi_w[i] == 0) {
      for (j = 0; j < i; j++) {
        free(w->shi_w[i]);
      }
      free(w->shp_p_w);
      free(w);
      CFL_ERROR_NULL("malloc failed for zshi_w[i]");
    }

    if (!strcmp("zeeman", sh->inter[i])) {
      /* Index of the Zeeman interaction in the inter array. */
      w->zi = i;
      /* Length in inv_data->b (total length m) of a single zeeman term.  For
       * Sz=1/2 this is 4. */
      w->msz = (sh->inv_data[i])->m/3;
    }
  }

  w->ninter = sh->ninter;

  return w;
}

void zshp_w_free(zshp_w *w) {
  int i;

  zshp_p_w_free(w->shp_p_w);
  for (i = 0; i < w->ninter; i++) {
    zshi_w_free(w->shi_w[i]);
  }
  free(w->shi_w);
  free(w);
}

/* 
 * Calculate the spin Hamiltonian parameters given a crystal field Hamiltonian.
 * This function wraps the projection and inversion function calls.
 * Additionally, the associated alloc and free functions handle all necessary
 * initialization operations and memory allocs/frees. 
 *
 * N.B.: repeat evaluations are fine, provided int_i increases monotonically to
 * ninter and then starts at 0 again. 
 *
 * Parameters
 * ----------
 *  a       An array of length 9 which will be overwritten with the spin
 *          Hamiltonian parameter matrix of the interaction specified with
 *          int_i upon exit.
 *  b       An array of length dshi^2, with dshi the dimension of the spin
 *          Hamiltonian specific to this interaction.  For a Zeeman interaction,
 *          this must be 3*dshi^2 in order to accomodate the three orientations
 *          magx, magy, and magz.  b will be overwritten with the spin
 *          Hamiltonian matrix elements upon exit.  Can be set to NULL if the
 *          matrix elements are not required.
 *  hz      Pointer to array containing the eigenvectors that diagonalize the
 *          Hamiltonian containing free-ion and crystal-field interactions.
 *  int_i   Index specifying for which interaction to calculate the parameter
 *          matrix.  The value is determined by the order of the **inter array
 *          used to create the sh object.  
 *  sh      The spin Hamiltonian object.
 *  shp_w   The parameter workspace.
 */
void zshp(double *a, complex double *b, complex double *hz, int int_i,
    zsh *sh, zshp_w *w) {
  int i;
  if (int_i < w->zi) {
    /* Before Zeeman interaction term, so inv index matches pro index. */
    zshp_p(hz, sh, int_i, w->shp_p_w);
    zshp_parse((sh->inv_data[int_i])->b, sh, int_i, 0, w->shp_p_w);
  }
  else if (int_i == w->zi) {
    /* Zeeman interaction term, loop through three projections. */
    for (i = 0; i < 3; i++) {
      zshp_p(hz, sh, int_i+i, w->shp_p_w);
      zshp_parse(&((sh->inv_data[int_i])->b[i*w->msz]), sh, int_i+i, 1,
          w->shp_p_w);
    }
  }
  else {
    /* After Zeeman term; add offset between inv and pro index. */
    zshp_p(hz, sh, int_i+2, w->shp_p_w);
    zshp_parse((sh->inv_data[int_i])->b, sh, int_i+2, 0, w->shp_p_w);
  }

  if (b != NULL) {
    memcpy(b, (sh->inv_data[int_i])->b, w->shi_w[int_i]->ldb*sizeof(complex double));
  }
  zshi(a, w->shi_w[int_i]);

}

