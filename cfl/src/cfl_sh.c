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
 * @file    cfl_sh.c
 * @brief   Spin Hamiltonian routines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>
#include <cfl_error.h>
#include <cfl_tensor.h>


/*
 * @brief Allocate storage for a spin Hamiltonian term. 
 *
 * @param[type]   The type of spin Hamiltonian term.
 * @param[data]   Data relevant to this type of spin Hamiltonian. 
 */
zsh *zsh_alloc(size_t n, state_t *states, sh_type_t type, sh_data_t *data) {
  zsh *sh;
  double complex *a;

  sh = (zsh *) malloc(sizeof(zsh));
  if (sh == 0) {
    CSL_ERROR_NULL("malloc failed for sh");
  }

  st = (label_t *) malloc(sizeof(label_t;));
  if (st == 0) {
    free(sh);
    CSL_ERROR_NULL("malloc failed for st");
  }

#if 0
  s_size = sizeof(states[0][0]);
  for (i=0; i<n; i++) {
    st[i] = (char *) malloc(s_size);
    if (st[i] == 0) {
      for (j=0; j<i; j++) {
        free(st[j]);
      }
      free(st);
      free(sh);
      CSL_ERROR_NULL("malloc failed for st[i]");
    }
    strcpy(st[i], states[i]);
  }
#endif

  a = (double complex *) calloc(n*n,sizeof(double complex));
  if (a == 0) {
    free(sh);
    free(st);
    CSL_ERROR_NULL("calloc failed for a");
  }

  sh->n = n;
  sh->states = st;
  sh->a = a;
  sh->type = type;
  sh->data = data;
}

/*
 * @brief Free spin Hamiltonian storage.
 */
void *zsh_free(zsh *sh) {
  free(sh->a);
  /* Add state freeing function here, once states are implemented centrally. */
  free(sh->st);
  free(sh);
}

/*
 * @brief Alloc workspace for the spin Hamiltonian projection.
 */
zshp_w *zshp_w_alloc(zt *t) {
  zshp_w *shp_w;
  size_t n = t->n;
  double complex *a; 
  double complex *m;

  shp_w = (zshp_w *) malloc(sizeof(zshp_w));
  if (shp_w == 0) {
    CSL_ERROR_NULL("malloc failed for shp_w");
  }

  m = (double complex *) calloc(n*n, sizeof(double complex));
  if (m == 0) {
    free(shp_w);
    CSL_ERROR_NULL("calloc failed for t");
  }
  
  /* Convert to dense storage, as required by the blas zhemm and ztrmm functions
   * in zshp. */
  crs_zhm2zha(t->matel, m)

  a = (double complex *) calloc(n*n, sizeof(double complex));
  if (a == 0) {
    free(shp_w);
    free(m);
    CSL_ERROR_NULL("calloc failed for a");
  }

  shp_w->m = m;
  shp_w->a = a;
  shp_w->nc = n;

  return shp_w;
}

/*
 * @brief Free workspace for spin Hamiltonian projection.
 */
zshp_w_free(zshp_w *shp_w) {
  free(shp_w->m);
  free(shp_w->a);
  free(shp_w);
}

/*
 * @brief Project out the spin Hamiltonian given an effective Hamiltonian and
 *        tensor. 
 *
 * @param[t]      The tensor for which to project out the spin Hamiltonian term.
 * @param[h]      A diagonalized Hamiltonian containing free-ion and
 *                crystal-field interactions.
 * @param[sh]     The spin Hamiltonian object.
 * @param[shp_w]  The projection workspace, allocated with zshp_w_alloc.
 * @param[l]      Integer specifying the initial level for which to project the
 *                spin Hamiltonian.
 */
void zshp(zt *t, zh *h, zsh *sh, zshp_w *shp_w, int l) {
  int i, j;
  const double complex alpha = 1;
  const double complex beta = 0;
  size_t n = t->n;
  
  /* The projection is a simmilarity transformation of the form V H V^dag, where
   * V is the eigenvector matrix of a Hamiltonian containing free-ion and
   * crystal-field interactions.  H are the matrix elements to project, i.e.,
   * Zeeman, hyperfine or quadrupole interaction elements; that is, matrix
   * elements that are diagonal in total angular momentum J. */
 
  /* Calculate VH. */
  cblas_zhemm(CblasColMajor, CblasRight, CblasUpper, n, n, &alpha, shp_w->m, n,
      h->z, n, &beta, shp_w->a, n);
  /* Calculate (VH) V^dag. */
  cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans,
      CblasNonUnit, n, n, &alpha, h->z, h->n, shp_w->a, n);

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      sh->a[i*n+j] = shp_w->a[(i+l[0])*n+j+l[0]];
    }
  }
}
