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
 *          "quadrupole", and each option must only be specified once.  If no
 *          "zeeman" interaction term is present one has to instead add the
 *          "magzs" interaction, which corresponds to a small magnetic field
 *          along the z direction.  This is required to sort the state labels of
 *          the projected spin Hamiltonian matrix elements.  
 * ninter   The number of interactions specified in inter.
 * sz       The spin projection S_z * 2; must be non-zero if inter contains
 *          "zeeman" or "hyperfine".  The factor of 2 ensures we're dealing with
 *          integer values.
 * iz       The nuclear spin projection I_z * 2; must be non-zero of inter
 *          contains "hyperfine" or "quadrupole". The factor of 2 ensures we're
 *          dealing with integer values.
 * a        An array of length ninter with entries corresponding to the
 *          inversion coefficient matrices with order matching that of inter.
 *          The coefficent matrix for a given interaction is A in Ax = b, where
 *          b is a columnv vector containing the matrix elements of the spin
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
  
  for (i=0; i<ninter; i++) {
    inv_data[i] = (zsh_inv_data *) malloc(sizeof(zsh_inv_data));
    if (inv_data[i] == 0) {
      for (j=0; j<i; j++) {
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
    else if (!strcmp("magzs", inter[i])) {
      continue;
    }
    else {
      CFL_ERROR_NULL("inter array contained invalid interaction type");
    }
    (inv_data[i])->a = a[i];
    (inv_data[i])->b = (complex double *) calloc(m, sizeof(complex double));
    if ((inv_data[i])->b == 0) {
      for (j=0; j<i; j++) {
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
    for (i=0; i<sh->ntensors; i++) {
      free((sh->pro_data[i])->pt);
      free(sh->pro_data[i]);
    }
    free(sh->pro_data);
  }
  for (i=0; i<sh->ninter; i++) {
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

  for (i=0; i<sh->ninter; i++) {
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
 *  sh      Pointer to the spin Hamlitonian for which to set pro_data.
 *  t       Pointer to array of tensors for which to project to spin Hamiltonian
 *          space.  The order of tensors in t must match the order of
 *          interactions used to alloc sh; for "zeeman" interactions three
 *          tensors are expected, in the order "magx", "magy", and "magz".  If
 *          no "zeeman" interaction term is present one has to instead add a
 *          tensor with with a small magnetic field along the z-axis, called
 *          magzs.  See the inter argument description of zsh_alloc.  l
 *          Integer specifying the initial level for which to project the spin
 *          Hamiltonian. 
 */
int zsh_set_pro(zsh *sh, zt **t, size_t l) {
  int i, j, ntensors, zeeman_index, zeeman_term;
  long thash;

  /* Check for zeeman interaction, in which case we expect tensors for three
   * magnetic field directions. */
  ntensors = sh->ninter;
  for (i=0; i<sh->ninter; i++) {
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

  zeeman_index = 0;
  zeeman_term = 0;
  thash = (t[0])->slabels->hash;
  for (i=0; i<ntensors; i++) {
    sh->pro_data[i] = (zsh_pro_data *) malloc(sizeof(zsh_pro_data));
    if (sh->pro_data[i] == 0) {
      for (j=0; j<i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data);
      CFL_ERROR_VAL("malloc failed for pro_data[i]", ENOMEM);
    }
    (sh->pro_data[i])->pt = (complex double *)
      calloc((t[i])->n*(t[i])->n, sizeof(complex double));
    if ((sh->pro_data[i])->pt == 0) {
      for (j=0; j<i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data[i]);
      free(sh->pro_data);
      CFL_ERROR_VAL("malloc failed for pro_data[i]->pt", ENOMEM);
    }
    else if (thash != (t[i])->slabels->hash) {
      for (j=0; j=i; j++) {
        free((sh->pro_data[j])->pt);
        free(sh->pro_data[j]);
      }
      free(sh->pro_data);
      CFL_ERROR_VAL("Tensor state labels passed to zsh_set_pro don't match",
          EINVAL);
    }

    /* Convert tensor matrix elements to dense storage, as required by the blas
     * zhemm and ztrmm functions in zshp_p. */
    zhcrs2zha((t[i])->matel, (sh->pro_data[i])->pt);
    
    /* Record the size of each spin Hamiltonian interaction term; for zeeman
     * interactions we need to record the same size for three tensors. */
    if (zeeman_term && zeeman_index < 2) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1);
      zeeman_index++;
    }
    else if (!strcmp("zeeman", sh->inter[i-zeeman_index])) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1);
      zeeman_term = 1;
    }
    else if (!strcmp("hyperfine", sh->inter[i-zeeman_index])) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1)*(sh->iz+1);
    }
    else if (!strcmp("quadrupole", sh->inter[i-zeeman_index])) {
      (sh->pro_data[i])->shi_dim = (sh->iz+1);
    }
    else if (!strcmp("magzs", sh->inter[i])) {
      (sh->pro_data[i])->shi_dim = (sh->sz+1);
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
  zsh_sort_t **sh_sort;

  if (sh->pro_data == NULL) {
    CFL_ERROR_NULL("sh does not contain projection data; run zsh_set_pro prior "
        "to zshp_p_w_alloc or zshp_w_alloc");
  }

  shp_p_w = (zshp_p_w *) malloc(sizeof(zshp_p_w));
  if (shp_p_w == 0) {
    CFL_ERROR_NULL("malloc failed for shp_p_w");
  }

  a = (complex double *) calloc(n*sh->dim, sizeof(complex double));
  //a = (complex double *) calloc(n*n, sizeof(complex double));
  if (a == 0) {
    free(shp_p_w);
    CFL_ERROR_NULL("calloc failed for a");
  }

  b = (complex double *) calloc(sh->dim*sh->dim,sizeof(complex double));
  //b = (complex double *) calloc(n*n,sizeof(complex double));
  if (b == 0) {
    free(shp_p_w);
    free(a);
    CFL_ERROR_NULL("calloc failed for b");
  }
  
  /* Create array of spin Hamiltonian state sorting data structs; the size is
   * the same as the dimension of the spin Hamiltonian. */
  sh_sort = (zsh_sort_t **) malloc(sh->dim*sizeof(zsh_sort_t *));
  if (sh_sort == 0) {
    free(shp_p_w);
    free(a);
    free(b);
    CFL_ERROR_NULL("malloc failed for sh_sort");
  }
  for (i=0; i<sh->dim; i++) {
    sh_sort[i] = (zsh_sort_t *) malloc(sizeof(zsh_sort_t));
    if (sh_sort[i] == 0) {
      for (j=0; j<i; j++) {
        free(sh_sort[j]);
      }
      free(sh_sort);
      free(shp_p_w);
      free(a);
      free(b);
      CFL_ERROR_NULL("malloc failed for sh_sort[i]");
    }
  }
  if (sh->iz != 0) {
    /* Determine index of iz label. */
    i=0;
    while (sh->pt_slabels->key[i]) {
      if (sh->pt_slabels->key[i] == 'I') {
        shp_p_w->iz_i = i;
        break;
      }
      i++;
    }
  } 

  shp_p_w->a = a;
  shp_p_w->b = b;
  shp_p_w->sh_sort = sh_sort;
  shp_p_w->sh_dim = sh->dim;

  return shp_p_w;
}

void zshp_p_w_free(zshp_p_w *shp_p_w) {
  int i;

  free(shp_p_w->a);
  free(shp_p_w->b);
  for (i=0; i<shp_p_w->sh_dim; i++) {
    free(shp_p_w->sh_sort[i]);
  }
  free(shp_p_w->sh_sort);
  free(shp_p_w);
}

/* 
 * Comparison function for nuclear spin label sorting.
 */
int zshp_state_cmp(const void *a, const void *b) {
  const zsh_sort_t *sa = *(const zsh_sort_t **) a;
  const zsh_sort_t *sb = *(const zsh_sort_t **) b;

  return (sa->iz < sb->iz) - (sa->iz > sb->iz);
}

/* 
 * Generate the state label sorting array.  This function must be run prior to
 * calling zshp_p whenever the eigenvectors used for the zshp_p call changes and
 * sh contains an interaction that depends on the nuclear spin.
 *
 * ----------
 *  hz      Pointer to array containing the eigenvectors that diagonalize the
 *          Hamiltonian containing free-ion and crystal-field interactions.
 *  pro_i   The projection index of the tensor according to which to sort Sz.
 *          If sh does not depend on Sz labels, set to -1. 
 *  sh      The spin Hamiltonian object.
 *  shp_p_w The projection workspace, allocated with zshp_p_w_alloc.
 */
void zshp_gen_sort(complex double *hz, int pro_i, zsh *sh, zshp_p_w *shp_p_w) {
  int i, j, col_offset, edi, odi, temp_i, pr_i;
  zsh_sort_t **sh_sort;
  sh_sort = shp_p_w->sh_sort;

  if (sh->iz != 0) {
    for (i=0; i<sh->dim; i++) {
      /* The principal component index. */
      pr_i = 0;
      /* hz is packed column wise; col_offset is the index of the first element
       * of the current column. */
      col_offset = (sh->l+i)*sh->pt_dim;
       
      for (j=0; j<sh->pt_dim; j++) {
        if (cabs(hz[col_offset+j]) > cabs(hz[col_offset+pr_i])) {
          pr_i = j;
        }
      }
      printf("i=%i, pr_i=%i, pr=%f, label=%i\n", i, pr_i, cabs(hz[col_offset+pr_i]), sh->pt_slabels->labels[pr_i][shp_p_w->iz_i]);  
      /* The i index will enumerate all unique spin Hamiltonian states.  We record
       * each i and the associated iz label of the corresponding principal
       * component. */
      (sh_sort[i])->index = i;
      (sh_sort[i])->iz = sh->pt_slabels->labels[pr_i][shp_p_w->iz_i];
    }

    /* Sort according to the nuclear spin label, iz. */
    qsort((void *) sh_sort, sh->dim, sizeof(zsh_sort_t *), zshp_state_cmp);
  }
  else {
    /* No nuclear spin; we still need to sort according to sz, so we assign only
     * the index to the sort structs. */
    for (i=0; i<sh->dim; i++) {
      (sh_sort[i])->index = i;
    }
  }
  printf("after iz sort:\n");
  for (i=0; i<sh->dim; i++) {
    printf("%i\n", sh_sort[i]->index);
  }
#if 0
  /* If the spin Hamiltonian depends on sz, sort according to it (pro_i != -1).
   * When sorting, we assume sz = 1, so sz degenerate blocks will be 2 by 2.  We
   * step through projected magz matrix elements and sort according to diagonal
   * values, from largest to smallest.
   */
  if (pro_i != -1) {
    for (i=0; i<sh->dim/2; i++) {
      /* Index of even diagonal elements. */
      edi = sh->dim*(sh_sort[i*2]->index)+sh_sort[i*2]->index;
      /* Index of odd diagonal elements. */
      odi = sh->dim*(sh_sort[i*2+1]->index)+sh_sort[i*2+1]->index;

      if (cabs(shp_p_w->b[edi]) < cabs(shp_p_w->b[odi])) {
        /* Swap index states for the current block in the sh_sort data. */
        temp_i = sh_sort[i*2]->index;
        sh_sort[i*2]->index = sh_sort[i*2+1]->index;
        sh_sort[i*2+1]->index = temp_i;
      }
    }
  }
#endif
  printf("after sz sort:\n");
  for (i=0; i<sh->dim; i++) {
    printf("%i\n", sh_sort[i]->index);
  }
}

/*
 * Read the projected spin Hamiltonian matrix elements and sort them.  This
 * function assumes that the caller has previously called zshp_gen_sort and
 * zshp_p using the desired crystal field Hamiltonian eigenvectors and pro_i. 
 *
 * Parameters
 * ----------
 *  a       Array of length shi_dim*shi_dim, with shi_dim the dimension of the
 *          spin Hamiltonian inversion term; this will be overwritten with the
 *          result upon exit.  
 *  sh      The spin Hamiltonian.
 *  pro_i   The index for which tensor to project out the spin Hamiltonian
 *          matrix elements.
 */
void zshp_parse(complex double *a, zsh *sh, int pro_i, zshp_p_w *shp_p_w) {
  int i, j, shi_dim, sh_dim;

  shi_dim = sh->pro_data[pro_i]->shi_dim;
  sh_dim = sh->dim;
  /* We read out the shi_dim*shi_dim block corresponding to the spin Hamiltonian
   * matrix elements specific to the interaction type.  Furthermore, we use the
   * index mapping, sh_sort, that sorts the state labels according to, firstly,
   * nuclear spin projection, and, secondly, spin projection.  This ensures that
   * when we go to invert the resulting array to calculate the spin Hamiltonian
   * parameters, our matrix element state labels match those calculated here. */
  for (i=0; i<shi_dim; i++) {
    for (j=0; j<shi_dim; j++) {
      a[i*shi_dim+j] = shp_p_w->b[(shp_p_w->sh_sort[i]->index)*sh_dim + j];
      //printf("%f ", a[i*shi_dim+j]);
    }
    //printf("\n");
  }
  for (i=0; i<sh_dim; i++) {
    for (j=0; j<sh_dim; j++) {
      printf("%f+%fI ", creal(shp_p_w->b[i*sh_dim + j]), cimag(shp_p_w->b[i*sh_dim + j]) );
    }
    printf("\n");
  }

  printf("\n");
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
void zshp_p(complex double *hz, zsh *sh, int pro_i, zshp_p_w *shp_p_w) {
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


///* Calculate H V. */
  //cblas_zhemm(CblasColMajor, CblasLeft, CblasUpper, d, d, &one, pd->pt, d, hz,
  //    d, &zero, shp_p_w->a, d);
  ///* Calculate V^dag (HV). */
  //cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, d, d, d, &one, hz, d,
  //    shp_p_w->a, d, &zero, shp_p_w->b, d);
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
void zshi(complex double *a, zshi_w *w) {
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
  for (i=0; i<9; i++) {
    a[i] = w->data->b[i];
  }
}

/*
 * Alloc storage for a crystal field Hamiltonian to spin Hamiltonian parameter
 * projection. 
 *
 * Parameters
 * ----------
 *  sh    The spin Hamiltonian object.
 */
zshp_w *zshp_w_alloc(zsh *sh) {
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
 
  /* We use the value of msz to check whether a zeeman term is present in zshp,
   * hence we initialize here. */
  w->msz = -1;
  w->magz_i = -1;
  for (i=0; i<sh->ninter; i++) {
    w->shi_w[i] = zshi_w_alloc(sh->inv_data[i]);
    if (w->shi_w[i] == 0) {
      for (j=0; j<i; j++) {
        free(w->shi_w[i]);
      }
      free(w->shp_p_w);
      free(w);
      CFL_ERROR_NULL("malloc failed for zshi_w[i]");
    }

    if (!strcmp("zeeman", sh->inter[i])) {
      /* The dimension of a single Zeeman term. */
      w->msz = (sh->inv_data[i])->m/3;
      w->magz_i = i;
    }
    else if (!strcmp("magzs", sh->inter[i])) {
      /* No Zeeman interaction, so record small magz tensor index. */
      w->magz_i = i;
    }
  }

  w->ninter = sh->ninter;
  w->zeeman_offset = 0;

  return w;
}


void zshp_w_free(zshp_w *w) {
  int i;

  zshp_p_w_free(w->shp_p_w);
  for (i=0; i<w->ninter; i++) {
    zshi_w_free(w->shi_w[i]);
  }
  free(w->shi_w);
  free(w);
}

/* 
 * Calculate the spin Hamiltonian parameters given a crystal field Hamiltonian.
 * This function wraps the projection, state label sorting, and inversion
 * function calls.  Additionally, the associated alloc and free functions handle
 * all necessary initializations operations and memory allocs/frees. 
 *
 * N.B.: repeat evaluations are fine, provided int_i increases monotonically to
 * ninter and then starts at 0 again. 
 *
 * Parameters
 * ----------
 *  a       An array of length 9 which will be overwritten with the spin
 *          Hamiltonian parameter matrix of the interaction specified with
 *          int_i upon exit.
 *  hz      Pointer to array containing the eigenvectors that diagonalize the
 *          Hamiltonian containing free-ion and crystal-field interactions.
 *  int_i   Index specifying for which interaction to calculate the parameter
 *          matrix.  The value is determined by the order of the **inter array
 *          used to create the sh object.  
 *  sh      The spin Hamiltonian object.
 *  shp_w   The parameter workspace.
 */
void zshp(complex double *a, complex double *hz, int int_i, zsh *sh, zshp_w *w) {
  int i;
  /* Generate sorting data every time we cycle through all interactions. */
  if (int_i == 0) {
    if (w->msz != -1) {
      /* Reset the offset between int_i and pro_i due to a zeeman term. */
      w->zeeman_offset = 0;
      /* zeeman interaction is present; we assume magz is the third zeeman
       * tensor. */
      zshp_p(hz, sh, w->magz_i+2, w->shp_p_w);
      zshp_gen_sort(hz, w->magz_i+2, sh, w->shp_p_w);
      /* Fill in the third 'block' of inversion data for zeeman int. */
      zshp_parse(&((sh->inv_data[w->magz_i])->b[2*w->msz]), sh, w->magz_i+2,
          w->shp_p_w);
    } 
    else {
      /* No zeeman interaction; sort using small magz or not at all w.r.t. Sz.*/
      zshp_p(hz, sh, w->magz_i, w->shp_p_w);
      zshp_gen_sort(hz, w->magz_i, sh, w->shp_p_w);
    }
  }

  if (!strcmp("zeeman", sh->inter[int_i])) {
    for (i=0; i<2; i++) {
      zshp_p(hz, sh, int_i+i, w->shp_p_w);
      zshp_parse(&((sh->inv_data[int_i])->b[i*w->msz]), sh, int_i+2,
          w->shp_p_w);
    }
    w->zeeman_offset = 2;
  }
  else {
    zshp_p(hz, sh, int_i+w->zeeman_offset, w->shp_p_w);
    zshp_parse((sh->inv_data[int_i])->b, sh, int_i+w->zeeman_offset, w->shp_p_w);
  }
  
  zshi(a, w->shi_w[int_i]);
}
