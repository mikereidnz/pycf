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

/* Double pointer comparison function for qsort. */
int dptr_cmp(const void *a, const void *b) {
  const double **da = (const double **) a;
  const double **db = (const double **) b;

  return (**da > **db) - (**da < **db);
}

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

  h->n = n;
  h->nt = nt;
  h->t = t;

  return h;
}

void zh_free(zh *h) {
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

/* Allocate workspace for a LAPACKE_zheevd diagonalization. 
 *
 * Parameters
 * ----------
 *
 */
zheevd_w *zheevd_w_alloc(char job, int n, double abstol) {
  int lda = n, ldz = n, il, iu, info;
  double vl, vu;
  int lwork, lrwork, liwork;
  complex double *work, wquery;
  double *rwork, rwquery;
  int *iwork, iwquery;
  zheevd_w *heevd_w;

  heevd_w = (zheevd_w *) malloc(sizeof(zheevd_w));
  if (heevd_w ==0) {
    CFL_ERROR_NULL("malloc failed for heevd_w");
  }

  heevd_w->isuppz = (int *) malloc(2*n*sizeof(int));
  if (heevd_w->isuppz == 0) {
    free(heevd_w);
    CFL_ERROR_NULL("malloc failed for isuppz");
  }
  /* zheevr workspace query. */
  lwork = -1;
  lrwork = -1;
  liwork = -1;

  //FIXME: changed h->a to NULL... if lapack throws errors, that is probably
  //why.
  info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', n, NULL, lda, vl, vu, il, iu,
      abstol, &(heevd_w->m), NULL, NULL, ldz, heevd_w->isuppz, &wquery,
      lwork, &rwquery, lrwork, &iwquery, liwork);
  if (info != 0) {
    free(heevd_w->isuppz);
    free(heevd_w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (int) creal(wquery);
  lrwork = (int) rwquery;
  liwork = iwquery;

  work = (complex double *) calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(heevd_w->isuppz);
    free(heevd_w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  rwork = (double *) calloc(lrwork,sizeof(double));
  if (rwork == 0) {
    free(heevd_w->isuppz);
    free(heevd_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for rwork");
  }
  iwork = (int *) calloc(liwork,sizeof(int));
  if (iwork == 0) {
    free(heevd_w->isuppz);
    free(heevd_w);
    free(work);
    free(rwork);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  heevd_w->work = work;
  heevd_w->lwork = lwork;
  heevd_w->rwork = rwork;
  heevd_w->lrwork = lrwork;
  heevd_w->iwork = iwork;
  heevd_w->liwork = liwork;

  return heevd_w;
}

void zheevd_w_free(zheevd_w *heevd_w) {
  free(heevd_w->isuppz);
  free(heevd_w->work);
  free(heevd_w->rwork);
  free(heevd_w->iwork);
  free(heevd_w);
}

/* Read RCM ordered CRS matrix into pre-allocated blocks. 
 *
 * Parameters
 * ----------
 *  nblocks     The number of blocks.
 *  blocks      Array of zblock structures to be filled. 
 *  crs_m       The block diagonalized CRS matrix. 
 */
inline void zh_parse_blocks(int nblocks, zblock **blocks, zcrs *crs_m) {
  int i, ii, j, jj, vi, bi, bd, bri;

  vi = 0;     /* Value index. */
  bi = 0;     /* Block index. */
  bri = 0;    /* Index of first row of current block. */
  for (bi=0; bi<nblocks; bi++) {
    bd = blocks[bi]->dim;             /* Current block dimension. */
    for (i=0; i<bd; i++) {
      for (j=0; j<bd; j++) {
        ii = bri+i;                   /* Complete array row index. */
        jj = bri+j;                   /* Complete array col index. */
        /* Ensure we're matching column indices on the current row. */
        if (vi == crs_m->row_ptr[ii+1]) {
          blocks[bi]->a[i*bd+j] = 0;
        }
        else if (crs_m->col_in[vi] == jj) {
          blocks[bi]->a[i*bd+j] = crs_m->val[vi];
          vi++;
        }
        else {
          blocks[bi]->a[i*bd+j] = 0;
        }
      }
    }
    bri += bd;
  }
}

/* Diagonalize blocks of RCM ordered Hamiltonian. 
 *
 * Parameters
 * ----------
 *  job     If 'N', only eigenvalues are computed and z is not referenced.  If
 *          'V' then both eigenvalues and eigenvectors are computed.
 *  w       Pointer to double valued array of length n to which eigenvalues
 *          will be written upon exit.  
 *  zb      Pointer to array of length nblocks containing pointers to complex
 *          double valued arrays of length n*n with n the length of the
 *          respective block.
 *  nblocks The number of blocks.
 *  blocks  Array of zblock structures.
 *  diag_w  The diagonalization workspace.
 *  abstol  The absolute error tolerance to which each eigenvector is required.
 */
inline void zh_diag_blocks(char job, double *w, complex double **zb, int nblocks, 
    zblock **blocks, zheevd_w *diag_w, double abstol) {
  int i, bri, bi;
  int n, lda, ldz, il, iu, info;
  double vl, vu;
  char lapack_err[] = "LAPACKE_zhpeevr failed with error code: 0";

  bri = 0; /* Index of first row of current block. */
  if (job == 'V') {
    for (bi=0; bi<nblocks; bi++) {
      n = blocks[bi]->dim;
      lda = blocks[bi]->dim;
      ldz = blocks[bi]->dim;
      info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'V', 'A', 'U', n, blocks[bi]->a, lda,
          vl, vu, il, iu, abstol, &(diag_w->m), &w[bri], zb[bi], ldz, diag_w->isuppz,
          diag_w->work, diag_w->lwork, diag_w->rwork, diag_w->lrwork, diag_w->iwork,
          diag_w->liwork);
      bri += blocks[bi]->dim;
    }
  }
  else {
    for (bi=0; bi<nblocks; bi++) {
      n = blocks[bi]->dim;
      lda = blocks[bi]->dim;
      ldz = blocks[bi]->dim;
      info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'N', 'A', 'U', n, blocks[bi]->a, lda,
          vl, vu, il, iu, abstol, &(diag_w->m), &w[bri], NULL, ldz, diag_w->isuppz,
          diag_w->work, diag_w->lwork, diag_w->rwork, diag_w->lrwork, diag_w->iwork,
          diag_w->liwork);
    bri += blocks[bi]->dim;
    }
  }

  if (info != 0) {
    sprintf(lapack_err, "LAPACKE_zheevr failed with error code: %i", info);
    CFL_ERROR_VOID(lapack_err);
  }
}

/* 
 * Parse blocks of eigenvectors given a permutation to yield the full
 * dimensioned eigenvector array. 
 *
 * Parameters
 * ----------
 *  z       Pointer to complex double valued array of length n^2 to which the
 *          eigenvectors will be written.
 *  zb      Pointer to array of length nblocks containing pointers to complex
 *          double valued arrays containing the eigenvectors of each block.
 *  n       The dimension of the complete Hamiltonian. 
 *  nblocks The number of blocks. 
 *  blocks  Array of zblock structures.
 *  w_perm  Permutation required to sort eigenvalues.
 */
inline void zh_parse_ev(complex double *z, complex double **zb, int n, 
    int nblocks, zblock **blocks, int *w_perm) {
  int bi, bri, i, ii, j, jj;

  //FIXME: looping to return z to 0... either memset, or some index array that
  //records the nz elements and only touches them...
  for (i=0; i<n*n; i++) {
    z[i] = 0;
  }
  
  bri = 0;   /* Index of first row of current block. */
  for (bi=0; bi<nblocks; bi++) {
    for (i=0; i<blocks[bi]->dim; i++) {
      for (j=0; j<blocks[bi]->dim; j++) {
        ii = w_perm[bri+i];
        jj = w_perm[bri+j];
        z[ii*n+jj] = zb[bi][i*blocks[bi]->dim+j]; 
      }
    }
    bri += blocks[bi]->dim;
  }
}

/* Perform an inplace permutation of a double valued array dx, according to 
 * dx(perm(j)) :=  dx(j), j=1,2,.., n. */
void dvperm(int n, double *dx, int *perm) {
  int ii, j, k, init, next; 
  int tmp, tmp1;

  k=-1; 
  init=-1;

  while (k < n) {
    init++;

    /* Test for end and whether the current value has been permuted; that is,
     * whether the current perm value is negative. */
    if (init >= n)
      break;
    else if (perm[init] < 0)
      continue;
    tmp = dx[init];
    ii = perm[init];
    perm[init] -= n;

    for (;;) {
      k++;
      /* Save the chased element. */
      tmp1 = dx[ii];
      dx[ii] = tmp;
      next = perm[ii];
      /* Test for end. */
      if (next < 0)
        break;
      else if (k >= n)
        break;
      /* tmp1 value also requires permutation. */
      tmp = tmp1;
      perm[ii] -= n;
      ii = next;
    }
  }
  /* Restore positive valued permutation vector. */
  for (j=0; j<n; j++) {
    perm[j] += n;
  }
}


/*
 * Allocate workspace for the Hamiltonian diagonalization. 
 *
 * Parameters
 * ----------
 *  job     If 'N', only eigenvalues are computed and z is not referenced.  If
 *          'V' then both eigenvalues and eigenvectors are computed.
 *  w       Pointer to double valued array of length n to which eigenvalues
 *          will be written upon exit.  
 *  z       Pointer to complex double valued array of length n^2 to which the
 *          eigenvectors will be written.
 *  h       The Hamiltonian to be diagonalized.
 */
zhd_w *zhd_w_alloc(char job, double *w, complex double *z, zh *h) {
  int i, j, k;
  zcrs *zcrs_h;
  signed char *node_mask;
  int *node_deg, *tmp_perm;
  int nblocks, max_bdim;
  int block_dim[CFL_MAX_BLOCK_NUM];

  zhcrs **coeff_w;
  zhd_w *hd_w;

  hd_w = (zhd_w *) malloc(sizeof(zhd_w));
  if (hd_w == 0) {
    CFL_ERROR_NULL("malloc failed for hd_w");
  }

  /* Allocation for matrix element scaling and addition. */
  if (h->nt>1) {
    coeff_w = (zhcrs **) malloc((h->nt-1)*sizeof(zhcrs *));
    hd_w->lcoeff_w = h->nt-1;
  }
  else {
    coeff_w = (zhcrs **) malloc((h->nt)*sizeof(zhcrs *));
    hd_w->lcoeff_w = h->nt;
  }
  if (coeff_w == 0) {
    free(hd_w);
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
      free(hd_w);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
    zhcrssam((h->t[0])->matel, (h->t[1])->matel, coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i=1; i<h->nt-1; i++) {
      coeff_w[i] = zhcrssam_alloc(coeff_w[i-1], (h->t[i+1])->matel);
      if (coeff_w[i] == 0) {
        free(hd_w);
        free(coeff_w);
        for (j=0; j<i; j++) {
          zhcrs_free(coeff_w[j]);
        }
        CFL_ERROR_NULL("alloc failed for coeff_w");
      }
      zhcrssam(coeff_w[i-1], (h->t[i+1])->matel, coeff_w[i], 1, h->coeff[i+1]);
    }
  }
  else {
    coeff_w[0] = zhcrssm_alloc((h->t[0])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
  }

  hd_w->coeff_w = coeff_w;

  /* Generate the reverse Cuthill-McKee ordering of the complete Hamiltonian. */
  zcrs_h = zhcrs2zcrs_alloc(coeff_w[hd_w->lcoeff_w-1]);
  if (zcrs_h == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    free(hd_w->coeff_w);
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
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for tmp_perm");
  }
  zhcrs2zcrs(coeff_w[hd_w->lcoeff_w-1], zcrs_h);

  RCM_FUNC(genrcmi)(zcrs_h->n, RCM_NO_SORT, zcrs_h->row_ptr, zcrs_h->col_in, tmp_perm,
      node_mask, node_deg, &nblocks, block_dim);

  //FIXME: don't need tmp_perm if I change the permutation in col and row swap...?
  /* Change permutation index to coordinate form. */
  for (i=0; i<zcrs_h->n; i++) {
    hd_w->rcm_perm[tmp_perm[i]] = i;
  }

  hd_w->zcrs_h = zcrs_h;

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
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for rcm_pj");
  }

  hd_w->rcm_rp_h = (zcrs *) zcrs_row_perm_alloc(hd_w->zcrs_h, hd_w->rcm_perm);
  if (hd_w->rcm_rp_h == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("zcrs_row_perm_alloc failed for rcm_rp_h");
  }

  hd_w->rcm_cp_h = (zcrs *) zcrs_col_perm_alloc(hd_w->rcm_rp_h, hd_w->rcm_perm, hd_w->rcm_pj);
  if (hd_w->rcm_cp_h == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    zcrs_free(hd_w->rcm_rp_h);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("zcrs_col_perm_alloc failed for rcm_cp_h");
  }

  zcrs_row_perm(hd_w->zcrs_h, hd_w->rcm_rp_h, hd_w->rcm_perm);
  zcrs_col_perm(hd_w->rcm_rp_h, hd_w->rcm_cp_h, hd_w->rcm_perm, hd_w->rcm_pj);

  /* Alloc space for blocks. */
  i = 0;
  hd_w->nblocks = nblocks;
  hd_w->blocks = (zblock **) malloc(nblocks*sizeof(zblock *));
  if (hd_w->blocks == 0) {
    for (i=0; i<hd_w->lcoeff_w; i++) {
      zhcrs_free(hd_w->coeff_w[i]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    zcrs_free(hd_w->rcm_rp_h);
    zcrs_free(hd_w->rcm_cp_h);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("malloc failed for hd_w->blocks");
  }
  if (job == 'V') {
    hd_w->zb = (complex double **) malloc(nblocks*sizeof(complex double *));
    if (hd_w->zb == 0) {
      for (i=0; i<hd_w->lcoeff_w; i++) {
        zhcrs_free(hd_w->coeff_w[i]);
      }
      zcrs_free(zcrs_h);
      free(hd_w->rcm_perm);
      zcrs_free(hd_w->rcm_rp_h);
      zcrs_free(hd_w->rcm_cp_h);
      free(hd_w->rcm_pj);
      free(hd_w->coeff_w);
      free(hd_w);
      CFL_ERROR_NULL("malloc failed for hd_w->zb");
    }
  } 
  else {
    hd_w->zb = NULL;
  }

  k=0;
  for (i=0; i<nblocks; i++) {
    hd_w->blocks[i] = (zblock *) malloc(sizeof(zblock));
    if (hd_w->blocks[i] == 0) {
      for (j=0; j<i; j++) {
        free(hd_w->blocks[j]->a);
        free(hd_w->blocks[j]);
      }
      free(hd_w->blocks);
      for (j=0; j<hd_w->lcoeff_w; j++) {
        zhcrs_free(hd_w->coeff_w[j]);
      }
      zcrs_free(zcrs_h);
      free(hd_w->rcm_perm);
      zcrs_free(hd_w->rcm_rp_h);
      zcrs_free(hd_w->rcm_cp_h);
      free(hd_w->rcm_pj);
      free(hd_w->coeff_w);
      if (job == 'V') {
        free(hd_w->zb);
      }
      free(hd_w);
      CFL_ERROR_NULL("malloc failed for hd_w->blocks[i]");
    }

    k += block_dim[i];
    hd_w->blocks[i]->dim = block_dim[i];
    hd_w->blocks[i]->a = (complex double *) calloc(block_dim[i]*block_dim[i],
        sizeof(complex double));
    if (hd_w->blocks[i]->a == 0) {
      for (j=0; j<i; j++) {
        free(hd_w->blocks[j]->a);
        free(hd_w->blocks[j]);
      }
      free(hd_w->blocks[i]);
      free(hd_w->blocks);
      for (j=0; j<hd_w->lcoeff_w; j++) {
        zhcrs_free(hd_w->coeff_w[j]);
      }
      zcrs_free(zcrs_h);
      free(hd_w->rcm_perm);
      zcrs_free(hd_w->rcm_rp_h);
      zcrs_free(hd_w->rcm_cp_h);
      free(hd_w->rcm_pj);
      free(hd_w->coeff_w);
      if (job == 'V') {
        free(hd_w->zb);
      }
      free(hd_w);
      CFL_ERROR_NULL("malloc failed for hd_w->blocks[i]->a");
    }
    if (job == 'V') {
      hd_w->zb[i] = (complex double *) calloc(block_dim[i]*block_dim[i],
          sizeof(complex double));
      if (hd_w->zb[i] == 0) { 
        for (j=0; j<i; j++) {
          free(hd_w->blocks[j]->a);
          free(hd_w->blocks[j]);
          free(hd_w->zb[i]);
        }
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        free(hd_w->blocks);
        for (j=0; j<hd_w->lcoeff_w; j++) {
          zhcrs_free(hd_w->coeff_w[j]);
        }
        zcrs_free(zcrs_h);
        free(hd_w->rcm_perm);
        zcrs_free(hd_w->rcm_rp_h);
        zcrs_free(hd_w->rcm_cp_h);
        free(hd_w->rcm_pj);
        free(hd_w->coeff_w);
        free(hd_w->zb);
        free(hd_w);
        CFL_ERROR_NULL("calloc failed for hd_w->zb[i]");
      }
    }
  }

  /* Set the absolute error tolerance to which each eigenvalue/eigenvector is
   * required to the safe minimum machine precision. */
  hd_w->abstol = LAPACKE_dlamch('S');

  /* Allocate single workspace for all diagonalizations, with the size
   * determined by the maximum block size. */
  max_bdim = 0;
  for (i=0; i<nblocks; i++) {
    if (hd_w->blocks[i]->dim > max_bdim) {
      max_bdim = block_dim[i];
    }
  }
  hd_w->diag_w = (zheevd_w *) zheevd_w_alloc(job, max_bdim, hd_w->abstol); 
  if (hd_w->diag_w == 0) {
    for (j=0; j<nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j=0; j<hd_w->lcoeff_w; j++) {
      zhcrs_free(hd_w->coeff_w[j]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    zcrs_free(hd_w->rcm_rp_h);
    zcrs_free(hd_w->rcm_cp_h);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    if (job == 'V') {
      for (j=0; j<nblocks; j++) {
        free(hd_w->zb[j]);
      }
      free(hd_w->zb);
    }
    free(hd_w);
    CFL_ERROR_NULL("zheevd_w_alloc failed for hd_w->diag_w");
  }

  zh_parse_blocks(nblocks, hd_w->blocks, hd_w->rcm_cp_h);
  zh_diag_blocks(job, w, hd_w->zb, nblocks, hd_w->blocks, hd_w->diag_w, hd_w->abstol);
  
  /* Determine the permutation required to sort eigenvalues from smallest to
   * largest. */
  double **wptr = (double **) malloc(zcrs_h->n*sizeof(double *));
  if (wptr == 0) {
    for (j=0; j<nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j=0; j<hd_w->lcoeff_w; j++) {
      zhcrs_free(hd_w->coeff_w[j]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    zcrs_free(hd_w->rcm_rp_h);
    zcrs_free(hd_w->rcm_cp_h);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    if (job == 'V') {
      for (j=0; j<nblocks; j++) {
        free(hd_w->zb[j]);
      }
      free(hd_w->zb);
    }
    zheevd_w_free(hd_w->diag_w);
    free(hd_w);
    CFL_ERROR_NULL("zheevd_w_alloc failed for hd_w->diag_w");
  }

  for (i=0; i<zcrs_h->n; i++) {
    wptr[i] = &w[i];
  }
  qsort(wptr, zcrs_h->n, sizeof(double *), dptr_cmp);

  hd_w->w_perm = (int *) calloc(zcrs_h->n, sizeof(int));
  if (hd_w->w_perm == 0) {
    for (j=0; j<nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j=0; j<hd_w->lcoeff_w; j++) {
      zhcrs_free(hd_w->coeff_w[j]);
    }
    zcrs_free(zcrs_h);
    free(hd_w->rcm_perm);
    zcrs_free(hd_w->rcm_rp_h);
    zcrs_free(hd_w->rcm_cp_h);
    free(hd_w->rcm_pj);
    free(hd_w->coeff_w);
    if (job == 'V') {
      for (j=0; j<nblocks; j++) {
        free(hd_w->zb[j]);
      }
      free(hd_w->zb);
    }
    zheevd_w_free(hd_w->diag_w);
    free(hd_w);
    free(wptr);
    CFL_ERROR_NULL("calloc failed for hd_w->w_perm");
  }
  for (i=0; i<zcrs_h->n; i++) {
    hd_w->w_perm[wptr[i] - w] = i;
  }
  free(wptr);
  
  /* Permute the eigenvalue vector. */ 
  dvperm(zcrs_h->n, w, hd_w->w_perm);
  
  /* Permute and parse eigenvectors, if requested. */
  if (job == 'V') {
    zh_parse_ev(z, hd_w->zb, zcrs_h->n, nblocks, hd_w->blocks, hd_w->w_perm);
  }

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
  zheevd_w_free(hd_w->diag_w);
  for (i=0; i<hd_w->nblocks; i++) {
    free(hd_w->blocks[i]->a);
    free(hd_w->blocks[i]);
  }
  free(hd_w->blocks);
  if (hd_w->zb != NULL) {
    for (i=0; i<hd_w->nblocks; i++) {
      free(hd_w->zb[i]);
    }
    free(hd_w->zb);
  }
  free(hd_w->w_perm);
  free(hd_w);
}


/*
 * Calculate the eigenvalues and corresponding eigenvectors of a Hamiltonian. 
 * 
 * Parameters
 * ----------
 *  job     If 'N', only eigenvalues are computed and z is not referenced.  If
 *          'V' then both eigenvalues and eigenvectors are computed.
 *  w       Pointer to double valued array of length n to which eigenvalues
 *          will be written.  
 *  z       Pointer to complex double valued array of length n^2 to which the
 *          eigenvectors will be written.
 *  h       The Hamiltonian. 
 *  hd_w    The work space for diagonalization; allocated using zhd_w_alloc.
 */
void zhd(char job, double *w, complex double *z, zh *h, zhd_w *hd_w) {
  int i;

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

  zcrs_row_perm(hd_w->zcrs_h, hd_w->rcm_rp_h, hd_w->rcm_perm);
  zcrs_col_perm(hd_w->rcm_rp_h, hd_w->rcm_cp_h, hd_w->rcm_perm, hd_w->rcm_pj);

  zh_parse_blocks(hd_w->nblocks, hd_w->blocks, hd_w->rcm_cp_h);
  zh_diag_blocks(job, w, hd_w->zb, hd_w->nblocks, hd_w->blocks, hd_w->diag_w,
      hd_w->abstol);

  /* Permute the eigenvalue vector. */ 
  dvperm(h->n, w, hd_w->w_perm);

  /* Permute and parse eigenvectors, if requested. */
  if (job == 'V') {
    zh_parse_ev(z, hd_w->zb, h->n, hd_w->nblocks, hd_w->blocks, hd_w->w_perm);
  }
}
