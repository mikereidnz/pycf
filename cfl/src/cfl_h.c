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

#include "cfl_error.h"
#include "cfl_tensor.h"
#include "cfl_h.h"


/* Integer pointer comparison function for qsort. */
int iptr_cmp(const void *a, const void *b) {
  const int **ia = (const int **) a;
  const int **ib = (const int **) b;

  return (**ia > **ib) - (**ia < **ib);
}

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
  for (i = 1; i < nt; i++) {
    if (t[0]->slabels->hash != t[i]->slabels->hash) {
      CFL_ERROR_NULL("Tensors have mismatching state labels")
    }
  }
  h->slabels = t[0]->slabels;

  h->n = n;
  h->nt = nt;
  h->t = t;
  h->coeff = NULL;

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
 *  job     If 'N', only eigenvalues are computed and z is not referenced.  If
 *          'V' then both eigenvalues and eigenvectors are computed.
 *  n       The dimension of matrix to be diagonalized; assumed to be symmetric.
 *  abstol  The absolute error tolerance to which each eigenvector is required.
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

  info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', n, NULL, lda, vl,
      vu, il, iu, abstol, &(heevd_w->m), NULL, NULL, ldz, heevd_w->isuppz,
      &wquery, lwork, &rwquery, lrwork, &iwquery, liwork);
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

/* 
 * Read block-diag ordered CSR matrix into pre-allocated blocks and diagonalize
 * them. 
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
 *  csr_m       The block diagonalized CSR matrix. 
 *  diag_w  The diagonalization workspace.
 *  abstol  The absolute error tolerance to which each eigenvector is required.
 */
inline void zh_diag_blocks(char job, double *w, complex double **zb, int nblocks, 
    zblock **blocks, zcsr *csr_m, zheevd_w *diag_w, double abstol) {
  int i, ii, j, jj, vi, bi, bd, bri;
  int lda, ldz, il, iu, info;
  double vl, vu;
  char lapack_err[] = "LAPACKE_zhpeevr failed with error code: 0";

  info = 0;   /* LAPACK return value. */
  vi = 0;     /* Value index. */
  bi = 0;     /* Block index. */
  bri = 0;    /* Index of first row of current block. */
  for (bi = 0; bi < nblocks; bi++) {
    bd = blocks[bi]->dim;             /* Current block dimension. */
    for (i = 0; i < bd; i++) {
      for (j = 0; j < bd; j++) {
        ii = bri+i;                   /* Complete array row index. */
        jj = bri+j;                   /* Complete array col index. */
        /* Ensure we're matching column indices on the current row. */
        if (vi == csr_m->row_ptr[ii+1]) {
          blocks[bi]->a[i*bd+j] = 0;
        }
        else if (csr_m->col_in[vi] == jj) {
          blocks[bi]->a[i*bd+j] = csr_m->val[vi];
          vi++;
        }
        else {
          blocks[bi]->a[i*bd+j] = 0;
        }
      }
    }

    lda = bd;
    ldz = bd;
    if (job == 'V') {
      info += LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'V', 'A', 'U', bd,
          blocks[bi]->a, lda, vl, vu, il, iu, abstol, &(diag_w->m), &w[bri],
          zb[bi], ldz, diag_w->isuppz, diag_w->work, diag_w->lwork,
          diag_w->rwork, diag_w->lrwork, diag_w->iwork, diag_w->liwork);
    }
    else {
      info += LAPACKE_zheevr_work(LAPACK_COL_MAJOR, 'N', 'A', 'U', bd,
          blocks[bi]->a, lda, vl, vu, il, iu, abstol, &(diag_w->m), &w[bri],
          NULL, ldz, diag_w->isuppz, diag_w->work, diag_w->lwork,
          diag_w->rwork, diag_w->lrwork, diag_w->iwork, diag_w->liwork);
    }
    bri += bd;
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
 *  z             Pointer to complex double valued array of length n^2 to which
 *                the eigenvectors will be written.
 *  zb            Pointer to array of length nblocks containing pointers to
 *                complex double valued arrays containing the eigenvectors of
 *                each block.
 *  n             The dimension of the complete Hamiltonian. 
 *  nblocks       The number of blocks. 
 *  blocks        Array of zblock structures.
 *  crd_blk_perm  Block diagonalization permutation in coordinate form.  
 *  w_perm        Permutation required to sort eigenvalues.
 */
inline void zh_parse_ev(complex double *z, complex double **zb, int n, 
    int nblocks, zblock **blocks, int *crd_blk_perm, int *w_perm) {
  int bi, bri, i, ii, j, jj;

  bri = 0;   /* Index of first row of current block. */
  for (bi = 0; bi < nblocks; bi++) {
    for (i = 0; i < blocks[bi]->dim; i++) {
      for (j = 0; j < blocks[bi]->dim; j++) {
        ii = w_perm[bri+i];
        jj = crd_blk_perm[bri+j];
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
  for (j = 0; j < n; j++) {
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
 *  h       The Hamiltonian to be diagonalized.
 */
zhd_w *zhd_w_alloc(char job, zh *h) {
  int i, j, k;
  zcsr *zcsr_h;
  int **lptr;
  int *labels;
  int nblocks, max_bdim;
  int *block_dim;

  zhcsr **coeff_w;
  zhd_w *hd_w;

  hd_w = (zhd_w *) malloc(sizeof(zhd_w));
  if (hd_w == 0) {
    CFL_ERROR_NULL("malloc failed for hd_w");
  }

  /* Allocation for matrix element scaling and addition. */
  if (h->nt>1) {
    coeff_w = (zhcsr **) malloc((h->nt-1)*sizeof(zhcsr *));
    hd_w->lcoeff_w = h->nt-1;
  }
  else {
    coeff_w = (zhcsr **) malloc((h->nt)*sizeof(zhcsr *));
    hd_w->lcoeff_w = h->nt;
  }
  if (coeff_w == 0) {
    free(hd_w);
    CFL_ERROR_NULL("malloc failed for coeff_w");
  }

  /* Allocation for summing matrix elements of tensors.  The zhsam function
   * calculates C for C = alpha A + beta C, for A, B, and C CSR matrices and
   * alpha and beta complex scalars.  The first two matrix elements are summed
   * directly with respective coefficients set for alpha and beta.  Further
   * matrix elements are then iteratively added to the previous result.  Since
   * zhcsrsam_alloc also calculates the row_ptr array and number of non-zero
   * elements of C, we have to run through the actual additions in order to
   * determine these values for each of the intermediate sums.  Finally, in case
   * there is only a single tensor, we use the scaling function zhcsrsm for
   * which we still have to allocate separate memory. 
   */
  if (h->nt>1) {
    coeff_w[0] = zhcsrsam_alloc((h->t[0])->matel, (h->t[1])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
    zhcsrsam((h->t[0])->matel, (h->t[1])->matel, coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i = 1; i < h->nt-1; i++) {
      coeff_w[i] = zhcsrsam_alloc(coeff_w[i-1], (h->t[i+1])->matel);
      if (coeff_w[i] == 0) {
        free(hd_w);
        free(coeff_w);
        for (j = 0; j < i; j++) {
          zhcsr_free(coeff_w[j]);
        }
        CFL_ERROR_NULL("alloc failed for coeff_w");
      }
      zhcsrsam(coeff_w[i-1], (h->t[i+1])->matel, coeff_w[i], 1, h->coeff[i+1]);
    }
  }
  else {
    coeff_w[0] = zhcsrsm_alloc((h->t[0])->matel);
    if (coeff_w[0] == 0) {
      free(hd_w);
      free(coeff_w);
      CFL_ERROR_NULL("alloc failed for coeff_w");
    }
  }

  hd_w->coeff_w = coeff_w;

  /* Find the connected components of the Hamiltonian. */
  zcsr_h = zhcsr2zcsr_alloc(coeff_w[hd_w->lcoeff_w-1]);
  if (zcsr_h == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("alloc failed for zcsr_h");
  }
  hd_w->zcsr_h = zcsr_h;

  hd_w->blk_perm = (int *) calloc(zcsr_h->n, sizeof(int));
  if (hd_w->blk_perm == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for blk_perm");
  }
  labels = (int *) malloc(zcsr_h->n*sizeof(int));
  if (labels == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for labels");
  }

  zhcsr2zcsr(coeff_w[hd_w->lcoeff_w-1], zcsr_h);
  nblocks = zcsr_cc(zcsr_h, labels);

  block_dim = (int *) calloc(nblocks, sizeof(int));
  if (block_dim == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    free(hd_w->coeff_w);
    free(hd_w);
    free(labels);
    CFL_ERROR_NULL("calloc failed for block_dim");
  }

  lptr = (int **) malloc(zcsr_h->n*sizeof(double *));
  if (lptr == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    free(hd_w->coeff_w);
    free(hd_w);
    free(labels);
    free(block_dim);
    CFL_ERROR_NULL("malloc failed for lptr");
  }

  for (i = 0; i < zcsr_h->n; i++) {
    block_dim[labels[i]] += 1;
    lptr[i] = &labels[i];
  }
  qsort(lptr, zcsr_h->n, sizeof(int *), iptr_cmp);

  for (i = 0; i < zcsr_h->n; i++) {
    hd_w->blk_perm[lptr[i] - labels] = i;
  }

  free(lptr);
  free(labels);

  hd_w->blk_pj = (int *) calloc(hd_w->zcsr_h->nnz+1, sizeof(int));
  if (hd_w->blk_pj == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    free(hd_w->coeff_w);
    free(hd_w);
    free(block_dim);
    CFL_ERROR_NULL("calloc failed for blk_pj");
  }
  hd_w->blk_rp_h = (zcsr *) zcsr_row_perm_alloc(hd_w->zcsr_h, hd_w->blk_perm);
  if (hd_w->blk_rp_h == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    free(hd_w->blk_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    free(block_dim);
    CFL_ERROR_NULL("zcsr_row_perm_alloc failed for blk_rp_h");
  }
  hd_w->blk_cp_h = (zcsr *) zcsr_col_perm_alloc(hd_w->blk_rp_h,
      hd_w->blk_perm, hd_w->blk_pj);
  if (hd_w->blk_cp_h == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    zcsr_free(hd_w->blk_rp_h);
    free(hd_w->blk_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    free(block_dim);
    CFL_ERROR_NULL("zcsr_col_perm_alloc failed for blk_cp_h");
  }

  zcsr_row_perm(hd_w->zcsr_h, hd_w->blk_rp_h, hd_w->blk_perm);
  zcsr_col_perm(hd_w->blk_rp_h, hd_w->blk_cp_h, hd_w->blk_perm, hd_w->blk_pj);

  /* Alloc space for blocks. */
  i = 0;
  hd_w->nblocks = nblocks;
  hd_w->blocks = (zblock **) malloc(nblocks*sizeof(zblock *));
  if (hd_w->blocks == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    zcsr_free(hd_w->blk_rp_h);
    zcsr_free(hd_w->blk_cp_h);
    free(hd_w->blk_pj);
    free(hd_w->coeff_w);
    free(hd_w);
    free(block_dim);
    CFL_ERROR_NULL("malloc failed for hd_w->blocks");
  }
  if (job == 'V') {
    hd_w->zb = (complex double **) malloc(nblocks*sizeof(complex double *));
    if (hd_w->zb == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->blk_perm);
      zcsr_free(hd_w->blk_rp_h);
      zcsr_free(hd_w->blk_cp_h);
      free(hd_w->blk_pj);
      free(hd_w->coeff_w);
      free(hd_w);
      free(block_dim);
      CFL_ERROR_NULL("malloc failed for hd_w->zb");
    }
    hd_w->crd_blk_perm = (int *) malloc(zcsr_h->n*sizeof(int));
    if (hd_w->crd_blk_perm == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->blk_perm);
      zcsr_free(hd_w->blk_rp_h);
      zcsr_free(hd_w->blk_cp_h);
      free(hd_w->blk_pj);
      free(hd_w->coeff_w);
      free(hd_w->zb);
      free(hd_w);
      free(block_dim);
      CFL_ERROR_NULL("malloc failed for hd_w->crd_blk_perm");
    }
    /* Find the block permutation in coordinate form. */
    for (i = 0; i < zcsr_h->n; i++) {
      hd_w->crd_blk_perm[hd_w->blk_perm[i]] = i;
    }
  } 
  else {
    hd_w->zb = NULL;
    hd_w->crd_blk_perm = NULL;
  }

  k=0;
  for (i = 0; i < nblocks; i++) {
    hd_w->blocks[i] = (zblock *) malloc(sizeof(zblock));
    if (hd_w->blocks[i] == 0) {
      for (j = 0; j < i; j++) {
        free(hd_w->blocks[j]->a);
        free(hd_w->blocks[j]);
      }
      free(hd_w->blocks);
      for (j = 0; j < hd_w->lcoeff_w; j++) {
        zhcsr_free(hd_w->coeff_w[j]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->blk_perm);
      zcsr_free(hd_w->blk_rp_h);
      zcsr_free(hd_w->blk_cp_h);
      free(hd_w->blk_pj);
      free(hd_w->coeff_w);
      if (job == 'V') {
        free(hd_w->zb);
        free(hd_w->crd_blk_perm);
      }
      free(hd_w);
      free(block_dim);
      CFL_ERROR_NULL("malloc failed for hd_w->blocks[i]");
    }

    k += block_dim[i];
    hd_w->blocks[i]->dim = block_dim[i];
    hd_w->blocks[i]->a = (complex double *) calloc(block_dim[i]*block_dim[i],
        sizeof(complex double));
    if (hd_w->blocks[i]->a == 0) {
      for (j = 0; j < i; j++) {
        free(hd_w->blocks[j]->a);
        free(hd_w->blocks[j]);
      }
      free(hd_w->blocks[i]);
      free(hd_w->blocks);
      for (j = 0; j < hd_w->lcoeff_w; j++) {
        zhcsr_free(hd_w->coeff_w[j]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->blk_perm);
      zcsr_free(hd_w->blk_rp_h);
      zcsr_free(hd_w->blk_cp_h);
      free(hd_w->blk_pj);
      free(hd_w->coeff_w);
      if (job == 'V') {
        free(hd_w->zb);
        free(hd_w->crd_blk_perm);
      }
      free(hd_w);
      free(block_dim);
      CFL_ERROR_NULL("malloc failed for hd_w->blocks[i]->a");
    }
    if (job == 'V') {
      hd_w->zb[i] = (complex double *) calloc(block_dim[i]*block_dim[i],
          sizeof(complex double));
      if (hd_w->zb[i] == 0) { 
        for (j = 0; j < i; j++) {
          free(hd_w->blocks[j]->a);
          free(hd_w->blocks[j]);
          free(hd_w->zb[i]);
        }
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        free(hd_w->blocks);
        for (j = 0; j < hd_w->lcoeff_w; j++) {
          zhcsr_free(hd_w->coeff_w[j]);
        }
        zcsr_free(zcsr_h);
        free(hd_w->blk_perm);
        zcsr_free(hd_w->blk_rp_h);
        zcsr_free(hd_w->blk_cp_h);
        free(hd_w->blk_pj);
        free(hd_w->coeff_w);
        free(hd_w->zb);
        free(hd_w->crd_blk_perm);
        free(hd_w);
        free(block_dim);
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
  for (i = 0; i < nblocks; i++) {
    if (hd_w->blocks[i]->dim > max_bdim) {
      max_bdim = block_dim[i];
    }
  }
  free(block_dim);

  hd_w->diag_w = (zheevd_w *) zheevd_w_alloc(job, max_bdim, hd_w->abstol); 
  if (hd_w->diag_w == 0) {
    for (j = 0; j < nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j = 0; j < hd_w->lcoeff_w; j++) {
      zhcsr_free(hd_w->coeff_w[j]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    zcsr_free(hd_w->blk_rp_h);
    zcsr_free(hd_w->blk_cp_h);
    free(hd_w->blk_pj);
    free(hd_w->coeff_w);
    if (job == 'V') {
      for (j = 0; j < nblocks; j++) {
        free(hd_w->zb[j]);
      }
      free(hd_w->zb);
      free(hd_w->crd_blk_perm);
    }
    free(hd_w);
    CFL_ERROR_NULL("zheevd_w_alloc failed for hd_w->diag_w");
  }

  hd_w->w_perm = (int *) calloc(zcsr_h->n, sizeof(int));
  if (hd_w->w_perm == 0) {
    for (j = 0; j < nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j = 0; j < hd_w->lcoeff_w; j++) {
      zhcsr_free(hd_w->coeff_w[j]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->blk_perm);
    zcsr_free(hd_w->blk_rp_h);
    zcsr_free(hd_w->blk_cp_h);
    free(hd_w->blk_pj);
    free(hd_w->coeff_w);
    if (job == 'V') {
      for (j = 0; j < nblocks; j++) {
        free(hd_w->zb[j]);
      }
      free(hd_w->zb);
      free(hd_w->crd_blk_perm);
    }
    zheevd_w_free(hd_w->diag_w);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for hd_w->w_perm");
  }
  /* We set the first element to -1, to allow zhd to check whether a previous
   * evaluation has found the w_perm array. */
  hd_w->w_perm[0] = -1;

  return hd_w;
}

void zhd_w_free(zhd_w *hd_w) {
  int i;

  for (i = 0; i < hd_w->lcoeff_w; i++) {
    zhcsr_free(hd_w->coeff_w[i]);
  }
  zcsr_free(hd_w->zcsr_h);
  free(hd_w->blk_perm);
  zcsr_free(hd_w->blk_rp_h);
  free(hd_w->blk_pj);
  zcsr_free(hd_w->blk_cp_h);
  free(hd_w->coeff_w);
  zheevd_w_free(hd_w->diag_w);
  for (i = 0; i < hd_w->nblocks; i++) {
    free(hd_w->blocks[i]->a);
    free(hd_w->blocks[i]);
  }
  free(hd_w->blocks);
  if (hd_w->zb != NULL) {
    for (i = 0; i < hd_w->nblocks; i++) {
      free(hd_w->zb[i]);
    }
    free(hd_w->zb);
    free(hd_w->crd_blk_perm);
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
  double **wptr;

  /* Multiply the tensor matrix elements by coefficients and sum them.  The
   * result is stored in hd_w->coeff_w[i], where i is the number of tensors -1.
   */
  if (h->nt>1) {
    zhcsrsam((h->t[0])->matel, (h->t[1])->matel, hd_w->coeff_w[0], h->coeff[0],
        h->coeff[1]);
    for (i = 1; i < hd_w->lcoeff_w; i++) {
      zhcsrsam(hd_w->coeff_w[i-1], (h->t[i+1])->matel, hd_w->coeff_w[i], 1,
          h->coeff[i+1]);
    }
  }
  else
    zhcsrsm((h->t[0])->matel, hd_w->coeff_w[0], h->coeff[0]);

  /* Convert the Hamiltonian from Hermitian CSR to standard CSR, then apply
   * block-diag permutation, and finally convert to dense storage. */
  zhcsr2zcsr(hd_w->coeff_w[hd_w->lcoeff_w-1], hd_w->zcsr_h);

  zcsr_row_perm(hd_w->zcsr_h, hd_w->blk_rp_h, hd_w->blk_perm);
  zcsr_col_perm(hd_w->blk_rp_h, hd_w->blk_cp_h, hd_w->blk_perm, hd_w->blk_pj);

  zh_diag_blocks(job, w, hd_w->zb, hd_w->nblocks, hd_w->blocks, hd_w->blk_cp_h,
      hd_w->diag_w, hd_w->abstol);

  /* Check whether we have to determine the permutation required to sort
   * eigenvalues from smallest to largest. */
  if (hd_w->w_perm[0] == -1) {
    wptr = (double **) malloc(h->n*sizeof(double *));
    if (wptr == 0) {
      CFL_ERROR_VOID("malloc failed for wptr");
    }

    for (i = 0; i < h->n; i++) {
      wptr[i] = &w[i];
    }
    qsort(wptr, h->n, sizeof(double *), dptr_cmp);

    for (i = 0; i < h->n; i++) {
      hd_w->w_perm[wptr[i] - w] = i;
    }
    free(wptr);
    
    if (job == 'V') {
      /* zh_parse_ev only ever touches the same elements; so we set z to zero
       * during the first zhd call. */
      for (i = 0; i < h->n*h->n; i++) {
        z[i] = 0;
      }
    }

  }

  /* Permute the eigenvalue vector. */ 
  dvperm(h->n, w, hd_w->w_perm);

  /* Permute and parse eigenvectors, if requested. */
  if (job == 'V') {
    zh_parse_ev(z, hd_w->zb, h->n, hd_w->nblocks, hd_w->blocks,
        hd_w->crd_blk_perm, hd_w->w_perm);
  }
}
