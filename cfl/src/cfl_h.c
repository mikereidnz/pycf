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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

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

#ifdef _OPENMP
  /* The number of cores available for diagonalization of this Hamiltonian.  If
   * not modified by the caller, zhd_w_alloc will default to all cores available
   * to the program. */
  h->num_procs = -1;
#endif /* _OPENMP */

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

/* Allocate workspace for a LAPACKE_zheevr diagonalization. 
 *
 * Parameters
 * ----------
 *  job     If 'N', only eigenvalues are computed and z is not referenced.  If
 *          'V' then both eigenvalues and eigenvectors are computed.
 *  n       The dimension of matrix to be diagonalized; assumed to be symmetric.
 *  abstol  The absolute error tolerance to which each eigenvector is required.
 */
zheevr_w *zheevr_w_alloc(char job, int n, double abstol) {
  int lda = n, ldz = n, il, iu, info;
  double vl, vu;
  int lwork, lrwork, liwork;
  complex double *work, wquery;
  double *rwork, rwquery;
  int *iwork, iwquery;
  zheevr_w *heevr_w;

  heevr_w = (zheevr_w *) malloc(sizeof(zheevr_w));
  if (heevr_w ==0) {
    CFL_ERROR_NULL("malloc failed for heevr_w");
  }

  heevr_w->isuppz = (int *) malloc(2*n*sizeof(int));
  if (heevr_w->isuppz == 0) {
    free(heevr_w);
    CFL_ERROR_NULL("malloc failed for isuppz");
  }
  /* zheevr workspace query. */
  lwork = -1;
  lrwork = -1;
  liwork = -1;

  info = LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', n, NULL, lda, vl,
      vu, il, iu, abstol, &(heevr_w->m), NULL, NULL, ldz, heevr_w->isuppz,
      &wquery, lwork, &rwquery, lrwork, &iwquery, liwork);

  if (info != 0) {
    free(heevr_w->isuppz);
    free(heevr_w);
    CFL_ERROR_NULL("LAPACKE workspace query failed");
  }

  lwork = (int) creal(wquery);
  lrwork = (int) rwquery;
  liwork = iwquery;

  work = (complex double *) calloc(lwork,sizeof(complex double));
  if (work == 0) {
    free(heevr_w->isuppz);
    free(heevr_w);
    CFL_ERROR_NULL("calloc failed for work");
  }
  rwork = (double *) calloc(lrwork,sizeof(double));
  if (rwork == 0) {
    free(heevr_w->isuppz);
    free(heevr_w);
    free(work);
    CFL_ERROR_NULL("calloc failed for rwork");
  }
  iwork = (int *) calloc(liwork,sizeof(int));
  if (iwork == 0) {
    free(heevr_w->isuppz);
    free(heevr_w);
    free(work);
    free(rwork);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  heevr_w->work = work;
  heevr_w->lwork = lwork;
  heevr_w->rwork = rwork;
  heevr_w->lrwork = lrwork;
  heevr_w->iwork = iwork;
  heevr_w->liwork = liwork;

  return heevr_w;
}

void zheevr_w_free(zheevr_w *heevr_w) {
  free(heevr_w->isuppz);
  free(heevr_w->work);
  free(heevr_w->rwork);
  free(heevr_w->iwork);
  free(heevr_w);
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
 *  csr_m   The block diagonalized CSR matrix. 
 *  hd_w    Hamiltonian diagonalization workspace. 
 */
inline void zh_diag_blocks(char job, double *w, zcsr *csr_m, zhd_w *hd_w) {
  int i, ii, j, jj, vi, bi, bd, tn;
  int lda, ldz, il, iu, info, *bri;
  double vl, vu;
  zheevr_w **diag_w;

  bri = hd_w->bri;        /* Index of first row of current block. */
  vi = 0;                 /* Value index. */
  for (bi = 0; bi < hd_w->nblocks; bi++) {
    bd = hd_w->blocks[bi]->dim;           /* Current block dimension. */
    for (i = 0; i < bd; i++) {
      for (j = 0; j < bd; j++) {
        ii = bri[bi]+i;                   /* Complete array row index. */
        jj = bri[bi]+j;                   /* Complete array col index. */
        /* Ensure we're matching column indices on the current row. */
        if (vi == csr_m->row_ptr[ii+1]) {
          hd_w->blocks[bi]->a[i*bd+j] = 0;
        }
        else if (csr_m->col_in[vi] == jj) {
          hd_w->blocks[bi]->a[i*bd+j] = csr_m->val[vi];
          vi++;
        }
        else {
          hd_w->blocks[bi]->a[i*bd+j] = 0;
        }
      }
    }
  }

  char lapack_err[] = "LAPACKE_zhpeevr failed with error code: 0";
  info = 0;               /* LAPACK return value. */
  diag_w = hd_w->diag_w;  /* Diagonalization workspace array. */

#ifdef _OPENMP
  if (hd_w->proc_limited) {
    /* Each core has a dedicated workspace, enumerated by the thread number. */
#pragma omp parallel for private(bi) schedule(dynamic)
    for (bi = 0; bi < hd_w->nblocks; bi++) {
      tn = omp_get_thread_num();
      bd = hd_w->blocks[bi]->dim;            
      lda = bd;
      ldz = bd;
      info += LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', bd,
          hd_w->blocks[bi]->a, lda, vl, vu, il, iu, hd_w->abstol, &(diag_w[tn]->m),
          &w[bri[bi]], hd_w->zb[bi], ldz, diag_w[tn]->isuppz, diag_w[tn]->work,
          diag_w[tn]->lwork, diag_w[tn]->rwork, diag_w[tn]->lrwork,
          diag_w[tn]->iwork, diag_w[tn]->liwork);
    }
  }
  else {
    /* Each block has a dedicated workspace, enumerated by bi. */
#pragma omp parallel for private(bi) schedule(dynamic)
    for (bi = 0; bi < hd_w->nblocks; bi++) {
      bd = hd_w->blocks[bi]->dim;            
      lda = bd;
      ldz = bd;
      info += LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', bd,
          hd_w->blocks[bi]->a, lda, vl, vu, il, iu, hd_w->abstol, &(diag_w[bi]->m),
          &w[bri[bi]], hd_w->zb[bi], ldz, diag_w[bi]->isuppz, diag_w[bi]->work,
          diag_w[bi]->lwork, diag_w[bi]->rwork, diag_w[bi]->lrwork,
          diag_w[bi]->iwork, diag_w[bi]->liwork);
    }
  }
#else
  /* OpenMP disabled; each block has a dedicated workspace, enumerated by bi. */
  for (bi = 0; bi < hd_w->nblocks; bi++) {
    bd = hd_w->blocks[bi]->dim;            
    lda = bd;
    ldz = bd;
    info += LAPACKE_zheevr_work(LAPACK_COL_MAJOR, job, 'A', 'U', bd,
        hd_w->blocks[bi]->a, lda, vl, vu, il, iu, hd_w->abstol, &(diag_w[0]->m),
        &w[bri[bi]], hd_w->zb[bi], ldz, diag_w[0]->isuppz, diag_w[0]->work,
        diag_w[0]->lwork, diag_w[0]->rwork, diag_w[0]->lrwork,
        diag_w[0]->iwork, diag_w[0]->liwork);
  }
#endif /* _OPENMP */

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
 *  n             The dimension of the complete Hamiltonian. 
 *  hd_w          Hamiltonian diagonalization workspace. 
 */
inline void zh_parse_ev(complex double *z, int n, zhd_w *hd_w) {
  int bi, i, ii, j, jj, *bri; 
  zblock **blocks;

  bri = hd_w->bri;        /* Index of first row of current block. */
  blocks = hd_w->blocks;  /* Array of blocks. */
  if (hd_w->nblocks != 1) {
#pragma omp parallel for private(bi) schedule(dynamic)
    for (bi = 0; bi < hd_w->nblocks; bi++) {
      for (i = 0; i < blocks[bi]->dim; i++) {
        for (j = 0; j < blocks[bi]->dim; j++) {
          ii = hd_w->w_perm[bri[bi]+i];
          jj = hd_w->crd_blk_perm[bri[bi]+j];
          z[ii*n+jj] = hd_w->zb[bi][i*blocks[bi]->dim+j]; 
        }
      }
    }
  }
  else {
    bi = 0;
    for (i = 0; i < blocks[bi]->dim; i++) {
      for (j = 0; j < blocks[bi]->dim; j++) {
        ii = hd_w->w_perm[bri[bi]+i];
        jj = j;
        z[ii*n+jj] = hd_w->zb[bi][i*blocks[bi]->dim+j]; 
      }
    }
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
  int i, j, k, nblocks, max_block_dim;
  zcsr *zcsr_h;
  int **lptr, *labels, *block_dim;

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

  labels = (int *) malloc(zcsr_h->n*sizeof(int));
  if (labels == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w);
    CFL_ERROR_NULL("calloc failed for labels");
  }

  zhcsr2zcsr(coeff_w[hd_w->lcoeff_w-1], zcsr_h);
  nblocks = zcsr_cc(zcsr_h, labels);
  hd_w->nblocks = nblocks;

  /* w_perm is the permutation that reorders eigenvalues to be monotonically
   * increasing. */
  hd_w->w_perm = (int *) calloc(zcsr_h->n, sizeof(int));
  if (hd_w->w_perm == 0) {
    for (j = 0; j < hd_w->lcoeff_w; j++) {
      zhcsr_free(hd_w->coeff_w[j]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w);
    free(labels);
    CFL_ERROR_NULL("calloc failed for hd_w->w_perm");
  }
  /* We set the first element to -1, to allow zhd to check whether a previous
   * evaluation has found the w_perm array. */
  hd_w->w_perm[0] = -1;

  /* Set the absolute error tolerance to which each eigenvalue/eigenvector is
   * required to the safe minimum machine precision. */
  hd_w->abstol = LAPACKE_dlamch('S');

  /* Alloc space for blocks. */
  block_dim = (int *) calloc(nblocks, sizeof(int));
  if (block_dim == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w);
    free(labels);
    CFL_ERROR_NULL("calloc failed for block_dim");
  }
  for (i = 0; i < zcsr_h->n; i++) {
    block_dim[labels[i]] += 1;
  }

  hd_w->blocks = (zblock **) malloc(nblocks*sizeof(zblock *));
  if (hd_w->blocks == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w->w_perm);
    free(hd_w);
    free(labels);
    free(block_dim);
    CFL_ERROR_NULL("malloc failed for hd_w->blocks");
  }
  hd_w->zb = (complex double **) malloc(nblocks*sizeof(complex double *));
  if (hd_w->zb == 0) {
    for (i = 0; i < hd_w->lcoeff_w; i++) {
      zhcsr_free(hd_w->coeff_w[i]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w->w_perm);
    free(hd_w);
    free(labels);
    free(block_dim);
    CFL_ERROR_NULL("malloc failed for hd_w->zb");
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
      free(hd_w->coeff_w);
      free(hd_w->w_perm);
      free(hd_w->zb);
      free(hd_w);
      free(labels);
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
      free(hd_w->coeff_w);
      free(hd_w->w_perm);
      free(hd_w->zb);
      free(hd_w);
      free(labels);
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
        free(hd_w->coeff_w);
        free(hd_w->w_perm);
        free(hd_w->zb);
        free(hd_w);
        free(labels);
        free(block_dim);
        CFL_ERROR_NULL("calloc failed for hd_w->zb[i]");
      }
    }
    else {
      hd_w->zb[i] = NULL;
    }
  }

  /* Allocate individual diag workspaces for each block.  While extravagant in
   * memory use, it allows for easy parallel diagonalization.  */
  max_block_dim = 0;
  for (i = 0; i < nblocks; i++) {
    if (max_block_dim < block_dim[i]) {
      max_block_dim = block_dim[i];
    }
  }

#ifdef _OPENMP
  if (h->num_procs == -1) {
    hd_w->ndiag_w = omp_get_num_procs();
  }
  else {
    hd_w->ndiag_w = h->num_procs;
  }

  if (hd_w->ndiag_w >= nblocks) {
    /* The limiting factor is the number of blocks; no point allocating space
     * for more diagonalizations than there is blocks. */
    hd_w->ndiag_w = nblocks;
    hd_w->proc_limited = 0;
  }
  else {
    /* The limiting factor is the number of available cores; we alloc diag
     * workspace for each core, of dimensions large enough to diag the biggest
     * block. */
    hd_w->proc_limited = 1;
  }
#endif /* _OPENMP */

  hd_w->diag_w = (zheevr_w **) malloc(nblocks*sizeof(zheevr_w *));
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
    free(hd_w->coeff_w);
    free(hd_w->w_perm);
    if (job == 'V') {
      for (j = 0; j < nblocks; j++) {
        free(hd_w->zb[j]);
      }
    }
    free(hd_w->zb);
    free(hd_w);
    free(labels);
    free(block_dim);
    CFL_ERROR_NULL("zheevr_w_alloc failed for hd_w->diag_w");
  }

#ifdef _OPENMP
  for (i = 0; i < hd_w->ndiag_w; i++) {
    if (hd_w->proc_limited) {
      hd_w->diag_w[i] = (zheevr_w *) zheevr_w_alloc(job, max_block_dim,
          hd_w->abstol);
    }
    else {
      hd_w->diag_w[i] = (zheevr_w *) zheevr_w_alloc(job, block_dim[i],
          hd_w->abstol);
    }
    if (hd_w->diag_w[i] == 0) {
      for (j = 0; j < i; j++) {
        zheevr_w_free(hd_w->diag_w[j]);
      }
      free(hd_w->diag_w);
      for (j = 0; j < nblocks; j++) {
        free(hd_w->blocks[j]->a);
        free(hd_w->blocks[j]);
      }
      free(hd_w->blocks);
      for (j = 0; j < hd_w->lcoeff_w; j++) {
        zhcsr_free(hd_w->coeff_w[j]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->coeff_w);
      free(hd_w->w_perm);
      if (job == 'V') {
        for (j = 0; j < nblocks; j++) {
          free(hd_w->zb[j]);
        }
      }
      free(hd_w->zb);
      free(hd_w);
      free(labels);
      free(block_dim);
      CFL_ERROR_NULL("zheevr_w_alloc failed for hd_w->diag_w[i]");
    }
  }
#else
  /* OpenMP is disabled; we alloc a single diag workspace, with dimension big
   * enough for the largest block. */
  hd_w->diag_w[0] = (zheevr_w *) zheevr_w_alloc(job, max_block_dim,
      hd_w->abstol);
  if (hd_w->diag_w[0] == 0) {
    free(hd_w->diag_w);
    for (j = 0; j < nblocks; j++) {
      free(hd_w->blocks[j]->a);
      free(hd_w->blocks[j]);
    }
    free(hd_w->blocks);
    for (j = 0; j < hd_w->lcoeff_w; j++) {
      zhcsr_free(hd_w->coeff_w[j]);
    }
    zcsr_free(zcsr_h);
    free(hd_w->coeff_w);
    free(hd_w->w_perm);
    if (job == 'V') {
      for (j = 0; j < nblocks; j++) {
        free(hd_w->zb[j]);
      }
    }
    free(hd_w->zb);
    free(hd_w);
    free(labels);
    free(block_dim);
    CFL_ERROR_NULL("zheevr_w_alloc failed for hd_w->diag_w[i]");
  }
#endif /* _OPENMP */ 

  if (nblocks != 1) {
    hd_w->blk_perm = (int *) calloc(zcsr_h->n, sizeof(int));
    if (hd_w->blk_perm == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->coeff_w);
      for (i = 0; i < nblocks; i++) {
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        zheevr_w_free(hd_w->diag_w[i]);
      }
      free(hd_w->blocks);
      free(hd_w->diag_w);
      if (hd_w->zb != NULL) {
        for (i = 0; i < nblocks; i++) {
          free(hd_w->zb[i]);
        }
      }
      free(hd_w->zb);
      free(hd_w->w_perm);
      free(hd_w);
      free(labels);
      free(block_dim);
      CFL_ERROR_NULL("calloc failed for blk_perm");
    }

    /* Pointer used to determine coordinates that sort the connected-components
     * labels, and thus block-diag the Hamiltonian.  */
    lptr = (int **) malloc(zcsr_h->n*sizeof(int *));
    if (lptr == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->coeff_w);
      for (i = 0; i < nblocks; i++) {
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        zheevr_w_free(hd_w->diag_w[i]);
      }
      free(hd_w->blocks);
      free(hd_w->diag_w);
      if (hd_w->zb != NULL) {
        for (i = 0; i < nblocks; i++) {
          free(hd_w->zb[i]);
        }
      }
      free(hd_w->zb);
      free(hd_w->w_perm);
      free(hd_w->blk_perm);
      free(hd_w);
      free(labels);
      free(block_dim);
      CFL_ERROR_NULL("malloc failed for lptr");
    }

    for (i = 0; i < zcsr_h->n; i++) {
      lptr[i] = &labels[i];
    }
    qsort(lptr, zcsr_h->n, sizeof(int *), iptr_cmp);

    for (i = 0; i < zcsr_h->n; i++) {
      hd_w->blk_perm[lptr[i] - labels] = i;
    }
    free(lptr);

    /* Alloc the row and column permuted csr matrices, and assosciated arrays,
     * used for block-diagonalizing the Hamiltonian. */
    hd_w->blk_pj = (int *) calloc(zcsr_h->nnz+1, sizeof(int));
    if (hd_w->blk_pj == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->coeff_w);
      for (i = 0; i < nblocks; i++) {
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        zheevr_w_free(hd_w->diag_w[i]);
      }
      free(hd_w->blocks);
      free(hd_w->diag_w);
      if (hd_w->zb != NULL) {
        for (i = 0; i < nblocks; i++) {
          free(hd_w->zb[i]);
        }
      }
      free(hd_w->zb);
      free(hd_w->w_perm);
      free(hd_w->blk_perm);
      free(hd_w);
      free(labels);
      free(block_dim);
      CFL_ERROR_NULL("calloc failed for blk_pj");
    }
    hd_w->blk_rp_h = (zcsr *) zcsr_row_perm_alloc(zcsr_h, hd_w->blk_perm);
    if (hd_w->blk_rp_h == 0) {
      for (i = 0; i < hd_w->lcoeff_w; i++) {
        zhcsr_free(hd_w->coeff_w[i]);
      }
      zcsr_free(zcsr_h);
      free(hd_w->coeff_w);
      for (i = 0; i < nblocks; i++) {
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        zheevr_w_free(hd_w->diag_w[i]);
      }
      free(hd_w->blocks);
      free(hd_w->diag_w);
      if (hd_w->zb != NULL) {
        for (i = 0; i < nblocks; i++) {
          free(hd_w->zb[i]);
        }
      }
      free(hd_w->zb);
      free(hd_w->w_perm);
      free(hd_w->blk_perm);
      free(hd_w->blk_pj);
      free(hd_w);
      free(labels);
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
      free(hd_w->coeff_w);
      for (i = 0; i < nblocks; i++) {
        free(hd_w->blocks[i]->a);
        free(hd_w->blocks[i]);
        zheevr_w_free(hd_w->diag_w[i]);
      }
      free(hd_w->blocks);
      free(hd_w->diag_w);
      if (hd_w->zb != NULL) {
        for (i = 0; i < nblocks; i++) {
          free(hd_w->zb[i]);
        }
      }
      free(hd_w->zb);
      free(hd_w->w_perm);
      free(hd_w->blk_perm);
      free(hd_w->blk_pj);
      zcsr_free(hd_w->blk_rp_h);
      free(hd_w);
      free(labels);
      free(block_dim);
      CFL_ERROR_NULL("zcsr_col_perm_alloc failed for blk_cp_h");
    }

    zcsr_row_perm(hd_w->zcsr_h, hd_w->blk_rp_h, hd_w->blk_perm);
    zcsr_col_perm(hd_w->blk_rp_h, hd_w->blk_cp_h, hd_w->blk_perm, hd_w->blk_pj);

    if (job == 'V') {
      /* Find the block permutation in coordinate form. */
      hd_w->crd_blk_perm = (int *) malloc(zcsr_h->n*sizeof(int));
      if (hd_w->crd_blk_perm == 0) {
        for (i = 0; i < hd_w->lcoeff_w; i++) {
          zhcsr_free(hd_w->coeff_w[i]);
        }
        zcsr_free(zcsr_h);
        free(hd_w->coeff_w);
        for (i = 0; i < hd_w->nblocks; i++) {
          free(hd_w->blocks[i]->a);
          free(hd_w->blocks[i]);
          zheevr_w_free(hd_w->diag_w[i]);
        }
        free(hd_w->blocks);
        free(hd_w->diag_w);
        if (hd_w->zb != NULL) {
          for (i = 0; i < hd_w->nblocks; i++) {
            free(hd_w->zb[i]);
          }
        }
        free(hd_w->zb);
        free(hd_w->w_perm);
        free(hd_w->blk_perm);
        free(hd_w->blk_pj);
        zcsr_free(hd_w->blk_rp_h);
        zcsr_free(hd_w->blk_cp_h);
        free(hd_w);
        free(labels);
        free(block_dim);
        CFL_ERROR_NULL("malloc failed for hd_w->crd_blk_perm");
      }
      for (i = 0; i < zcsr_h->n; i++) {
        hd_w->crd_blk_perm[hd_w->blk_perm[i]] = i;
      }
    }
    else {
      hd_w->crd_blk_perm = NULL;
    }
  }

  free(labels);

  /* Calculate the index of the first row of each block. */
  k = block_dim[0];
  block_dim[0] = 0;
  for (i = 1; i < nblocks; i++) {
    j = block_dim[i];
    block_dim[i] = block_dim[i-1] + k;
    k = j;
  }
  hd_w->bri = block_dim;

  return hd_w;
}

void zhd_w_free(zhd_w *hd_w) {
  int i;

  for (i = 0; i < hd_w->lcoeff_w; i++) {
    zhcsr_free(hd_w->coeff_w[i]);
  }
  zcsr_free(hd_w->zcsr_h);
  free(hd_w->coeff_w);
  for (i = 0; i < hd_w->nblocks; i++) {
    free(hd_w->blocks[i]->a);
    free(hd_w->blocks[i]);
  }
  free(hd_w->blocks);

#ifdef _OPENMP
  for (i = 0; i < hd_w->ndiag_w; i++) {
    zheevr_w_free(hd_w->diag_w[i]);
  }
#else 
  zheevr_w_free(hd_w->diag_w[0]);
#endif /* _OPENMP */
  free(hd_w->diag_w);
  if (hd_w->zb[0] != NULL) {
    for (i = 0; i < hd_w->nblocks; i++) {
      free(hd_w->zb[i]);
    }
  }
  free(hd_w->zb);
  if (hd_w->nblocks != 1) {
    free(hd_w->blk_perm);
    zcsr_free(hd_w->blk_rp_h);
    free(hd_w->blk_pj);
    zcsr_free(hd_w->blk_cp_h);
    if (hd_w->crd_blk_perm != NULL) {
      free(hd_w->crd_blk_perm);
    }
  }
  free(hd_w->bri);
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
  else {
    zhcsrsm((h->t[0])->matel, hd_w->coeff_w[0], h->coeff[0]);
  }

  /* Convert the Hamiltonian from Hermitian CSR to standard CSR, then apply
   * block-diag permutation, and finally convert to dense storage. */
  zhcsr2zcsr(hd_w->coeff_w[hd_w->lcoeff_w-1], hd_w->zcsr_h);

  if (hd_w->nblocks != 1) {
    zcsr_row_perm(hd_w->zcsr_h, hd_w->blk_rp_h, hd_w->blk_perm);
    zcsr_col_perm(hd_w->blk_rp_h, hd_w->blk_cp_h, hd_w->blk_perm, hd_w->blk_pj);
    zh_diag_blocks(job, w, hd_w->blk_cp_h, hd_w);
  }
  else {
    zh_diag_blocks(job, w, hd_w->zcsr_h, hd_w);
  }

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
    zh_parse_ev(z,  h->n, hd_w);
  }
}
