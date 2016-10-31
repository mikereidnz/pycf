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
 * Diagonalization, and associated, routines for crystal-field and spin
 * Hamiltonians.
 */

#ifndef _CFL_H_H_
#define _CFL_H_H_

#include <complex.h>
#include "cfl_csr.h"
#include "cfl_tensor.h"

typedef struct {
  /* Block dimension. */
  int dim;
  /* Pointer to matrix elements in dense column major form. */
  complex double *a;
} zblock;

/* The Hamiltonian structure. */
typedef struct {
  /* Dimension of the Hamiltonian. */
  int n;
  /* Number of tensors. */
  int nt;
  /* State labels. */
  sl *slabels;
  /* Array of length nt containing the hash values for each tensor name. */
  uint32_t *tnh;
  /* The Hamiltonian hash; consists of a hash of the tnh array. */
  uint32_t hh;
  /* Pointer to array of pointers to complex valued tensors. */
  zt **t;
  /* Tensor coefficients. */
  complex double *coeff;
#ifdef _OPENMP
  /* Allows the zhd_w_alloc caller to specify the number of cores available for
   * diagonalization of this Hamiltonian.  If not modified by the caller,
   * zhd_w_alloc will default to all cores available to the program. */
  int num_procs;
#endif /* _OPENMP */
} zh;

/* Workspace for LAPACKE_zheevr. */
typedef struct {
/* The support of the eigenvectors in Z. */
  int *isuppz;
  /* Workspace for LAPACKE_zheevr. */
  complex double *work;
  /* Dimensions of LAPACKE_zheevr work. */
  int lwork;
  /* LAPACKE_zheevr RWORK. */
  double *rwork;
  /* Dimensions of LAPACKE_zheevr rwork. */
  int lrwork;
  /* LAPACKE_zheevr IWORK. */
  int *iwork;
  /* Dimensions of LAPACKE_zheevr iwork. */
  int liwork;
  /* The total number of eigenvalues found by zheevr. */
  int m;
} zheevr_w;

/* Workspace type declaration for Hamiltonian diagonalization. */
typedef struct {
  /* Array of Hermitian CSR matrices of individual tensors. */
  zhcsr **hcsr_ma;
  /* Workspace for summing the tensors for currently set coefficients. */
  zhcsrsama_data *coeff_w;
  /* Storage for non-Hermitian CRS representation. */
  zcsr *zcsr_h;
  /* Permutation to obtain block-diagonalized ordering of the Hamiltonian. */
  int *blk_perm;
  /* The block-diagonalization permutation in coordinate form; used for
   * reconstructing eigenvectors after diagonalization. */
  int *crd_blk_perm;
  /* The block_perm ordering row permuted Hamiltonian. */
  zcsr *blk_rp_h;
  /* Storage for value array permutation required for block_perm cperm. */
  int *blk_pj;
  /* The block_perm column permuted Hamiltonian. */
  zcsr *blk_cp_h;
  /* The absolute error tolerance to which each eigenvector is required. */
  double abstol;
  /* LAPACKE_zheevr diagonalization workspace. */
  zheevr_w **diag_w;
  /* Permutation required to sort eigenvalues. */
  int *w_perm;
  /* The number of blocks. */
  int nblocks;
  /* The index of the first row of each block. */
  int *bri;
  /* Array of blocks. */
  zblock **blocks;
  /* Eigenvectors of blocks; only available if eigenvector evaluation is
   * requested. */
  complex double **zb;
#ifdef _OPENMP
  /* The parallelization limiting factor is the number of processors, not the
   * number of blocks. */
  int proc_limited;
#endif
  /* The number of diagonalization workspaces. */
  int ndiag_w;
} zhd_w;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zh *zh_alloc(int n, int nt, zt **t); 
void zh_free(zh *h);
void zh_set_coeff(zh *h, complex double *coeff);
zhd_w *zhd_w_alloc(char job, zh *h);
void zhd_w_free(zhd_w *hd_w);
void zhd(char job, double *w, complex double *z, zh *h, zhd_w *hd_w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_H_ */
