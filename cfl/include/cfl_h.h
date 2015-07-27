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
 * Diagonalization, and associated, routines for crystal-field and spin
 * Hamiltonians.
 */

#ifndef _CFL_H_H_
#define _CFL_H_H_

#include <complex.h>
#include "cfl_crs.h"
#include "cfl_tensor.h"

/* The Hamiltonian structure. */
typedef struct {
  /* Dimension of the Hamiltonian. */
  int n;
  /* Number of tensors. */
  int nt;
  /* State labels. */
  sl *slabels;
  /* Pointer to array of pointers to complex valued tensors. */
  zt **t;
  /* Tensor coefficients. */
  complex double *coeff;
  /* Pointer to matrix of the complete Hamiltonian in dense col major form. */
  complex double *a;
} zh;


/* Work space type declaration for Hamiltonian diagonalization. */
typedef struct {
  /* Workspace for summing the tensors for currently set coefficents. */
  crs_zhm **coeff_w;
  /* Length of coeff_w array. */
  int lcoeff_w;
  /* The total number of eigenvalues found by zheevr. */
  int m;
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
  /* LAPACKE_zheevd IWORK. */
  int *iwork;
  /* Dimensions of LAPACKE_zheevr iwork. */
  int liwork;
} zhd_w;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zh *zh_alloc(int n, int nt, zt **t); 
void zh_free(zh *h);
void zh_set_coeff(zh *h, complex double *coeff);
zhd_w *zhd_w_alloc(zh *h);
void zhd_w_free(zhd_w *hd_w);
void zhd(double *w, complex double *z, zh *h, zhd_w *hd_w);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_H_H_ */
