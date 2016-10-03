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

#ifndef _CFL_ZEFOZ_H_ 
#define _CFL_ZEFOZ_H_

#include "cfl_h.h"

/* Workspace type declaration for Hamiltonian diagonalization. */
typedef struct {
  /* The Hamiltonian. */
  zh *h;
  /* Indices of the Zeeman tensors in coeff, in order x, y, and z. */
  int *zi;
  /* The Zeeman tensor matrix elements, in order x, y, and z. */
  /* Storage for eigenvalues. */
  double *w;
  /* Storage for eigenvectors. */
  double complex *z;
  /* Hamiltonian diagonalization workspace. */
  zhd_w *hd_w;
  /* Workspace for computing the inner product in gradient eval. */
  double complex *inprod_w;
  /* Workspace for dgetri inversion, and Jacobian/gradient dot product. */
  double *dwork;
  /* The size of the dwork array. */
  int dlwork;
  /* The pivot indices the LU factorization and dgetri inversion used when
   * inverting C. */
  int ipiv[3];
  /* The Zeeman gradient vector. */
  double v[3];
  /* The Zeeman curvature tensor. */
  double C[9];
} zefoz_w;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
zefoz_w *zefoz_alloc(zh *h, int *zi);
void zefoz_free(zefoz_w *work);
void zefoz_iter(int k, int l, double *B, double complex **m, zefoz_w *work);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_ZEFOZ_H_ */
