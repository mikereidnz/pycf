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
  /* Storage for the magnetic field strength of a given iteration. */
  double B[3];
  /* The pivot indices the LU factorization and dgetri inversion used when
   * inverting C. */
  int ipiv[3];
  /* The Zeeman gradient vector. */
  double v[3];
  /* The Zeeman curvature tensor. */
  double C[10];
} zefoz_d;

typedef struct {
  /* Field strengths, in consecutive blocks of three (x, y, and z). */
  double *B;
  /* Field gradients, in consecutive blocks of three (x, y, and z). */
  double *v;
  /* Zefoz point counter. */
  int ctr;
  /* Size of zefoz point array. */
  int size;
} zefoz_a;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
zefoz_d *zefoz_alloc(zh *h, int *zi);
void zefoz_free(zefoz_d *data);
zefoz_a *zefoz_a_alloc(int init_size);
void zefoz_a_free(zefoz_a *za);
void zefoz_search(zefoz_a *za, double xtol, int k, int l, double *Bi, 
    double *Bf, int *na, complex double **m, zefoz_d *data);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_ZEFOZ_H_ */
