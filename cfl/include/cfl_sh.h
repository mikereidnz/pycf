/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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
 * @file    cfl_sh.h
 * @brief   Spin Hamiltonian routines.
 */

#ifndef _CFL_SH_H_
#define _CFL_SH_H_

#include "cfl_tensor.h"
#include "cfl_h.h"

/* Spin Hamiltonian projection data. */
typedef struct {
  /* Matrix elements of tensor to project in dense storage. */
  complex double *td;
  /* Dimension of the tensor. */
  size_t tn;
  /* Integer specifying the initial level for which to project the spin
   * Hamiltonian. */
  size_t l;
  /* Flag used to free memory if alloced. */
  int set_flag;
} zsh_pro_data;


/* Spin Hamiltonian inversion data. */
typedef struct {
  /* Pointer to the inversion coefficient matrix A. */
  complex double *a;
  /* The number of rows of A and length of b. */
  size_t m;
  /* The number of columns of B, and the length of x. */
  size_t n;
} zsh_inv_data;

/* Spin Hamiltonian structure definition. */
typedef struct {
  /* Dimension of the spin Hamiltonian. */
  size_t n;
  /* Pointer to term type character array. */
  char *type;
  /* State labels corresponding to eigenvalues. */
  sl *states;
  /* Projection data. */
  zsh_pro_data *pro_data;
} zsh;

/* Definition of spin Hamiltonian projection workspace type. */
typedef struct {
  /* Dimension of the complete Hamiltonian. */
  size_t nc;
  /* Array used for storing intermediate values. */
  complex double *a;
  /* Array used for storing the final values of the projection. */
  complex double *b;
} zshp_w; 

/* The spin Hamiltonian inversion workspace. */
typedef struct {
  /* Spin Hamiltonian inversion data. */
  zsh_inv_data *data;
  /* Storage for inversion coefficient matrix; since zgels overwrites this upon
   * exit we can't pass the pointer to a stored in the zsh_inv_data object
   * directly. */
  complex double *a;
  /* The size of a */
  size_t a_size;
  /* Length of workspace. */
  int lwork;
  /* Pointer to workspace required by zgels. */
  complex double *work;
} zshi_w;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zsh *zsh_alloc(size_t n, char *type);
void zsh_free(zsh *sh);
zshp_w *zshp_w_alloc(zsh *sh);
void zshp_w_free(zshp_w *shp_w);
zsh_inv_data *zsh_inv_data_alloc(complex double *a, size_t m, size_t n);
void zsh_inv_data_free(zsh_inv_data *d);
zshi_w *zshi_w_alloc(zsh_inv_data *d);
void zshi_w_free(zshi_w *w);
void zsh_set_pro(zsh *sh, zt *t, int l);
void zsh_set_inv(zsh *sh, complex double *a, size_t m, size_t n); 
void zshp(complex double *a, complex double *hz, zsh *sh, zshp_w *shp_w);
void zshi(complex double *a, zshi_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
