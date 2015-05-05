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
 * @file    cfl_sh.h
 * @brief   Spin Hamiltonian routines.
 */

#ifndef _CFL_SH_H_
#define _CFL_SH_H_

#include "cfl_tensor.h"
#include "cfl_h.h"

/* Data type for sorting projection states according to sz and iz labels. */
typedef struct {
  /* Spin Hamiltonian element index. */
  size_t index;
  /* sz value (2*S_z). */
  int sz;
  /* iz value (2*I_z). */
  int iz;
} zsh_sort_t;

/* Data for spin Hamiltonian inversion, solving Ax=b. */
typedef struct {
  /* Pointer to the inversion coefficient matrix A. */
  complex double *a;
  /* Pointer to the spin Hamiltonian matrix elements, b. */
  complex double *b;
  /* The number of columns of A, and the length of x; the number of rows of A
   * and the length of b is always 9. */
  size_t m;
} zsh_inv_data;

/* Data for projection from full dimension tensor matrix elements to spin
 * Hamiltonian space. */
typedef struct {
  /* The dimension of the spin Hamiltonian inversion term. */
  int shi_dim;
  /* The matrix elements of the full dimension tensor to project. */
  complex double *pt;
} zsh_pro_data;

/* Spin Hamiltonian structure definition. */
typedef struct {
  /* Dimension of the complete spin Hamiltonian. */
  size_t dim;
  /* State labels corresponding to eigenvalues; currently not implemented. */
  sl *slabels;
  /* Array of strings specifing the type of interactions described by the spin
   * Hamiltonian. */
  char **inter;
  /* The number of interactions described by the spin Hamiltonian. */
  int ninter;
  /* The spin projection S_z * 2 (to ensure integer values). */
  int sz;
  /* The nuclear spin projection I_z * 2 (to ensure integer values). */
  int iz;
  /* The spin Hamiltonian inversion data. */
  zsh_inv_data **inv_data;
  /* The number of tensors to project. */
  int ntensors;
  /* Integer specifying the initial level for which to project the spin
   * Hamiltonian. */
  size_t l;
  /* The dimension of the tensor to project. */
  size_t pt_dim;
  /* Projection data. */
  zsh_pro_data **pro_data;
  /* The projection tensor state labels. */
  sl *pt_slabels;
} zsh;

/* Definition of spin Hamiltonian projection workspace type. */
typedef struct {
  /* Array used for storing intermediate values. */
  complex double *a;
  /* Array used for storing the final values of the projection. */
  complex double *b;
  /* Data for sorting w.r.t. Iz and Sz labels of projection result. */
  zsh_sort_t **sh_sort;
  /* The complete spin Hamiltonian dimension; required for freeing sh_sort. */
  size_t sh_dim;
  /* The index of the sz label in sh->pt_slabels. */
  char sz_i;
  /* The index of the iz label in sh->pt_slabels. */
  char iz_i;
} zshph_w; 

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

/* Workspace for crystal field Hamiltonian to spin Hamiltonian parameter
 * projection. */
typedef struct {
  /* Pointer to the spin Hamiltonian projection workspace. */
  zshph_w *shph_w;
  /* Array of pointers to the spin Hamiltonian inversion workspaces. */
  zshi_w **shi_w;
  /* The number of interactions; required for freeing zshi_w. */
  int ninter;
} zshp_w;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zsh *zsh_alloc(char **inter, size_t ninter, int sz, int iz, int l, complex double **a);
void zsh_free(zsh *sh);
zshph_w *zshph_w_alloc(zsh *sh);
void zshph_w_free(zshph_w *shph_w);
zsh_inv_data *zsh_inv_data_alloc(complex double *a, size_t m, size_t n);
void zsh_inv_data_free(zsh_inv_data *d);
zshi_w *zshi_w_alloc(zsh_inv_data *d);
void zshi_w_free(zshi_w *w);
int zsh_set_pro(zsh *sh, zt **t, size_t l);
void zsh_set_inv(zsh *sh, complex double *a, size_t m, size_t n); 
void zshph(complex double *a, complex double *hz, size_t l, size_t pt_dim, zsh_pro_data *pd, zshph_w *shph_w);
void zshi(complex double *a, zshi_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
