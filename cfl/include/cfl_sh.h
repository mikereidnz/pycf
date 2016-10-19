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


#ifndef _CFL_SH_H_
#define _CFL_SH_H_

#include "cfl_tensor.h"
#include "cfl_h.h"

/* Data type for sorting projection states according to sz and iz labels. */
typedef struct {
  /* Spin Hamiltonian element index. */
  size_t index;
  /* iz value (2*I_z). */
  int iz;
} zsh_sort_t;

/* Data for spin Hamiltonian inversion, solving Ax=b. */
typedef struct {
  /* Pointer to the inversion coefficient matrix A. */
  complex double *a;
  /* Pointer to the spin Hamiltonian matrix elements, b. */
  complex double *b;
  /* The number of rows of A, and the length of b; the number of columns of A
   * and the length of x is always 9. */
  size_t m;
} zsh_inv_data;

/* Data for projection from full dimension tensor matrix elements to spin
 * Hamiltonian space. */
typedef struct {
  /* The dimension of the spin Hamiltonian inversion term. */
  int shi_dim;
  /* The matrix elements of the full dimension tensor to project. */
  complex double *pt;
  /* The coupling coefficient; should be unity for Zeeman, and the nuclear
   * dipole or nuclear quadrupole coupling strength for HYP and QUAD. */
  double coupling;
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
  /* The first and second entry points to the **pro_data entries of the
   * hyperfine and quadrupole interaction, respectively. */
  int pd_map[2];
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
  /* The index of the iz label in sh->pt_slabels. */
  char iz_i;
} zshp_p_w; 

/* Workspace for symmeterizing spin Hamiltonian parameter tensors using an SVD
 * of the form A = U * SIGMA * conjugate-transpose(V). */
typedef struct {
  /* The complex valued LAPACK workspace for the SVD. */
  double *work;
  /* The dimension of the array work. */
  int lwork;
  /* Copy of the input parameter matrix a. */
  double tmp_a[9];
  /* The singular value matrix. */
  double s[9];
  /* The unitary matrix U. */
  double u[9];
  /* The transposed unitary matrix V. */
  double vt[9];
} svd_sym_w;

/* The spin Hamiltonian inversion workspace. */
typedef struct {
  /* Flag indicating whether to symmeterize param tensor using an SVD. */
  char job;
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
  /* The leading dimension of array b. ldb >= MAX(1,M,N) */
  int ldb;
  /* Workspace for SVD decomposition. */
  svd_sym_w *svd_w;
} zshi_w;

/* Workspace for crystal field Hamiltonian to spin Hamiltonian parameter
 * projection. */
typedef struct {
  /* Pointer to the spin Hamiltonian projection workspace. */
  zshp_p_w *shp_p_w;
  /* Array of pointers to the spin Hamiltonian inversion workspaces. */
  zshi_w **shi_w;
  /* The number of interactions; required for freeing zshi_w. */
  int ninter;
  /* The offset between int_i and pro_i due to there being three pro_i indices
   * per int_i for a zeeman interaction. */
  int zeeman_offset;
  /* The dimension of a single Zeeman term. */
  int msz;
 /* The tensor index of the magz or magzs tensor, used for sorting matrix
  * elements w.r.t. Sz. */
  int magz_i;
} zshp_w;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zsh *zsh_alloc(char **inter, size_t ninter, int sz, int iz, complex double **a);
void zsh_free(zsh *sh);
int zsh_set_pro(zsh *sh, zt **t, int l, double *coupling);
void zsh_set_inv(zsh *sh, complex double *b, char *inter);
zshp_p_w *zshp_p_w_alloc(zsh *sh);
void zshp_p_w_free(zshp_p_w *shp_p_w);
void zshp_gen_sort(complex double *hz, int pro_i, zsh *sh, zshp_p_w *shp_p_w);
void zshp_parse(complex double *a, zsh *sh, int pro_i, zshp_p_w *shp_p_w);
void zshp_p(complex double *hz, zsh *sh, int pro_i, zshp_p_w *shp_p_w);
svd_sym_w *svd_sym_w_alloc();
void svd_sym_w_free(svd_sym_w *w);
void svd_sym(double *a, svd_sym_w *w);
zshi_w *zshi_w_alloc(char job, zsh_inv_data *d);
void zshi_w_free(zshi_w *w);
void zshi(double *a, zshi_w *w);
zshp_w *zshp_w_alloc(char job, zsh *sh);
void zshp_w_free(zshp_w *w);
void zshp(double *a, complex double *b, complex double *hz, int int_i,
    zsh *sh, zshp_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
