/*
 * Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * @file    cfl_sh.h
 * @brief   Spin Hamiltonian routines.
 */

#ifndef _CFL_SH_H_
#define _CFL_SH_H_

#include <cfl_tensor.h>
#include <cfl_h.h>

/* 
 * @brief State label type. 
 */
typedef struct {
  /* The length of labels */
  size_t l;
  /* Pointer to arrays of length l of state labels. */
  char **states;
  /* Pointer to hash of states. */
  char *state_hash;
} state_t;

/* Spin Hamiltonian projection data. */
typedef struct {
  /* Perturbation matrix elements. */
  zt *tensor;
  /* Integer specifying first level of the complete Hamiltonian which
   * corresponds to a spin Hamiltonian element. */
  int l;
} zsh_p_data;

/* Spin Hamiltonian inversion data. */
typedef struct {
  /* Pointer to the inversion coefficient matrix A. */
  double complex *a;
  /* The number of rows of A and length of b. */
  size_t m;
  /* The number of columns of B, and the length of x. */
  size_t n;
} zsh_inv_data;

/*
 * @brief Spin Hamiltonian structure definition.
 */
typedef struct {
  /* Dimension of the spin Hamiltonian. */
  size_t n;
  /* Pointer to term type character array. */
  char *type;
  /* Pointer to spin Hamiltonian matrix element array. */
  double complex *a;
  /* State labels corresponding to eigenvalues. */
  state_t *states;
  /* Projection data. */
  zsh_p_data *p_data;
} zsh;

/*
 * @brief Definition of spin Hamiltonian projection workspace type.
 */
typedef struct {
  /* Dimension of the complete Hamiltonian. */
  size_t nc;
  /* Pointer to matrix elements of tensor to project in dense storage. */
  double complex *m;
  /* Pointer to array used for storing intermediate values. */
  double complex *a;
  /* Pointer to array used for storing the final values of the projection. */
  double complex *b;
} zshp_w; 

/*
 * The spin Hamiltonian inversion workspace. 
 */
typedef struct {
  /* Spin Hamiltonian inversion data. */
  zsh_inv_data *data;
  /* Length of workspace. */
  int lwork;
  /* Pointer to workspace required by zgels. */
  double complex *work;
} zshi_w;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zsh *zsh_alloc(size_t n, char *type);
void zsh_free(zsh *sh);
zshp_w *zshp_w_alloc(zsh *sh);
zsh_inv_data *zsh_inv_data_alloc(double complex *a, size_t m, size_t n);
void zsh_inv_data_free(zsh_inv_data *d);
void zshp(zsh *sh, zshp_w *shp_w);
zshi_w *zshi_w_alloc(zsh_inv_data *d);
void zshi_w_free(zshi_w *w);
void zsh_set_inv(zsh *sh, double complex *a, size_t m, size_t n); 
void zshi(double complex *a, zshi_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
