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

#include "cfl_tensor.h"
#include "cfl_h.h"

/* Spin Hamiltonian projection data. */
typedef struct {
  /* Matrix elements of tensor to project in dense storage. */
  double complex *td;
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
  double complex *a;
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
  double complex *a;
  /* Array used for storing the final values of the projection. */
  double complex *b;
} zshp_w; 

/* The spin Hamiltonian inversion workspace. */
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
void zshp_w_free(zshp_w *shp_w);
zsh_inv_data *zsh_inv_data_alloc(double complex *a, size_t m, size_t n);
void zsh_inv_data_free(zsh_inv_data *d);
zshi_w *zshi_w_alloc(zsh_inv_data *d);
void zshi_w_free(zshi_w *w);
void zsh_set_pro(zsh *sh, zt *t, int l);
void zsh_set_inv(zsh *sh, double complex *a, size_t m, size_t n); 
void zshp(double complex *a, double complex *hz, zsh *sh, zshp_w *shp_w);
void zshi(double complex *a, zshi_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
