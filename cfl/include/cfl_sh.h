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

/*
 * @brief Spin Hamiltonian structure definition.
 */
typedef struct {
  /* Dimension of the spin Hamiltonian. */
  size_t n;
  /* State labels corresponding to eigenvalues. */
  state_t *states;
  /* Pointer to matrix elements stored in a contiguous array. */
  double complex *a;
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
  /* The number of rows of the inversion matrix. */
  size_t m;
  /* The number of columns of the inversion matrix. */
  size_t n;
  /* Pointer to coefficient array, of size m by n. */
  double complex *a;
  /* Length of workspace. */
  int lwork;
  /* Pointer to workspace required by zgels. */
  double complex *work;
} zshi_w;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

zsh *zsh_alloc(size_t n);
void zsh_free(zsh *sh);
zshp_w *zshp_w_alloc(zt *t);
void zshp(zh *h, zsh *sh, zshp_w *shp_w, int l);
zshi_w *zshi_w_alloc(double complex *a, double complex *b, size_t m, size_t n);
void zshi_w_free(zshi_w *w);
void zshi(double complex *b, zshi_w *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
