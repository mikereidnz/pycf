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

/*
 * @brief Spin Hamiltonian term types.
 */
typedef enum {
  bgs = 0,
  ias = 1,
  iqi = 2
} sh_type_t;

/*
 * @brief Struct containing information of bgs terms.
 */
typedef struct {
  /* The total spin angular momentum quantum number. */
  float s;
  /* Pointer to magnetic field array. */
  double complex *b;
  /* Pointer to S matrix elements in a dense array. */
  double complex *s_matel;
} bgs_t;

/*
 * @brief Struct containing information of ias terms.
 */
typedef struct {
  /* The total spin angular momentum quantum number. */
  float s;
  /* The total nuclear angular momentum quantum number. */
  float i;
  /* Pointer to S matrix elements in a dense array. */
  double complex *s_matel;
  /* Pointer to I matrix elements in a dense array. */
  double complex *i_matel;
} ias_t;

/*
 * @brief Struct containing information of iqi terms.
 */
typedef struct {
  /* The total nuclear angular momentum quantum number. */
  float i;
  /* Pointer to I matrix elements in a dense array. */
  double complex *i_matel;
} iqi_t;

/*
 * @brief Union holding different term data structs.
 */
typedef union {
  bgs_t *bgs;
  ias_t *ias;
  iqi_t *iqi;
} sh_data_t;

/*
 * @brief Spin Hamiltonian structure definition.
 */
typedef struct {
  /* Dimension of the spin Hamiltonian. */
  size_t n;
  /* State labels corresponding to eigenvalues. */
  char **states;
  /* Pointer to hash of state labels. */
  char *state_hash;
  /* Pointer to matrix elements stored in a contigious array. */
  double complex *a;
  /* Spin Hamiltonian data type. */
  sh_type_t type;
  /* Pointer to union of data types. */
  sh_data_t data;
} zsh;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
zsh *zsh_alloc(sh_type_t type, sh_data_t data, char **states);
void zsh_free(zsh *sh);
zshp_w *zshp_w_alloc(zsh *sh);
void zshp_w_free(zshp_w *shp_w);
void zshp(zsh *sh, zshp_w *shp_w);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_SH_H_ */
