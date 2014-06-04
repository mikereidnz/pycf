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
 * @file    cfl_spinh.c
 * @brief   Spin Hamiltonian routines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <cfl_error.h>

/*
 * @brief Allocate storage for a spin Hamiltonian term. 
 *
 * @param[type]   The type of spin Hamiltonian term.
 * @param[data]   Data relevant to this type of spin Hamiltonian. 
 */
zsh *zsh_alloc(size_t n, char **states, sh_type_t type, sh_data_t data) {

  zsh *sh;
  double complex *a;
  char **s;



  sh = (zsh *) malloc(sizeof(zsh));
  if (sh == 0) {
    CSL_ERROR_NULL("malloc failed for sh");
  }

  s = (char *) malloc(n*sizeof(states[0][0]));
  if (s == 0) {
    free(sh);
    CSL_ERROR_NULL("malloc failed for s");
  }

  a = (double complex *) calloc(n*n,sizeof(double complex));
  if (a == 0) {
    free(sh);
    free(s);
    CSL_ERROR_NULL("calloc failed for a");
  }

  sh->type = type;
  sh->data = data;



}
