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
 * @file    cfl_tensor.c
 * @brief   Diagonalization, and associated, routines for crystal-field and spin
 *          Hamiltonians.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <cfl_error.h>
#include <cfl_crs.h>
#include <cfl_tensor.h>

/*
 * @brief Allocate storage for complex valued tensors. 
 *
 * @param[name]   A unique identifier of the tensor. 
 * @param[a]      Pointer to array containing the matrix elements. 
 * @param[n]      The dimension of the matrix elemet matrix.
 */
zt *zt_alloc(char *name, double complex *a, size_t n) {
  zt *t;
  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  crs_zhm *ma = crs_zhm_alloc(a, n);
  if (ma == 0) {
    free(t);
    CFL_ERROR_NULL("alloc failed for crs_zhm");
  }

  t->name = name;
  t->n = n;
  t->matel = ma;

  return t;
}

/*
 * @brief Free storage allocated for a zt.
 *
 * @param[t]     Pointer to the zt struct.
 */
void zt_free(zt *t) {
  crs_zhm_free(t->matel);
  free(t);
}


/*
 * @brief Add and scale the matrix elements of two tensors, write the result to
 *        a newly allocated tensor, and return a pointer to it. 
 *
 * @param[name] Name of the resulting third tensor. 
 * @param[t1]   Pointer to the first tensor struct. 
 * @param[t2]   Pointer to the second tensor struct.
 * @param[s1]   A complex valued scale factor for the first tensor.
 * @param[s2]   A complex valued scale factor for the second tensor.
 */
zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2) {
  zt *t;

  if (t1->n != t2->n) {
    CFL_ERROR_VOID("tensor dimensions do not match");
  }

  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  t->matel = crs_zhsam_alloc(t1->matel, t2->matel);
  if (t == 0) {
    free(t);
    CFL_ERROR_NULL("failed to alloc t");
  }
  crs_zhsam(t1->matel, t2->matel, t->matel, s1, s2);

  t->name = name;
  t->n = t1->n;

  return t;
}

/*
 * @brief Allocate storage for a new tensor, and write to it the scaled matrix
 *        elements of the provided tensor.  
 *
 * @param[name] The name of the new tensor. 
 * @param[t]    Pointer to the input tensor.
 * @param[s]    A complex valued scale factor.
 */
zt *zt_s(char *name, zt *t, double complex s) {
  zt *ts;

  ts = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  ts->matel = crs_zhsm_alloc(t->matel);
  if (ts == 0) {
    free(ts);
    CFL_ERROR_NULL("alloc failed for ts");
  }
  crs_zhsm(t->matel, ts->matel, s);

  ts->name = name;
  ts->n = t->n;

  return ts;
} 
