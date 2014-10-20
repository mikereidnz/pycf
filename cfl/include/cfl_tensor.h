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

#ifndef _CFL_TENSOR_H_
#define _CFL_TENSOR_H_

#include <cfl_crs.h>

/* State label type. */
typedef struct {
  /* The number of states. */
  size_t n;
  /* Pointer to arrays of length l of state labels. */
  char **states;
  /* Pointer to hash of states. */
  long hash;
} sl;

 
/* The tensor structure for complex valued matrix elements. */
typedef struct {
  /* Pointer to tensor name character array. */
  char *name;
  /* Dimension of the matrix elements. */
  int n;
  /* State labels of the tensor. */
  sl *states;
  /* Pointer to the matrix elements stored in CRS form. */
  crs_zhm *matel;
} zt; 


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

sl *sl_alloc(size_t n, char **states);
void sl_free(sl *l);
zt *zt_alloc(char *name, complex double *a, size_t n, sl *states);
void zt_free(zt *t);
zt *zt_sa(char *name, zt *t1, zt *t2, complex double s1, complex double s2);
zt *zt_s(char *name, zt *t, complex double s);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_TENSOR_H_ */

