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
 * Diagonalization, and associated, routines for crystal-field and spin
 * Hamiltonians.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cfl_error.h>
#include <cfl_crs.h>
#include <cfl_tensor.h>

/*
 * Implement the djb2 hash algorithm for arrays of strings. For details on djb2,
 * see: http://www.cse.yorku.ca/~oz/hash.html
 *
 * Parameters
 * ----------
 *  n         The length of the array.
 *  str_a     Array of strings to hash. 
 */
long state_hash(size_t n, char **str_a) {
  unsigned long hash = 5381;
  int i, c;
  char *str;
   
  for (i=0; i<n; i++) {
    str = str_a[i];
    while (c = *str++) {
      /* hash * 33 + c */
      hash = ((hash << 5) + hash) + c;
    }
  }
  return hash;
}

/*
 * Allocate storage for state labels. 
 *
 * Parameters
 * ----------
 *  n         The number of states.
 *  states    Array of strings containing state labels; it is assumed that all
 *            state label strings are of the same length. 
 */
sl *sl_alloc(size_t n, char **states) {
  sl *l;
  int i, j;
  size_t sl_len;

  l = (sl *) malloc(sizeof(sl));
  if (l == 0) {
    CFL_ERROR_NULL("malloc failed for sl");
  }
  sl_len = strlen(states[0])+1;

  l->states = (char **) malloc(n*sizeof(char *));
  if (l->states == 0) {
    free(l);
    CFL_ERROR_NULL("malloc failed for l.states");
  }

  for (i=0; i<n; i++) {
    l->states[i] = (char *) malloc(sl_len*sizeof(char));
    if (l->states[i] == 0) {
      for (j=0; j<i; j++) {
        free(l->states[j]);
      }
      free(l->states);
      free(l);
      CFL_ERROR_NULL("malloc failed for l.states[i]");
    }
    strcpy(l->states[i], states[i]);
  }
  
  l->hash = state_hash(n, l->states);
  l->n = n;
  
  return l;
}

void sl_free(sl *l) {
  int i;

  for (i=0; i<l->n; i++) {
    free(l->states[i]);
  }
  free(l->states);
  free(l);
}

/*
 * Allocate storage for complex valued tensors. 
 *
 * Parameters
 * ----------
 *  name    A unique identifier of the tensor. 
 *  a       Pointer to array containing the matrix elements. 
 *  n       The dimension of the matrix element matrix.
 *  states  Pointer to state labels struct.
 */
zt *zt_alloc(char *name, double complex *a, size_t n, sl *states) {
  zt *t;
  size_t sl_len;
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
  t->states = states;
  t->matel = ma;
  
  return t;
}

void zt_free(zt *t) {
  crs_zhm_free(t->matel);
  free(t);
}


/*
 * Add and scale the matrix elements of two tensors, write the result to a newly
 * allocated tensor, and return a pointer to it. 
 *
 * Parameters
 * ----------
 *  name    Name of the resulting third tensor. 
 *  t1      Pointer to the first tensor struct. 
 *  t2      Pointer to the second tensor struct.
 *  s1      A complex valued scale factor for the first tensor.
 *  s2      A complex valued scale factor for the second tensor.
 */
zt *zt_sa(char *name, zt *t1, zt *t2, double complex s1, double complex s2) {
  zt *t;

  if (t1->n != t2->n) {
    CFL_ERROR_NULL("dimensions of tensors to be added do not match");
  }
  else if (t1->states->hash != t2->states->hash) {
    CFL_ERROR_NULL("state labels of tensors to be added don't match");
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
  t->states = t1->states;

  return t;
}

/*
 * Allocate storage for a new tensor, and write to it the scaled matrix elements
 * of the provided tensor.  
 *
 * Parameters
 * ----------
 *  name    The name of the new tensor. 
 *  t       Pointer to the input tensor.
 *  s       A complex valued scale factor.
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
  ts->states = t->states;

  return ts;
} 
