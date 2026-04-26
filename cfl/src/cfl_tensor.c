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
 * Overview
 * ========
 *
 * Tensor data storage used by cfl Hamiltonians and spin Hamiltonians.  State
 * label hashes can be used to efficiently check whether tensors span the same
 * state space.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <limits.h>

#include "cfl_config.h"
#include "cfl_error.h"
#include "cfl_csr.h"
#include "cfl_tensor.h"


/* 32-bit FNV hash function for a buffer due to Fowler, Noll and Vo.  For
 * further details, see: http://www.isthe.com/chongo/tech/comp/fnv/
 *
 * Parameters
 * ----------
 *  buf       The start of the buffer to be hashed. 
 *  len       The length of buf in octets.
 */
uint32_t fnv_hash(void *buf, int len) {
  unsigned char *bp = (unsigned char *)buf;	/* start of buffer */
  unsigned char *be;
  uint32_t hval;
  
  /* Validate preconditions */
  if (buf == NULL) {
    CFL_ERROR_VAL("fnv_hash: buf is NULL", 0);
  }
  if (len < 0) {
    CFL_ERROR_VAL("fnv_hash: len must be non-negative", 0);
  }
  
  be = bp + len;		        /* beyond end of buffer */
  hval = FNV1_32_INIT;
  /* FNV-1 hash each octet in the buffer. */
  while (bp < be) {
    /* multiply by the 32 bit FNV magic prime mod 2^32 */
    hval *= FNV_32_PRIME;
    /* xor the bottom with the current octet */
    hval ^= (uint32_t)*bp++;
  }
  
  /* return our new hash value */
  return hval;
}

/*
 * Allocate storage for state labels. 
 *
 * Parameters
 * ----------
 *  n         The number of states.
 *  key       String identifying the type of each state label.  Valid keys are:
 *            S, L, J, M and I, and the order in which they are listed must
 *            correspond to the order used in the label array for each state. 
 *  labels    Char array corresponding to the value of each state label.
 *            The order is dictade by the key array.  To avoid half integers,
 *            label values are always stored as twice their real value.  N.B.:
 *            label arrays are not strings, since 0 is a perfectly valid state
 *            label yet would yield a premature string termination. 
 */
sl *sl_alloc(int n, char *key, int **labels) {
  sl *l;
  int i, j;
  int nl;

  l = (sl *) malloc(sizeof(sl));
  if (l == 0) {
    CFL_ERROR_NULL("malloc failed for sl");
  }
  nl = strlen(key);

  l->key = (char *) malloc((nl+1)*sizeof(char));
  if (l->key == 0) {
    free(l);
    CFL_ERROR_NULL("malloc failed for l.key");
  }
  strncpy(l->key, key, nl);
  l->key[nl] = '\0';

  l->labels = (int **) malloc(n*sizeof(int *));
  if (l->labels == 0) {
    free(l->key);
    free(l);
    CFL_ERROR_NULL("malloc failed for l.labels");
  }

  for (i = 0; i < n; i++) {
    l->labels[i] = (int *) malloc(nl*sizeof(int));
    if (l->labels[i] == 0) {
      for (j = 0; j < i; j++) {
        free(l->labels[j]);
      }
      free(l->key);
      free(l->labels);
      free(l);
      CFL_ERROR_NULL("malloc failed for l.labels[i]");
    }
    memcpy(l->labels[i], labels[i], nl*sizeof(int));
  }

  l->lh = (uint32_t *) malloc(n*sizeof(uint32_t));
  if (l->lh == 0) {
    for (i = 0; i < n; i++) {
      free(l->labels[i]);
    }
    free(l->key);
    free(l->labels);
    free(l);
    CFL_ERROR_NULL("malloc failed for l->lh");
  }

  for (i = 0; i < n; i++) {
    size_t hash_size = (size_t)nl * sizeof(int);
    if (hash_size > INT_MAX) {
      CFL_ERROR_NULL("label array too large for hashing");
    }
    l->lh[i] = fnv_hash(l->labels[i], (int)hash_size);
  }

  {
    size_t hash_size = (size_t)n * sizeof(uint32_t);
    if (hash_size > INT_MAX) {
      CFL_ERROR_NULL("label hash array too large for hashing");
    }
    l->th = fnv_hash(l->lh, (int)hash_size);
  }
  l->n = n;

  return l;
}

void sl_free(sl *l) {
  int i;

  for (i = 0; i < l->n; i++) {
    free(l->labels[i]);
  }
  free(l->key);
  free(l->labels);
  free(l->lh);
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
 *  slabels Pointer to state labels struct.
 */
zt *zt_alloc(char *name, complex double *a, int n, sl *slabels) {
  int sl_len;
  zhcsr *ma;
  zt *t;

  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  ma = zhcsr_gen(a, n);
  if (ma == 0) {
    free(t);
    CFL_ERROR_NULL("alloc failed for zhcsr");
  }

  t->name = name;
  t->n = n;
  t->slabels = slabels;
  t->matel = ma;

  return t;
}


/*
 * Allocate storage for complex valued tensors for matrix elements stored in CSR
 * form. 
 *
 * Parameters
 * ----------
 *  name      A unique identifier of the tensor. 
 *  n         The dimension of the matrix element matrix.
 *  row_ptr   The CSR row pointer. 
 *  col_in    The CSR column index array. 
 *  val       Array containing the values of the non-zero elements.
 *  slabels   Pointer to state labels struct.
 */
zt *zt_csr_alloc(char *name, int n, int *row_ptr, int *col_in, 
    complex double *val, sl *slabels) {
  int sl_len;
  zhcsr *ma;
  zt *t;

  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }
  ma = zhcsr_alloc(n, row_ptr, col_in, val);
  if (ma == 0) {
    free(t);
    CFL_ERROR_NULL("alloc failed for zhcsr");
  }

  t->name = name;
  t->n = n;
  t->slabels = slabels;
  t->matel = ma;

  return t;
}


void zt_free(zt *t) {
  zhcsr_free(t->matel);
  free(t);
}

/* Wraper for cython to get the matrix elements of a tensor in dense storage;
 * defining c types with other user defined c types (zhcsr) seems annoying
 * in cython. */
void zt_get_matel(zt *t, complex double *a) {
  zhcsr2zha(t->matel, a);
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
zt *zt_sa(char *name, zt *t1, zt *t2, complex double s1, complex double s2) {
  zt *t;

  if (t1->n != t2->n) {
    CFL_ERROR_NULL("dimensions of tensors to be added do not match");
  }
  else if (t1->slabels->th != t2->slabels->th) {
    CFL_ERROR_NULL("state labels of tensors to be added don't match");
  }

  t = (zt *) malloc(sizeof(zt));
  if (t == 0) {
    CFL_ERROR_NULL("malloc failed for zt");
  }

  t->matel = zhcsrsam_alloc(t1->matel, t2->matel);
  if (t->matel == 0) {
    free(t);
    CFL_ERROR_NULL("failed to alloc t->matel");
  }
  zhcsrsam(t1->matel, t2->matel, t->matel, s1, s2);

  t->name = name;
  t->n = t1->n;
  t->slabels = t1->slabels;

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
zt *zt_s(char *name, zt *t, complex double s) {
  zt *ts;

  ts = (zt *) malloc(sizeof(zt));
  if (ts == 0) {
    CFL_ERROR_NULL("malloc failed for ts");
  }

  ts->matel = zhcsrsm_alloc(t->matel);
  if (ts->matel == 0) {
    free(ts);
    CFL_ERROR_NULL("alloc failed for ts->matel");
  }
  zhcsrsm(t->matel, ts->matel, s);

  ts->name = name;
  ts->n = t->n;
  ts->slabels = t->slabels;

  return ts;
} 
