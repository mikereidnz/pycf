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
#include <cfl_error.h>
#include <cfl_crs.h>
#include <cfl_tensor.h>

/*
 * Implement the djb2 hash algorithm for array of char arrays.  For details
 * on djb2, see: http://www.cse.yorku.ca/~oz/hash.html
 *
 * Parameters
 * ----------
 *  n         The length of the array of char arrays.
 *  nc        The length of the char arrays.
 *  a         Array of char arrays to hash. 
 */
long state_hash(size_t n, size_t nc, char **a) {
  unsigned long hash = 5381;
  int i, j;
   
  for (i=0; i<n; i++) {
    for (j=0; j<nc; j++) {
      hash = ((hash << 5) + hash) + a[i][j];
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
 *  key       String identifying the type of each state label.  Valid keys are:
 *            S, L, J, M and I, and the order in which they are listed must
 *            correspond to the order used in the label array for each state. 
 *  labels    Char array corresponding to the value of each state label.
 *            The order is dictade by the key array.  To avoid half integers,
 *            label values are always stored as twice their real value.  N.B.:
 *            label arrays are not strings, since 0 is a perfectly valid state
 *            label yet would yield a premature string termination. 
 */
sl *sl_alloc(size_t n, char *key, char **labels) {
  sl *l;
  int i, j;
  size_t nl;

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
  strcpy(l->key, key);

  l->labels = (char **) malloc(n*sizeof(char *));
  if (l->labels == 0) {
    free(l->key);
    free(l);
    CFL_ERROR_NULL("malloc failed for l.labels");
  }

  for (i=0; i<n; i++) {
    l->labels[i] = (char *) malloc(nl*sizeof(char));
    if (l->labels[i] == 0) {
      for (j=0; j<i; j++) {
        free(l->labels[j]);
      }
      free(l->key);
      free(l->labels);
      free(l);
      CFL_ERROR_NULL("malloc failed for l.labels[i]");
    }
    memcpy(l->labels[i], labels[i], nl*sizeof(char));
  }
  
  l->hash = state_hash(n, nl, l->labels);
  l->n = n;
  
  return l;
}

void sl_free(sl *l) {
  int i;

  for (i=0; i<l->n; i++) {
    free(l->labels[i]);
  }
  free(l->key);
  free(l->labels);
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
zt *zt_alloc(char *name, complex double *a, size_t n, sl *slabels) {
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
  t->slabels = slabels;
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
zt *zt_sa(char *name, zt *t1, zt *t2, complex double s1, complex double s2) {
  zt *t;

  if (t1->n != t2->n) {
    CFL_ERROR_NULL("dimensions of tensors to be added do not match");
  }
  else if (t1->slabels->hash != t2->slabels->hash) {
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
  ts->slabels = t->slabels;

  return ts;
} 
