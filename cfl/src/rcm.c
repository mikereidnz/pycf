/*
   Copyright (C) 2015 Sebastian Horvath (sebastian.horvath@gmail.com)

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


/* Implementation of the Reverse Cuthill McKee (RCM) ordering algorithm [1]
 * tailored to conveniently work with cfl_crs data types.  This implementation
 * is takes inspiration from the c implementation by David Fritzsche [2] and the
 * cython implementation of qutip [3]. 
 *
 *
 * [1] E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric
 *     Matrices", ACM '69 Proceedings of the 1969 24th national conference,
 *     (1969).
 * [2] https://math.temple.edu/~daffi/software/rcm/ 
 * [3] https://github.com/qutip/qutip/blob/master/qutip/cy/graph_utils.pyx
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_sort_int.h>

#include "cfl_error.h"
#include "cfl_crs.h"

#include "rcm.h"


/* Calculate the number of degrees of each node (that is, the number of non-zero
 * elements per row). 
 *
 * Parameters
 * ----------
 *  deg         Array of length n, where n is the number of rows in m, which
 *              will be overwritten with the degree of each node.
 *  m           The sparse matrix in CRS form for which to calculate the degrees
 *              per node.
 */
inline void degrees(int *deg, zcrs *m) {
  int i;

  printf("row_ptr\n");
  for (i=0; i<m->n; i++) {
    printf("%i ", m->row_ptr[i]);
  }
  printf("\n");

  for (i=0; i<m->n; i++) {
    deg[i] = (m->row_ptr[i+1] - m->row_ptr[i]);
  }
}


/* Integer index sort comparison function for quicksort. */
int int_index_cmp_fn(const void *a, const void *b) {
  const int_index_sort_t *ia = *(const int_index_sort_t **) a;
  const int_index_sort_t *ib = *(const int_index_sort_t **) b;

  return (ia->val > ib->val) - (ia->val < ib->val);
}


/* Insertion sort for rcm_sort; sorts edges of a given node according to their
 * labels. 
 *
 * Parameters
 * ----------
 *  sz    The size of the permutation array. 
 *  p     The permutation array to be sorted; overwritten with the result upon
 *        exit.
 *  deg   The degree of each node.
 */
inline void insertion_sort(int sz, int *p, int *deg) {
  int k, l;
  int nbr;

  for (k = sz-1; k > 0; k--) {
    nbr = p[k-1];
    for (l = k; l < sz && deg[p[l]] < deg[nbr]; l++) {
      p[l-1] = p[l];
    }
    p[l-1] = nbr;
  }
}

/* Reverse Cuthill McKee sorting function tailored to conveniently work with
 * cfl_crs data types.
 *
 * Parameters
 * ----------
 *  p       Array of length n, where n is the number of rows in m, which will be
 *          overwritten with the indices that will permute m to RCM form.
 *  m       The sparse matrix in CRS form for which to determine the RCM
 *          sorting.
 *  w       The work space allocated with rcm_work_alloc. 
 */
void rcm_sort(int *p, zcrs *m) {
  int i, j;
  int row, node, level_start, level_end, nbr, nbr_cc, nbr_cc_pr;
  int *deg;
  int_index_sort_t **mask;

  deg = (int *) calloc(m->n, sizeof(int));
  if (deg == 0) {
    CFL_ERROR_VOID("calloc failed for deg");
  }
  mask = (int_index_sort_t **) malloc(m->n*sizeof(int_index_sort_t *));
  if (mask == 0) {
    free(deg);
    CFL_ERROR_VOID("malloc failed for mask");
  }
  degrees(deg, m);
  for (i=0; i<m->n; i++) {
    mask[i] = (int_index_sort_t *) malloc(sizeof(int_index_sort_t));
    if (mask[i] == 0) {
      for (j=0; j<i; j++) {
        free(mask[j]);
      }
      free(mask);
      free(deg);
      CFL_ERROR_VOID("malloc failed for mask[i]");
    }
    mask[i]->index = i;
    mask[i]->val = deg[i];
  }
  
  printf("mask pre sort:\n");
  for (i=0; i<28; i++) {
    printf("%i ", mask[i]->index);
  }
  printf("\n");
  for (i=0; i<28; i++) {
    printf("%i ", deg[i]);
  }
  printf("\n");


  qsort(mask, m->n, sizeof(int_index_sort_t *), int_index_cmp_fn);
  printf("deg post sort:\n");
  for (i=0; i<28; i++) {
    printf("%i ", mask[i]->index);
  }
  printf("\n");

  row = 0;
  for (row=0; row<m->n; row++) {
    if (mask[row]->val != -1) {
      p[row] = mask[row]->index;
      mask[row]->val = -1; 
      level_start = row;
      level_end = row+1;
      row++;
      do {
        /* Iterate over all nodes in the current level */
        for (i=level_start; i<level_end; ++i) {
          node = p[i];
          printf("node=%i\n", node);
          /* Record the number of neighbors count in this set of connected
           * components for the last node. */
          nbr_cc_pr = nbr_cc;
          /* Find the non-masked neighbors of the current node. */
          for (j=m->row_ptr[node]; j<m->row_ptr[node+1]; j++) {
            nbr = m->col_in[j];
            if (mask[nbr]->val != -1) {
              mask[nbr]->val = -1;
              p[nbr_cc] = nbr;
              nbr_cc++;
            }
            printf("nbr_cc=%i\n", nbr_cc);
          }
          if (nbr_cc - nbr_cc_pr > 1) {
            insertion_sort(nbr_cc-nbr_cc_pr, p+nbr_cc_pr, deg);
          }
        }
        level_start = level_end;
        level_end = nbr_cc;
      } while (level_end - level_start > 0);
    }
  }
  free(deg);
  for (i=0; i<m->n; i++) {
    free(mask[i]);
  }
  free(mask);
}
