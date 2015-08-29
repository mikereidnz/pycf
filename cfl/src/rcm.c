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


/* Implementation of the Reverse Cuthill McKee (RCM) ordering algorithm [1]. We
 * follow the qutip implentation [2]. 
 *
 * [1] E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric
 *     Matrices", ACM '69 Proceedings of the 1969 24th national conference,
 *     (1969).
 * [2] https://github.com/qutip/qutip/blob/master/qutip/cy/graph_utils.pyx
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cfl_error.h"
#include "cfl_crs.h"

#include "rcm.h"

/* Calculate the number of degrees of each node (that is, the number of non-zero
 * elements per row). 
 *
 * Parameters
 * ----------
 *  degrees     Array of length n, where n is the number of rows in m, which
 *              will be overwritten with the degree of each node.
 *  m           The sparse matrix in CRS form for which to calculate the degrees
 *              per node.
 *  max         Pointer to variable which will be overwritten by the highest
 *              degree node encountered.
 *  max_index   Pointer to variable which will be overwritten by the index of
 *              the highest degree node encountered.
 *
 */
inline void node_degrees(size_t *degrees, zcrs *m, size_t *max, size_t *max_index) {
  int i, j;

  for (i=0; i<zcrs->n; i++) {
    degrees[i] = (zcrs->row_ptr[i+1]-zcrs->row_ptr[i]);
    
    /* Add an additional degree if the row contains a non-zero diagonal element.
     */
    for (j=zcrs->row_ptr[i]; j<zcrs->row_ptr[i+1]; j++) {
      if (zcrs->col_in[j] == i) {
        degrees[i] += 1;
      }
    }
    if (degrees[i] > *max) {
      *max = degrees[i];
      *max_index = i;
    }
  }
}

/* Perform the RCM sort. 
 *
 * Parameters
 * ----------
 *  p       Array of length n, where n is the number of rows in m, which will be
 *          overwritten with the indices that will permute m to RCM form.
 *  m       The sparse matrix in CRS form for which to determine the RCM
 *          sorting.
 *  w       The work space allocated with rcm_work_alloc. 
 */
void rcm(size_t *p, zcrs *m, rcm_work *w) {
  int i, j, row, node;
  size_t max_node, max_node_index;
  
  node_degrees(w->node_degrees, m, &max_node, &max_node_index);

  node = 0;
  for (row=0; row<zcrs->n; row++) {
    //FIXME: require inds equivalent, since otherwise I don't have a seed
    //whenever I encounter a new block... better go sort some nodes.
    p[node] = max

  }

}
