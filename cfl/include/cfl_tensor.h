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

#ifndef _CFL_TENSOR_H_
#define _CFL_TENSOR_H_

#include <cfl_crs.h>

/* State label type. */
typedef struct {
  /* The number of states. */
  size_t n;
  /* String identifying the type of label.  Valid characters are S, L, J, M, and
   * I, and their position in key specifies the location in each label. */
  char *key;
  /* Array of length l containing arrays of length strlen(key). */
  char **labels;
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
  sl *slabels;
  /* Pointer to the matrix elements stored in CRS form. */
  crs_zhm *matel;
} zt; 

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */

sl *sl_alloc(size_t n, char *key, char **labels);
void sl_free(sl *l);
zt *zt_alloc(char *name, complex double *a, size_t n, sl *slabels);
void zt_free(zt *t);
zt *zt_sa(char *name, zt *t1, zt *t2, complex double s1, complex double s2);
zt *zt_s(char *name, zt *t, complex double s);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_TENSOR_H_ */

