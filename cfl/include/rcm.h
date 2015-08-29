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

#ifndef _RCM_H_ 
#define _RCM_H_

#include "cfl_crs.h"

typedef struct {
  /* The degree of each node in the CRS matrix. */
  size_t *node_degrees;
  /* Mask used to keep track of which nodes we've traveresed. */
  size_t *mask;
} rcm_work;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
rcm_work *rcm_work_alloc(zcrs *m);
rcm_work_free(rcm_work *w);
void rcm(size_t *p, zcrs *m, rcm_work *w);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _RCM_H_ */
