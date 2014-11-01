/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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

#ifndef _BASINHOPPING_H_ 
#define _BASINHOPPING_H_

#include <gsl/gsl_rng.h>
#include "cfl_min.h"

/* Storage for minimum energy. */
typedef struct {
  /* Minimum energy. */
  double e;
  /* Parameter array corresponding to minimum energy state. */
  double *x;
} emin_t;

/* Stepsize struct. */
typedef struct {
  /* Pointer to stepsize array. */
  double *stepsize;
  /* Number of steps taken. */
  size_t nstep;
  /* Number of accepted steps. */
  float naccept;
  /* Target acceptance rate. */
  float target_accept_rate;
  /* Interval for how often to update stepsize. */
  size_t interval;
} bh_step_data;

/* Workspace allocation for basinhopping procedure. */
typedef struct {
  /* The number of parameters of the objective function. */
  size_t n;
  /* Internal storage for previous iteration parameter list. */
  double *x;
  /* The target number of iterations. */ 
  size_t niter;
  /* Pointer to local minimization cfl_min_obj. */
  cfl_min_obj *lmin_obj;
  /* Pointer to data holding stepsize information. */
  bh_step_data *step_data;
  /* Pointer to parameter bounds. */
  cfl_min_bounds *bounds;
  /* The current energy. */
  double e;
  /* Lowest energy state found. */
  emin_t *emin;
  /* Random number generator. */
  gsl_rng *rng; 
} bh_work;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
bh_work *bh_work_alloc(size_t niter, double *stepsize, float target_accept_rate,
    int step_adapt_int, cfl_min_obj *lmin_obj, cfl_min_bounds *bounds);
void bh_work_free(void *work);
int bh_min(double *x, double *fmin, void *work);
cfl_min_obj *cfl_bh_min_setup(size_t niter, double *stepsize, float target_accept_rate,
    int step_adapt_int, cfl_min_bounds *bounds, cfl_min_obj *lmin);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _BASINHOPPING_H_ */
