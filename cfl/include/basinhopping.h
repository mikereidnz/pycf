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


#ifndef _BASINHOPPING_H_ 
#define _BASINHOPPING_H_

#include <gsl/gsl_rng.h>
#include "cfl_min_wrap.h"

/*
 * Basinhopping algorithm, as described in Wales, D J, and Doye J P K, Journal
 * of Physical Chemistry A, 1997, 101, 5111. This implementation is based on the
 * python implementation by the SciPy community (scipy.optimize.basinhopping).
 */

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
  size_t naccept;
  /* Target acceptance rate. */
  float target_accept_rate;
  /* Interval for how often to update stepsize. */
  size_t interval;
  /* Multiplicative factor whereby the stepsize is updated if the target rate is
   * not being met. */
  float factor;
} bh_step_data;

/* Workspace allocation for basinhopping procedure. */
typedef struct {
  /* The number of parameters of the objective function. */
  size_t n;
  /* Internal storage for previous itteration parameter list. */
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
  /* The current temperature. */
  double T;
  /* Lowest energy state found. */
  emin_t *emin;
  /* Random number generator. */
  gsl_rng *rng; 
} bh_work;

/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
bh_work *bh_work_alloc(size_t niter, cfl_min_obj *lmin_obj, cfl_min_bounds *bounds);
void bh_work_free(void *work);
void bh_set_step(bh_work *w, double *stepsize, float target_accept_rate,
    size_t interval, float factor);
int bh_min(double *x, double *fmin, void *work);
cfl_min_obj *cfl_bh_min_setup(size_t niter, cfl_min_bounds *bounds, cfl_min_obj *lmin);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _BASINHOPPING_H_ */
