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

/*
 * Overview
 * ========
 * Basinhopping algorithm, as described in Wales, D J, and Doye J P K, Journal
 * of Physical Chemistry A, 1997, 101, 5111. This implementation is based on the
 * python implementation by the SciPy community (scipy.optimize.basinhopping).
 *
 * Depends on cfl_min data types.  Specifically, cfl_min_obj and cfl_min_bounds.
 * The easiest method to run an optimization is to create a cfl_min_obj object
 * using cfl_bh_min_setup and then passing it to cfl_min, prior to freeing it
 * with cfl_min_free (see cfl_min.c).  Alternatively, manually alloc workspace
 * and call bh_min.  
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_multimin.h>
#include "cfl_error.h"
#include "cfl_min.h"
#include "basinhopping.h"

/* 
 * Allocate workspace for the basinhopping procedure. 
 *
 * Parameters
 * ----------
 *
 *  n           The number of parameters to be varied.
 *  niter       The number of basinhopping iterations to complete.
 *  lmin_f      Pointer to the local minimization routine. 
 *  lmin_data   Pointer to the workspace for the local minimization routine.
 *  bounds      Pointer to a bounds object; in case of no bounds, pass a NULL
 *              pointer.
 */
bh_work *bh_work_alloc(size_t niter, cfl_min_obj *lmin_obj, cfl_min_bounds *bounds) {
  int i;
  size_t n = lmin_obj->n;
  bh_work *w;
  double *x;

  w = (bh_work *) malloc(sizeof(bh_work));
  if (w == 0) {
    CFL_ERROR_NULL("malloc failed for w");
  }
  x = (double *) calloc(n,sizeof(double));
  if (x == 0) {
    free(w);
    CFL_ERROR_NULL("calloc failed for x");
  }
  const gsl_rng_type *rt;
  gsl_rng_env_setup();
  rt = gsl_rng_default;
  w->rng = gsl_rng_alloc(rt);
  if (w->rng == 0) {
    free(w);
    free(x);
    CFL_ERROR_NULL("gsl_rng_alloc failed for rng");
  }
  w->emin = (emin_t *) malloc(sizeof(emin_t));
  if (w->emin == 0) {
    gsl_rng_free(w->rng);
    free(w);
    free(x);
    CFL_ERROR_NULL("malloc failed for emin");
  }
  w->emin->x = (double *) calloc(n,sizeof(double));
  if (w->emin->x == 0) {
    gsl_rng_free(w->rng);
    free(w->emin);
    free(w);
    free(x);
    CFL_ERROR_NULL("calloc failed for emin->x");
  }
  w->step_data = (bh_step_data *) malloc(sizeof(bh_step_data));
  if (w->step_data == 0) {
    gsl_rng_free(w->rng);
    free(w->emin);
    free(w->emin->x);
    free(w);
    free(x);
    CFL_ERROR_NULL("malloc failled for w->step_data");
  }
  w->step_data->stepsize = (double *) malloc(n*sizeof(double));
  if (w->step_data->stepsize == 0) {
    gsl_rng_free(w->rng);
    free(w->emin);
    free(w->emin->x);
    free(w->step_data);
    free(w);
    free(x);
    CFL_ERROR_NULL("malloc failed for w->step_data->stepsize");
  }

  /* Initialize parameters to defaults. */
  for (i=0; i<n; i++) {
    w->step_data->stepsize[i] = 0.5;
  }
  w->T = 1.0;
  w->step_data->nstep = 0;
  w->step_data->naccept = 0;
  w->step_data->target_accept_rate = 0.5;
  w->step_data->interval = 20;
  w->step_data->factor = 0.9;

  w->x = x;
  w->n = n;
  w->niter = niter;
  w->lmin_obj = lmin_obj;
  w->bounds = bounds;
  
  return w;
}

void bh_work_free(void *work) {
  bh_work *w = (bh_work *) work;
  free(w->x);
  gsl_rng_free(w->rng);
  free(w->emin->x);
  free(w->emin);
  free(w->step_data->stepsize);
  free(w->step_data);
  free(w);
}


/* The Metropolis criterion. */ 
inline int metropolis(double T, double e_new, double e_old, gsl_rng *r) {
  double p, u;
  p = fmin(1, exp(-(e_new - e_old)/T));
  u = gsl_rng_uniform(r);

  if (p>=u) 
    return 0;
  else 
    return 1;
}

/* Check that the boundary constraints have been satisfied. */
inline int bh_bounds_check(double *x, bh_work *w) {
  int i;
  
  for (i=0; i<w->n; i++) {
    if (x[i] > w->bounds->u[i] || x[i] < w->bounds->l[i]) {
      return 1;
    }
  }

  return 0;
}

/* Set the stepsize manually.  To disable adaptive stepsize adjustment, set
 * accept_rate, interval and factor to 0. 
 */
void bh_set_step(bh_work *w, double *stepsize, float target_accept_rate,
    size_t interval, float factor) {
  w->step_data->stepsize;
  w->step_data->target_accept_rate;
  w->step_data->interval;
  w->step_data->factor;
}

/* Add a random number in range [0, stepsize) to w->x and assign to x. */
inline void bh_rnd_disp(double *x, bh_work *w) {
  int i;

  for (i=0; i<w->n; i++) {
    x[i] = w->x[i] * gsl_rng_uniform(w->rng)*w->step_data->stepsize[i];
  }
}

/* Take a basinhopping step; checks whether adaptive stepsize is enabled, and,
 * if so, adjust the stepsize to meet the set target_accept_rate every interval
 * number of steps. */
inline void bh_takestep(double *x, bh_work *w) {
  int i;
  float accept_rate;

  if (w->step_data->target_accept_rate == 0) {
    /* We're not using adaptive stepsize. */
    bh_rnd_disp(x, w);
  }
  else {
    w->step_data->nstep++;
    if (w->step_data->nstep % w->step_data->interval == 0) {
      accept_rate = w->step_data->naccept/w->step_data->nstep;
      if (accept_rate > w->step_data->target_accept_rate) {
        /* We're accepting too many steps; increase the stepsize to escape
         * the basin. */
        for (i=0; i<w->n; i++) {
          w->step_data->stepsize[i] /= w->step_data->factor;
        }
      } 
      else {
        /* We're accepting too few steps; decrease the stepsize. */
        for (i=0; i<w->n; i++) {
          w->step_data->stepsize[i] *= w->step_data->factor;
        }
      }
    }
    bh_rnd_disp(x, w);
  }
}


/*
 * The basinhopping routine. 
 *
 * Parameters
 * ---------- 
 *  x       The initial parameter array; if the routine succeeds, this is
 *          overwritten with the result upon exit.
 *  fmin    Pointer to a double valued variable; if successful, this will be
 *          overwritten with the objective function value for the best-fit
 *          parameters. 
 *  w       Pointer to the workspace allocated with bh_work_alloc. 
 */
int bh_min(double *x, double *fmin, void *work) {

  bh_work *w = (bh_work *) work;
  int i, status, test;
  size_t n = w->n;
  size_t lmin_fail = 0;
  double e;

  /* Perform initial minimization. */
  status = w->lmin_obj->min_f(x, &e, w->lmin_obj->min_data);
  if (status) {
    lmin_fail++;
  }
  w->e = e;
  memcpy(w->x, x, n*sizeof(double));
  w->emin->e = e;
  memcpy(w->emin->x, x, n*sizeof(double));
  
  for (i=0; i<w->niter; i++) {
    bh_takestep(x, w);
    status = w->lmin_obj->min_f(x, &e, w->lmin_obj->min_data);
    if (status) {
      lmin_fail++;
    }
    test = metropolis(w->T, e, w->e, w->rng);
    if (w->bounds != NULL) {
      test += bh_bounds_check(x, w);
    }
    if (test == 0) {
      w->e = e;
      memcpy(w->x, x, n*sizeof(double));
      w->step_data->naccept++;
      if (e < w->emin->e) {
        w->emin->e = e;
        memcpy(w->emin->x, x, n*sizeof(double));
      }
    }
  }

  /* Set the solution to x and fmin. */
  for (i=0; i<n; i++) {
    x[i] = w->emin->x[i];
  }
  *fmin = w->emin->e;

  return lmin_fail;
}

/* Generate cfl_min_obj settings object for the basinhopping minimization
 * routine.
 *
 * Parameters
 * ----------
 *  niter     The number of basinhopping iterations to complete. 
 *  bounds    Pointer to a bounds object; in case of no bounds, pass a NULL
 *            pointer.
 *  lmin      cfl_min_obj to be used for the local minimization.
 */
cfl_min_obj *cfl_bh_min_setup(size_t niter, cfl_min_bounds *bounds, cfl_min_obj *lmin) {

  bh_work *bh_w;
  cfl_min_obj *obj;
  bh_w = bh_work_alloc(niter, lmin, bounds);
  if (bh_w == 0) {
    CFL_ERROR_NULL("bh_work_alloc failed for bh_w");
  }
  
  obj = (cfl_min_obj *) malloc(sizeof(cfl_min_obj));
  if (obj == 0) {
    bh_work_free(bh_w);
    CFL_ERROR_NULL("malloc failed for obj");
  }
  obj->min_data = bh_w;
  obj->n = lmin->n;
  obj->min_f = &bh_min;
  obj->min_obj_free = bh_work_free;

  return obj;
}
