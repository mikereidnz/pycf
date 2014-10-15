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
 * Basinhopping algorithm, as described in Wales, D J, and Doye J P K, Journal
 * of Physical Chemistry A, 1997, 101, 5111. This implementation is based on the
 * python implementation by the SciPy community (scipy.optimize.basinhopping).
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>

#include <gsl/gsl_multimin.h>
#include "cfl_error.h"
#include "cfl_min_wrap.h"
#include "basinhopping.h"

/* 
 * Allocate workspace for the basinhopping procedure. 
 *
 * Parameters
 * ----------
 *
 *  n       The number of parameters to be varied.
 *  niter   The number of basinhopping iterations to complete.
 *  lmin_f  Pointer to the local minimization routine. 
 *  lmin_w  Pointer to the workspace for the local minimization routine.
 *  bounds  Pointer to a bounds object; in case of no bounds, pass a NULL
 *          pointer.
 */
bh_work *bh_work_alloc(size_t n, size_t niter, int (*lmin_f)(double *x, double *fmin, void *w), void *lmin_w, cfl_min_bounds *bounds) {
  int i;
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
  w->lmin_f = lmin_f;
  w->lmin_w = lmin_w;
  w->bounds = bounds;
  
  return w;
}

void bh_work_free(bh_work *w) {
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
    return 1;
  else 
    return 0;
}

/* Check that the boundary constraints have been satisfied. */
inline int bh_bounds_check(double *x, bh_work *w) {
  int i;
  int check = 0;
  
  for (i=0; i<w->n; i++) {
    if (x[i] > w->bounds->u[i] || x[i] < w->bounds->l[i]) {
      check++;
    }
  }

  if (check == 0)
    return 0;
  else
    return 1;
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
 *  fmin    Pointer to a single double; if successful, this will be overwritten
 *          with the objective function value for the best-fit parameters. 
 *  w       Pointer to the workspace allocated with bh_work_alloc. 
 */
int bh_min(double *x, double *fmin, void *work) {

  bh_work *w = (bh_work *) work;
  int i, status, test;
  size_t n = w->n;
  size_t lmin_fail = 0;
  double e;

  /* Perform initial minimization. */
  status = w->lmin_f(x, &e, w->lmin_w);
  if (status) {
    lmin_fail++;
  }
  w->e = e;
  memcpy(w->x, x, n*sizeof(double));
  w->emin->e = e;
  memcpy(w->emin->x, x, n*sizeof(double));
  
  for (i=0; i<w->niter; i++) {
    bh_takestep(x, w);
    status = w->lmin_f(x, &e, w->lmin_w);
    if (status) {
      lmin_fail++;
    }
    test = metropolis(w->T, e, w->e, w->rng);
    if (w->bounds != NULL) {
      test += bh_bounds_check(x, w);
    }
    if (test) {
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

/* Complete basinhopping fitting function, which manages memory and local
 * minimization function calls. 
 *
 * Parameters
 * ----------
 *  obj_f     Pointer to the objective function.
 *  x0        The initial parameter array; if the routine succeeds, this is
 *            overwritten with the result upon exit.
 *  nx        The number of parameters to be varied.
 *  data      Generic data to be passed to the objective function.
 *  niter     The number of basinhopping iterations to complete. 
 *  bounds    Pointer to a bounds object; in case of no bounds, pass a NULL
 *            pointer.
 *  lmintype  The local minimization type; implemented options are:
 *              + gsl_nmsimplex2rand
 *              + gsl_nmsimplex2 
 *              + gsl_conjugate_fr 
 *              + gsl_conjugate_pr
 *              + gsl_vector_bfgs2 
 */
int bh_fit(double (*obj_f)(size_t n, double *x, double *grad, void *data),
    double *x0, size_t nx, void *data, size_t niter, cfl_min_bounds *bounds, bh_lmin
    lmintype) {
  int status;
  double fmin;
  int (*lmin_f)(double *x, double *fmin, void *w);
  void (*lmin_work_free)(void *work);
  void *bh_lmin_w;
 
  switch (lmintype) {
    case gsl_nmsimplex2rand:
      bh_lmin_w =(void *) gsl_multimin_f_alloc(obj_f, nx, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      lmin_f = &gsl_multimin_f;
      lmin_work_free = gsl_multimin_f_free;
      break;
    case gsl_nmsimplex2:
      bh_lmin_w = (void *) gsl_multimin_f_alloc(obj_f, nx, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      lmin_f = &gsl_multimin_f;
      lmin_work_free = gsl_multimin_f_free;
      break;
    case gsl_conjugate_fr:
      bh_lmin_w = (void *) gsl_multimin_fndf_alloc(obj_f, nx, data,
          gsl_multimin_fdfminimizer_conjugate_fr);
      lmin_f = &gsl_multimin_fndf;
      lmin_work_free = gsl_multimin_fndf_free;
      break;
    case gsl_conjugate_pr:
      bh_lmin_w = (void *) gsl_multimin_fndf_alloc(obj_f, nx, data,
          gsl_multimin_fdfminimizer_conjugate_pr);
      lmin_f = &gsl_multimin_fndf;
      lmin_work_free = gsl_multimin_fndf_free;
      break;
    case gsl_vector_bfgs2:
      bh_lmin_w = (void *) gsl_multimin_fndf_alloc(obj_f, nx, data,
          gsl_multimin_fdfminimizer_vector_bfgs2);
      lmin_f = &gsl_multimin_fndf;
      lmin_work_free = gsl_multimin_fndf_free;
  }

  bh_work *bh_w;
  bh_w = bh_work_alloc(nx, niter, lmin_f, bh_lmin_w, bounds);
  status = bh_min(x0, &fmin, bh_w);
  bh_work_free(bh_w);

  lmin_work_free(bh_lmin_w);

  return status;
}
