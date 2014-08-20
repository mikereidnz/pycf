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
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include <cfl_error.h>
#include <basinhopping.h>
/* ISSUES:
 *  + local minimization routines only handle doubles; need to split parameters
 *    in to real and complex components in a sensible way... 
 *  + The paramater array passed to cfl_h expects all parameters in the order
 *    tensors are added to the hamiltonian, yet we're only varying a subset of
 *    them.  We need to copy the parameters to be varied into their correct
 *    position of the parameter array... probably want to avoid simply pointing
 *    to the correct location to avoid aliasing issues with the fortran
 *    subroutines.  To write the real valued parameter list to the complex
 *    valued tensor prefactor list read the nth and n+1th value of the index
 *    array; if they match, the two values should be written as the real and
 *    complex part of the appropriate tensor prefactor list element.  If they
 *    don't match, write the values to the tensor prefactor list elements
 *    specified and increment the copying index by two. 
 *
 */

/*
 * Wrapper for gsl minimization; used to construct a function of type
 * gsl_multimin_function. 
 *
 * Parameters
 * ----------
 *  v     Parameter vector. 
 *  data  Is cast to type gsl_multimin_data.   
 */
double gsl_multimin_f_wrapper(const gsl_vector *v, void *data) {
  int i;
  double fval;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }
  
  return gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->data);
}

/*
 * Allocate workspace for using gsl_multimin, which employs the Nelder-Mead
 * Simplex algorithm to perform a local minimization. 
 *
 * Parameters
 * ----------
 *  f     The objective function with generic, gsl independent, arguments. 
 *  n     The number of arguments of f.
 *  data  Generic data to be passed to f. 
 */
gsl_multimin_work *gsl_multimin_alloc(double (*f)(size_t n, double *x, void *data), size_t n, void *data) {
  gsl_multimin_work *w;
  double *x;
  gsl_multimin_data *gsl_data;
  gsl_multimin_function *gsl_f;
  gsl_vector *v;
  gsl_vector *ssv;
  const gsl_multimin_fminimizer_type *T;
  gsl_multimin_fminimizer *s;

  w = (gsl_multimin_work *) malloc(sizeof(gsl_multimin_work));
  if (w == 0) {
    CFL_ERROR_NULL("malloc failed for w");
  }
  gsl_data = (gsl_multimin_data *) malloc(sizeof(gsl_multimin_data));
  if (gsl_data == 0) {
    free(w);
    CFL_ERROR_NULL("malloc failed for gsl_data");
  }
  x = (double *) calloc(n,sizeof(double));
  if (x == 0) {
    free(w);
    free(gsl_data);
    CFL_ERROR_NULL("calloc failed for x");
  } 

  gsl_data->f = f;
  gsl_data->n = n;
  gsl_data->x = x;
  gsl_data->data = data;

  gsl_f = (gsl_multimin_function *) malloc(sizeof(gsl_multimin_function));
  if (gsl_f == 0) {
    free(w);
    free(gsl_data);
    free(x);
    CFL_ERROR_NULL("malloc failed for gsl_f");
  }
  
  gsl_f->f = gsl_multimin_f_wrapper;
  gsl_f->n = n;
  gsl_f->params = (void *)gsl_data;

  v = gsl_vector_alloc(n);
  if (v == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(gsl_f);
    CFL_ERROR_NULL("gsl_vector_alloc failed for v");
  }
  ssv = gsl_vector_alloc(n);
  if (ssv == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(gsl_f);
    free(v);
    CFL_ERROR_NULL("gsl_vector_alloc failed for ssv");
  }

  T = gsl_multimin_fminimizer_nmsimplex2; 
  s = gsl_multimin_fminimizer_alloc(T, n);

  /* Initialize parameters. */
  gsl_vector_set_all(ssv, 1.0);
  
  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->ssv = ssv;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_free(gsl_multimin_work *w) {
  free(w->gsl_data->x);
  gsl_multimin_fminimizer_free(w->s);
  free(w->f);
  gsl_vector_free(w->v);
  gsl_vector_free(w->ssv);
  free(w->gsl_data);
  free(w);
}

/*
 * Run gsl_multimin, which employs the Nelder-Mead Simplex algorithm to perform
 * a local minimization.  In future, this should wrap both the derivative based
 * an derivative free gsl_multimin functions into a common interface. 
 *
 * Parameters
 * ----------
 *  x     Pointer to the initial parameter estimates; if the optimization
 *        succeeds, this will be overwritten with the best-fit parameters.
 *  fmin  Poiter to a single double; if successfull, this will be overwritten
 *        with the objective function value for the best-fit parameters. 
 *  w     Pointer to the workspace allocated with gsl_multimin_alloc. 
 */
int gsl_multimin(double *x, double *fmin, gsl_multimin_work *w) {
  size_t iter = 0;
  int i, status;
  double size;

  /* Set initial parameters to gsl_vector. */
  for (i=0; i<w->gsl_data->n; i++) {
    gsl_vector_set(w->v, i, x[i]);
  }

  /* Run the minimization. */
  gsl_multimin_fminimizer_set(w->s, w->f, w->v, w->ssv);
  do {
    iter++;
    status = gsl_multimin_fminimizer_iterate(w->s);

    if (status)
      break;

    /* Test for convergence. */
    size = gsl_multimin_fminimizer_size(w->s);
    status = gsl_multimin_test_size(size, 1e-2);

  } while (status == GSL_CONTINUE && iter < 100);

  /* Set the solution to x and fmin. */
  for (i=0; i<w->gsl_data->n; i++) {
    x[i] = w->gsl_data->x[i];
  }
  *fmin = w->s->fval;

  if (status == GSL_SUCCESS) 
    return 0;
  else 
    return 1;
}


/* Allocate workspace for the basinhopping procedure. */
bh_work *bh_work_alloc(double (*f)(size_t n, double *x, void *data), size_t n, void *data, size_t niter, bh_bounds *bounds) {
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
  w->lm_work = gsl_multimin_alloc(f, n, data);
  if (w->lm_work == 0) {
    free(w->rng);
    free(w);
    free(x);
    CFL_ERROR_NULL("gsl_multimin_alloc failed for lm_work");
  }
  w->emin = (emin_t *) malloc(sizeof(emin_t));
  if (w->emin == 0) {
    gsl_rng_free(w->rng);
    gsl_multimin_free(w->lm_work);
    free(w);
    free(x);
    CFL_ERROR_NULL("malloc failed for emin");
  }
  w->emin->x = (double *) calloc(n,sizeof(double));
  if (w->emin->x == 0) {
    gsl_rng_free(w->rng);
    gsl_multimin_free(w->lm_work);
    free(w->emin);
    free(w);
    free(x);
    CFL_ERROR_NULL("calloc failed for emin->x");
  }
  w->step_data = (bh_step_data *) malloc(sizeof(bh_step_data));
  if (w->step_data == 0) {
    gsl_rng_free(w->rng);
    gsl_multimin_free(w->lm_work);
    free(w->emin);
    free(w->emin->x);
    free(w);
    free(x);
    CFL_ERROR_NULL("malloc failled for w->step_data");
  }
  w->step_data->stepsize = (double *) calloc(n,sizeof(double));
  if (w->step_data->stepsize == 0) {
    gsl_rng_free(w->rng);
    gsl_multimin_free(w->lm_work);
    free(w->emin);
    free(w->emin->x);
    free(w->step_data);
    free(w);
    free(x);
    CFL_ERROR_NULL("calloc failed for w->step_data->stepsize");
  }

  /* Initialize parameters to defaults. */
  w->T = 1.0;
  w->step_data->nstep = 0;
  w->step_data->naccept = 0;
  w->step_data->target_accept_rate = 0.5;
  w->step_data->interval = 50;
  w->step_data->factor = 0.9;


  w->x = x;
  w->n = n;
  w->niter = niter;
  w->bounds = bounds;
  
  return w;
}

void bh_work_free(bh_work *w) {
  gsl_rng_free(w->rng);
  gsl_multimin_free(w->lm_work);
  free(w->emin->x);
  free(w->emin);
  free(w->step_data->stepsize);
  free(w->step_data);
  free(w);
}

inline void dacpy(double *a, double *b, size_t n) {
  int i;
  for (i=0; i<n; i++) {
    a[i] = b[i];
  }
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

/* Set the stepsize manually.  To disable adaptive stepsize adjustment, set
 * accept_rate, interval and factor to 0. 
 */
void bh_set_stepsize(bh_work *w, double *stepsize, float target_accept_rate,
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
 *  x     Pointer to the initial parameter list; if the routine succeeds, this
 *        is overwritten with the result upon exit.
 *  w     Pointer to the workspace allocated with bh_work_alloc. 
 */
void basinhopping(double *x, bh_work *w) {
  size_t n = w->n;
  int i, status, test;
  size_t lmin_fail = 0;
  double e;

  /* Perform initial minimization. */
  status = gsl_multimin(x, &e, w->lm_work);
  if (status) {
    lmin_fail++;
  }
  w->emin->e = e;
  dacpy(w->emin->x, w->x, n); 

  for (i=0; i<w->niter; i++) {
    bh_takestep(x, w);
    status = gsl_multimin(x, &e, w->lm_work);
    if (status) {
      lmin_fail++;
    }
    test = metropolis(w->T, e, w->e, w->rng);
    if (test) {
      w->e = e;
      dacpy(w->x, x, n);
      w->step_data->naccept++;
      if (e < w->emin->e) {
        w->emin->e = e;
        dacpy(w->emin->x, x, n);
      }
    }
  }
}
