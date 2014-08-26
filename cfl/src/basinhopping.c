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
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <cfl_error.h>
#include <basinhopping.h>


/* Utility function for copying arrays. */
inline void dacpy(double *a, double *b, size_t n) {
  int i;
  for (i=0; i<n; i++) {
    a[i] = b[i];
  }
}

/* Wrapper for gsl minimization; used to construct a function of type
 * gsl_multimin_function. */
double gsl_multimin_f_wrapper(const gsl_vector *v, void *data) {
  int i;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  return gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->grad, gsl_data->data);
}

/* Wrapper for gsl minimization with gradient based algorithms; returns the
 * gradient only. */
void gsl_multimin_df_wrapper(const gsl_vector *v, void *data, gsl_vector *df) {
  int i;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->grad, gsl_data->data);

  for (i=0; i<gsl_data->n; i++) {
    gsl_vector_set(df, i, gsl_data->grad[i]);
  }
}

/* Wrapper for gsl minimization with gradient based algorithms; returns the
 * gradient and the function value. */
void gsl_multimin_fdf_wrapper(const gsl_vector *v, void *data, double *f, gsl_vector *df) {
  int i;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  *f = gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->grad, gsl_data->data);

  for (i=0; i<gsl_data->n; i++) {
    gsl_vector_set(df, i, gsl_data->grad[i]);
  }
}

/* Wrapper for gsl minimization with gradient based algorithms; numerically
 * estimates the gradient, and returns it. */
void gsl_multimin_ndf_wrapper(const gsl_vector *v, void *data, gsl_vector *df) {
  int i, status;
  double result, abserr;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  /* Copy x to differentiation workspace to prevent x from being modified. */
  dacpy(gsl_data->df_work, gsl_data->x, gsl_data->n);
  for (i=0; i<gsl_data->n; i++) {
    gsl_data->dfi = i;
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i], 1e-9, &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i], 1e-9, &result, &abserr);
      gsl_vector_set(df, i, result);
    }
  }
}

/* Wrapper for gsl minimization with gradient based algorithms; numerically
 * estimates the gradient and returns it along with the function value. */
void gsl_multimin_fndf_wrapper(const gsl_vector *v, void *data, double *f, gsl_vector *df) {
  int i, status;
  double result, abserr;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  *f = gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->grad, gsl_data->data);

  /* Copy x to differentiation workspace to prevent x from being modified. */
  dacpy(gsl_data->df_work, gsl_data->x, gsl_data->n);
  for (i=0; i<gsl_data->n; i++) {
    gsl_data->dfi = i;
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i], 1e-9, &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i], 1e-9, &result, &abserr);
      gsl_vector_set(df, i, result);
    }
  }
}

/* Wrapper function for numerically calculating the derivative of an objective
 * function using gsl numerical derivative facilities. */
inline double gsl_numerical_df_wrapper(double x, void *data) {
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;
  
  gsl_data->df_work[gsl_data->dfi] = x;
  return gsl_data->f(gsl_data->n, gsl_data->df_work, NULL, gsl_data->data);
}

/*
 * Allocate workspace for using gsl_multimin with derivative free algorithms.
 *
 * Parameters
 * ----------
 *  f     The objective function with generic, gsl independent, arguments. 
 *  n     The number of arguments of f.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative free options are:
 *          + gsl_multimin_fminimizer_nmsimplex2
 *          + gsl_multimin_fminimizer_nmsimplex2rand
 */
gsl_multimin_f_work *gsl_multimin_f_alloc(double (*f)(size_t n, double *x, double *grad, void *data), size_t n, void *data, const gsl_multimin_fminimizer_type *T) {
  gsl_multimin_f_work *w;
  double *x;
  gsl_multimin_data *gsl_data;
  gsl_multimin_function *gsl_f;
  gsl_vector *v;
  gsl_vector *ssv;
  gsl_multimin_fminimizer *s;

  w = (gsl_multimin_f_work *) malloc(sizeof(gsl_multimin_f_work));
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

  s = gsl_multimin_fminimizer_alloc(T, n);
  gsl_vector_set_all(ssv, 1.0);
  w->epsabs = 1e-2;
  
  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->ssv = ssv;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_f_free(gsl_multimin_f_work *w) {
  free(w->gsl_data->x);
  gsl_multimin_fminimizer_free(w->s);
  free(w->f);
  gsl_vector_free(w->v);
  gsl_vector_free(w->ssv);
  free(w->gsl_data);
  free(w);
}


/*
 * Allocate workspace for using gsl_multimin with derivative based algorithms.
 *
 * Parameters
 * ----------
 *  f     The objective function with generic, gsl independent, arguments. 
 *  n     The number of arguments of f.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative based options are:
 *          + gsl_multimin_fdfminimizer_conjugate_fr
 *          + gsl_multimin_fdfminimizer_conjugate_pr
 *          + gsl_multimin_fdfminimizer_vector_bfgs2
 *          + gsl_multimin_fdfminimizer_steepest_descent
 */
gsl_multimin_fdf_work *gsl_multimin_fdf_alloc(double (*f)(size_t n, double *x, double *grad, void *data), size_t n, void *data, const gsl_multimin_fdfminimizer_type *T) {
  gsl_multimin_fdf_work *w;
  double *x;
  double *grad;
  gsl_multimin_data *gsl_data;
  gsl_multimin_function_fdf *gsl_f;
  gsl_vector *v;
  gsl_multimin_fdfminimizer *s;

  w = (gsl_multimin_fdf_work *) malloc(sizeof(gsl_multimin_fdf_work));
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
  grad = (double *) calloc(n,sizeof(double));
  if (x == 0) {
    free(w);
    free(gsl_data);
    free(x);
    CFL_ERROR_NULL("calloc failed for grad");
  } 

  gsl_data->f = f;
  gsl_data->n = n;
  gsl_data->x = x;
  gsl_data->grad = grad;
  gsl_data->data = data;

  gsl_f = (gsl_multimin_function_fdf *) malloc(sizeof(gsl_multimin_function_fdf));
  if (gsl_f == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    CFL_ERROR_NULL("malloc failed for gsl_f");
  }
  
  gsl_f->f = gsl_multimin_f_wrapper;
  gsl_f->df = gsl_multimin_df_wrapper;
  gsl_f->fdf = gsl_multimin_fdf_wrapper;
  gsl_f->n = n;
  gsl_f->params = (void *)gsl_data;

  v = gsl_vector_alloc(n);
  if (v == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    free(gsl_f);
    CFL_ERROR_NULL("gsl_vector_alloc failed for v");
  }

  s = gsl_multimin_fdfminimizer_alloc(T, n);
  w->ss = 0.01;
  w->tol = 1e-4;
  w->epsabs = 1e-3;

  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_fdf_free(gsl_multimin_fdf_work *w) {
  free(w->gsl_data->x);
  free(w->gsl_data->grad);
  gsl_multimin_fdfminimizer_free(w->s);
  free(w->f);
  gsl_vector_free(w->v);
  free(w->gsl_data);
  free(w);
}


/*
 * Allocate workspace for using gsl_multimin with derivative based algorithms
 * and numerical derivative estimation.
 *
 * Parameters
 * ----------
 *  f     The objective function with generic, gsl independent, arguments. 
 *  n     The number of arguments of f.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative based options are:
 *          + gsl_multimin_fdfminimizer_conjugate_fr
 *          + gsl_multimin_fdfminimizer_conjugate_pr
 *          + gsl_multimin_fdfminimizer_vector_bfgs2
 *          + gsl_multimin_fdfminimizer_steepest_descent
 */
gsl_multimin_fndf_work *gsl_multimin_fndf_alloc(double (*f)(size_t n, double *x, double *grad, void *data), size_t n, void *data, const gsl_multimin_fdfminimizer_type *T) {
  int i;
  gsl_multimin_fndf_work *w;
  double *x;
  double *grad;
  gsl_function *dfa;
  double *df_work;
  gsl_multimin_data *gsl_data;
  gsl_multimin_function_fdf *gsl_f;
  gsl_vector *v;
  gsl_multimin_fdfminimizer *s;

  w = (gsl_multimin_fndf_work *) malloc(sizeof(gsl_multimin_fndf_work));
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
  grad = (double *) calloc(n,sizeof(double));
  if (x == 0) {
    free(w);
    free(gsl_data);
    free(x);
    CFL_ERROR_NULL("calloc failed for grad");
  } 
  dfa = (gsl_function *) malloc(n*sizeof(gsl_function));
  if (dfa == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    CFL_ERROR_NULL("malloc failed for dfa");
  }
  df_work = (double *) calloc(n,sizeof(double));
  if (df_work == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    free(dfa);
    CFL_ERROR_NULL("calloc failed for df_work");
  }

  gsl_data->f = f;
  gsl_data->n = n;
  gsl_data->x = x;
  gsl_data->grad = grad;
  gsl_data->dfa = dfa;
  gsl_data->df_work = df_work;
  gsl_data->data = data;

  gsl_f = (gsl_multimin_function_fdf *) malloc(sizeof(gsl_multimin_function_fdf));
  if (gsl_f == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    free(dfa);
    free(df_work);
    CFL_ERROR_NULL("malloc failed for gsl_f");
  }
  
  gsl_f->f = gsl_multimin_f_wrapper;
  gsl_f->df = gsl_multimin_ndf_wrapper;
  gsl_f->fdf = gsl_multimin_fndf_wrapper;
  gsl_f->n = n;
  gsl_f->params = (void *)gsl_data;

  for(i=0; i<n; i++) {
    dfa[i].function = &gsl_numerical_df_wrapper; 
    dfa[i].params = (void *)gsl_data;
  }

  v = gsl_vector_alloc(n);
  if (v == 0) {
    free(w);
    free(gsl_data);
    free(x);
    free(grad);
    free(dfa);
    free(df_work);
    free(gsl_f);
    CFL_ERROR_NULL("gsl_vector_alloc failed for v");
  }

  s = gsl_multimin_fdfminimizer_alloc(T, n);
  w->ss = 0.01;
  w->tol = 1e-4;
  w->epsabs = 1e-3;

  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_fndf_free(gsl_multimin_fndf_work *w) {
  free(w->gsl_data->x);
  free(w->gsl_data->grad);
  free(w->gsl_data->dfa);
  free(w->gsl_data->df_work);
  gsl_multimin_fdfminimizer_free(w->s);
  free(w->f);
  gsl_vector_free(w->v);
  free(w->gsl_data);
  free(w);
}


/*
 * Run gsl_multimin, for derivative free minimization routines.  Any value
 * written to the grad pointer by an objective function will be neglected.  
 *
 * Parameters
 * ----------
 *  x     Pointer to the initial parameter estimates; if the optimization
 *        succeeds, this will be overwritten with the best-fit parameters.
 *  fmin  Poiter to a single double; if successful, this will be overwritten
 *        with the objective function value for the best-fit parameters. 
 *  work  Pointer to the workspace allocated with gsl_multimin_f_alloc. 
 */
int gsl_multimin_f(double *x, double *fmin, void *work) {
  size_t iter = 0;
  int i, status;
  double size;
  gsl_multimin_f_work *w = (gsl_multimin_f_work *)work;

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
    status = gsl_multimin_test_size(size, w->epsabs);

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

/*
 * Run gsl_multimin, for derivative based minimization routines.  The objective
 * function must write the derivative w.r.t. each variable at a given x to the
 * grad pointer. 
 *
 * Parameters
 * ----------
 *  x     Pointer to the initial parameter estimates; if the optimization
 *        succeeds, this will be overwritten with the best-fit parameters.
 *  fmin  Poiter to a single double; if successful, this will be overwritten
 *        with the objective function value for the best-fit parameters. 
 *  work  Pointer to the workspace allocated with gsl_multimin_fdf_alloc. 
 */
int gsl_multimin_fdf(double *x, double *fmin, void *work) {
  size_t iter = 0;
  int i, status;
  double size;
  gsl_multimin_fdf_work *w = (gsl_multimin_fdf_work *)work;

  /* Set initial parameters to gsl_vector. */
  for (i=0; i<w->gsl_data->n; i++) {
    gsl_vector_set(w->v, i, x[i]);
  }

  /* Run the minimization. */
  gsl_multimin_fdfminimizer_set(w->s, w->f, w->v, w->ss, w->tol);
    do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(w->s);

      if (status)
        break;

      status = gsl_multimin_test_gradient(w->s->gradient, w->epsabs);
    } while (status == GSL_CONTINUE && iter < 100);

  /* Set the solution to x and fmin. */
  for (i=0; i<w->gsl_data->n; i++) {
    x[i] = w->gsl_data->x[i];
  }
  *fmin = w->s->f;

  if (status == GSL_SUCCESS) 
    return 0;
  else 
    return 1;
}


/*
 * Run gsl_multimin, for derivative based minimization routines. Derivatives are
 * estimated numerically using the gsl_deriv_central and, in case of failure of
 * the central derivative, the gsl_deriv_forward functions.  Any result written
 * to the grad pointer of an objective function will be ignored. 
 *
 * Parameters
 * ----------
 *  x     Pointer to the initial parameter estimates; if the optimization
 *        succeeds, this will be overwritten with the best-fit parameters.
 *  fmin  Poiter to a single double; if successful, this will be overwritten
 *        with the objective function value for the best-fit parameters. 
 *  work  Pointer to the workspace allocated with gsl_multimin_fndf_alloc. 
 */
int gsl_multimin_fndf(double *x, double *fmin, void *work) {
  size_t iter = 0;
  int i, status;
  double size;
  gsl_multimin_fndf_work *w = (gsl_multimin_fndf_work *)work;

  /* Set initial parameters to gsl_vector. */
  for (i=0; i<w->gsl_data->n; i++) {
    gsl_vector_set(w->v, i, x[i]);
  }

  /* Run the minimization. */
  gsl_multimin_fdfminimizer_set(w->s, w->f, w->v, w->ss, w->tol);
    do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(w->s);

      if (status)
        break;

      status = gsl_multimin_test_gradient(w->s->gradient, w->epsabs);
    } while (status == GSL_CONTINUE && iter < 100);

  /* Set the solution to x and fmin. */
  for (i=0; i<w->gsl_data->n; i++) {
    x[i] = w->gsl_data->x[i];
  }
  *fmin = w->s->f;

  if (status == GSL_SUCCESS) 
    return 0;
  else 
    return 1;
}

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
bh_work *bh_work_alloc(size_t n, size_t niter, int (*lmin_f)(double *x, double *fmin, void *w), void *lmin_w, bh_bounds *bounds) {
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
 *  x       Pointer to the initial parameter list; if the routine succeeds, this
 *          is overwritten with the result upon exit.
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
  dacpy(w->x, x, n);
  w->emin->e = e;
  dacpy(w->emin->x, x, n);
  
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
      dacpy(w->x, x, n);
      w->step_data->naccept++;
      if (e < w->emin->e) {
        w->emin->e = e;
        dacpy(w->emin->x, x, n);
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
