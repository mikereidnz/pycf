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

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <nlopt.h>

#include "cfl_config.h"
#include "cfl_error.h"
#include "basinhopping.h"
#include "cfl_min.h"

/* Overview
 * ========
 *
 * All implemented/wrapped minimization routines are most easily run by creating
 * a cfl_min_obj object and passing it to cfl_min to run the minimization.
 * There are different functions for creating cfl_min_obj objects, in
 * particular, one for wrapped gsl minimizations, one for wrapped nlopt
 * minimizations, and one for the basinhopping algorithm.  All cfl_min_obj
 * objects must be freed with a call to cfl_min_free.  This interface of a
 * common minimization object type for all algorithms was chosen so as to allow
 * one to easily test different local minimization routines with the
 * basinhopping algorithm in addition to exposing a common interface to
 * extensions linked to cfl.  It is implemented by passing a pointer to the
 * minimization function along with a void data struct; the minimization
 * function then casts the void data to the data struct type appropriate to that
 * min algorithm.
 *
 * All objective functions must be of the form double (*f)(size_t n, double *x,
 * double *grad, void *data), with n the number of parameters to be varied; x an
 * array of the parameters; grad an array that should be NULL for derivative
 * free (or numerical derivative estimation wrapper) functions, and overwritten
 * with the gradient for each variable upon function return; and data can be any
 * additional information the function may require.  The function should return
 * the objective function value for the provided parameters.
 *
 * This file contains common cfl minimization functions, in addition to wrapping
 * both gsl and nlopt minimization algorithms.  See basinhopping.c for the
 * basinhopping implementation.  
 *
 * gsl multimin
 * ------------
 * There are three types of gsl multimin wrappers, denoted by the suffixes f,
 * df, and ndf.  These, respectively, stand for wrappers of derivative free
 * routines, wrappers for gradient based routines that expect an objective
 * function that returns a derivative, and wrappers for gradient based routines
 * with numerical derivative estimation.  The execution routine consists of
 * workspace allocation, minimization, and workspace freeing.  Since gsl
 * minimization routines require custom data structures (gsl functions and
 * vectors) there are dedicated wrapper functions for objective functions
 * following the cfl min argument convention.  
 *
 * nlopt
 * -----
 * Since the nlopt interface is quite similar to cfl_min the setup function
 * simply creates an nlopt object and sets the implemented optional parameters.
 *
 */


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
void gsl_multimin_fdf_wrapper(const gsl_vector *v, void *data, double *f,
    gsl_vector *df) {
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
  memcpy(gsl_data->df_work, gsl_data->x, gsl_data->n*sizeof(double));
  for (i=0; i<gsl_data->n; i++) {
    gsl_data->dfi = i;
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i], GSL_DERIV_H,
        &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i],
          GSL_DERIV_H, &result, &abserr);
      gsl_vector_set(df, i, result);
    }
  }
}

/* Wrapper for gsl minimization with gradient based algorithms; numerically
 * estimates the gradient and returns it along with the function value. */
void gsl_multimin_fndf_wrapper(const gsl_vector *v, void *data, double *f,
    gsl_vector *df) {
  int i, status;
  double result, abserr;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  *f = gsl_data->f(gsl_data->n, gsl_data->x, gsl_data->grad, gsl_data->data);

  /* Copy x to differentiation workspace to prevent x from being modified. */
  memcpy(gsl_data->df_work, gsl_data->x, gsl_data->n*sizeof(double));
  for (i=0; i<gsl_data->n; i++) {
    gsl_data->dfi = i;
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i], GSL_DERIV_H,
        &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i],
          GSL_DERIV_H, &result, &abserr);
      gsl_vector_set(df, i, result);
    }
  }
}

/* Wrapper function for numerically calculating the derivative of an objective
 * function using gsl numerical derivative facilities. */
double gsl_numerical_df_wrapper(double x, void *data) {
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
 *  n     The number of parameters to be varied.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative free options are:
 *          + gsl_multimin_fminimizer_nmsimplex2
 *          + gsl_multimin_fminimizer_nmsimplex2rand
 */
gsl_multimin_f_work *gsl_multimin_f_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fminimizer_type *T) {
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
  
  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->ssv = ssv;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_f_free(void *work) {
  gsl_multimin_f_work *w = (gsl_multimin_f_work *) work;
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
 *  n     The number of parameters to be varied.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative based options are:
 *          + gsl_multimin_fdfminimizer_conjugate_fr
 *          + gsl_multimin_fdfminimizer_conjugate_pr
 *          + gsl_multimin_fdfminimizer_vector_bfgs2
 *          + gsl_multimin_fdfminimizer_steepest_descent
 */
gsl_multimin_fdf_work *gsl_multimin_fdf_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fdfminimizer_type *T) {
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

  gsl_f = (gsl_multimin_function_fdf *)malloc(sizeof(gsl_multimin_function_fdf));
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

  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_fdf_free(void *work) {
  gsl_multimin_fdf_work *w = (gsl_multimin_fdf_work *) work;
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
 *  n     The number of parameters to be varied.
 *  data  Generic data to be passed to f. 
 *  T     The type of optimization algorithm.  Derivative based options are:
 *          + gsl_multimin_fdfminimizer_conjugate_fr
 *          + gsl_multimin_fdfminimizer_conjugate_pr
 *          + gsl_multimin_fdfminimizer_vector_bfgs2
 *          + gsl_multimin_fdfminimizer_steepest_descent
 */
gsl_multimin_fndf_work *gsl_multimin_fndf_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fdfminimizer_type *T) {
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

  gsl_f = (gsl_multimin_function_fdf *)malloc(sizeof(gsl_multimin_function_fdf));
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

  w->s = s;
  w->f = gsl_f;
  w->v = v;
  w->gsl_data = gsl_data;
  
  return w;
}

void gsl_multimin_fndf_free(void *work) {
  gsl_multimin_fndf_work *w = (gsl_multimin_fndf_work *) work;
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
    status = gsl_multimin_test_size(size, GSL_EPSABS);

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
  gsl_multimin_fdfminimizer_set(w->s, w->f, w->v, GSL_SS, GSL_TOL);
    do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(w->s);

      if (status)
        break;

      status = gsl_multimin_test_gradient(w->s->gradient, GSL_DERIV_EPSABS);
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
  gsl_multimin_fdfminimizer_set(w->s, w->f, w->v, GSL_SS, GSL_TOL);
    do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(w->s);

      if (status)
        break;

      status = gsl_multimin_test_gradient(w->s->gradient, GSL_DERIV_EPSABS);
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

/* Wrapper for nlopt minimization. */
int nlopt_min_f(double *x, double *min, void *data) {
  return nlopt_optimize((nlopt_opt )data, x, min);
}

void nlopt_free(void *data) {
  nlopt_opt opt = (nlopt_opt )data;
  nlopt_destroy(opt);
}

/*
 * Generate cfl_min_obj settings object for nlopt based minimization routines. 
 *  
 * Parameters
 * ----------
 *  obj_f       Pointer to the objective function.
 *  cov_f       Function to calculate the covariance matrix; set to NULL if not
 *              required.
 *  n           The number of parameters to be varied.
 *  data        Generic data to be passed to the objective function.
 *  algorithm   The minimization algorithm.  Implemented options are:
 *                + nlopt_cobyla
 *                + nlopt_bobyqa
 *                + nlopt_sbplx
 *  xtol        Stopping criteria for relative tolerance in parameters x.
 *  bounds      Linear bounds on the parameters.
 */
cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad,
      void *data), void (*cov_f)(double *x0, double *cov, struct cfl_min_obj
      *obj), size_t n, void *data, nlopt_min_alg algorithm, double xtol,
      cfl_min_bounds *bounds) {
  cfl_min_obj *obj;
  nlopt_opt opt;

  obj = (cfl_min_obj *) malloc(sizeof(cfl_min_obj));
  if (obj == 0) {
    CFL_ERROR_NULL("malloc failed for obj");
  }
  switch (algorithm) {
    case nlopt_cobyla:
      opt = nlopt_create(NLOPT_LN_COBYLA, n);
      break;
    case nlopt_bobyqa:
      opt = nlopt_create(NLOPT_LN_BOBYQA, n);
      break;
    case nlopt_sbplx:
      opt = nlopt_create(NLOPT_LN_SBPLX, n);
      break;
    case nlopt_crs2_lm:
      opt = nlopt_create(NLOPT_GN_CRS2_LM, n);
      break;
    case nlopt_esch:
      opt = nlopt_create(NLOPT_GN_ESCH, n);
  }
  if (opt == 0) {
    free(obj);
    CFL_ERROR_NULL("nlopt_create failed for opt");
  }

  if (bounds != NULL) {
    nlopt_set_lower_bounds(opt, bounds->l);
    nlopt_set_upper_bounds(opt, bounds->u);
  }

  nlopt_set_min_objective(opt, (nlopt_func) f, data);
  nlopt_set_xtol_rel(opt, xtol);

  obj->min_f = &nlopt_min_f;
  obj->n = n;
  obj->min_data = (void *)opt;
  obj->min_obj_free = nlopt_free;
  obj->obj_f_data = data;
  obj->cov_f = cov_f;

  return obj;
}

/*
 * Generate cfl_min_obj settings object for gsl based minimization routines. 
 *
 * Parameters
 * ----------
 *  obj_f       Pointer to the objective function.
 *  cov_f       Function to calculate the covariance matrix; set to NULL if not
 *              required.
 *  n           The number of parameters to be varied.
 *  data        Generic data to be passed to the objective function.
 *  algorithm   The minimization algorithm; implemented options are:
 *              + gsl_nmsimplex2rand
 *              + gsl_nmsimplex2 
 *              + gsl_conjugate_fr 
 *              + gsl_conjugate_pr
 *              + gsl_vector_bfgs2
 */
cfl_min_obj *cfl_gsl_min_setup(double (*obj_f)(size_t n, double *x, double
      *grad, void *data), void (*cov_f)(double *x0, double *cov, struct
      cfl_min_obj *obj), size_t n, void *data, gsl_min_alg algorithm) {
  int (*min_f)(double *x, double *fmin, void *w);
  void (*min_obj_free)(void *obj);
  void *min_data;
  cfl_min_obj *obj;

  obj = (cfl_min_obj *) malloc(sizeof(cfl_min_obj));
  if (obj == 0) {
    CFL_ERROR_NULL("malloc failed for obj");
  }

  switch (algorithm) {
    case gsl_nmsimplex2rand:
      min_data =(void *) gsl_multimin_f_alloc(obj_f, n, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      min_f = &gsl_multimin_f;
      min_obj_free = gsl_multimin_f_free;
      break;
    case gsl_nmsimplex2:
      min_data = (void *) gsl_multimin_f_alloc(obj_f, n, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      min_f = &gsl_multimin_f;
      min_obj_free = gsl_multimin_f_free;
      break;
    case gsl_conjugate_fr:
      min_data = (void *) gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_conjugate_fr);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
      break;
    case gsl_conjugate_pr:
      min_data = (void *) gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_conjugate_pr);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
      break;
    case gsl_vector_bfgs2:
      min_data = (void *) gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_vector_bfgs2);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
  }

  obj->min_data = min_data;
  obj->n = n;
  obj->min_f = min_f;
  obj->min_obj_free = min_obj_free;
  obj->obj_f_data = data;
  obj->cov_f = cov_f;

  return obj;
}


/*
 * Perform minimization for a cfl_min_obj object. 
 *
 * Parameters
 * ----------
 *  x0      The starting values of the parameters to be fit. 
 *  fmin    Point to a double valued variable which will be overwritten with the
 *          objective function value upon return.
 *  cov     Pointer to space that will be overwritten with the covariance
 *          matrix; set to NULL to disable. 
 *  obj     The cfl_min_obj for which to run the minimization.
 */
int cfl_min(double *x0, double *fmin, double *cov, cfl_min_obj *obj) {
  int status;

  status = obj->min_f(x0, fmin, obj->min_data);

  if (cov != NULL) {
    if (obj->cov_f == NULL) {
      CFL_ERROR_VAL("Non NULL cov argument yet cov_f as not been specified", 1);
    }
    obj->cov_f(x0, cov, obj);
  }

  return status;
}

void cfl_min_free(cfl_min_obj *obj) {
   obj->min_obj_free(obj->min_data);
   free(obj);
}
  


