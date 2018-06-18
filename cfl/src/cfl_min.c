/*
    Copyright (C) 2014-2018 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>

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
 * common minimization object type for all algorithms was chosen to provide a
 * large number of options for local minimization routines with the basinhopping
 * algorithm in addition to exposing a common interface to extensions linked to
 * cfl.  It is implemented by passing a pointer to the minimization function
 * along with a void data struct; the minimization function then casts the void
 * data to the data struct type appropriate to that min algorithm.
 *
 * All objective functions must be of the form double (*f)(size_t n, double *x,
 * double *grad, void *data), with n the number of parameters to be varied; x an
 * array of the parameters; and data can be any additional information the
 * function may require.  The function should return the objective function
 * value for the provided parameters.
 *
 * This file contains common cfl minimization functions, in addition to wrapping
 * both gsl and nlopt minimization algorithms.  See basinhopping.c for the
 * basinhopping implementation.  
 *
 * gsl multimin
 * ------------
 * There are two types of gsl multimin wrappers, denoted by the suffixes f, and
 * ndf.  These, respectively, stand for wrappers of derivative free routines,
 * and wrappers for gradient based routines with numerical derivative
 * estimation.  The execution routine consists of workspace allocation,
 * minimization, and workspace freeing.  Since gsl minimization routines require
 * custom data structures (gsl functions and vectors) there are dedicated
 * wrapper functions for objective functions following the cfl min argument
 * convention.  
 *
 * nlopt
 * -----
 * Since the cfl_min interface is quite similar to nlopt, the setup function
 * simply creates an nlopt object and sets the implemented optional parameters.
 *
 * gsl nonlinear least squares
 * ---------------------------
 * Layout describing how this section works: 
 *   + cfl_nls_setup returns a generic cfl_min_obj, the common data type for all
 *   of the optimization procedures.  The setup function also allocates
 *   workspace and performs some initializations. 
 *   + cfl_min_obj contains a pointer to the optimization function which is
 *   called when the object is passed to cfl_min; in this case, the optimization
 *   function is gsl_nls_f. 
 *   + gsl_nls_f calls the multifit driver, using the preallocated workspace and
 *   settings; these settings include information about the number of
 *   parameters, number of observables, the objective function to be called gsl
 *   multifit, as well as any data to be passed to the objective function.
 *   + the objective function is gsl_nls_f_wrapper, which knows how to handle
 *   the cfl_nls_data type to unpack which optimization function to call from
 *   cfl_h_fit, as well as pass data destined to that function.
 *
 * This complexity is necessary since we want the ability to run any of the
 * objective functions, with any of the optimization functions.  The further
 * (perhaps unjustifiable) abstraction of the minimization routine interface was
 * choosen to enable one to pass any of the local optimization routines to
 * basinhopping.  
 */
 
 
 


/* Wrapper for gsl minimization; used to construct a function of type
 * gsl_multimin_function. */
double gsl_multimin_f_wrapper(const gsl_vector *v, void *data) {
  int i;
  gsl_multimin_data *gsl_data = (gsl_multimin_data *)data;

  for (i=0; i<gsl_data->n; i++) {
    gsl_data->x[i] = gsl_vector_get(v, i);
  }

  return gsl_data->f(gsl_data->n, gsl_data->x, NULL, gsl_data->data);
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
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i],
        GSL_MIN_DERIV_H, &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i],
          GSL_MIN_DERIV_H, &result, &abserr);
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

  *f = gsl_data->f(gsl_data->n, gsl_data->x, NULL, gsl_data->data);

  /* Copy x to differentiation workspace to prevent x from being modified. */
  memcpy(gsl_data->df_work, gsl_data->x, gsl_data->n*sizeof(double));
  for (i=0; i<gsl_data->n; i++) {
    gsl_data->dfi = i;
    status = gsl_deriv_central(&(gsl_data->dfa[i]), gsl_data->x[i], GSL_MIN_DERIV_H,
        &result, &abserr);
    if (status) {
      gsl_vector_set(df, i, result);
    } 
    else {
      status = gsl_deriv_forward(&(gsl_data->dfa[i]), gsl_data->x[i],
          GSL_MIN_DERIV_H, &result, &abserr);
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
  gsl_f->params = gsl_data;

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
  gsl_f->params = gsl_data;

  for(i=0; i<n; i++) {
    dfa[i].function = &gsl_numerical_df_wrapper; 
    dfa[i].params = gsl_data;
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
    status = gsl_multimin_test_size(size, GSL_MIN_EPSABS);

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
  gsl_multimin_fdfminimizer_set(w->s, w->f, w->v, GSL_MIN_SS, GSL_MIN_TOL);
    do {
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(w->s);

      if (status)
        break;

      status = gsl_multimin_test_gradient(w->s->gradient, GSL_MIN_DERIV_EPSABS);
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
 * Generate cfl_min_obj settings object for gsl based minimization routines. 
 *
 * Parameters
 * ----------
 *  obj_f       Pointer to the objective function.
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
      *grad, void *data), size_t n, void *data, gsl_min_alg algorithm) {
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
      min_data = gsl_multimin_f_alloc(obj_f, n, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      min_f = &gsl_multimin_f;
      min_obj_free = gsl_multimin_f_free;
      break;
    case gsl_nmsimplex2:
      min_data = gsl_multimin_f_alloc(obj_f, n, data,
          gsl_multimin_fminimizer_nmsimplex2rand);
      min_f = &gsl_multimin_f;
      min_obj_free = gsl_multimin_f_free;
      break;
    case gsl_conjugate_fr:
      min_data = gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_conjugate_fr);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
      break;
    case gsl_conjugate_pr:
      min_data = gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_conjugate_pr);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
      break;
    case gsl_vector_bfgs2:
      min_data = gsl_multimin_fndf_alloc(obj_f, n, data,
          gsl_multimin_fdfminimizer_vector_bfgs2);
      min_f = &gsl_multimin_fndf;
      min_obj_free = gsl_multimin_fndf_free;
  }

  obj->min_data = min_data;
  obj->n = n;
  obj->min_f = min_f;
  obj->min_obj_free = min_obj_free;
  obj->obj_f_data = data;

  return obj;
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
 *  n           The number of parameters to be varied.
 *  data        Generic data to be passed to the objective function.
 *  algorithm   The minimization algorithm.  Implemented options are:
 *                + nlopt_cobyla
 *                + nlopt_bobyqa
 *                + nlopt_sbplx
 *  xtol        Stopping criteria for relative tolerance in parameters x.
 *              Criterion is disabled if non-positive.
 *  maxtime     Stopping criteria - maximum time in seconds (not absolute, may
 *              be slightly exceeded depnding on optimization function
 *              evaluation time.  Criterion is disabled if non-positive. 
 *  bounds      Linear bounds on the parameters.
 */
cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad,
      void *data), size_t n, void *data, nlopt_min_alg algorithm, double xtol,
    double maxtime, cfl_min_bounds *bounds) {
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

  nlopt_set_min_objective(opt, (nlopt_func)f, data);
  nlopt_set_xtol_rel(opt, xtol);
  nlopt_set_maxtime(opt, maxtime);

  obj->min_f = &nlopt_min_f;
  obj->n = n;
  obj->min_data = opt;
  obj->min_obj_free = nlopt_free;
  obj->obj_f_data = data;

  return obj;
}


/* Function to be called by GSL nonlinear least-squares to perform fit. */
int gsl_nls_f_wrapper(const gsl_vector *x, void *data, gsl_vector *y) {
  cfl_nls_data *d = (cfl_nls_data *) data;
  int i;
  
  /* These GSL vectors are a pain; they are vector views with memory blocks that
   * aren't contiguous (alloced by gsl_multifit_nlinear_winit), but all of the
   * cfl_h_fit machinery requires contiguous blocks... so we need to write/read
   * them into contiguous blocks. */
  
  for (i=0; i<d->p; i++) {
    d->x[i] = gsl_vector_get(x, i);
  }
  d->f(d->x, d->data, d->y);
  for (i=0; i<d->n; i++) {
    gsl_vector_set(y, i, d->y[i]);
  }

  return GSL_SUCCESS;
}


/* Function that actually runs the least-squares fit; assigned to min_f. */
int gsl_nls_f(double *x0, double *fmin, void *data) {
  int i, info, status;
  gsl_matrix *J;
  cfl_nls_data *d = (cfl_nls_data *) data;

  gsl_vector_view x = gsl_vector_view_array(x0, d->p);
  gsl_vector_view wts = gsl_vector_view_array(d->wts, d->n);

  gsl_multifit_nlinear_winit (&x.vector, &wts.vector, &(d->fdf), d->w);
  status = gsl_multifit_nlinear_driver(d->niter, d->xtol, d->gtol,
      d->ftol, NULL, NULL, &info, d->w);
  for (i=0; i<d->p; i++) {
    x0[i] = gsl_vector_get(d->w->x, i);
  }

  /* compute covariance of best fit parameters */
  J = gsl_multifit_nlinear_jac(d->w);
  gsl_matrix_view covar = gsl_matrix_view_array(d->covar, d->p, d->p);
  gsl_multifit_nlinear_covar(J, GSL_COV_EPSREL, &covar.matrix);
  
  return status;
}

void gsl_nls_free(void *data) {
  cfl_nls_data *d = (cfl_nls_data *) data;
  
  gsl_multifit_nlinear_free(d->w);
  free(d->x);
  free(d->y);
  free(d);
}


/*
 * Generate cfl_min_obj settings object for gsl based non-linear least squares. 
 *
 * Parameters
 * ----------
 *  obj_f               Pointer to the objective function.
 *  n                   The number of observables.
 *  p                   The number of parameters to be varied.
 *  data                Generic data to be passed to the objective function.
 *  wts                 Array of length n, specifying the weighting for each
 *                      observable.
 *  xtol, gtol, ftol    Tolerances; see GSL reference, section 39.8. 
 *  covar               Pointer to p*p array; will be overwritten by the
 *                      covariance matrix on exit.
 *  niter               The maximum number of iterations.
 */
cfl_min_obj *cfl_gsl_nls_setup(void (*f)(double *x, void *data, double *y), int n,
    int p, void *data, double *wts, double xtol, double gtol, double ftol,
    double *covar, int niter) {
  cfl_min_obj *obj; 
  cfl_nls_data *d;

  obj = (cfl_min_obj *) malloc(sizeof(cfl_min_obj));
  if (obj == 0) {
    CFL_ERROR_NULL("malloc failed for obj");
  }

  d = (cfl_nls_data *) malloc(sizeof(cfl_nls_data));
  if (d == 0) {
    free(obj);
    CFL_ERROR_NULL("malloc failed for d");
  }
  d->x = (double *) calloc(p,sizeof(double));
  if (d->x == 0) {
    free(obj);
    free(d);
    CFL_ERROR_NULL("malloc failed for d->x");
  }
  d->y = (double *) calloc(n,sizeof(double));
  if (d->y == 0) {
    free(obj);
    free(d->x);
    free(d);
    CFL_ERROR_NULL("malloc failed for d->y");
  }
  obj->min_f = &gsl_nls_f;
  obj->n = p;
  obj->min_data = d;
  obj->min_obj_free = &gsl_nls_free;
  obj->obj_f_data = data;

  d->T = gsl_multifit_nlinear_trust;
  d->fdf_params = gsl_multifit_nlinear_default_parameters();
  d->fdf_params.solver = gsl_multifit_nlinear_solver_svd;
  d->wts = wts;
  d->niter = niter;
  d->xtol = xtol;
  d->gtol = gtol;
  d->ftol = ftol;
  d->n = n;
  d->p = p;
  d->covar = covar;

  /* d->data is the data that will be passed to the actual objective function,
   * that is, cfl_h_fit.c functions.  We need to assign this to d here, since
   * the gsl fdf generic data field (fdf.params void ptr assigned below) has to
   * be occupied by d itself, which also contains a ptr to the objective
   * function to be call (this call is performed by gsl_nls_f_wrapper). */
  d->data = data;
  d->f = f;

  /* define the function to be minimized */
  d->fdf.f = gsl_nls_f_wrapper;
  d->fdf.df = NULL;   /* set to NULL for finite-difference Jacobian */
  d->fdf.fvv = NULL;     /* not using geodesic acceleration */
  d->fdf.n = n;
  d->fdf.p = p;
  d->fdf.params = d;

  /* allocate workspace with default parameters */
  d->w = gsl_multifit_nlinear_alloc (d->T, &(d->fdf_params), n, p);
  if (d->w == 0) {
    free(d);
    CFL_ERROR_NULL("gsl_multifit_nlinear_alloc failed for w");
  }
  
  return obj;
}

/*
 * Allocate workspace for simulated annealing. 
 *
 * Parameters
 * ----------
 *  f           Pointer to the objective function
 *  data        Data to be passed to the objective function. 
 *  n           The number of parameters to be varied. 
 *  niter       The total number of iterations to perform.
 *  bounds      Pointer to a bounds object; in case of no bounds, pass a NULL
 *              pointer.
 *  stepsize    Array of length n.  Multiplicative factor for stepsize of
 *              magnitude tan(M_PI*0.999*(u-0.5)), with u a random number in the
 *              interval (0...1], for each parameter in x.
 *  Tstart      The temperature to start for the simulated annealing cycle. 
 *  Tend        The stopping temperature. 
 *  maxtime     Stopping criteria - maximum time in seconds (not absolute, may
 *              be slightly exceeded depending on optimization function
 *              evaluation time.  Criterion is disabled if non-positive. 
 */
siman_data *siman_data_alloc(double (*f)(size_t n, double *x, double *grad, void
      *data), void *data, int n, int niter, cfl_min_bounds *bounds, double
    *stepsize, double Tstart, double Tend, double maxtime) {
  int i, j;
  double *xnew;
  siman_data *d;
  const gsl_rng_type *rt;

  d = (siman_data *) malloc(sizeof(siman_data));
  if (d == 0) {
    CFL_ERROR_NULL("malloc failed for d");
  }
  xnew = (double *) calloc(n,sizeof(double));
  if (xnew == 0) {
    free(d);
    CFL_ERROR_NULL("calloc failed for x");
  }

  gsl_rng_env_setup();
  rt = gsl_rng_default;
  d->rng = gsl_rng_alloc(rt);
  if (d->rng == 0) {
    free(d);
    free(xnew);
    CFL_ERROR_NULL("gsl_rng_alloc failed for rng");
  }

  d->x_accept = (double **) malloc(sizeof(double *)*niter);
  if (d->x_accept == 0) {
    free(xnew);
    gsl_rng_free(d->rng);
    free(d);
    CFL_ERROR_NULL("malloc failed for d->x_accept");
  }

  for (i=0; i<niter; i++) {
    d->x_accept[i] = (double *) malloc(sizeof(double)*n);
    if (d->x_accept[i] == 0) {
      for(j=0; j<i; j++) {
        free(d->x_accept[j]);
      }
      free(d->x_accept);
      gsl_rng_free(d->rng);
      free(xnew);
      free(d);
    }
  }
  
  d->f = f; 
  d->data = data; 
  d->xnew = xnew;
  d->n = n;
  d->niter = niter;
  d->bounds = bounds;
  d->stepsize = stepsize;
  d->Tstart = Tstart;
  d->Tend = Tend;
  d->maxtime = maxtime;
  
  return d;
}

void siman_data_free(void *data) {
  int i;
  siman_data *d = (siman_data *) data;
  for(i=0; i<d->niter; i++) {
    free(d->x_accept[i]);
  }
  free(d->x_accept);
  free(d->xnew);
  gsl_rng_free(d->rng);
  free(d);
}


/* Run simulated annealing optimization. */
int siman_f(double *x, double *fmin, void *data) {
  int i, j, naccept;
  double u, chi2, T, Tdec, delta, tmp_x;
  siman_data *d = (siman_data *) data;

  T = d->Tstart;
  
  Tdec = exp(log(d->Tend/d->Tstart)/(d->niter));
  d->accepted_chi2 = d->f(d->n, x, NULL, d->data);

  naccept = 0;
  for (i=0; i<d->niter; i++) {
    memcpy(d->xnew, x, sizeof(double)*d->n);
    
    u = gsl_rng_uniform(d->rng);
    j = (int) floor(u*d->n);

    u = gsl_rng_uniform(d->rng);

    delta = tan(M_PI*0.999*(u-0.5))*d->stepsize[j];
    if (d->bounds != NULL) {
      tmp_x = delta+d->xnew[j];
      while (!(tmp_x < d->bounds->u[j] && tmp_x > d->bounds->l[j])) {
        u = gsl_rng_uniform(d->rng);
        delta = tan(M_PI*0.999*(u-0.5))*d->stepsize[j];
        tmp_x = delta+d->xnew[j];
      }
    }
    d->xnew[j] += delta;   

    chi2 = d->f(d->n, x, NULL, d->data);
    
    u = gsl_rng_uniform(d->rng);
    if (u < exp((d->accepted_chi2-chi2)/(2*T))) {
        x[j] = d->xnew[j];
        memcpy(d->x_accept[naccept], x, sizeof(double)*d->n);
        d->accepted_chi2 = chi2;
        naccept++;
    }

    if (T > 1) {
        T *= Tdec; 
    }
  }
  d->naccept = naccept;

  return naccept;
}


/*
 * Generate cfl_min_obj settings object for simulated annealing optimization. 
 *  
 * Parameters
 * ----------
 *  obj_f       Pointer to the objective function.
 *  n           The number of parameters to be varied.
 *  data        Generic data to be passed to the objective function.
 *  niter       The number of iterations to perform. 
 *  bounds      Pointer to a bounds object; in case of no bounds, pass a NULL
 *              pointer.
 *  stepsize    Array of length n.  Multiplicative factor for stepsize of
 *              magnitude tan(M_PI*0.999*(u-0.5)), with u a random number in the
 *              interval (0...1], for each parameter in x.
 *  Tstart      The temperature to start for the simulated annealing schedule.  
 *  Tend        The stopping temperature. 
 *  maxtime     Stopping criteria - maximum time in seconds (not absolute, may
 *              be slightly exceeded depending on optimization function
 *              evaluation time.  Criterion is disabled if non-positive. 
 */
cfl_min_obj *cfl_siman_min_setup(double (*f)(size_t n, double *x, double *grad,
      void *data), size_t n, void *data, int niter, cfl_min_bounds *bounds,
    double *stepsize, double Tstart, double Tend, double maxtime) {
  cfl_min_obj *obj;
  siman_data *d;

  obj = (cfl_min_obj *) malloc(sizeof(cfl_min_obj));
  if (obj == 0) {
    CFL_ERROR_NULL("malloc failed for obj");
  }
  
  d = siman_data_alloc(f, data, n, niter, bounds, stepsize, Tstart, Tend, maxtime);
  if (d == 0) {
    free(obj);
    CFL_ERROR_NULL("malloc failed for d");
  }

  obj->min_f = &siman_f;
  obj->n = n;
  obj->min_data = d;
  obj->min_obj_free = &siman_data_free;
  obj->obj_f_data = data;
  
  return obj;
}


/* Generic freeing function, to be used for all objects of type cfl_min_obj. */
void cfl_min_free(cfl_min_obj *obj) {
   obj->min_obj_free(obj->min_data);
   free(obj);
}

/*
 * Perform minimization for a cfl_min_obj object. 
 *
 * Parameters
 * ----------
 *  x0      The starting values of the parameters to be fit. 
 *  fmin    Point to a double valued variable which will be overwritten with the
 *          objective function value upon return.
 *  obj     The cfl_min_obj for which to run the minimization.
 */
int cfl_min(double *x0, double *fmin, cfl_min_obj *obj) {
  int status;
  
  status = obj->min_f(x0, fmin, obj->min_data);

  return status;
}


