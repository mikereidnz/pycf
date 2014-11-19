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

#ifndef _CFL_MIN_H_ 
#define _CFL_MIN_H_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <nlopt.h>

typedef enum {
  gsl_nmsimplex2rand = 0,
  gsl_nmsimplex2 = 1,
  gsl_conjugate_fr = 2,
  gsl_conjugate_pr = 3,
  gsl_vector_bfgs2 = 4
} gsl_min_alg;

typedef enum {
  nlopt_cobyla = 1,
  nlopt_bobyqa = 2, 
  nlopt_sbplx = 3, 
  nlopt_crs2_lm = 4,
  nlopt_esch = 5
} nlopt_min_alg;


/* Local minimization object. */
typedef struct cfl_min_obj {
  /* Pointer to the cfl minimization function.  Note: this is not the objective
   * function. */
  int (*min_f)(double *x, double *fmin, void *d);
  /* Number of parameters. */
  size_t n;
  /* Pointer to data required by min_f */
  void *min_data;
  /* Pointer to the cfl_min_obj object freeing function. */
  void (*min_obj_free)(void *obj);
  /* Pointer to data for minimization objective function; duplicated here since
   * nlopt hides it using an opaque pointer. */
  void *obj_f_data;
  /* Pointer to function that calculates the covariance matrix. */
  void (*cov_f)(double *x0, double *cov, struct cfl_min_obj *obj);
} cfl_min_obj;

/* Storage for optimization bounds. */
typedef struct {
  /* Lower bounds. */
  double *l;
  /* Upper bounds. */
  double *u;
} cfl_min_bounds;

/* Data type for gsl_min_wrapper; passed to the minimization wrapper which then
 * extracts the parameter data from the gsl_vector and calls the objective
 * function with gsl independent arguments.  Similarly, it is used for passing
 * data to the wrapper function that numerically estimates derivatives. */
typedef struct {
  /* Pointer to the objective function. */
  double (*f)(size_t n, double *x, double *grad, void *data); 
  /* Number of parameters. */
  size_t n;
  /* Pointer to parameter list. */
  double *x;
  /* Pointer to gradient list. */
  double *grad;
  /* Pointer to array of derivative functions. */
  gsl_function *dfa;
  /* Pointer to numerical differentiation workspace. */
  double *df_work;
    /* Index of variable w.r.t. which to differentiate. */
  size_t dfi;
  /* Data to be passed to the objective function. */
  void *data;
} gsl_multimin_data;


/* Work space allocation and initialization data type for gradient free gsl
 * multimin functions. */
typedef struct {
  /* Pointer to minimizer. */
  gsl_multimin_fminimizer *s;
  /* Objective function in gsl form. */
  gsl_multimin_function *f;
  /* Pointer to parameter gsl_vector. */
  gsl_vector *v;
  /* Pointer to step size gsl_vector. */
  gsl_vector *ssv;
  /* Pointer to gsl_min_wrapper data struct. */
  gsl_multimin_data *gsl_data;
} gsl_multimin_f_work;

/* Work space allocation and initialization data type for gradient based gsl
 * multimin functions. */
typedef struct {
  /* Pointer to minimizer. */
  gsl_multimin_fdfminimizer *s;
  /* Objective function in gsl form. */
  gsl_multimin_function_fdf *f;
  /* Pointer to parameter gsl_vector. */
  gsl_vector *v;
  /* Pointer to gsl_min_wrapper data struct. */
  gsl_multimin_data *gsl_data;
} gsl_multimin_fdf_work;

/* Work space allocation and initialization data type for gradient based gsl
 * multimin functions with numerical derivative estimation. */
typedef struct {
  /* Pointer to minimizer. */
  gsl_multimin_fdfminimizer *s;
  /* Objective function in gsl form. */
  gsl_multimin_function_fdf *f;
  /* Pointer to parameter gsl_vector. */
  gsl_vector *v;
  /* Pointer to gsl_min_wrapper data struct. */
  gsl_multimin_data *gsl_data;
} gsl_multimin_fndf_work;


/* Function prototypes. */
#ifdef __cplusplus
extern "C" { 
#endif /* __cplusplus */
gsl_multimin_f_work *gsl_multimin_f_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fminimizer_type *T); 
gsl_multimin_fdf_work *gsl_multimin_fdf_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fdfminimizer_type *T);
gsl_multimin_fndf_work *gsl_multimin_fndf_alloc(double (*f)(size_t n, double *x,
      double *grad, void *data), size_t n, void *data, const
    gsl_multimin_fdfminimizer_type *T);
void gsl_multimin_f_free(void *work);
void gsl_multimin_fdf_free(void *work);
void gsl_multimin_fndf_free(void *work);
int gsl_multimin_f(double *x, double *fmin, void *work);
int gsl_multimin_fdf(double *x, double *fmin, void *work);
int gsl_multimin_fndf(double *x, double *fmin, void *work);
cfl_min_obj *cfl_nlopt_min_setup(double (*f)(size_t n, double *x, double *grad,
      void *data), void (*cov_f)(double *x0, double *cov, struct cfl_min_obj
        *obj), size_t n, void *data, nlopt_min_alg algorithm, double xtol,
    cfl_min_bounds *bounds);
cfl_min_obj *cfl_gsl_min_setup(double (*obj_f)(size_t n, double *x, double
      *grad, void *data), void (*cov_f)(double *x0, double *cov, struct
    cfl_min_obj *obj), size_t n, void *data, gsl_min_alg algorithm);
int cfl_min(double *x0, double *fmin, double *cov, cfl_min_obj *obj);
void cfl_min_free(cfl_min_obj *obj);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_MIN_H_ */



