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

#ifndef _CFL_MIN_WRAP_H_ 
#define _CFL_MIN_WRAP_H_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

/* Data type for gsl_min_wrapper; passed to the minimization wrapper which then
 * extracts the parameter data from the gsl_vector and calls the objective
 * function with gsl independent arguments.  Similarily, it is used for passing
 * data to the wrapper function that numerically estimatiates derivatives. */
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
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _CFL_MIN_WRAP_H_ */



