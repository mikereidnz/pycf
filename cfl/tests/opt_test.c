#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include "cfl_tensor.h"
#include "cfl_h.h"
#include "cfl_sh.h"

#include "cfl_min.h"
#include "basinhopping.h"
#include "cfl_h_fit.h"

/*
 * Check the equality of two complex valued arrays.
 *
 * Parameters
 * ----------
 * a  Pointer to first array. 
 * b  Pointer to second array.
 * n  Length of arrays a and b.
 *
 */
void zequ_chk(complex double *a, complex double *b, size_t n) {
  int i;
  int p = 0;

  for (i=0; i<n; i++) {
    if (cabs(a[i]-b[i]) >= 1e-8) {
      p = 1;
    }
  }
  if (p==0) {
    printf("pass\n");
  }
  else {
    printf("fail\n");
  }
}

/*
 * Check the equality of two double valued arrays.
 *
 * Parameters
 * ----------
 * a  Pointer to first array. 
 * b  Pointer to second array.
 * n  Length of arrays a and b.
 *
 */
void dequ_chk(double *a, double *b, size_t n) {
  int i;
  int p = 0;

  for (i=0; i<n; i++) {
    if (a[i]-b[i] >= 1e-2) {
      p = 1;
    }
  }
  if (p==0) {
    printf("pass\n");
  }
  else {
    printf("fail\n");
  }
}

double gsl_test_f1(size_t n, double *x, double *grad, void *params) {
  double *p = (double *)params;

  return p[2] * (x[0] - p[0]) * (x[0] - p[0]) +
           p[3] * (x[1] - p[1]) * (x[1] - p[1]) + p[4]; 
}

double gsl_test_f2(size_t n, double *x, double *grad, void *params) {
  double *p = (double *)params;

  grad[0] = 2.0 * p[2] * (x[0] - p[0]);
  grad[1] = 2.0 * p[3] * (x[1] - p[1]); 
  
  return p[2] * (x[0] - p[0]) * (x[0] - p[0]) +
           p[3] * (x[1] - p[1]) * (x[1] - p[1]) + p[4]; 
}

double bh_test_f1(size_t n, double *x, double *grad, void *params) {
  double *p = (double *)params;

  return cos(p[0] * x[0] - p[1]) + (x[1] + p[2]) * x[1] + (x[0] + p[2]) * x[0] + 1.010876184442655;
};


double bh_test_f2(size_t n, double *x, double *grad, void *params) {
  double *p = (double *)params;
  
  return cos(p[0] * x[0] - p[1]) + (x[0] + p[2]) * x[0] + cos(p[0] * x[1] - p[1]) + (x[1] + p[2]) * x[1] + x[0] * x[1] + 1.963879482144252;
}


int main (void)
{

  /*=========================================================================*/
  /* gsl minimization routines test.                                         */
  /*=========================================================================*/
  int status;
  double gsl_result[2] = {1.0, 2.0};
 
  /* Position of the minimum (1,2), scale factors 10, 20, height 30. */
  double gsl_par[5] = {1.0, 2.0, 10.0, 20.0, 30.0};
  double gsl_x1[2] = {10.0, -5.0};
  double gsl_x2[2] = {10.0, -5.0};
  double gsl_x3[2] = {10.0, -5.0};
  double fmin;

  /* Derivative free Nelder-Mead simplex. */
  gsl_multimin_f_work *gsl_w1;
  gsl_w1 = gsl_multimin_f_alloc(&gsl_test_f1, 2, gsl_par, gsl_multimin_fminimizer_nmsimplex2);

  status = gsl_multimin_f(gsl_x1, &fmin, (void *)gsl_w1);

  if (status) {
    printf("Warning: gsl_multimin_f minimization failure\n");
  }

  printf("gsl_multimin_f:\n");
  dequ_chk(gsl_result, gsl_x1, 2);
  gsl_multimin_f_free(gsl_w1);

  /* Fletcher-Reeves conjugate gradient algorithm. */
  gsl_multimin_fdf_work *gsl_w2;
  gsl_w2 = gsl_multimin_fdf_alloc(&gsl_test_f2, 2, gsl_par, gsl_multimin_fdfminimizer_conjugate_fr);

  status = gsl_multimin_fdf(gsl_x2, &fmin, (void *)gsl_w2);

  if (status) {
    printf("Warning: gsl_multimin_fdf minimization failure\n");
  }

  printf("gsl_multimin_fdf:\n");
  dequ_chk(gsl_result, gsl_x2, 2);
  gsl_multimin_fdf_free(gsl_w2);

  /* Vector Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm with numerical
   * derivative estimation. */
  gsl_multimin_fndf_work *gsl_w3;
  gsl_w3 = gsl_multimin_fndf_alloc(&gsl_test_f1, 2, gsl_par, gsl_multimin_fdfminimizer_vector_bfgs2);

  status = gsl_multimin_fndf(gsl_x3, &fmin, (void *)gsl_w3);

  if (status) {
    printf("Warning: gsl_multimin_fndf minimization failure\n");
  }

  printf("gsl_multimin_fndf:\n");
  dequ_chk(gsl_result, gsl_x3, 2);
  gsl_multimin_fndf_free(gsl_w3);

  /*=========================================================================*/
  /* nlopt test.                                                             */
  /*=========================================================================*/
  double nlopt_x1[2] = {10.0, -5.0};
  cfl_min_obj *nlopt_min_obj;

  nlopt_min_obj = cfl_nlopt_min_setup(&gsl_test_f1, NULL, 2, gsl_par, nlopt_cobyla, 1e-6, NULL);
  status = cfl_min(nlopt_x1, &fmin, NULL, nlopt_min_obj);

  printf("nlopt cobyla:\n");
  dequ_chk(gsl_result, nlopt_x1, 2);
  cfl_min_free(nlopt_min_obj);

  /*=========================================================================*/
  /* basinhopping test.                                                      */
  /*=========================================================================*/

  double bh_result1[2] = {-0.19472980, -0.10130833};
  double bh_result2[2] = {-0.19415263, -0.19415263};
  double bh_par[3] = {14.5, 0.3, 0.2};
  double bh_x1[2] =  {-20, 13};
  double bh_x2[2] =  {-20, 13};

  double bounds_l[2] = {-10, -10};
  double bounds_u[2] = {10, 10};
  cfl_min_bounds bounds;

  bounds.l = bounds_l;
  bounds.u = bounds_u;

  /* Basinhopping test with derivative free gsl local minimization. */
  cfl_min_obj *lmin_obj1, *bhmin_obj1;

  lmin_obj1 = cfl_gsl_min_setup(&bh_test_f1, NULL, 2, bh_par, gsl_nmsimplex2);
  bhmin_obj1 = cfl_bh_min_setup(5, NULL, 0.5, 10, &bounds, lmin_obj1);
  status = cfl_min(bh_x1, &fmin, NULL, bhmin_obj1);
  printf("bh with gsl_nmsimplex2 local minimization:\n");
  dequ_chk(bh_result1, bh_x1, 2);

  cfl_min_free(bhmin_obj1);
  cfl_min_free(lmin_obj1);

  /* Basinhopping test with gsl local minimization using numerical derivative
   * gradient estimation. */
  cfl_min_obj *lmin_obj2, *bhmin_obj2;

  lmin_obj2 = cfl_gsl_min_setup(&bh_test_f2, NULL, 2, bh_par, gsl_vector_bfgs2);
  bhmin_obj2 = cfl_bh_min_setup(5, NULL, 0.5, 10, &bounds, lmin_obj2);
  status = cfl_min(bh_x2, &fmin, NULL, bhmin_obj2);
  printf("bh with gsl_vector_bfgs2 local minimization:\n");
  dequ_chk(bh_result2, bh_x2, 2);

  cfl_min_free(bhmin_obj2);
  cfl_min_free(lmin_obj2);


  /*=========================================================================*/
  /* h_fit test.                                                             */
  /*=========================================================================*/
  int i;
  /* Testing hamiltonian and spin hamiltonian fitting for Ce:LiYF4. Tensor
   * matrix elements and solutions externally calculated using pyemp. */
  complex double ce_eavg_a[196] = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0};

  complex double ce_zeta_a[196] = {1.50000015467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -2.00000020623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.50000015467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.00000020623, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.50000015467, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, -2.00000020623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1.50000015467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.00000020623,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.50000015467, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, -2.00000020623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1.50000015467, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2.00000020623, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.50000015467, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.50000015467};
  
  complex double ce_C20_a[196] = {-0.333333308417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -0.285714264357, 0.116642359985, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.116642359985, -0.0476190440595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.0571428528714, 0.0903507835368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0903507835368, 0.142857132179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.228571411486, 0.0329914414876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0329914414876, 0.238095220298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.228571411486, -0.0329914414876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0329914414876, 0.238095220298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0571428528714, -0.0903507835368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0903507835368, 0.142857132179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.285714264357, -0.116642359985, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.116642359985, -0.0476190440595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.333333308417};
  
  complex double ce_C40_a[196] = {0.0909089176865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0.0476189568834, -0.106038314953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -0.106038314953, -0.168830847132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -0.14285687065, 0.109515900766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.109515900766, -0.0389609647228, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0952379137668, 0.0749804115686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0749804115686, 0.116882894168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0952379137668, -0.0749804115686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0749804115686, 0.116882894168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.14285687065, -0.109515900766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.109515900766, -0.0389609647228, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0476189568834, 0.106038314953, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.106038314953, -0.168830847132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0909089176865};
  
  complex double ce_C44_a[196] = {0, 0, 0, 0, 0, 0, 0, 0.148453640934,
    0.128564624333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178173821773,
    -0.102442744767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15870361777,
    0.188199339398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178173821773,
    -0.15870361777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.102442744767,
    0.188199339398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.148453640934,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.128564624333, 0.148453640934, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.128564624333, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.178173821773, 0.15870361777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -0.102442744767, 0.188199339398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.178173821773, 0.102442744767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.15870361777, 0.188199339398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.148453640934, 0.128564624333, 0, 0, 0, 0, 0, 0, 0};
  
  complex double ce_C60_a[196] = {-0.0116550046289, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.0285488142907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0285488142907, 0.0582750231447, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, -0.110569082302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.110569082302,
    -0.10489504166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.201870601798,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.201870601798, 0.0582750231447, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.201870601798, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -0.201870601798, 0.0582750231447, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.110569082302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.110569082302, -0.10489504166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0285488142907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0285488142907,
    0.0582750231447, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0116550046289};
  
  complex double ce_C64_a[196] = {0, 0, 0, 0, 0, 0, 0, -0.127674178862,
    -0.110569082302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.185017462665, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.238856517219,
    0.0755330628389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.238856517219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.185017462665,
    0.0755330628389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127674178862,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.110569082302, -0.127674178862, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.110569082302, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0.238856517219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.185017462665, 0.0755330628389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, -0.185017462665, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.238856517219,
    0.0755330628389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127674178862,
    -0.110569082302, 0, 0, 0, 0, 0, 0, 0};
  
  complex double ce_magx_a[196] = {0, 0.463984001202, -1.51229633083, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.463984001202, 0, 0, -0.957944299092,
    -0.101249609846, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.51229633083, 0, 0,
    0.392138052742, -1.98006068835, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.957944299092, 0.392138052742, 0, 0, -1.21171434268, -0.175369468499, 0,
    0, 0, 0, 0, 0, 0, 0, -0.101249609846, -1.98006068835, 0, 0, 0.320179379315,
    -2.21377514936, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.21171434268,
    0.320179379315, 0, 0, -1.28521714286, -0.248009880777, 0, 0, 0, 0, 0, 0, 0,
    0, -0.175369468499, -2.21377514936, 0, 0, 0.248009880777, -2.28637714286, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, -1.28521714286, 0.248009880777, 0, 0,
    -1.21171434268, -0.320179379315, 0, 0, 0, 0, 0, 0, 0, 0, -0.248009880777,
    -2.28637714286, 0, 0, 0.175369468499, -2.21377514936, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, -1.21171434268, 0.175369468499, 0, 0, -0.957944299092,
    -0.392138052742, 0, 0, 0, 0, 0, 0, 0, 0, -0.320179379315, -2.21377514936, 0,
    0, 0.101249609846, -1.98006068835, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.957944299092, 0.101249609846, 0, 0, -0.463984001202, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -0.392138052742, -1.98006068835, 0, 0, -1.51229633083, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -0.463984001202, -1.51229633083, 0};
  
  complex double ce_magy_a[196] = {0, 0-0.463984001202*I, 0+1.51229633083*I, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0+0.463984001202*I, 0, 0, 0+0.957944299092*I,
    0+0.101249609846*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-1.51229633083*I, 0, 0,
    0-0.392138052742*I, 0+1.98006068835*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0-0.957944299092*I, 0+0.392138052742*I, 0, 0, 0+1.21171434268*I,
    0+0.175369468499*I, 0, 0, 0, 0, 0, 0, 0, 0, 0-0.101249609846*I,
    0-1.98006068835*I, 0, 0, 0-0.320179379315*I, 0+2.21377514936*I, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0-1.21171434268*I, 0+0.320179379315*I, 0, 0,
    0+1.28521714286*I, 0+0.248009880777*I, 0, 0, 0, 0, 0, 0, 0, 0,
    0-0.175369468499*I, 0-2.21377514936*I, 0, 0, 0-0.248009880777*I,
    0+2.28637714286*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-1.28521714286*I,
    0+0.248009880777*I, 0, 0, 0+1.21171434268*I, 0+0.320179379315*I, 0, 0, 0, 0,
    0, 0, 0, 0, 0-0.248009880777*I, 0-2.28637714286*I, 0, 0, 0-0.175369468499*I,
    0+2.21377514936*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-1.21171434268*I,
    0+0.175369468499*I, 0, 0, 0+0.957944299092*I, 0+0.392138052742*I, 0, 0, 0,
    0, 0, 0, 0, 0, 0-0.320179379315*I, 0-2.21377514936*I, 0, 0,
    0-0.101249609846*I, 0+1.98006068835*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0-0.957944299092*I, 0+0.101249609846*I, 0, 0, 0+0.463984001202*I, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0-0.392138052742*I, 0-1.98006068835*I, 0, 0,
    0+1.51229633083*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-0.463984001202*I,
    0-1.51229633083*I, 0};
  
  complex double ce_magz_a[196] = {4.00116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2.14202857143, 0.350738936998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.350738936998, 2.85797142857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.28521714286, 0.45280202062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.45280202062, 1.71478285714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.428405714286, 0.496019761555, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.496019761555, 0.571594285714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.428405714286, 0.496019761555, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.496019761555, -0.571594285714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1.28521714286, 0.45280202062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.45280202062, -1.71478285714, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -2.14202857143, 0.350738936998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.350738936998, -2.85797142857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -4.00116};

  complex double celiyf4_coeff[7] = {1535.12773615, 625.699030356,
    297.890587979, -1328.15222293, -1282.47659014, -191.510006575,
    -1743.14238515+692.866175947*I};
  
  double celiyf4_diag_res[14] = {1.475068, 1.475068, 213.798927, 213.798927,
    414.393675, 414.393675, 2215.463472, 2215.463472, 2312.050866, 2312.050866,
    2430.140843, 2430.140843, 3158.571302, 3158.571302};
  
  double ce_ex[6] = {0, 216, 2216.10, 2312.80, 2428.80, 3157.80};
  
  complex double ce_zeeman_inv[108] = {0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0-0.5*I, 0+0.5*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-0.5*I,
    0+0.5*I, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0-0.5*I, 0+0.5*I, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, -0.5};

  complex double ce_zeeman_term[12] = {0, 0.7365, 0.7365, 0, 0, -0.7365*I,
    0.7365*I, 0, 1.3825, 0, 0, -1.3825};
  
  complex double ce_gvalues[9] = {1.473, 0, 0, 0, 1.473, 0, 0, 0, 2.765};
 
  /* State label preparation. */
  int nstates = 14;
  char label_key[] = "SLJM";
  static int l_array[14][4] = {
    {1,3,7,7}, {1,3,5,5}, {1,3,7,5}, {1,3,5,3}, {1,3,7,3}, {1,3,5,1}, {1,3,7,1},
    {1,3,5,-1}, {1,3,7,-1}, {1,3,5,-3}, {1,3,7,-3}, {1,3,5,-5}, {1,3,7,-5},
    {1,3,7,-7}
  };

  int **l;
  l = (int **) malloc(nstates*sizeof(int *));
  if (l == 0) {
      printf("Error; label array **l malloc failed\n");
  }
  for (i=0; i<nstates; i++) {
    l[i] = l_array[i];
  }

  sl *states;
  states = sl_alloc(nstates, label_key, l);

  /* Tensor allocs. */
  zt *eavg, *zeta, *C20, *C40, *C44, *C60, *C64, *magx, *magy, *magz;
  eavg = (zt *) zt_alloc("eavg", ce_eavg_a, 14, states);
  zeta = (zt *) zt_alloc("zeta", ce_zeta_a, 14, states);
  C20 = (zt *) zt_alloc("C20", ce_C20_a, 14, states);
  C40 = (zt *) zt_alloc("C40", ce_C40_a, 14, states);
  C44 = (zt *) zt_alloc("C44", ce_C44_a, 14, states);
  C60 = (zt *) zt_alloc("C60", ce_C60_a, 14, states);
  C64 = (zt *) zt_alloc("C64", ce_C64_a, 14, states);
  magx = (zt *) zt_alloc("magx", ce_magx_a, 14, states);
  magy = (zt *) zt_alloc("magy", ce_magy_a, 14, states);
  magz = (zt *) zt_alloc("magz", ce_magz_a, 14, states);

  zt *tensors[7] = {eavg, zeta, C20, C40, C44, C60, C64};

  double *w;
  complex double *z;
  w = (double *) calloc(nstates,sizeof(double));
  z = (complex double *) calloc(nstates*nstates,sizeof(complex double));
  zh *h;
  zhd_w *hd_w;

  /* Check diagonalization routine. */
  h = zh_alloc(nstates, 7, tensors);
  zh_set_coeff(h, celiyf4_coeff);
  hd_w = zhd_w_alloc('N', w, z, h);
  //zhd(w, z, h, hd_w);
  zhd_w_free(hd_w);

  printf("Ce:LiYF4 diagonalization:\n");
  dequ_chk(celiyf4_diag_res, w, 14);
#if 0
  /* Inversion test. */
  zsh_inv_data celiyf4_inv_data;
  celiyf4_inv_data.a = ce_zeeman_inv;
  celiyf4_inv_data.m = 12;
  celiyf4_inv_data.n = 9;

  zshi_w *celiyf4_inv_work = zshi_w_alloc(&celiyf4_inv_data);
  zshi(ce_zeeman_term, celiyf4_inv_work);
  printf("Ce:LiYF4 inversion:\n");
  zequ_chk(ce_gvalues, ce_zeeman_term, 9);
  zshi_w_free(celiyf4_inv_work);
#endif
  /* Manually prepare array of parameter structs. */
  param_type efit_p0;
  efit_p0.type = 'r';
  efit_p0.index = 0;
  param_type efit_p1;
  efit_p1.type = 'r';
  efit_p1.index = 1;
  param_type efit_p2;
  efit_p2.type = 'r';
  efit_p2.index = 2;
  param_type efit_p3;
  efit_p3.type = 'r';
  efit_p3.index = 3;
  param_type efit_p4;
  efit_p4.type = 'r';
  efit_p4.index = 4;
  param_type efit_p5;
  efit_p5.type = 'r';
  efit_p5.index = 6;
  param_type **p = (param_type **) malloc(6*sizeof(param_type *));
  p[0] = &efit_p0;
  p[1] = &efit_p1;
  p[2] = &efit_p2;
  p[3] = &efit_p3;
  p[4] = &efit_p4;
  p[5] = &efit_p5;

  /* Set up the experimental data struct. */
  double ce_x0[7] = {2000, 900, 200, -1000, -1000, -100};
  ex_data ce_ex_data;
  int ex_index[6] = {1, 2, 7, 8, 11, 13};
  ce_ex_data.n = 6;
  ce_ex_data.e = ce_ex;
  ce_ex_data.li = ex_index;

  /* Run energy level fit. */
  efit_data *efit_d;
  cfl_min_obj *efit_lmin_obj, *efit_min_obj;
  
  efit_d = efit_data_alloc(h, celiyf4_coeff, &ce_ex_data, 6, p);
  efit_lmin_obj = cfl_gsl_min_setup(&efit_obj, &efit_cov, 6, efit_d, gsl_vector_bfgs2);
  efit_min_obj = cfl_bh_min_setup(1, NULL, 0.5, 10, NULL, efit_lmin_obj);

  double *cov;
  cov = malloc(36*sizeof(double));

  status = cfl_min(ce_x0, &fmin, cov, efit_min_obj);

  cfl_min_free(efit_min_obj);
  cfl_min_free(efit_lmin_obj);
  efit_data_free(efit_d);

  /* Alloc and free test for mev fit. */
  mev_data *mev_d1, *mev_d2;
  mevfit_data *mev_fd;
  
  double fs1[3] = {0, 1, 0};
  double fs2[3] = {1, 0, 1};
  int fi[3] = {6, 7, 8};

  mev_d1 = mev_data_alloc(h, 1.0, fs1, fi, &ce_ex_data);
  mev_d2 = mev_data_alloc(h, 0.5, fs2, fi, &ce_ex_data);
  mev_data *meva[2] = {mev_d1, mev_d2};

  mev_fd = mevfit_data_alloc(2, meva, celiyf4_coeff, 6, p);

  mevfit_data_free(mev_fd);
  mev_data_free(mev_d1);
  mev_data_free(mev_d2);
  zh_free(h);

  printf("Energy level only fit:\n");
  printf("fmin = %.6f\n", fmin);
  for (i=0; i<6; i++) {
    printf("%.5f\n", ce_x0[i]);
  }
  /* Spin Hamiltonian and energy level fit. */
  zt *sh_tensors[8] = {eavg, zeta, C20, C40, C44, C60, C64, magz};
  zt *shpro_tensors[3] = {magx, magy, magz};

  complex double sh_celiyf4_coeff[8] = {1535.1277, 625.6990, 297.8906,
    -1328.1522, -1282.4766, -191.5100, -1743.1424+692.8662*I, 0.0001};

  complex double *inv_a[] = {ce_zeeman_inv};
  char *inter[] = {"zeeman"};
  zsh *ce_sh;
  eshfit_data *eshfit_d;
  cfl_min_obj *eshfit_lmin_obj, *eshfit_min_obj;
  shx_data ce_zeeman_exp_data;


  ce_zeeman_exp_data.pa = ce_gvalues;
  ce_zeeman_exp_data.chisq_weight = 4e6;
  shx_data *shx[1] = {&ce_zeeman_exp_data};

  h = zh_alloc(14, 8, sh_tensors);
  zh_set_coeff(h, sh_celiyf4_coeff);

  ce_sh = zsh_alloc(inter, 1, 1, 0, inv_a);
  zsh_set_pro(ce_sh, shpro_tensors, 0);
  eshfit_d = eshfit_data_alloc(h, NULL, sh_celiyf4_coeff, &ce_ex_data, ce_sh,
      shx, 6, p);
  eshfit_lmin_obj = cfl_gsl_min_setup(&eshfit_obj, &eshfit_cov, 6, eshfit_d,
      gsl_vector_bfgs2);
  eshfit_min_obj = cfl_bh_min_setup(1, NULL, 0.5, 10, NULL, eshfit_lmin_obj);
  status = cfl_min(ce_x0, &fmin, cov, eshfit_min_obj);

  cfl_min_free(eshfit_min_obj);
  cfl_min_free(eshfit_lmin_obj);
  eshfit_data_free(eshfit_d);
  zsh_free(ce_sh);
  zh_free(h);
#if 0
  zsh *ce_x_sh, *ce_y_sh, *ce_z_sh;
  ce_x_sh = zsh_alloc(2, "magx");
  ce_y_sh = zsh_alloc(2, "magy");
  ce_z_sh = zsh_alloc(2, "magz");
  zsh_set_pro(ce_x_sh, magx, 0);
  zsh_set_pro(ce_y_sh, magy, 0);
  zsh_set_pro(ce_z_sh, magz, 0);
  zsh *sh_a[3] = {ce_x_sh, ce_y_sh, ce_z_sh};
  
  zsh_inv_data ce_inv_data;

  ce_inv_data.a = ce_zeeman_inv;
  ce_inv_data.m = 12;
  ce_inv_data.n = 9;
  shx_data ce_zeeman_exp_data;
  ce_zeeman_exp_data.pa = ce_gvalues;
  ce_zeeman_exp_data.chisq_weight = 4e6;
  ce_zeeman_exp_data.inv_data = &ce_inv_data;
  shx_data *shx[1] = {&ce_zeeman_exp_data};
  
  /* Testing objective function used when there are no spin Hamiltonian terms in
   * the projection Hamiltonian. */
  eshfit_data *eshfit_d;
  cfl_min_obj *eshfit_lmin_obj, *eshfit_min_obj;
  
  eshfit_d = eshfit_data_alloc(sh_a, 3, 0, h, NULL, celiyf4_coeff, &ce_ex_data,
      shx, 6, p);

  eshfit_lmin_obj = cfl_gsl_min_setup(&eshfit_obj, &eshfit_cov, 6, eshfit_d, gsl_vector_bfgs2);
  eshfit_min_obj = cfl_bh_min_setup(1, NULL, 0.5, 10, NULL, eshfit_lmin_obj);

  status = cfl_min(ce_x0, &fmin, cov, eshfit_min_obj);

  cfl_min_free(eshfit_min_obj);
  cfl_min_free(eshfit_lmin_obj);
  eshfit_data_free(eshfit_d);

  printf("Energy level and spin Hamiltonian fit (eshfit_obj):\n");
  printf("fmin = %.6f\n", fmin);
  for (i=0; i<6; i++) {
    printf("%.5f\n", ce_x0[i]);
  }

  /* Testing objective function used when spin Hamiltonian terms are also part
   * of the projection Hamiltonian. */
  eshfit_d = eshfit_data_alloc(sh_a, 3, 0, h, h, celiyf4_coeff, &ce_ex_data,
      shx, 6, p);

  eshfit_lmin_obj = cfl_nlopt_min_setup(&eshfit_hpro_obj, &eshfit_hpro_cov, 6, eshfit_d, nlopt_sbplx, 1e-6, NULL);
  eshfit_min_obj = cfl_bh_min_setup(1, NULL, 0.5, 10, NULL, eshfit_lmin_obj);

  status = cfl_min(ce_x0, &fmin, cov, eshfit_min_obj);

  cfl_min_free(eshfit_min_obj);
  cfl_min_free(eshfit_lmin_obj);


  printf("Energy level and spin Hamiltonian fit (eshfit_hpro_obj):\n");
  printf("fmin = %.6f\n", fmin);
  for (i=0; i<6; i++) {
    printf("%.5f\n", ce_x0[i]);
  }

  free(cov);
  zsh_free(ce_x_sh);
  zsh_free(ce_y_sh);
  zsh_free(ce_z_sh);
#endif

  free(w);
  free(z);

  free(p);
  zt_free(eavg);
  zt_free(zeta);
  zt_free(C20);
  zt_free(C40);
  zt_free(C44);
  zt_free(C60);
  zt_free(C64);
  zt_free(magx);
  zt_free(magy);
  zt_free(magz);
  sl_free(states);
  free(cov);
  free(l);
  
  return 0;
}  
