#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>

#include <basinhopping.h>


/*
 * @brief   Check the equality of two double valued arrays.
 *
 * @param[a]  Pointer to first array. 
 * @param[b]  Pointer to second array.
 * @param[n]  Length of arrays a and b.
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

double gsl_test_f(size_t n, double *x, void *params) {
  double *p = (double *)params;
  
  return p[2] * (x[0] - p[0]) * (x[0] - p[0]) +
           p[3] * (x[1] - p[1]) * (x[1] - p[1]) + p[4]; 
}

double bh_test_f1(size_t n, double *x, void *params) {
  double *p = (double *)params;

  return cos(p[0] * x[0] - p[1]) + (x[1] + p[2]) * x[1] + (x[0] + p[2]) * x[0] + 1.010876184442655;
};


double bh_test_f2(size_t n, double *x, void *params) {
  double *p = (double *)params;
  
  return cos(p[0] * x[0] - p[1]) + (x[0] + p[2]) * x[0] + cos(p[0] * x[1] - p[1]) + (x[1] + p[2]) * x[1] + x[0] * x[1] + 1.963879482144252;
}

int main (void)
{

  /*=========================================================================*/
  /* gsl Nelder-Mead simplex test.                                           */
  /*=========================================================================*/
  int status;
  double gsl_result[2] = {1.0, 1.99};
 
  /* Position of the minimum (1,2), scale factors 
     10,20, height 30. */
  double gsl_par[5] = {1.0, 2.0, 10.0, 20.0, 30.0};
  double gsl_x[2] = {1.0096476861, 1.9991639022};
  //double gsl_x[2] = {200.0, 7.0};
  double fmin;

  gsl_multimin_f_work *gsl_w;
  gsl_w = gsl_multimin_f_alloc(&gsl_test_f, 2, gsl_par);

  status = gsl_multimin_f(gsl_x, &fmin,(void *)gsl_w);

  if (status) {
    printf("gsl minimization failed\n");
  }

  printf("gsl_multimin_f:\n");
  dequ_chk(gsl_result, gsl_x, 2);
  gsl_multimin_f_free(gsl_w);

  /*=========================================================================*/
  /* basin hopping test.                                                     */
  /*=========================================================================*/

  double bh_result[2] = {-0.19415263, -0.19415263};
  double bh_par[3] = {14.5, 0.3, 0.2};
  double bh_x[2] =  {-20, 13};

  double bounds_l[2] = {-10, -10};
  double bounds_u[2] = {10, 10};
  bh_bounds bounds;

  bounds.l = bounds_l;
  bounds.u = bounds_u;

  /* NOTE: 
   * The 1d test taken from the scipy implementation passes with the gradient
   * free algorithm (which is what it was used to test in the scipy case), but
   * the 2d test fails (this was used to test a gradient based local
   * minimization example in scipy).  It seems like an issue with the local
   * minimization, rather than the basinhopping routine.  Also, the interval
   * option for the adaptive step size seems to be quite critical... this will
   * become especially relevant when we fit cf stuff, given every bh iteration
   * will be fiercely expensive.  
   */
  

  gsl_multimin_f_work *bh_multimin_w1;
  bh_multimin_w1 = gsl_multimin_f_alloc(&bh_test_f1, 2, bh_par);

  bh_work *bh_w1;
  bh_w1 = bh_work_alloc(&bh_test_f1, 2, bh_par, 300, NULL);
  status = bh_min(bh_x, &fmin, bh_w1, &gsl_multimin_f, (void *)bh_multimin_w1);
  printf("x0=%.6f, x1=%.6f, fmin=%.6f\n", bh_x[0], bh_x[1], fmin);
  bh_work_free(bh_w1);
  
  gsl_multimin_f_free(bh_multimin_w1);

  gsl_multimin_f_work *bh_multimin_w2;
  bh_multimin_w2 = gsl_multimin_f_alloc(&bh_test_f2, 2, bh_par);

  bh_work *bh_w2;
  bh_w2 = bh_work_alloc(&bh_test_f2, 2, bh_par, 300, NULL);
  status = bh_min(bh_x, &fmin, bh_w2, &gsl_multimin_f, (void *)bh_multimin_w2);
  printf("x0=%.6f, x1=%.6f, fmin=%.6f\n", bh_x[0], bh_x[1], fmin);
  bh_work_free(bh_w2);
  
  gsl_multimin_f_free(bh_multimin_w2);

  return 0;
}  
