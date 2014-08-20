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
    if (a[i]-b[i] >= pow(10,-8)) {
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

double test_f(size_t n, double *x, void *params) {
  double *p = (double *)params;
  
  return p[2] * (x[0] - p[0]) * (x[0] - p[0]) +
           p[3] * (x[1] - p[1]) * (x[1] - p[1]) + p[4]; 
}

int main (void)
{

  /*=========================================================================*/
  /* gsl Nelder-Mead simplex test.                                           */
  /*=========================================================================*/
  int status;
  double min_result[2] = {1.0096476861, 1.9991639022};
 
  /* Position of the minimum (1,2), scale factors 
     10,20, height 30. */
  double par[5] = { 1.0, 2.0, 10.0, 20.0, 30.0 };
  double x[2] = {5.0, 7.0};
  double fmin;

  gsl_multimin_work *work;
  work = gsl_multimin_alloc(&test_f, 2, par);

  status = gsl_multimin(x, &fmin, work);

  if (status) {
    printf("gsl minimization faild\n");
  }

  printf("gsl_multimin:\n");
  dequ_chk(min_result, x, 2);
  gsl_multimin_free(work);


  return 0;
}  
