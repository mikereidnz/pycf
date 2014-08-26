#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <basinhopping.h>
#include <cfl_tensor.h>
#include <cfl_h.h>
#include <cfl_sh.h>
#include <test_data.h>

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

  grad[0] = -p[0] * sin(p[0] * x[0] - p[1]) + 2. * x[0] + p[2] + x[1];
  grad[1] = -p[0] * sin(p[0] * x[1] - p[1]) + 2. * x[1] + p[2] + x[0];
  
  return cos(p[0] * x[0] - p[1]) + (x[0] + p[2]) * x[0] + cos(p[0] * x[1] - p[1]) + (x[1] + p[2]) * x[1] + x[0] * x[1] + 1.963879482144252;
}

double bh_test_f3(size_t n, double *x, double *grad, void *params) {
  double *p = (double *)params;
  
  return cos(p[0] * x[0] - p[1]) + (x[0] + p[2]) * x[0] + cos(p[0] * x[1] - p[1]) + (x[1] + p[2]) * x[1] + x[0] * x[1] + 1.963879482144252;
}


int main (void)
{

  /*=========================================================================*/
  /* gsl Nelder-Mead simplex test.                                           */
  /*=========================================================================*/
  int status;
  double gsl_result[2] = {1.0, 2.0};
 
  /* Position of the minimum (1,2), scale factors 10, 20, height 30. */
  double gsl_par[5] = {1.0, 2.0, 10.0, 20.0, 30.0};
  double gsl_x1[2] = {10.0, -5.0};
  double gsl_x2[2] = {10.0, -5.0};
  double gsl_x3[2] = {10.0, -5.0};
  double fmin;

  gsl_multimin_f_work *gsl_w1;
  gsl_w1 = gsl_multimin_f_alloc(&gsl_test_f1, 2, gsl_par, gsl_multimin_fminimizer_nmsimplex2);

  status = gsl_multimin_f(gsl_x1, &fmin, (void *)gsl_w1);

  if (status) {
    printf("Warning: gsl_multimin_f minimization failure\n");
  }

  printf("gsl_multimin_f:\n");
  dequ_chk(gsl_result, gsl_x1, 2);
  gsl_multimin_f_free(gsl_w1);

  gsl_multimin_fdf_work *gsl_w2;
  gsl_w2 = gsl_multimin_fdf_alloc(&gsl_test_f2, 2, gsl_par, gsl_multimin_fdfminimizer_conjugate_fr);

  status = gsl_multimin_fdf(gsl_x2, &fmin, (void *)gsl_w2);

  if (status) {
    printf("Warning: gsl_multimin_fdf minimization failure\n");
  }

  printf("gsl_multimin_fdf:\n");
  dequ_chk(gsl_result, gsl_x2, 2);
  gsl_multimin_fdf_free(gsl_w2);

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
  /* basinhopping test.                                                      */
  /*=========================================================================*/

  double bh_result1[2] = {-0.19472980, -0.10130833};
  double bh_result2[2] = {-0.19415263, -0.19415263};
  double bh_result3[2] = {-0.19415263, -0.19415263};
  double bh_par[3] = {14.5, 0.3, 0.2};
  double bh_x1[2] =  {-20, 13};
  double bh_x2[2] =  {-20, 13};
  double bh_x3[2] =  {-20, 13};

  double bounds_l[2] = {-10, -10};
  double bounds_u[2] = {10, 10};
  bh_bounds bounds;

  bounds.l = bounds_l;
  bounds.u = bounds_u;

  gsl_multimin_f_work *bh_multimin_w1;
  bh_multimin_w1 = gsl_multimin_f_alloc(&bh_test_f1, 2, bh_par, gsl_multimin_fminimizer_nmsimplex2);

  bh_work *bh_w1;
  bh_w1 = bh_work_alloc(2, 300, &gsl_multimin_f, (void *)bh_multimin_w1, NULL);
  status = bh_min(bh_x1, &fmin, bh_w1);
  printf("bh with gsl_multimin_f local minimization:\n");
  dequ_chk(bh_result1, bh_x1, 2);
  bh_work_free(bh_w1);
  
  gsl_multimin_f_free(bh_multimin_w1);

  gsl_multimin_fdf_work *bh_multimin_w2;
  bh_multimin_w2 = gsl_multimin_fdf_alloc(&bh_test_f2, 2, bh_par, gsl_multimin_fdfminimizer_vector_bfgs2);

  bh_work *bh_w2;
  bh_w2 = bh_work_alloc(2, 300, &gsl_multimin_fdf, (void *)bh_multimin_w2, NULL);
  status = bh_min(bh_x2, &fmin, bh_w2);
  printf("bh with gsl_multimin_fdf local minimization:\n");
  dequ_chk(bh_result2, bh_x2, 2);
  bh_work_free(bh_w2);
  
  gsl_multimin_fdf_free(bh_multimin_w2);

  gsl_multimin_fndf_work *bh_multimin_w3;
  bh_multimin_w3 = gsl_multimin_fndf_alloc(&bh_test_f3, 2, bh_par, gsl_multimin_fdfminimizer_vector_bfgs2);

  bh_work *bh_w3;
  bh_w3 = bh_work_alloc(2, 300, &gsl_multimin_fndf, (void *)bh_multimin_w3, NULL);
  status = bh_min(bh_x3, &fmin, bh_w3);
  printf("bh with gsl_multimin_fndf local minimization:\n");
  dequ_chk(bh_result3, bh_x3, 2);
  bh_work_free(bh_w3);
  
  gsl_multimin_fndf_free(bh_multimin_w3);

  /*=========================================================================*/
  /* h_fit test.                                                             */
  /*=========================================================================*/

  /* Testing hamiltonian and spin hamiltonian fitting for Ce:LiYF4. Tensor
   * matrix elements and solutions externally calculated using pyemp. */

  int i;
  /* Tensor allocs. */
  zt *eavg, *zeta, *C20, *C40, *C44, *C60, *C64;
  eavg = (zt *) zt_alloc("eavg", ce_eavg_a, 14);
  zeta = (zt *) zt_alloc("zeta", ce_zeta_a, 14);
  C20 = (zt *) zt_alloc("C20", ce_C20_a, 14);
  C40 = (zt *) zt_alloc("C40", ce_C40_a, 14);
  C44 = (zt *) zt_alloc("C44", ce_C44_a, 14);
  C60 = (zt *) zt_alloc("C60", ce_C60_a, 14);
  C64 = (zt *) zt_alloc("C64", ce_C64_a, 14);

  zt *tensors[7];
  tensors[0] = eavg;
  tensors[1] = zeta;
  tensors[2] = C20;
  tensors[3] = C40;
  tensors[4] = C44;
  tensors[5] = C60;
  tensors[6] = C64;

  /* Dummy state label preparation. */
  int nstates = 14;
  char *s[nstates];
  for (i=0; i<nstates; i++) {
    s[i] = malloc(nstates*sizeof(char));
    if (s[i] == 0) 
      printf("Error; label array s malloc failed\n");
    sprintf(s[i], "l=%i", i);
  }


  double *w;
  double complex *z;
  w = (double *) calloc(nstates,sizeof(double));
  z = (double complex *) calloc(nstates*nstates,sizeof(double complex));
  zh *h;
  zhd_w *hd_w;

  h = zh_alloc(nstates, 7, s, tensors);
  zh_set_coeff(h, celiyf4_coeff);
  hd_w = zhd_w_alloc(h);
  zhd(w, z, h, hd_w);
  zhd_w_free(hd_w);

  printf("Ce:LiYF4 diagonalization:\n");
  dequ_chk(celiyf4_diag_res, w, 14);


  zh_free(h);
  for (i=0; i<nstates; i++) {
    free(s[i]);
  }
  free(w);
  free(z);

  zt_free(eavg);
  zt_free(zeta);
  zt_free(C20);
  zt_free(C40);
  zt_free(C44);
  zt_free(C60);
  zt_free(C64);
  return 0;
}  
