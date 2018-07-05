#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>

#include "cfl_tensor.h"
#include "cfl_h.h"

#include "cfl_min.h"
#include "cfl_h_fit.h"


int main (void) {

  /*=========================================================================*/
  /* h_fit test.                                                             */
  /*=========================================================================*/
  int i, status;

  double fmin;
  /* Testing hamiltonian and spin hamiltonian fitting for Ce:LiYF4. Tensor
   * matrix elements and solutions externally calculated using pyemp. */
  
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
  
  double ce_ex[6] = {0, 216, 2216.10, 2312.80, 2428.80, 3157.80};
  
  complex double celiyf4_coeff[7] = {1535.12773615, 625.699030356,
    297.890587979, -1328.15222293, -1282.47659014, -191.510006575,
    -1743.14238515+692.866175947*I};

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
  zt *C20, *C40, *C44, *C60;

  C20 = (zt *) zt_alloc("C20", ce_C20_a, 14, states);
  C40 = (zt *) zt_alloc("C40", ce_C40_a, 14, states);
  C44 = (zt *) zt_alloc("C44", ce_C44_a, 14, states);
  C60 = (zt *) zt_alloc("C60", ce_C60_a, 14, states);
 
  zt *tensors[4] = {C20, C40, C44, C60};
  
  /* Manually prepare array of parameter structs. */
  param_type efit_p0;
  efit_p0.type = 'r';
  efit_p0.xi = 0;
  efit_p0.ci = 0;
  param_type efit_p1;
  efit_p1.type = 'r';
  efit_p1.xi = 1;
  efit_p1.ci = 1;
  param_type efit_p2;
  efit_p2.type = 'r';
  efit_p2.xi = 2;
  efit_p2.ci = 2;
  param_type efit_p3;
  efit_p3.type = 'r';
  efit_p3.xi = 3;
  efit_p3.ci = 3;
  param_type efit_p4;
  efit_p4.type = 'r';
  efit_p4.xi = 4;
  efit_p4.ci = 4;
  param_type efit_p5;
  efit_p5.type = 'r';
  efit_p5.xi = 5;
  efit_p5.ci = 5;
  param_type **p = (param_type **) malloc(6*sizeof(param_type *));
  p[0] = &efit_p0;
  p[1] = &efit_p1;
  p[2] = &efit_p2;
  p[3] = &efit_p3;
  p[4] = &efit_p4;
  p[5] = &efit_p5;


  
  /* Set up the experimental data struct. */
  double ce_x0[4] = {2000, 900, 200, 1000};

  double bounds_u[4] = {2500, 1500, 5000, 2500};
  double bounds_l[4] = {10, 10, 0, 0};
  cfl_min_bounds bounds;
  bounds.l = bounds_l;
  bounds.u = bounds_u;

  zh *h;
  h = zh_alloc(nstates, 4, tensors);
  zh_set_coeff(h, celiyf4_coeff);

  ex_data ce_ex_data;
  int ex_index[6] = {1, 2, 7, 8, 11, 13};
  double weights[6] = {1, 1, 1, 1, 1, 1}; 
  double stepsize[4] = {1, 1, 1, 1};
  ce_ex_data.n_obs = 4;
  ce_ex_data.n_a = 4;
  ce_ex_data.n_d = 0;
  ce_ex_data.e = ce_ex;
  ce_ex_data.la = ex_index;
  ce_ex_data.ild = NULL;
  ce_ex_data.fld = NULL;
  ce_ex_data.chisq_weight = 1.0;
  
  int niter = 3000000;
  double *xaccept = (double *) calloc(4*niter,sizeof(double));
  double *chi2accept = (double *) calloc(niter,sizeof(double));
  /* Run energy level fit. */
  efit_data *efit_d;
  cfl_min_obj *efit_min_obj;
   
  double xtol = 1e-8;
  double gtol = 1e-8;
  double ftol = 0.0;
  efit_d = efit_data_alloc('N', h, &ce_ex_data, 4, p);
  efit_min_obj = cfl_siman_min_setup(&efit_obj, 4, efit_d, niter, &bounds,
      stepsize, 10000, 80, 1.0000005, 2, chi2accept, xaccept, -1);
  status = cfl_min(ce_x0, &fmin, efit_min_obj);
  
  printf("fmin = %f\n", fmin);
  printf("x0 = ");
  for (i=0; i<4; i++) {
    printf("%f ", ce_x0[i]);
  }
  printf("\n");
  free(chi2accept);
  free(xaccept);
  cfl_min_free(efit_min_obj);
  efit_data_free(efit_d);

  zh_free(h);

  free(p);

  zt_free(C20);
  zt_free(C40);
  zt_free(C44);
  zt_free(C60);

  sl_free(states);
  free(l);
  
  return 0;
}  
