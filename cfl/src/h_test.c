#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <gsl/gsl_cblas.h>

#include <cfl_crs.h>
#include <cfl_tensor.h>
#include <cfl_h.h>
#include <cfl_sh.h>

#include <test_data.h>

/* This file contains tests for:
 *  * tensor allocation and scaling/addition functions
 *  * hamiltonian allocation, and diagonalization
 *  * spin hamiltonian allocation and projection
 *
 * Tests in this file depend on a functional crs implementation, so run crs_test
 * as a prerequisite. 
 */

/*
 * @brief   Check the equality of two complex valued arrays.
 *
 * @param[a]  Pointer to first array. 
 * @param[b]  Pointer to second array.
 * @param[n]  Length of arrays a and b.
 *
 */
void zequ_chk(double complex *a, double complex *b, size_t n) {
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

/*
 * @brief   Print an n by n complex valued matrix a, stored as a one dim array.
 *
 * @param[a]  Pointer to a.
 * @param[n]  Dimension of the matrix a. 
 */
void mprint(double complex *a, size_t n) {
  int i,j;

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      printf("%.2f+%.2fI ", creal(a[i*n+j]), cimag(a[i*n+j]));
    }
    printf("\n");
  }
}

int main (void)
{
  int i, j;

  double complex alpha = 2;
  double complex beta = 4;
  double complex a[16] = {0, I, 2*I, 3*I, -I, 1, 1+2*I, 1+3*I, -2*I, 1-2*I, 2,
    2+3*I, -3*I, 1-3*I, 2-3*I, 3};
  double complex b[16] = {0, I, 0, 0, -I, 0, 1+2*I, 0, 0, 1-2*I, 0, 2+3*I, 0, 0,
    2-3*I, 0};

  /* Tensor allocs; neglecting state labels for now. */
  zt *t1, *t2;
  t1 = (zt *) zt_alloc("aten", a, 4);
  t2 = (zt *) zt_alloc("bten", b, 4);

  /*=========================================================================*/
  /* Tensor tests.                                                           */
  /*=========================================================================*/
  
  /* zt_sa test. */
  double complex *c;
   
  double complex ztsa_res[16] = {0, 6*I, 4*I, 6*I, -6*I, 2, 6+12*I, 2+6*I, -4*I,
    6-12*I, 4, 12+18*I, -6*I, 2-6*I, 12-18*I, 6};
  
  c = (double complex *) calloc(16,sizeof(double complex));
  if (c==0) {
    printf("Error; failed to calloc c");
  }

  zt *t3;
  t3 = zt_sa("cten", t1, t2, alpha, beta);
  crs_zhm2zha(t3->matel, c);

  printf("zt_sa:\n");
  zequ_chk(ztsa_res, c, 16);
  zt_free(t3);

  /* zt_s test. */
  double complex zts_res[16] = {0, 2*I, +4*I, 6*I, -2*I, 2, 2+4*I, 2+6*I, -4*I,
    2-4*I, 4, 4+6*I, -6*I, 2-6*I, 4-6*I, 6};

  t3 = zt_s("cten", t1, alpha);
  crs_zhm2zha(t3->matel, c);

  printf("zt_s:\n");
  zequ_chk(zts_res, c, 16);
  zt_free(t3);


  /*=========================================================================*/
  /* Hamiltonian tests.                                                      */
  /*=========================================================================*/
  
  /* State label preparation. */
  char *s[4];
  for (i=0; i<4; i++) {
    s[i] = malloc(4*sizeof(char));
    if (s[i] == 0) 
      printf("Error; label array s malloc failed\n");
    sprintf(s[i], "l=%i", i);
  }

  double *w;
  double complex *z;
  w = (double *) calloc(4,sizeof(double));
  z = (double complex *) calloc(4*4,sizeof(double complex));

  /* Test Hamiltonian alloc and diag with two tensors. */
  zt *tens[2];
  double complex coeff[2];
  double hdiag_res[4] = {-19.89945633, -7.16829888, 5.43631787, 33.63143733};
  tens[0] = t1;
  tens[1] = t2;
  coeff[0] = alpha;
  coeff[1] = beta;
  zh *h;
  zhd_w *hd_w;

  h = zh_alloc(4, 2, s, tens);
  zh_set_coeff(h, coeff);
  hd_w = zhd_w_alloc(h);
  zhd(w, z, h, hd_w);
  zhd_w_free(hd_w);
 
  printf("hdiag multiple tensors:\n");
  dequ_chk(hdiag_res, w, 4);

  /* Test diagonalization of Hamiltonian with a single tensor. */
  zt *tens2[1];
  double complex coeff2[1];
  double h2diag_res[4] = {-7.48348091, -0.42411223, 1.65736632, 18.25022682};
  tens2[0] = t1;
  coeff2[0] = alpha;
  zh *h2;
  zhd_w *hd_w2;
  
  h2 = zh_alloc(4, 1, s, tens2);
  zh_set_coeff(h2, coeff2);
  hd_w2 = zhd_w_alloc(h2);
  zhd(w, z, h2, hd_w2);
  zhd_w_free(hd_w2);

  printf("hdiag single tensor:\n");
  dequ_chk(h2diag_res, w, 4);
  zh_free(h2);

  free(c);

  /*=========================================================================*/
  /* Spin Hamiltonian projection test.                                       */
  /*=========================================================================*/

  double complex zshp_res[4] = {-3.7417404568, 0, 0, -0.2120561172};
  double complex *sha = (double complex *) calloc(4,sizeof(double complex));
  zsh *sh;
  zshp_w *shp_w;

  sh = zsh_alloc(2, "test");

  zsh_set_pro(sh, t1, 0);
  shp_w = zshp_w_alloc(sh);
  zshp(sha, z, sh, shp_w);

  printf("zshp:\n");
  zequ_chk(zshp_res, sha, 4);
  zshp_w_free(shp_w);
  zsh_free(sh);
  free(sha);

  zh_free(h);
  free(w);
  free(z);
  for (i=0; i<4; i++) {
    free(s[i]);
  }

  zt_free(t1);
  zt_free(t2);

  /*=========================================================================*/
  /* Spin Hamiltonian inversion test.                                        */
  /*=========================================================================*/

  /* The inversion matrix and hyperfine term of the Hamiltonian were externally
   * calculated for Er:YSO, with experimental A-tensor data source from O.
   * Guillot_Noel et al, Journal of Alloys and Compounds, 451, (2008) 62. */
  zsh_inv_data hyp_inv_data;
  hyp_inv_data.a = euyso_hyp_inv;
  hyp_inv_data.m = 256;
  hyp_inv_data.n = 9;

  double complex hyp_inv_result[9] = {69.35, -580.73, -248.83, -580.73, 696.30,
    682.49, -248.83, 682.49, 495.54};

  zshi_w *hyp_work = zshi_w_alloc(&hyp_inv_data);
  zshi(euyso_hyp_term, hyp_work);

  printf("Hyperfine inversion test for Eu:YSO:\n");
  zequ_chk(hyp_inv_result, euyso_hyp_term, 9);
  
  zshi_w_free(hyp_work);
  return 0;
}  
