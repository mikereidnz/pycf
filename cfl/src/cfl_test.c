#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <cfl_crs.h>
#include <cfl_h.h>

int main (void)
{
  printf("This is a test; a testing test.\n");
  int i, j;
  /* State label preparation and test. */
  char *s[4];
  for (i=0; i<4; i++) {
    s[i] = malloc(4*sizeof(char));
    if (s[i] == 0) 
      printf("Error; label array s malloc failed\n");
    sprintf(s[i], "l=%i", i);
  }

  /* CRS tests. */
  double complex a[20] = {0.0, 4.0, 1.0, 10.0, 0.0, 6.0, 2.0, 3.0, 0.0, 0.0, 5.0, 0.0, 9.0, 7.0, 8.0, 0.0};  
  double complex b[20] = {0.0, 4.0, 0.0, 0.0, 5.0, 8.0, 2.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 7.0, 8.0, 1.0};  
  double complex *c;
  c = (double complex *) calloc(20,sizeof(double complex));
  if (c==0) {
    printf("Error; failed to alloc memory for array c.");
  }

  crs_zhm *ma = crs_zhm_alloc(a, 4);
  crs_zhm *mb = crs_zhm_alloc(b, 4);

  double complex alpha = 2;
  double complex beta = 1;

  //crs_zhm *mc = crs_zhsam_alloc(ma, mb);

  //crs_zhsam(ma, mb, mc, alpha, beta);
  //print_crs_hm(ma);
  //print_crs_hm(mb);
  //print_crs_hm(mc);


  //for (i=0; i<ma->n+1; i++) {
  //  printf("row_ptr a = %i\n", ma->row_ptr[i]);
  //}
  //for (i=0; i<mb->n+1; i++) {
  //  printf("row_ptr b = %i\n", mb->row_ptr[i]);
  //}
  //for (i=0; i<mc->n+1; i++) {
  //  printf("row_ptr c = %i\n", mc->row_ptr[i]);
  //}
  //crs_zhm *md = crs_zhsm_alloc(ma);
 // crs_zhsm(ma, md, alpha);



  /* Packed Hermitian diagonal conversion test. */
  double complex *hpa;
  hpa = (double complex *) calloc(10,sizeof(double complex));
  //crs_zhm2zhpa(ma, hpa);
  
  //print_crs_hm(ma);
  //for (i=0; i<10; i++) {
  //  printf("hpa val=%.2f\n", hpa[i]);
  //}

  free(hpa);
  crs_zhm_free(ma);
  crs_zhm_free(mb);
  //crs_zhm_free(mc);
  //crs_zhm_free(md);
  free(c);


  /*=========================================================================*/
  double *w;
  double complex *z;
  w = (double *) calloc(4,sizeof(double));
  z = (double complex *) calloc(4*4,sizeof(double complex));

  ztensor *t1, *t2;
  t1 = (ztensor *) ztensor_alloc("aten", a, 4);
  t2 = (ztensor *) ztensor_alloc("bten", b, 4);
  ztensor *tens[2];
  double complex coeff[2];
  
  tens[0] = t1;
  tens[1] = t2;
  coeff[0] = alpha;
  coeff[1] = beta;
  
  /* Hamiltonian creation. */
  zh *h;
  zhd_w *hd_w;
  int hdim = 4;
  h = zh_alloc(hdim, 2, s, tens, w, z);

  //h->coeff = coeff;
  zh_set_coeff(h, coeff);
  hd_w = zhd_w_alloc(h);
  zhd(h, hd_w);
  zhd_w_free(hd_w);

  for (i=0; i<4; i++) {
    printf("Eigenvalue: %.4f\n", w[i]);
    printf("Eigenvector:\n");
    for (j=0; j<4; j++) {
      printf(" %.4f\n", z[4*i+j]);
    }
  }

  free(w);
  free(z);
  zh_free(h);
  ztensor_free(t1);
  ztensor_free(t2);
  

  for (i=0; i<4; i++) {
    free(s[i]);
  }

  return 0;
}  
