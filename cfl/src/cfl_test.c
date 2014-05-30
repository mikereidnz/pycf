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
  
  /* crs_zhsm test. */
  //crs_zhm *md = crs_zhsm_alloc(ma);
  //crs_zhsm(ma, md, beta);

  //for(i=0; i<md->nnz; i++) {
  //  printf("val=%.2f, col_in=%i\n", md->val[i], md->col_in[i]);
  //}

  //for(i=0; i<md->n+1; i++) {
  //  printf("row_ptr=%i\n", md->row_ptr[i]);
  //}





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

  zt *t1, *t2;
  t1 = (zt *) zt_alloc("aten", a, 4);
  t2 = (zt *) zt_alloc("bten", b, 4);
  zt *tens[2];
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

  zh_free(h);


  

  /* zt_sa test */
  zt *t3;
  zh *h2;
  zhd_w *hd_w2;
  t3 = zt_sa("cten", t1, t2, beta, beta);
  zt *tens2[1];
  tens2[0] = t2;
  //tens2[1] = t1;
  double complex coeff2[1];
  coeff2[0] = beta;
  //coeff2[1] = alpha;

  printf("n = %i, nnz = %i\n", t3->matel->n, t3->matel->nnz);

  for (i=0; i<8; i++) {
    printf("val = %.2f, col_in = %i\n", t3->matel->val[i], t3->matel->col_in[i]);
  }
  for (i=0; i<5; i++) {
    printf("row_ptr = %i\n", t3->matel->row_ptr[i]);
  }

  printf("n=%i, name=%s\n", t3->n, t3->name);
  
  h2 = zh_alloc(hdim, 1, s, tens2, w, z);
  zh_set_coeff(h2, coeff2);
  hd_w2 = zhd_w_alloc(h2);
  zhd(h2, hd_w2);
  zhd_w_free(hd_w2);
  zh_free(h2);


  free(w);
  free(z);
  zt_free(t1);
  zt_free(t2);
  zt_free(t3);

  

  for (i=0; i<4; i++) {
    free(s[i]);
  }

  return 0;
}  
