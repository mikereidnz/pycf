#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <complex.h>
#include <cfl_crs.h>
#include <cfl_h.h>

equ_chk(double complex *a, double complex *b, size_t n) {
  int i;
  int p = 0;

  for (i=0; i<n; i++) {
    if (cabs(a[i]-b[i]) != 0) {
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

int main (void)
{
  int i, j;

  /* CRS tests. */
  double complex a[16] = {0, 0+I, 0+2*I, 0+3*I, 0-I, 1, 1+2*I, 1+3*I, 0-2*I,
    1-2*I, 2, 2+3*I, 0-3*I, 1-3*I, 2-3*I, 3};
  double complex b[16] = {0, 0+I, 0, 0, 0-I, 0, 1+2*I, 0, 0, 1-2*I, 0, 2+3*I, 0,
    0, 2-3*I, 0};

  /* Allocs used by multiple crs tests. */
  crs_zhm *ma = crs_zhm_alloc(a, 4);
  crs_zhm *mb = crs_zhm_alloc(b, 4);

  /* crs_zhm2zha test. */
  double complex *aa;
  aa = (double complex *) calloc(16,sizeof(double complex));
  if (aa==0) {
    printf("Error; failed to calloc aa");
  }
  crs_zhm2zha(ma, aa);
  printf("crs_zhm2zha:\n");
  equ_chk(a, aa, 16);
  free(aa);

  /* crs_zhm2zhpa test. */
  double complex bp[10] = {0, 0+1*I, 0, 0, 0, 1+2*I, 0, 0, 2+3*I, 0};
  double complex *bbp;
  bbp = (double complex *) calloc(10,sizeof(double complex));
  crs_zhm2zhpa(mb, bbp);

  printf("crs_zhm2zhpa:\n");
  equ_chk(bp, bbp, 10);
  free(bbp);

  /* crs_zhsam test. */
  double complex zhsam_res[16] = {0, 0+3*I, 0+2*I, 0+3*I, 0-3*I, 1, 3+6*I,
    1+3*I, 0-2*I, 3-6*I, 2+0*I, 6+9*I, 0-3*I, 1-3*I, 6-9*I, 3};
  double complex *c;
  double complex alpha = 1;
  double complex beta = 2;

  c = (double complex *) calloc(16,sizeof(double complex));
  if (c==0) {
    printf("Error; failed to calloc c.");
  }

  crs_zhm *mc = crs_zhsam_alloc(ma, mb);
  crs_zhsam(ma, mb, mc, alpha, beta);
  crs_zhm2zha(mc, c);
  printf("crs_zhsam (depends on crs_zhm2zha):\n");
  equ_chk(c, zhsam_res, 16);
  
  crs_zhm_free(mc);
  free(c);

  /* crs_zhsm test. */
  double complex zhsm_res[16] = {0, 0+2*I, 0+4*I, 0+6*I, 0-2*I, 2, 2+4*I, 2+6*I,
    0-4*I, 2-4*I, 4, 4+6*I, 0-6*I, 2-6*I, 4-6*I, 6};
  double complex *d;
 
  d = (double complex *) calloc(16,sizeof(double complex));
  if (d==0) {
    printf("Error; failed to calloc d.");
  }

  crs_zhm *md = crs_zhsm_alloc(ma);
  crs_zhsm(ma, md, beta);
  crs_zhm2zha(md, d);

  printf("crs_zhsm (depends on crs_zhm2zha):\n");
  equ_chk(d, zhsm_res, 16);

  crs_zhm_free(md);
  free(d);

  /* Remaining frees. */
  crs_zhm_free(ma);
  crs_zhm_free(mb);

  
  return 0;
}  
