#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "cfl_crs.h"
#include "cfl_h.h"

void equ_chk(complex double *a, complex double *b, size_t n) {
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
  complex double a[16] = {0, 0+I, 0+2*I, 0+3*I, 0-I, 1, 1+2*I, 1+3*I, 0-2*I,
    1-2*I, 2, 2+3*I, 0-3*I, 1-3*I, 2-3*I, 3};
  complex double b[16] = {0, 0+I, 0, 0, 0-I, 0, 1+2*I, 0, 0, 1-2*I, 0, 2+3*I, 0,
    0, 2-3*I, 0};

  /* Allocs used by multiple crs tests. */
  zhcrs *ma = zhcrs_alloc(a, 4);
  zhcrs *mb = zhcrs_alloc(b, 4);

  /* zhcrs2zha test. */
  complex double *aa;
  aa = (complex double *) calloc(16,sizeof(complex double));
  if (aa==0) {
    printf("Error; failed to calloc aa");
  }
  zhcrs2zha(ma, aa);
  printf("zhcrs2zha:\n");
  equ_chk(a, aa, 16);

  /* zhcrs2zcrs test. */
  zcrs *mac = zhcrs2zcrs_alloc(ma);
  zhcrs2zcrs(ma, mac);
  zcrs2zha(mac, aa);

  printf("zhcrs2zcrs: (depends on zcrs2zha)\n");
  equ_chk(a, aa, 16);

  zcrs_free(mac);
  free(aa);

  /* zhcrs2zhpa test. */
  complex double bp[10] = {0, 0+1*I, 0, 0, 0, 1+2*I, 0, 0, 2+3*I, 0};
  complex double *bbp;
  bbp = (complex double *) calloc(10,sizeof(complex double));
  zhcrs2zhpa(mb, bbp);

  printf("zhcrs2zhpa:\n");
  equ_chk(bp, bbp, 10);
  free(bbp);

  /* zhcrssam test. */
  complex double zhsam_res[16] = {0, 0+3*I, 0+2*I, 0+3*I, 0-3*I, 1, 3+6*I,
    1+3*I, 0-2*I, 3-6*I, 2+0*I, 6+9*I, 0-3*I, 1-3*I, 6-9*I, 3};
  complex double *c;
  complex double alpha = 1;
  complex double beta = 2;

  c = (complex double *) calloc(16,sizeof(complex double));
  if (c==0) {
    printf("Error; failed to calloc c.");
  }

  zhcrs *mc = zhcrssam_alloc(ma, mb);
  zhcrssam(ma, mb, mc, alpha, beta);
  zhcrs2zha(mc, c);
  printf("zhcrssam (depends on zhcrs2zha):\n");
  equ_chk(c, zhsam_res, 16);
  
  zhcrs_free(mc);
  free(c);

  /* zhcrssm test. */
  complex double zhsm_res[16] = {0, 0+2*I, 0+4*I, 0+6*I, 0-2*I, 2, 2+4*I, 2+6*I,
    0-4*I, 2-4*I, 4, 4+6*I, 0-6*I, 2-6*I, 4-6*I, 6};
  complex double *d;
 
  d = (complex double *) calloc(16,sizeof(complex double));
  if (d==0) {
    printf("Error; failed to calloc d.");
  }

  zhcrs *md = zhcrssm_alloc(ma);
  zhcrssm(ma, md, beta);
  zhcrs2zha(md, d);

  printf("zhcrssm (depends on zhcrs2zha):\n");
  equ_chk(d, zhsm_res, 16);

  zhcrs_free(md);
  free(d);

  /* Remaining frees. */
  zhcrs_free(ma);
  zhcrs_free(mb);

  
  return 0;
}  
