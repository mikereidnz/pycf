#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "cfl_config.h"
#include "cfl_csr.h"
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

void int_equ_chk(int *a, int *b, size_t n) {
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

void print_matrix(char *desc, int m, int n, complex double *a) {
  int i, j;
  printf( "\n %s\n", desc );
  for(i=0; i<m; i++) {
    for(j=0; j<n; j++)
      printf( " (%6.2f,%6.2f)", creal(a[i+j*n]), cimag(a[i+j*n]));
    printf( "\n" );
  }
}

int main (void) {
  int i, j;

  /* CRS tests. */
  complex double a[16] = {0, 0+I, 0+2*I, 0+3*I, 0-I, 1, 1+2*I, 1+3*I, 0-2*I,
    1-2*I, 2, 2+3*I, 0-3*I, 1-3*I, 2-3*I, 3};
  complex double b[16] = {0, 0+I, 0, 0, 0-I, 0, 1+2*I, 0, 0, 1-2*I, 0, 2+3*I, 0,
    0, 2-3*I, 0};

  /* Allocs used by multiple csr tests. */
  zhcsr *ma = zhcsr_gen(a, 4);
  zhcsr *mb = zhcsr_gen(b, 4);

  /* zhcsr2zha test. */
  complex double *aa;
  aa = (complex double *) calloc(16,sizeof(complex double));
  if (aa==0) {
    printf("Error; failed to calloc aa");
  }
  zhcsr2zha(ma, aa);
  printf("zhcsr2zha:\n");
  equ_chk(a, aa, 16);

  /* zhcsr2zcsr test. */
  zcsr *mac = zhcsr2zcsr_alloc(ma);
  zhcsr2zcsr(ma, mac);
  zcsr2zha(mac, aa);

  printf("zhcsr2zcsr (depends on zcsr2zha):\n");
  equ_chk(a, aa, 16);

  /* zhcsr_alloc test. */
  zhcsr *maa;
  maa = zhcsr_alloc(mac->n, mac->row_ptr, mac->col_in, mac->val);
  zcsr *maac = zhcsr2zcsr_alloc(maa);
  zhcsr2zcsr(ma, maac);
  zcsr2zha(maac, aa);

  printf("zhcsr_alloc (depends on zcsr2zha and zhcsr2zcsr):\n");
  equ_chk(a, aa, 16);
  zhcsr_free(maa);
  zcsr_free(maac); 
  
  /* zhcsr2zhpa test. */
  complex double bp[10] = {0, 0+1*I, 0, 0, 0, 1+2*I, 0, 0, 2+3*I, 0};
  complex double *bbp;
  bbp = (complex double *) calloc(10,sizeof(complex double));
  zhcsr2zhpa(mb, bbp);

  printf("zhcsr2zhpa:\n");
  equ_chk(bp, bbp, 10);
  free(bbp);

  /* zhcsrsam test. */
  complex double zhsam_res[16] = {0, 0+3*I, 0+2*I, 0+3*I, 0-3*I, 1, 3+6*I,
    1+3*I, 0-2*I, 3-6*I, 2+0*I, 6+9*I, 0-3*I, 1-3*I, 6-9*I, 3};
  complex double *c;
  complex double alpha = 1;
  complex double beta = 2;

  c = (complex double *) calloc(16,sizeof(complex double));
  if (c==0) {
    printf("Error; failed to calloc c.");
  }

  zhcsr *mc = zhcsrsam_alloc(ma, mb);
  zhcsrsam(ma, mb, mc, alpha, beta);
  zhcsr2zha(mc, c);

  printf("zhcsrsam (depends on zhcsr2zha):\n");
  equ_chk(c, zhsam_res, 16);

  zhcsr_free(mc);
  free(c);
  zcsr_free(mac);
  free(aa);

  /* Cerium EAVG + C44 zhcsrsam test. */
  complex double eavg_a[196] = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0};

  complex double c44_a[196] = {5.0, 0, 0, 0, 0, 0, 0, 0.148453640934,
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

  complex double ce_zhsam_res[196] = {10.0, 0, 0, 0, 0, 0, 0, 0.148453640934,
    0.128564624333, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0.178173821773,
    -0.102442744767, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0.15870361777,
    0.188199339398, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0.178173821773,
    -0.15870361777, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0.102442744767,
    0.188199339398, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0, -0.148453640934,
    0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0.128564624333, 0.148453640934, 0,
    0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0, 0.128564624333, 0, 0, 0, 0, 0, 0, 0,
    5.0, 0, 0, 0, 0, 0, 0, 0.178173821773, 0.15870361777, 0, 0, 0, 0, 0, 0, 5.0,
    0, 0, 0, 0, 0, -0.102442744767, 0.188199339398, 0, 0, 0, 0, 0, 0, 0, 5.0, 0,
    0, 0, 0, 0, 0, 0.178173821773, 0.102442744767, 0, 0, 0, 0, 0, 0, 5.0, 0, 0,
    0, 0, 0, -0.15870361777, 0.188199339398, 0, 0, 0, 0, 0, 0, 0, 5.0, 0, 0, 0,
    0, 0, 0, -0.148453640934, 0.128564624333, 0, 0, 0, 0, 0, 0, 5.0};

  complex double ce_alpha = 5;
  complex double ce_beta = 1;
  complex double *ce_res_a;

  zhcsr *ce_eavg = zhcsr_gen(eavg_a, 14);
  zhcsr *ce_c44 = zhcsr_gen(c44_a, 14);

  zhcsr *ce_res = zhcsrsam_alloc(ce_eavg, ce_c44);
  zhcsrsam(ce_eavg, ce_c44, ce_res, ce_alpha, ce_beta);

  ce_res_a = (complex double *) calloc(196,sizeof(complex double));
  zhcsr2zha(ce_res, ce_res_a);
  
  printf("ce zhsam:\n");
  equ_chk(ce_res_a, ce_zhsam_res, 196);

  /* Cerium EAVG + C44 zhcsrsama test. */
  zhcsrsama_data *zhcsrsama_d;
  zhcsr *csr_ma[2] = {ce_eavg, ce_c44};
  complex double ce_ca[2] = {ce_alpha, ce_beta};
  
  zhcsrsama_d = zhcsrsama_alloc(2, csr_ma);
  zhcsrsama(csr_ma, ce_ca, zhcsrsama_d); 

  printf("ce zhcsrsama:\n");
  zhcsr2zha(zhcsrsama_d->hcsr_m, ce_res_a);
  equ_chk(ce_res_a, ce_zhsam_res, 196);
  
  zhcsrsama_free(zhcsrsama_d);
  zhcsr_free(ce_eavg);
  zhcsr_free(ce_c44);
  zhcsr_free(ce_res);
  free(ce_res_a);

  /* zhcsrsm test. */
  complex double zhsm_res[16] = {0, 0+2*I, 0+4*I, 0+6*I, 0-2*I, 2, 2+4*I, 2+6*I,
    0-4*I, 2-4*I, 4, 4+6*I, 0-6*I, 2-6*I, 4-6*I, 6};
  complex double *d;

  d = (complex double *) calloc(16,sizeof(complex double));
  if (d==0) {
    printf("Error; failed to calloc d.");
  }

  zhcsr *md = zhcsrsm_alloc(ma);
  zhcsrsm(ma, md, beta);
  zhcsr2zha(md, d);

  printf("zhcsrsm (depends on zhcsr2zha):\n");
  equ_chk(d, zhsm_res, 16);

  zhcsr_free(md);
  free(d);

  /* Remaining frees. */
  zhcsr_free(ma);
  zhcsr_free(mb);

  complex double ce_C20_a[784] = {-0.333333308417, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.333333308417, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, -0.285714264357, 0, 0.116642359985, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.285714264357, 0,
    0.116642359985, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0.116642359985, 0, -0.0476190440595, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.116642359985, 0,
    -0.0476190440595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0571428528714, 0, 0.0903507835368, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0571428528714, 0, 0.0903507835368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0903507835368, 0, 0.142857132179, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0903507835368, 0, 0.142857132179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.228571411486, 0,
    0.0329914414876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.228571411486, 0, 0.0329914414876, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0329914414876, 0,
    0.238095220298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.0329914414876, 0, 0.238095220298, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.228571411486, 0, -0.0329914414876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.228571411486, 0, -0.0329914414876,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.0329914414876, 0, 0.238095220298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0329914414876, 0, 0.238095220298,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0.0571428528714, 0, -0.0903507835368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0571428528714, 0,
    -0.0903507835368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -0.0903507835368, 0, 0.142857132179, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0903507835368, 0,
    0.142857132179, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, -0.285714264357, 0, -0.116642359985, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.285714264357, 0, -0.116642359985, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.116642359985, 0, -0.0476190440595, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.116642359985, 0, -0.0476190440595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.333333308417, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -0.333333308417};

  int n = 28;
  zhcsr *c20_zhm = zhcsr_gen(ce_C20_a, n);
  zcsr *c20_zm = zhcsr2zcsr_alloc(c20_zhm);
  zhcsr2zcsr(c20_zhm, c20_zm);

  complex double ce_C20_aa[784];
  zcsr2zha(c20_zm, ce_C20_aa);
  printf("zhcsr2zcsr with I=1/2 Ce C20 (depends on zcsr2zha):\n");
  equ_chk(ce_C20_a, ce_C20_aa, n);


  /* zhcsr_alloc test. */
  zhcsr *c20_zhmm = zhcsr_alloc(c20_zm->n, c20_zm->row_ptr, c20_zm->col_in, c20_zm->val);
  zcsr *c20_zmm = zhcsr2zcsr_alloc(c20_zhmm);
  zhcsr2zcsr(c20_zhmm, c20_zmm);

  zcsr2zha(c20_zmm, ce_C20_aa);
  printf("zhcsr_alloc with I=1/2 Ce C20 (depends on zcsr2zha and zhcsr2zcsr):\n");
  equ_chk(ce_C20_a, ce_C20_aa, n);

  zhcsr_free(c20_zhmm);
  zcsr_free(c20_zmm);

  /* Connected components test. */
  int p[n];
  int nblocks;

  int cc_a[28] = {0,  1,  2,  3,  2,  3,  4,  5,  4,  5,  6,  7,  6,  7,  8,  9,
    8,  9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15};
  int labels[28];
  nblocks = zcsr_cc(c20_zm, labels);
  printf("zcsr_cc test:\n");
  int_equ_chk(cc_a, labels, n);

  int ix[9] = {5, 9, 4, 7, 8, 1, 2, 6, 3};
  int ix_perm[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int perm[9] = {4, 8, 3, 6, 7, 0, 1, 5, 2};
  int perm_orig[9] = {4, 8, 3, 6, 7, 0, 1, 5, 2};
  ivperm(9, ix, perm);

  printf("ivperm:\n");
  int_equ_chk(ix, ix_perm, 9);
  printf("ivperm test 2:\n");
  int_equ_chk(perm_orig, perm, 9);

  complex double pma[16] = {1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 1, 4, 0, 0, 4, 1};
  complex double pmaa[16] = {1, 2, 0, 0, 2, 1, 0, 4, 0, 0, 1, 0, 0, 4, 0, 1};
  int pm_p[4] = {0, 2, 1, 3}; 
  zhcsr *pmh;
  zcsr *cpm, *rpm;
  int *pmj;

  pmh = zhcsr_gen(pma, 4);
  zcsr *pm = zhcsr2zcsr_alloc(pmh);
  zhcsr2zcsr(pmh, pm);

  pmj = (int *) calloc(pm->nnz+1, sizeof(int));
  cpm = (zcsr *) zcsr_col_perm_alloc(pm, pm_p, pmj);
  rpm = (zcsr *) zcsr_row_perm_alloc(cpm, pm_p);

  zcsr_col_perm(pm, cpm, pm_p, pmj);
  zcsr_row_perm(cpm, rpm, pm_p);
  zcsr_col_perm(pm, cpm, pm_p, pmj);
  zcsr_row_perm(cpm, rpm, pm_p);
  zcsr2zha(rpm, pma);

  printf("cpm and rpm:\n");
  equ_chk(pma, pmaa, 16);

  zhcsr_free(pmh);
  zcsr_free(pm);

  free(pmj);
  zcsr_free(cpm);
  zcsr_free(rpm);

  zcsr *c20_cpm, *c20_rpm;
  int *pj;

  pj = (int *) calloc(c20_zm->nnz+1, sizeof(int));

  int *pi;
  pi = (int *) calloc(c20_zm->n, sizeof(int));

  for (i=0; i<c20_zm->n; i++) {
    pi[p[i]] = i;
  }
  c20_rpm = (zcsr *) zcsr_row_perm_alloc(c20_zm, pi);
  c20_cpm = (zcsr *) zcsr_col_perm_alloc(c20_rpm, pi, pj);
  
  zcsr_row_perm(c20_zm, c20_rpm, pi);
  zcsr_col_perm(c20_rpm, c20_cpm, pi, pj);

  complex double *c20_a = (complex double *) calloc( c20_zm->n*c20_zm->n, sizeof(complex double));
  zcsr2zha(c20_cpm, c20_a);

  free(pi);
  free(pj);
  zcsr_free(c20_rpm);
  zcsr_free(c20_cpm);
  zhcsr_free(c20_zhm);
  zcsr_free(c20_zm);
  free(c20_a);

  return 0;
}  
