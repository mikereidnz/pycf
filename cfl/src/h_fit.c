/*
 * Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include <cfl_h.h>
#include <cfl_sh.h>
#include <cfl_error.h>
#include <h_fit.h>


/*
 * Alloc workspace for fitting both energy levels.
 *
 * Parameters
 * ----------
 *  h       Pointer to the Hamiltonian.  
 *  ex      Array of experimental energy level data. 
 *  n_zx    The number of complex valued parameters to be fit to the Hamiltonian
 *  p       Array of pointers to parameters to be fit.
 */
efit_data *efit_data_alloc(zh *h, double *ex, size_t n_zx, param_type **p) {
  efit_data *data;

  data = (efit_data *) malloc(sizeof(efit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for data");
  }

  data->hd_w = zhd_w_alloc(h);
  if (data->hd_w == 0) {
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }

  data->evect = (double complex *) calloc(h->n*h->n,sizeof(double complex));
  if (data->evect == 0) {
    zhd_w_free(data->hd_w);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->evect");
  }
  data->eval = (double *) calloc(h->n,sizeof(double));
  if (data->eval == 0) {
    zhd_w_free(data->hd_w);
    free(data->evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->eval");
  }
  data->coeff = (double complex *) calloc(h->n_zx,sizeof(double complex));
  if (data->coeff == 0) {
    zhd_w_free(data->hd_w);
    free(data->evect);
    free(data->eval);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->coeff");
  }

  data->h = h;
  data->ex = ex;
  data->p = p;

  return data;
}

void efit_data_free(efit_data *data) {
  zhd_w_free(data->hd_w);
  free(data->evect);
  free(data->eval);
  free(data->coeff);
  free(data);
}


/*
 * Alloc workspace for fitting both energy levels and spin hamiltonians.
 *
 * Parameters
 * ----------
 *  sh      Array of pointers to spin Hamiltonians.  
 *  nsh     The number of spin Hamiltonians to be fit.
 *  h       Pointer to the complete Hamiltonian.  
 *  hfo     Pointer to the first order Hamiltonian.
 *  ex      Array of experimental energy level data. 
 *  shx     Array of pointers to spin Hamiltonian experimental data. 
 *  n_zx    The number of complex valued parameters to be fit to the complete
 *          Hamiltonian h.
 *  n_fozx  The number of complex valued parameters to be fit to the first order
 *          Hamiltonian hfo.
 *  p       Array of pointers to parameters to be fit.
 */
eshfit_data *eshfit_data_alloc(zsh **sh, size_t nsh, zh *h, zh *hfo, double *ex, shx_data **shx, size_t n_zx, size_t n_fozx; param_type **p) {
  int i,j;
  eshfit_data *data;

  data = (eshfit_data *) malloc(sizeof(eshfit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for eshfit_data");
  }

  data->hd_w = zhd_w_alloc(h);
  if (data->hd_w == 0) {
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }
  data->hfod_w = zhd_w_alloc(hfo);
  if (data->hfod_w == 0) {
    free(data->hd_w);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hfod_w");
  }
  data->h_evect = (double complex *) calloc(h->n*h->n,sizeof(double complex));
  if (data->h_evect == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_evect");
  }
  data->h_eval = (double *) calloc(h->n,sizeof(double));
  if (data->h_eval == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data->h_evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_eval");
  }
  data->hfo_evect = (double complex *) calloc(hfo->n*hfo->n,sizeof(double complex));
  if (data->hfo_evect == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->hfo_evect");
  }
  data->hfo_eval = (double *) calloc(hfo->n,sizeof(double));
  if (data->hfo_eval == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data->hfo_evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->hfo_eval");
  }
  data->h_coeff = (double complex *) calloc(n_zx,sizeof(double complex));
  if (data->h_coeff == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data->hfo_evect);
    free(data->hfo_eval);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_coeff");
  }
  data->hfo_coeff = (double complex *) calloc(n_fozx,sizeof(double complex));
  if (data->hfo_coeff == 0) {
    zhd_w_free(data->hd_w);
    zhd_w_free(data->hfod_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data->hfo_evect);
    free(data->hfo_eval);
    free(data->h_coeff);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->hfo_coeff");
  }
  for (i=0; i<nsh; i++) {
    data->shp_w_array[i] = zshp_w_alloc(shx_data[i]->ten);
    if (data->shp_w_array == 0) {
      for (j=0; j<i; j++) {
        zshp_w_free(data->shp_w_array[j]);
        zshi_w_free(data->shi_w_array[j]);
      }
      zhd_w_free(data->hd_w);
      zhd_w_free(data->hfod_w);
      free(data->h_evect);
      free(data->h_eval);
      free(data->hfo_evect);
      free(data->hfo_eval);
      free(data->h_coeff);
      free(data->hfo_coeff);
      free(data);
      CFL_ERROR_NULL("zshp_w_alloc failed for data->shp_w_array");
    }
    data->shi_w_array[i] = zshi_w_alloc(sh[i]->inv_data);
    if (data->shi_w_array[i] == 0) {
      for (j=0; j<i; j++) {
        zshp_w_free(data->shp_w_array[j]);
        zshi_w_free(data->shi_w_array[j]);
      }
      zshp_w_free(data->shp_w_array[i]);
      zhd_w_free(data->hd_w);
      zhd_w_free(data->hfod_w);
      free(data->h_evect);
      free(data->h_eval);
      free(data->hfo_evect);
      free(data->hfo_eval);
      free(data->h_coeff);
      free(data->hfo_coeff);
      free(data);
      CFL_ERROR_NULL("zshi_w_alloc failed for data->shi_w_array");
    }
    data->sh_pa[i] = (double complex *) calloc(9,sizeof(double complex));
    if (data->sh_pa[i] == 0) {
      for (j=0; j<i; j++) {
        zshp_w_free(data->shp_w_array[j]);
        zshi_w_free(data->shi_w_array[j]);
        free(data->sh_pa[j]);
      }
      zshp_w_free(data->shp_w_array[i]);
      zshi_w_free(data->shi_w_array[i]);
      zhd_w_free(data->hd_w);
      zhd_w_free(data->hfod_w);
      free(data->h_evect);
      free(data->h_eval);
      free(data->hfo_evect);
      free(data->hfo_eval);
      free(data->h_coeff);
      free(data->hfo_coeff);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->sh_pa");
  }

  data->sh = sh;
  data->h = h;
  data->hfo = hfo;
  data->ex = ex;
  data->shx = shx;
  data->p = p;
}


void eshfit_data_free(eshfit_data *data) {
  int i;

  for (i=0; i<data->nsh; i++) {
    zshp_w_free(data->shp_w_array[i]);
    zshi_w_free(data->shi_w_array[i]);
    free(data->sh_pa[i]);
  }
  zhd_w_free(data->hd_w);
  zhd_w_free(data->hfod_w);
  free(data->h_evect);
  free(data->h_eval);
  free(data->hfo_evect);
  free(data->hfo_eval);
  free(data->h_coeff);
  free(data->hfo_coeff);
  free(data);
}


/* Chi^2 for energy levels. */
inline double echisq(double *eval, double *ex, size_t n) {
  int i;
  double chisq=0;

  for (i=0; i<n; i++) {
    chisq += abs(eval[i] - ex[i]);
  }

  return chisq;
}

/* Chi^2 for real parameters. 
 *
 * Parameters
 * ----------
 *  pa    The theoretical parameter array.
 *  x_pa  The experimental parameter array. 
 *  n     The number of parameters.
 */
inline double dchisq(double *pa, double *x_pa, size_t n) {
  int i;
  double chisq=0;

  for (i=0; i<n; i++) {
    chisq += pow(pa[i]-x_pa[i]);
  }

  return chisq;
}

/* Chi^2 for complex parameters. 
 *
 * Parameters
 * ----------
 *  pa    The theoretical parameter array.
 *  x_pa  The experimental parameter array. 
 *  n     The number of parameters.
 */
inline double zchisq(complex double *pa, complex double *x_pa, size_t n) {
  int i;
  double chisq=0;

  for (i=0; i<n; i++) {
    chisq += pow(cabs(pa[i]-x_pa[i]));
  }

  return chisq;
}

/*
 * Objective function for fit to energy levels only.
 */
double efit_obj(size_t n, double *x, double *grad, void *data) {
  int i, zi;
  efit_data *d = data;
  
  i = 0;
  for(zi=0; zi<d->n_zx; zi++) {
    if (d->p.type[zi] == 'c') {
      /* Parameter is a complex number. */
      d->coeff[d->p.index[zi]] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (d->p.type[zi] == 'i') {
      /* Parameter is a purely imaginary number. */
      d->coeff[d->p.index[zi]] = x[i]*I;
      i++;
    }
    else {
      /* Parameter is a purely real number. */
      d->coeff[d->p.index[zi]] = x[i];
      i++;
    }
  }

  zh_set_coeff(d->h, d->coeff);
  zhd(d->eval, d->evect, d->h, d->hd_w);

  return dchisq(d->eval, d->ex, d->h->n);
}

/* 
 * Objective function for fit to both energy levels and spin hamiltonians. 
 */
void eshfit_obj(size_t n, double *x, double *grad, void *data) {
  int i, zi;
  eshfit_data *d = data;
  double chisq;

  i = 0;
  /* Assign parameters that are varied to the coefficient arrays of both the
   * first order and the complete Hamiltonian; that is, all non-second order
   * parameters. */
  for(zi=0; zi<d->n_fozx; zi++) {
    if (d->p.type[zi] == 'c') {
      /* Parameter is a complex number. */
      d->h_coeff[d->p.index[zi]] = x[i]+x[i+1]*I;
      d->hfo_coeff[d->p.index[zi]] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (d->p.type[zi] == 'i') {
      /* Parameter is a purely imaginary number. */
      d->h_coeff[d->p.index[zi]] = x[i]*I;
      d->hfo_coeff[d->p.index[zi]] = x[i]*I;
      i++;
    }
    else {
      /* Parameter is a purely real number. */
      d->h_coeff[d->p.index[zi]] = x[i];
      d->hfo_coeff[d->p.index[zi]] = x[i];
      i++;
    }
  }
  /* Assign any remaining second order parameters to coefficient array of the
   * complete Hamiltonian. */
  for(zi=d->n_fozx; zi<d->n_zx; zi++) {
    if (d->p.type[zi] == 'c') {
      /* Parameter is a complex number. */
      d->h_coeff[d->p.index[zi]] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (d->p.type[zi] == 'i') {
      /* Parameter is a purely imaginary number. */
      d->h_coeff[d->p.index[zi]] = x[i]*I;
      i++;
    }
    else {
      /* Parameter is a purely real number. */
      d->h_coeff[d->p.index[zi]] = x[i];
      i++;
    }
  }

  /* Calculate the energy level chi^2. */
  zh_set_coeff(d->h, d->h_coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chisq = dchisq(d->h_eval, d->ex, d->h->n);

  /* Diagonalize the first order Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zh_set_coeff(d->hfo, d->hfo_coeff);
  zhd(d->hfo_eval, d->hfo_evect, d->hfo, d->hfod_w);

  for (i=0; i<d->nsh; i++) {
    zshp(d->sh_array[i], d->shx[i]->ten, d->shx[i]->l, d->shp_w_array[i]);
    zshi(sh_pa[i], d->shi_w_array[i]);
    chisq += shx[i]->chisq_weight * zchisq(sh_pa[i], shx[i]->pa);
  }

  return chisq;
}


