/*
    Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
 
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "cfl_h.h"
#include "cfl_sh.h"
#include "cfl_error.h"
#include "basinhopping.h"
#include "cfl_h_fit.h"

/*
 * Overview:
 * =========
 *
 * cfl_h_fit.c provides three objective functions for fitting crystal field
 * parameters to energy levels and spin Hamiltonian data.  These are: efit_obj,
 * eshfit_obj, and eshfit_hpro_obj, which are, respectively, used for fitting to
 * energy levels, in addition to spin Hamiltonian data for cases where the
 * complete Hamiltonian does not contain any terms that also occur in the spin
 * Hamiltonian, and energy levels in addition to spin Hamiltonian data for cases
 * where the complete Hamiltonian contains terms that also occur in the spin
 * Hamiltonian.
 *
 * The objective functions can be directly passed to all cfl_min algorithms (see
 * cfl_min.c).  In order to facilitate this, objective functions parse the real
 * double valued parameter array employed by the minimization routines to obtain
 * complex valued tensor coefficients.  This then allows for the crystal field
 * Hamiltonian to be diagonalized, and any necessary projections to the spin
 * Hamiltonian space performed, to complete the optimization.  
 *
 * The basic work flow consists of workspace allocation using the function
 * appropriate to the problem being solved, which is then passed to the
 * objective function via the additional data argument.  Upon completion of the
 * minimization, the workspace must be freed again.
 *
 */

/* TODO:
 *  + Try adaptive chi^2 sigma using annealing.
 *  + Note: it should be written in the overview that weighting should be
 *  adjusted via sh parameters only. 
 */

/*
 * Alloc data for fitting to energy levels.
 *
 * Parameters
 * ----------
 *  h       Pointer to the Hamiltonian.  
 *  coeff   Coefficient array for h.
 *  ex      Experimental energy level data. 
 *  n_zx    The number of complex valued parameters to be fit to the Hamiltonian
 *  p       Array of pointers to parameters to be fit.
 */
efit_data *efit_data_alloc(zh *h, complex double *coeff, ex_data *ex, size_t n_zx,
    param_type **p) {
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

  data->evect = (complex double *) calloc(h->n*h->n,sizeof(complex double));
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

  data->h = h;
  data->coeff = coeff;
  data->ex = ex;
  data->n_zx = n_zx;
  data->p = p;

  return data;
}

void efit_data_free(efit_data *data) {
  zhd_w_free(data->hd_w);
  free(data->evect);
  free(data->eval);
  free(data);
}

/*
 * Alloc data for fitting to both energy levels and spin Hamiltonians.  
 *
 * We can get away with providing a single coefficient array even if a separate
 * hpro is specified, since both h and hpro are aware of the number of tensors
 * they are composed of, and they will not read beyond that number of
 * coefficients.  Furthermore, set_coeff does not modify the coefficient array.
 *
 * Parameters
 * ----------
 *  sh_a    Array of pointers to spin Hamiltonians. If the inversion involves a
 *          Zeeman term then this function expects three linearly independent
 *          Zeeman terms in a row.    
 *  nsh     The number of spin Hamiltonians to be fit.
 *  nzeeman The index of the first Zeeman term; for cases without Zeeman
 *          interaction, set to -1.
 *  h       Pointer to the complete Hamiltonian.  
 *  hpro    Pointer to the projection Hamiltonian; can be NULL if identical to
 *          h.
 *  coeff   Tensor coefficient array.
 *  ex      Experimental energy level data.  
 *  shx     Array of pointers to spin Hamiltonian experimental data.  These must
 *          be in the same order as the terms in sh.  For Zeeman terms, the
 *          experimental data position is expected to coincide with the position
 *          of the first Zeeman term in sh.
 *  n_zx    The number of complex valued parameters to be fit to the complete
 *          Hamiltonian h.
 *  p       Array of pointers to parameters to be fit.
 */
eshfit_data *eshfit_data_alloc(zsh **sh_a, size_t nsh, size_t nzeeman, zh *h, zh
    *hpro, complex double *coeff, ex_data *ex, shx_data
    **shx, size_t n_zx, param_type **p) {
  int i,j;
  size_t ninv;
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
  data->h_evect = (complex double *) calloc(h->n*h->n,sizeof(complex double));
  if (data->h_evect == 0) {
    zhd_w_free(data->hd_w);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_evect");
  }
  data->h_eval = (double *) calloc(h->n,sizeof(double));
  if (data->h_eval == 0) {
    zhd_w_free(data->hd_w);
    free(data->h_evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_eval");
  }
  data->shp_w_array = (zshp_w **) malloc(nsh*sizeof(zshp_w *));
  if (data->shp_w_array == 0) {
    zhd_w_free(data->hd_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->shp_w_array *");
  }

  /* Allocate spin Hamiltonian projection space. */
  for (i=0; i<nsh; i++) {
    data->shp_w_array[i] = zshp_w_alloc(sh_a[i]);
    if (data->shp_w_array == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<i; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      free(data);
      CFL_ERROR_NULL("zshp_w_alloc failed for data->shp_w_array");
    }
  }

  /* Determine the number of inversions; a Zeeman inversion requires three
   * terms. */
  if (nzeeman != -1) {
    ninv = nsh-2;
  }
  else {
    ninv = nsh;
  }

  data->shi_w_array = (zshi_w **) malloc(ninv*sizeof(zshi_w *));
  if (data->shp_w_array == 0) {
    zhd_w_free(data->hd_w);
    free(data->h_evect);
    free(data->h_eval);
    for (j=0; j<nsh; j++) {
      zshp_w_free(data->shp_w_array[j]);
    }
    free(data->shp_w_array);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->shi_w_array *");
  }
  data->sh_pa = (complex double **) malloc(ninv*sizeof(complex double *));
  if (data->sh_pa == 0) {
    zhd_w_free(data->hd_w);
    free(data->h_evect);
    free(data->h_eval);
    for (j=0; j<nsh; j++) {
      zshp_w_free(data->shp_w_array[j]);
    }
    free(data->shp_w_array);
    free(data->shi_w_array);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->sh_pa *");
  }

  for (i=0; i<ninv; i++) {
    data->shi_w_array[i] = zshi_w_alloc(shx[i]->inv_data);
    if (data->shi_w_array[i] == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<nsh; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      for (j=0; j<i; j++) {
        zshi_w_free(data->shi_w_array[j]);
        free(data->sh_pa[j]);
      }
      free(data->shi_w_array);
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("zshi_w_alloc failed for data->shi_w_array");
    }
    /* Size m for Zeeman shx is set to three times the size of a single term. */
    data->sh_pa[i] = (complex double *) calloc(shx[i]->inv_data->m,sizeof(complex
          double));
    if (data->sh_pa[i] == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<nsh; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      for (j=0; j<=i; j++) {
        zshi_w_free(data->shi_w_array[j]);
      }
      free(data->shi_w_array);
      for (j=0; j<i; j++) {
        free(data->sh_pa[j]);
      }
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->sh_pa");
    }
  }

  if (hpro != NULL) {
    /* Only alloc data if we require a separate projection Hamiltonian. */
    data->hprod_w = zhd_w_alloc(hpro);
    if (data->hprod_w == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<nsh; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      for (j=0; j<ninv; j++) {
        zshi_w_free(data->shi_w_array[j]);
      }
      free(data->shi_w_array);
      for (j=0; j<ninv; j++) {
        free(data->sh_pa[j]);
      }
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("zhd_w_alloc failed for data->hprod_w");
    }
    data->hpro_evect = (complex double *) calloc(hpro->n*hpro->n,sizeof(complex
          double));
    if (data->hpro_evect == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<nsh; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      for (j=0; j<ninv; j++) {
        zshi_w_free(data->shi_w_array[j]);
      }
      free(data->shi_w_array);
      for (j=0; j<ninv; j++) {
        free(data->sh_pa[j]);
      }
      free(data->sh_pa);
      free(data->hprod_w);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->hpro_evect");
    }
    data->hpro_eval = (double *) calloc(hpro->n,sizeof(double));
    if (data->hpro_eval == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<nsh; j++) {
        zshp_w_free(data->shp_w_array[j]);
      }
      free(data->shp_w_array);
      for (j=0; j<ninv; j++) {
        zshi_w_free(data->shi_w_array[j]);
      }
      free(data->shi_w_array);
      for (j=0; j<ninv; j++) {
        free(data->sh_pa[j]);
      }
      free(data->sh_pa);
      free(data->hprod_w);
      free(data->hpro_evect);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->hpro_eval");
    }
  }

  data->sh_a = sh_a;
  data->h = h;
  data->hpro = hpro;
  data->coeff = coeff;
  data->ex = ex;
  data->nsh = nsh;
  data->nzeeman = nzeeman;
  data->ninv = ninv;
  data->shx = shx;
  data->n_zx = n_zx;
  data->p = p;

  return data;
}

void eshfit_data_free(eshfit_data *data) {
  int i;

  zhd_w_free(data->hd_w);
  free(data->h_evect);
  free(data->h_eval);
  if (data->hpro != NULL) {
    zhd_w_free(data->hprod_w);
    free(data->hpro_evect);
    free(data->hpro_eval);
  }
  for (i=0; i<data->nsh; i++) {
    zshp_w_free(data->shp_w_array[i]);
  }
  free(data->shp_w_array);
  for (i=0; i<data->ninv; i++) {
    zshi_w_free(data->shi_w_array[i]);
    free(data->sh_pa[i]);
  }
  free(data->shi_w_array);
  free(data->sh_pa);
  free(data);
}


/* Chi^2 for energy levels. 
 *
 * Parameters
 * ----------
 *  e         The theoretical energy array.
 *  ex_data   Pointer to the experimental data struct.
 */
inline double echisq(double *e, ex_data *d) {
  int i;
  double chisq=0;

  for (i=0; i<d->n; i++) {
    chisq += pow(e[d->li[i]] - d->e[i], 2);
  }

  return chisq;
}

/* Chi^2 for spin Hamiltonian data. 
 *
 * Parameters
 * ----------
 *  pa    The theoretical parameter array.
 *  xpa   The experimental parameter array. 
 *  n     The number of parameters.
 */
inline double shchisq(complex double *pa, complex double *xpa) {
  int i;
  double chisq=0;

  for (i=0; i<9; i++) {
    chisq += pow(cabs(pa[i] - xpa[i]), 2);
  }
  
  return chisq;
}

/* Parse an array of doubles into an array of complex doubles using param_type
 * data. 
 *
 * Parameters
 * ----------
 *  n_zx      The number of complex parameters
 *  p         Array of param_type data.
 *  coeff     Complex array which will be overwritten with the parsed data.
 *  x         Source of data. 
 */
inline void parse_param_data(size_t n_zx, param_type **p, complex double *coeff,
    double *x) {
  int i, zi;

  i = 0;
  for(zi=0; zi<n_zx; zi++) {
    if (p[zi]->type == 'c') {
      /* Parameter is a complex number. */
      coeff[p[zi]->index] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (p[zi]->type == 'i') {
      /* Parameter is a purely imaginary number. */
      coeff[p[zi]->index] = x[i]*I;
      i++;
    }
    else {
      /* Parameter is a purely real number. */
      coeff[p[zi]->index] = x[i];
      i++;
    }
  }
}

/* Objective function for fit to energy levels only. */
double efit_obj(size_t n, double *x, double *grad, void *data) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->eval, d->evect, d->h, d->hd_w);

  return echisq(d->eval, d->ex);
}

/*  Objective function for fit to both energy levels and spin Hamiltonians in
 *  case the complete Hamiltonian is the same as the projection Hamiltonian. */
double eshfit_obj(size_t n, double *x, double *grad, void *data) {
  int i, j, sh_index;
  eshfit_data *d = data;
  double chisq;

  parse_param_data(d->n_zx, d->p, d->coeff, x);

  /* Calculate the energy level chi^2. */
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chisq = echisq(d->h_eval, d->ex);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  sh_index = 0;
  for (i=0; i<d->ninv; i++) {
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->h_evect, d->sh_a[sh_index],
            d->shp_w_array[sh_index]);
        sh_index++;
      }
    }
    else {
      zshp(d->sh_pa[i], d->h_evect, d->sh_a[sh_index],
          d->shp_w_array[sh_index]);
      sh_index++;
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }

  return chisq;
}

/*  Objective function for fit to both energy levels and spin Hamiltonians. */
double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) {
  int i, j, sh_index;
  eshfit_data *d = data;
  double chisq;

  parse_param_data(d->n_zx, d->p, d->coeff, x);

  /* Calculate the energy level chi^2. */
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chisq = echisq(d->h_eval, d->ex);

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zh_set_coeff(d->hpro, d->coeff);
  zhd(d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  sh_index = 0;
  for (i=0; i<d->ninv; i++) {
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->hpro_evect, d->sh_a[sh_index],
            d->shp_w_array[sh_index]);
        sh_index++;
      }
    }
    else {
      zshp(d->sh_pa[i], d->hpro_evect, d->sh_a[sh_index],
          d->shp_w_array[sh_index]);
      sh_index++;
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
  return chisq;
}

/*  Function used to get an initial estimate of chi^2 values, in scenario where
 *  the complete Hamiltonian is the same as the projection Hamiltonian. */
void eshfit_chi2(size_t n, double *x, double *grad, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);

  /* Calculate the energy level chi^2. */
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chi2[0] = echisq(d->h_eval, d->ex);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  sh_index = 0;
  for (i=0; i<d->ninv; i++) {
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->h_evect, d->sh_a[sh_index],
            d->shp_w_array[sh_index]);
        sh_index++;
      }
    }
    else {
      zshp(d->sh_pa[i], d->h_evect, d->sh_a[sh_index],
          d->shp_w_array[sh_index]);
      sh_index++;
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    chi2[i+1] += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Function used to get an initial estimate of chi^2 values. */
void eshfit_hpro_chi2(size_t n, double *x, double *grad, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);

  /* Calculate the energy level chi^2. */
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chi2[0] = echisq(d->h_eval, d->ex);

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zh_set_coeff(d->hpro, d->coeff);
  zhd(d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  sh_index = 0;
  for (i=0; i<d->ninv; i++) {
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->hpro_evect, d->sh_a[sh_index],
            d->shp_w_array[sh_index]);
        sh_index++;
      }
    }
    else {
      zshp(d->sh_pa[i], d->hpro_evect, d->sh_a[sh_index],
          d->shp_w_array[sh_index]);
      sh_index++;
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    chi2[i+1] = d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}
