/*
    Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
 
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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_deriv.h>

#include "cfl_h.h"
#include "cfl_sh.h"
#include "cfl_error.h"
#include "cfl_config.h"
#include "cfl_min.h"
#include "cfl_h_fit.h"
//FIXME: currently pro data is not available when eshfit_alloc is called,
//therefore the zshp workspace is not being allocated. Also, check free
//functions. Okay, pro data should already be set.. adjust the zshp_alloc
//function, then remove this and add zshp_alloc to eshfit_alloc. Also, make sure
//this new zshp call sequence will not completely break the way we do covariant
//matrix estimates.

/*
 * Overview:
 * =========
 *
 * cfl_h_fit.c provides three objective functions for fitting crystal field
 * parameters to energy levels and spin Hamiltonian data.  These are: efit_obj,
 * eshfit_obj, and eshfit_hpro_obj, which are, respectively, used for fitting
 * to:
 *    + energy levels; 
 *    + energy levels in addition to spin Hamiltonian data for cases where the
 *      complete Hamiltonian does not contain any terms that also occur in the
 *      spin Hamiltonian;
 *    + and energy levels in addition to spin Hamiltonian data for cases
 *      where the complete Hamiltonian contains terms that also occur in the
 *      spin Hamiltonian.
 * Note that in order to correctly fit spin Hamiltonian data the (complete or
 * dedicated projection) Hamiltonian must have a small magnetic field term in
 * the z-direction, since the chi^2 algorithm assumes that the spin projection
 * values are ordered.
 *
 * The objective functions can be directly passed to all cfl_min algorithms (see
 * cfl_min.c).  In order to facilitate this, objective functions parse the real
 * double valued parameter array employed by the minimization routines to obtain
 * complex valued tensor coefficients.  These complex coefficients can then be
 * used as prefactors for tensor matrix elements to form the Hamiltonian. 
 *
 * Fitting is performed by a weighted chi^2 method.  The weighting can be setup
 * by calling the appropriate *fit_chi2 function prior to fitting, which will
 * return both the energy level and the spin Hamiltonian chi^2 contrbutions.
 * Additionally, such a call sets the weighting of the energy level chi^2 such
 * that it is ~1 to ensure we stay within machine precision.  Any weighting
 * between the energy level chi^2 and spin Hamiltonian chi^2 contributions can
 * then be set using shx_data->chisq_weight variable, which must be specified
 * relative to the energy level weighting.
 *
 * The covariance matrix is calculated following Press et al, Numerical recipes,
 * 3rd edition, section 15.2.  Sigmas are evaluated by assuming a model fit
 * (Press et al, page 780).  All weighting factors are set to unity, since while
 * the weighting is useful to force certain solutions, it cannot affect the
 * quality of the final fit. 
 *
 * The basic work flow consists of workspace allocation using the function
 * appropriate to the problem being solved, running the corresponding *fit_chi2
 * function, and minimizing the objective function by passing the workspace via
 * the additional void *data argument.  Upon completion of the minimization, the
 * workspace must be freed.
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
  data->echisq_weight = 1;

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
 *  h       Pointer to the complete Hamiltonian.  
 *  hpro    Pointer to the projection Hamiltonian; can be NULL if identical to
 *          h.
 *  coeff   Tensor coefficient array.
 *  ex      Experimental energy level data.  
 *  sh      Pointers to spin Hamiltonian.    
 *  shx     Array of pointers to spin Hamiltonian experimental data.  These must
 *          be in the same order as the terms in sh.  For Zeeman terms, the
 *          experimental data position is expected to coincide with the position
 *          of the first Zeeman term in sh.
 *  n_zx    The number of complex valued parameters to be fit to the complete
 *          Hamiltonian h.
 *  p       Array of pointers to parameters to be fit.
 */
eshfit_data *eshfit_data_alloc(zh *h, zh *hpro, complex double *coeff, ex_data
    *ex, zsh *sh, shx_data **shx, size_t n_zx, param_type **p) {
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

  data->sh_pa = (complex double **) malloc(sh->ninter*sizeof(complex double *));
  if (data->sh_pa == 0) {
    zhd_w_free(data->hd_w);
    free(data->h_evect);
    free(data->h_eval);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->sh_pa");
  }
  
  for (i=0; i<sh->ninter; i++) {
    data->sh_pa[i] = (complex double *) calloc(9,sizeof(complex double));
    if (data->sh_pa[i] == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (j=0; j<i; j++) {
        free(data->sh_pa[j]);
      }
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->sh_pa");
    }
  }
  
  /* Only alloc data if we require a separate projection Hamiltonian. */
  if (hpro != NULL) {
    data->hprod_w = zhd_w_alloc(hpro);
    if (data->hprod_w == 0) {
      zhd_w_free(data->hd_w);
      free(data->h_evect);
      free(data->h_eval);
      for (i=0; i<sh->ninter; i++) {
        free(data->sh_pa[i]);
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
      for (i=0; i<sh->ninter; i++) {
        free(data->sh_pa[i]);
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
      for (i=0; i<sh->ninter; i++) {
        free(data->sh_pa[i]);
      }
      free(data->sh_pa);
      free(data->hprod_w);
      free(data->hpro_evect);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->hpro_eval");
    }
  }


  data->h = h;
  data->hpro = hpro;
  data->coeff = coeff;
  data->ex = ex;
  data->sh = sh;
  data->shx = shx;
  data->n_zx = n_zx;
  data->p = p;
  data->echisq_weight = 1;

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
 */
inline double shchisq(complex double *pa, complex double *xpa) {
  int i;
  double chisq=0;

  for (i=0; i<9; i++) {
    chisq += pow(cabs(pa[i]) - cabs(xpa[i]), 2);
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

  return d->echisq_weight * echisq(d->eval, d->ex);
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
  chisq = d->echisq_weight * echisq(d->h_eval, d->ex);

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
  chisq = d->echisq_weight * echisq(d->h_eval, d->ex);

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

/*  Function used to get an initial estimate of chi^2 values, for energy level
 *  fit only. */
void efit_chi2(double *x, void *data, double *chi2) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->eval, d->evect, d->h, d->hd_w);
  chi2[0] = echisq(d->eval, d->ex);
  d->echisq_weight = 1/chi2[0];
}

/*  Function used to get an initial estimate of chi^2 values, in scenario where
 *  the complete Hamiltonian is the same as the projection Hamiltonian. */
void eshfit_chi2(double *x, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chi2[0] = echisq(d->h_eval, d->ex);
  d->echisq_weight = 1/chi2[0];

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
    chi2[i+1] += shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Function used to get an initial estimate of chi^2 values. */
void eshfit_hpro_chi2(double *x, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->coeff, x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  chi2[0] = echisq(d->h_eval, d->ex);
  d->echisq_weight = 1/chi2[0];

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
    chi2[i+1] = shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Returns the value of a single, specified, observable w.r.t. a given
 * parameter; used for covariance matrix evaluation of energy level fits. */
double efit_cov_df(double x, void *data) {
  cov_data *cov_d = (cov_data *)data;
  efit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p, d->coeff, cov_d->df_x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->eval, d->evect, d->h, d->hd_w);

  /* Return the value of the specified energy level. */
  return d->eval[d->ex->li[cov_d->obs_index]];
}

/* Returns the value of a single, specified, observable w.r.t. a given
 * parameter; used for covariance matrix evaluation of energy level and sh fits.
 */
double eshfit_cov_df(double x, void *data) {
  int i, j, sh_element;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p, d->coeff, cov_d->df_x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);
  
  if (cov_d->obs_index >= d->ex->n) {
    /* i, the SH term index (sh_index increments three times per i for Zeeman),
     * increments once every 6 observables for spin Hamiltonian observables. */
    /* FIXME: should increment only by 5 for quadrupole terms, since quadrupole
     * param matrices must be traceless once diagonal... */
    i = (cov_d->obs_index - d->ex->n)/6;
    /* The current element of the spin Hamiltonian upper diagonal; ranges from 0
     * to 5. */
    sh_element = (cov_d->obs_index - d->ex->n) - i*6;
    
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->h_evect, d->sh_a[cov_d->sh_index],
            d->shp_w_array[cov_d->sh_index]);
        cov_d->sh_index++;
      }
      if (sh_element < 6) {
        /* We have more elements to return with the current sh_index. */
        cov_d->sh_index -= 3;
      }
    }
    else {
      zshp(d->sh_pa[i], d->h_evect, d->sh_a[cov_d->sh_index],
          d->shp_w_array[cov_d->sh_index]);
      if (sh_element == 5) {
        /* We are returning the last element of the current sh_index. */
        cov_d->sh_index++;
      }
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    /* Return the upper diagonal entries of the spin Hamiltonian parameter
     * matrix. */
    if (sh_element < 3) {
      return d->sh_pa[i][sh_element];
    }
    else if (sh_element < 5) {
      return d->sh_pa[i][sh_element+1];
    }
    else {
      return d->sh_pa[i][8];
    }
  }
  else {
    /* Energy level case; return the value of the specified level.*/
    return d->h_eval[d->ex->li[cov_d->obs_index]];
  }
}

/* Returns the value of a single, specified, observable w.r.t. a given
 * parameter; used for covariance matrix evaluation of energy level and sh fits
 * when the projection Hamiltonian is different to the complete Hamiltonian.
 */
double eshfit_hpro_cov_df(double x, void *data) {
  int i, j, sh_element;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p, d->coeff, cov_d->df_x);
  zh_set_coeff(d->h, d->coeff);
  zhd(d->h_eval, d->h_evect, d->h, d->hd_w);

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zh_set_coeff(d->hpro, d->coeff);
  zhd(d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  if (cov_d->obs_index >= d->ex->n) {
    /* i, the SH term index (sh_index increments three times per i for Zeeman),
     * increments once every 6 observables for spin Hamiltonian observables. */
    /* FIXME: should increment only by 5 for quadrupole terms, since quadrupole
     * param matrices must be traceless once diagonal... */
    i = (cov_d->obs_index - d->ex->n)/6;
    /* The current element of the spin Hamiltonian upper diagonal; ranges from 0
     * to 5. */
    sh_element = (cov_d->obs_index - d->ex->n) - i*6;
    
    if (i == d->nzeeman) {
      /* The dimension of a single Zeeman term. */
      size_t dz = d->shx[i]->inv_data->m/3;
      for (j=0; j<3; j++) {
        zshp(&(d->sh_pa[i][j*dz]), d->hpro_evect, d->sh_a[cov_d->sh_index],
            d->shp_w_array[cov_d->sh_index]);
        cov_d->sh_index++;
      }
      if (sh_element < 6) {
        /* We have more elements to return with the current sh_index. */
        cov_d->sh_index -= 3;
      }
    }
    else {
      zshp(d->sh_pa[i], d->hpro_evect, d->sh_a[cov_d->sh_index],
          d->shp_w_array[cov_d->sh_index]);
      if (sh_element == 5) {
        /* We are returning the last element of the current sh_index. */
        cov_d->sh_index++;
      }
    }
    zshi(d->sh_pa[i], d->shi_w_array[i]);
    /* Return the upper diagonal entries of the spin Hamiltonian parameter
     * matrix. */
    if (sh_element < 3) {
      return d->sh_pa[i][sh_element];
    }
    else if (sh_element < 5) {
      return d->sh_pa[i][sh_element+1];
    }
    else {
      return d->sh_pa[i][8];
    }
  }
  else {
    /* Energy level case; return the value of the specified level.*/
    return d->h_eval[d->ex->li[cov_d->obs_index]];
  }
}

/* Common steps for covariance matrix estimation for *fit_cov functions. */
inline void covariance_helper(size_t m, size_t n, gsl_function F, double *x0,
    void *data, double sigma, double *cov_inv) {

  int i, j, k;
  double result, abserr;
  double *a;
  cov_data *cov_d;

  a = (double *) calloc(m*n, sizeof(double));
  if (a == 0) {
    CFL_ERROR_VOID("calloc failed for a");
  }
  cov_d = (cov_data *) malloc(sizeof(cov_data));
  if (cov_d == 0) {
    free(a);
    CFL_ERROR_VOID("malloc failed for cov_data");
  }
  cov_d->df_x = (double *) malloc(n*sizeof(double));
  if (cov_d->df_x == 0) {
    free(a);
    free(cov_d);
    CFL_ERROR_VOID("malloc failed for cov_d->df_x");
  }
  cov_d->obj_f_data = data;
  cov_d->sh_index = 0;

  /* Create copy of x0, since the derivative function modifies the parameter
   * value w.r.t. which we're differentiating. */
  memcpy(cov_d->df_x, x0, n*sizeof(double));
  F.params = cov_d;
  int status;
  for (i=0; i<n; i++) {
    cov_d->par_index = i;
    for (j=0; j<m; j++) {
      cov_d->obs_index = j;
      status = gsl_deriv_central(&F, x0[i], COV_DERIV_H, &result, &abserr);
      if (status) {
        CFL_ERROR_VOID("Derivative failure during covariance matrix estimation.\
            Disable covariance estimation matrix estimation, or attempt to \
            change COV_DERIV_H in cfl_conf.h and recompiling.");
        printf("deriv failure\n");
      }
      a[i*n+j] = result/sigma;
      /* Restore original value of the modified df_x element. */
      cov_d->df_x[i] = x0[i];
    }
  }

  /* Calculate a^T a. */
  for (k=0; k<n; k++) {
    for (i=0; i<n; i++) {
      cov_inv[i*n+k] = 0;
      for (j=0; j<m; j++) {
        cov_inv[i*n+k] += (a[j*n+k] * a[j*n+i]);
      }
    }
  }

  free(a);
  free(cov_d->df_x);
  free(cov_d);
}

/* Estimate the covariance matrix for an energy level fit. 
 *
 * Parameters
 * ----------
 *  x0      The parameters found by the minimization.
 *  cov_inv Pointer to space that will be overwritten with the inverse
 *          covariance matrix. 
 *  obj     The cfl_min_obj for which the minimization was run.
 */
void efit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) {
  size_t m, n;
  double sigma, chisq;
  gsl_function F;
  efit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables. */
  m = d->ex->n;

  F.function = &efit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * energy levels (pg. 780, Press et al. 3rd edition). */
  efit_chi2(x0, d, &chisq);
  sigma = sqrt(chisq/(m-n));

  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}

/* Estimate the covariance matrix for an energy level and spin Hamiltonian fit
 * for which the projection Hamiltonian is the same as the complete Hamiltonian. 
 *
 * Parameters
 * ----------
 *  x0      The parameters found by the minimization.
 *  cov_inv Pointer to space that will be overwritten with the inverse
 *          covariance matrix. 
 *  obj     The cfl_min_obj for which the minimization was run.
 */
void eshfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) {
  size_t m, n;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;
  
  /* The number of parameters. */
  n = obj->n;
  /* FIXME: The number of observables; we count 6 observables per spin
   * Hamiltonian term.  This is not quite correct for quadrupole terms, since
   * they are tracless after diagonalization. */
  m = d->ex->n + d->ninv*6;
  
  F.function = &eshfit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  eshfit_chi2(x0, d, chisq);
  sigma = sqrt((chisq[0] + chisq[1])/(m-n));
  
  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}

/* Estimate the covariance matrix for an energy level and spin Hamiltonian fit
 * for which the projection Hamiltonian is different from the complete
 * Hamiltonian. 
 *
 * Parameters
 * ----------
 *  x0      The parameters found by the minimization.
 *  cov_inv Pointer to space that will be overwritten with the inverse
 *          covariance matrix.  
 *  obj     The cfl_min_obj for which the minimization was run.
 */
void eshfit_hpro_cov(double *x0, double *cov_inv, cfl_min_obj *obj) {
  size_t m, n;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* FIXME: The number of observables; we count 6 observables per spin
   * Hamiltonian term.  This is not quite correct for quadrupole terms, since
   * they are tracless after diagonalization. */
  m = d->ex->n + d->ninv*6;
  
  F.function = &eshfit_hpro_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  eshfit_chi2(x0, d, chisq);
  sigma = sqrt((chisq[0] + chisq[1])/(m-n));

  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}
