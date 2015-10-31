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

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#include "cfl_h.h"
#include "cfl_sh.h"
#include "cfl_error.h"
#include "cfl_config.h"
#include "cfl_min.h"
#include "cfl_h_fit.h"

/*
 * Overview:
 * =========
 *
 * cfl_h_fit.c provides several objective functions for fitting crystal field
 * parameters to energy levels and spin Hamiltonian data.  These are: efit_obj,
 * mhfit_obj, eshfit_obj, and eshfit_hpro_obj, which are, respectively, used for
 * fitting to:
 *    + energy levels; 
 *    + energy levels of multiple, distinct, Hamiltonians;
 *    + energy levels in addition to spin Hamiltonian data for cases where the
 *      complete Hamiltonian does not contain any interactions that also occur
 *      in the spin Hamiltonian;
 *    + and energy levels in addition to spin Hamiltonian data for cases
 *      where the complete Hamiltonian contains interactions that also occur in
 *      the spin Hamiltonian.
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
 * 3rd edition, section 15.2.  sigmas are evaluated by assuming a model fit
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
 *  ex      Experimental energy level data. 
 *  n_zx    The number of complex valued parameters to be fit to the Hamiltonian
 *  p       Array of pointers to parameters to be fit.
 */
efit_data *efit_data_alloc(zh *h, ex_data *ex, int n_zx, param_type **p) {
  efit_data *data;

  if (h->coeff == NULL) {
    CFL_ERROR_NULL("h is missing coefficients; set with zh_set_coeff prior to" \
        "calling efit_data_alloc");
  }
  data = (efit_data *) malloc(sizeof(efit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for data");
  }

  data->eval = (double *) calloc(h->n,sizeof(double));
  if (data->eval == 0) {
    free(data);
    CFL_ERROR_NULL("calloc failed for data->eval");
  }

  data->hd_w = zhd_w_alloc('N', h);
  if (data->hd_w == 0) {
    free(data->eval);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }

  data->h = h;
  data->ex = ex;
  data->n_zx = n_zx;
  data->p = p;
  data->echisq_weight = 1;

  return data;
}

void efit_data_free(efit_data *data) {
  zhd_w_free(data->hd_w);
  free(data->eval);
  free(data);
}

/* Alloc data for fitting to multiple Hamiltonians. 
 *
 * Parameters
 * ----------
 *  n           The number of Hamiltonians. 
 *  input_data  Array of eigenvalue data structs. 
 *  ha          Array of length n containing pointers to Hamiltonians.
 *  weights     Array of length n with each entry specifying the chi^2 weighting
 *              of the corresponding ha entry.
 *  bc_blockdim The barycenter block dimension for each corresponding ha entry.
 *              If 0, no barycenter shift is applied.  For entries of value n,
 *              the barycenter shift for n dimensional blocks of energy levels
 *              is calculated and subtracted from the theoretical eigenvalues
 *              prior to the chi^2 evaluation.  This is useful for ensuring that
 *              magnetic or hyperfine data available for a subset of CF levels
 *              is not dominated by a shift of the entire multiplet.  If
 *              non-zero, the experimental data must be in blocks of the
 *              specified size with no missing levels. 
 *  exa         Array of pointers to experimental energy level data. 
 *  n_zx        The number of complex valued parameters to be fit to the
 *              complete Hamiltonians.
 *  p           Array of length n to arrays of pointers to parameter type
 *              structs. 
 */
mhfit_data *mhfit_data_alloc(int n, zh **ha, double *weights, 
    int *bc_blockdim, ex_data **exa, int n_zx, param_type ***p) {
  int i, j, nhd_w;
  int num_procs;
  mhfit_data *data;
  long *lwork;
  int *iwork;

  data = (mhfit_data *) malloc(sizeof(mhfit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for mhfit_data");
  }

  iwork = (int *) calloc(n,sizeof(int));
  if (iwork == 0) {
    free(data);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  lwork = (long *) calloc(n,sizeof(long));
  if (lwork == 0) {
    free(iwork);
    free(data);
    CFL_ERROR_NULL("calloc failed for lwork");
  }
  data->hi = (int *) calloc(n,sizeof(int));
  if (data->hi == 0) {
    free(iwork);
    free(lwork);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->hi");
  }

#ifdef _OPENMP
  /* If we evaluate Hamiltonians in parallel, we need a workspace for each
   * Hamiltonian. */
  nhd_w = n;
  num_procs = omp_get_num_procs();
  if (num_procs > nhd_w) {
    num_procs = num_procs/nhd_w;
  } 
  else {
    num_procs = 1;
  }
  for (i = 0; i < n; i++) {
    iwork[i] = i;
    data->hi[i] = i;
    ha[i]->num_procs = num_procs;
  }
#else
  /* We only need a diag workspace for each unique Hamiltonian. */
  nhd_w = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (j >= nhd_w) {
        data->hi[i] = j;
        lwork[nhd_w] = ha[i]->slabels->hash;
        iwork[j] = i;
        nhd_w++;
        break;
      }
      else if (ha[i]->slabels->hash == lwork[j]) {
        data->hi[i] = j;
        break;
      }
    }
  }
#endif /* _OPENMP */

  data->hd_w = (zhd_w **) malloc(nhd_w*sizeof(zhd_w *));
  if (data->hd_w == 0) {
    free(data->hi);
    free(data);
    free(iwork);
    free(lwork);
    CFL_ERROR_NULL("malloc failed for data->hd_w");
  }
  data->h_eval = (double **) malloc(nhd_w*sizeof(double *));
  if (data->h_eval == 0) {
    free(data->hi);
    free(data->hd_w);
    free(data);
    free(iwork);
    free(lwork);
    CFL_ERROR_NULL("malloc failed for data->h_eval");
  }

  for (i = 0; i < nhd_w; i++) {
    data->h_eval[i] = (double *) calloc(ha[iwork[i]]->n,sizeof(double));
    if (data->h_eval[i] == 0) {
      for (j = 0; j < i; j++) {
        free(data->h_eval[j]);
        free(data->hd_w[j]);
      }
      free(data->hi);
      free(data);
      free(iwork);
      free(lwork);
      CFL_ERROR_NULL("calloc failed for data->h_eval[i]");
    }

    data->hd_w[i] = (zhd_w *) zhd_w_alloc('N', ha[iwork[i]]);
    if (data->hd_w[i] == 0) {
      for (j = 0; j < i; j++) {
        free(data->h_eval[j]);
        free(data->hd_w[j]);
      }
      free(data->h_eval[i]);
      free(data->hi);
      free(data);
      free(iwork);
      free(lwork);
      CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w[i]");
    }
  }

  free(iwork);
  free(lwork);

  data->n = n;
  data->ha = ha;
  data->weights = weights;
  data->bc_blockdim = bc_blockdim;
  data->exa = exa;
  data->nhd_w = nhd_w;
  data->n_zx = n_zx;
  data->p = p;
  
  return data;
}

void mhfit_data_free(mhfit_data *data) {
  int i;

  free(data->hi);
  for (i = 0; i < data->nhd_w; i++) {
    free(data->h_eval[i]);
    zhd_w_free(data->hd_w[i]);
  }
  free(data->h_eval);
  free(data->hd_w);
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
 *          h.  The tensor order of hpro must match the tensor order of h, since
 *          they share the same coefficent array; hpro will ignore any
 *          coefficients that are soley required by h.  Furthermore, if the
 *          caller has set hpro->coeff, the caller must retain a copy of this
 *          pointer, since eshfit_data_alloc will alias hpro->coeff with
 *          h->coeff.  
 *  ex      Experimental energy level data.  
 *  sh      Pointer to spin Hamiltonian.    
 *  shx     Array of pointers to spin Hamiltonian experimental data.  These must
 *          be in the same order as the terms in sh.  For Zeeman terms, the
 *          experimental data position is expected to coincide with the position
 *          of the first Zeeman term in sh.
 *  n_zx    The number of complex valued parameters to be fit to both the
 *          complete Hamiltonian h and the spin Hamiltonian sh.
 *  n_ushx  The number of parameters that are unique to the spin Hamiltonian sh;
 *          that is, not in the Hamiltonian h.
 *  p       Array of pointers to parameters to be fit.
 */
eshfit_data *eshfit_data_alloc(zh *h, zh *hpro, ex_data *ex, zsh *sh, 
    shx_data **shx, int n_zx, int n_ushx, param_type **p) {
  int i,j;
  eshfit_data *data;

  if (h->coeff == NULL) {
    CFL_ERROR_NULL("h is missing coefficients; set with zh_set_coeff prior to" \
        "calling eshfit_data_alloc");
  }
  data = (eshfit_data *) malloc(sizeof(eshfit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for eshfit_data");
  }
  data->h_evect = (complex double *) calloc(h->n*h->n,sizeof(complex double));
  if (data->h_evect == 0) {
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_evect");
  }
  data->h_eval = (double *) calloc(h->n,sizeof(double));
  if (data->h_eval == 0) {
    free(data->h_evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->h_eval");
  }
  data->hd_w = zhd_w_alloc('V', h);
  if (data->hd_w == 0) {
    free(data->h_evect);
    free(data->h_eval);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }
  data->shp_w = zshp_w_alloc(sh);
  if (data->shp_w == 0) {
    free(data->h_evect);
    free(data->h_eval);
    zhd_w_free(data->hd_w);
    free(data);
    CFL_ERROR_NULL("zshp_w_alloc failed for data->shp_w");
  }
  data->sh_pa = (complex double **) malloc(sh->ninter*sizeof(complex double *));
  if (data->sh_pa == 0) {
    free(data->h_evect);
    free(data->h_eval);
    zhd_w_free(data->hd_w);
    free(data->shp_w);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->sh_pa");
  }
  for (i = 0; i < sh->ninter; i++) {
    data->sh_pa[i] = (complex double *) calloc(9,sizeof(complex double));
    if (data->sh_pa[i] == 0) {
      free(data->h_evect);
      free(data->h_eval);
      zhd_w_free(data->hd_w);
      for (j = 0; j < i; j++) {
        free(data->sh_pa[j]);
      }
      free(data->shp_w);
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->sh_pa[i]");
    }
  }

  /* Only alloc data if we require a separate projection Hamiltonian. */
  if (hpro != NULL) {
    data->hpro_evect = (complex double *) calloc(hpro->n*hpro->n,sizeof(complex
          double));
    if (data->hpro_evect == 0) {
      free(data->h_evect);
      free(data->h_eval);
      zhd_w_free(data->hd_w);
      for (i = 0; i < sh->ninter; i++) {
        free(data->sh_pa[i]);
      }
      free(data->shp_w);
      free(data->sh_pa);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->hpro_evect");
    }
    data->hpro_eval = (double *) calloc(hpro->n,sizeof(double));
    if (data->hpro_eval == 0) {
      free(data->h_evect);
      free(data->h_eval);
      zhd_w_free(data->hd_w);
      for (i = 0; i < sh->ninter; i++) {
        free(data->sh_pa[i]);
      }
      free(data->shp_w);
      free(data->sh_pa);
      free(data->hpro_evect);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->hpro_eval");
    }
    data->hprod_w = zhd_w_alloc('V', hpro);
    if (data->hprod_w == 0) {
      free(data->h_evect);
      free(data->h_eval);
      zhd_w_free(data->hd_w);
      for (i = 0; i < sh->ninter; i++) {
        free(data->sh_pa[i]);
      }
      free(data->shp_w);
      free(data->sh_pa);
      free(data->hpro_evect);
      free(data->hpro_eval);
      free(data);
      CFL_ERROR_NULL("zhd_w_alloc failed for data->hprod_w");
    }
    /* Alias the coeff ptrs of h and hpro. */
    hpro->coeff = h->coeff;
  }

  data->h = h;
  data->hpro = hpro;
  data->ex = ex;
  data->sh = sh;
  data->shx = shx;
  data->n_zx = n_zx;
  data->n_ushx = n_ushx;
  data->p = p;
  data->echisq_weight = 1;

  return data;
}

void eshfit_data_free(eshfit_data *data) {
  int i;

  free(data->h_evect);
  free(data->h_eval);
  zhd_w_free(data->hd_w);
  if (data->hpro != NULL) {
    free(data->hpro_evect);
    free(data->hpro_eval);
    zhd_w_free(data->hprod_w);
  }
  zshp_w_free(data->shp_w);
  for (i = 0; i < data->sh->ninter; i++) {
    free(data->sh_pa[i]);
  }
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
  double chisq;

  chisq = 0;
  for (i = 0; i < d->n; i++) {
    chisq += pow(d->e[i] - e[d->li[i]], 2);
  }

  return chisq;
}

/* Chi^2 for multi Hamiltonian fit.  Optionally accounts for barycenter shifts
 * between blocks of experimental data. 
 *
 * Parameters
 * ----------
 *  e           The theoretical energy array.
 *  ex_data     Pointer to the experimental data struct.
 *  bc_blockdim The barycenter block dimension for this set of ex_data. 
 */
inline double mhchisq(double *e, ex_data *d, int bc_blockdim) {
  int i, j;
  double chisq, bc_shift;
  
  chisq = 0;
  if (bc_blockdim != 0) {
    bc_shift = 0;
    for (i = 0; i < d->n; i++) {
      if (i % bc_blockdim == 0) {
        bc_shift = 0;
        for (j = i; j < i+bc_blockdim; j++) {
          bc_shift += d->e[j] - e[d->li[j]];
        }
        bc_shift /= bc_blockdim;
      }
      chisq += pow(d->e[i] - (e[d->li[i]] - bc_shift), 2);
    }
  }
  else {
    for (i = 0; i < d->n; i++) {
      chisq += pow(d->e[i] - e[d->li[i]], 2);
    }
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
  double chisq;

  chisq = 0;
  for (i = 0; i < 9; i++) {
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
inline void parse_param_data(int n_zx, param_type **p, complex double *coeff,
    double *x) {
  int i, ii;

  i = 0;
  for(ii = 0; ii < n_zx; ii++) {
    if (p[ii]->type == 'c') {
      /* Parameter is a complex number. */
      coeff[p[ii]->index] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (p[ii]->type == 'i') {
      /* Parameter is a purely imaginary number. */
      coeff[p[ii]->index] = x[i]*I;
      i++;
    }
    else {
      /* Parameter is a purely real number. */
      coeff[p[ii]->index] = x[i];
      i++;
    }
  }
}

/* Parse an array of doubles into an array of complex doubles using param_type
 * data for a Hamiltonian coeff array.  Furthermore, we parse the nuclear dipole
 * and quadrupole coupling constants to sh->proj_data, if these interactions are
 * present.  If these are not present in the Hamiltonian, then these must be the
 * last parameters in x.
 *
 * Parameters
 * ----------
 *  n_zx      The number of complex parameters.
 *  n_ushx    The number of parameters unique to sh; that is, not in coeff.
 *  p         Array of param_type data.
 *  coeff     Complex array which will be overwritten with the parsed data.
 *  sh        Pointer to spin Hamiltonian.    
 *  x         Source of data. 
 */
inline void sh_parse_param_data(int n_zx, int n_ushx, param_type **p, 
    complex double *coeff, zsh *sh, double *x) {
  int i, ii;
  
  i = 0;
  for(ii = 0; ii < n_zx-n_ushx; ii++) {
    if (p[ii]->type == 'c') {
      /* Parameter is a complex number. */
      coeff[p[ii]->index] = x[i]+x[i+1]*I;
      i+=2;
    }
    else if (p[ii]->type == 'i') {
      /* Parameter is a purely imaginary number. */
      coeff[p[ii]->index] = x[i]*I;
      i++;
    }
    else if (p[ii]->type == 'r') {
      /* Parameter is a purely real number. */
      coeff[p[ii]->index] = x[i];
      i++;
    }
    else if (p[ii]->type == 'h') {
      /* Nuclear dipole coupling constant. */
      sh->pro_data[sh->pd_map[0]]->coupling = x[i];
      coeff[p[ii]->index] = x[i];
    }
    else if (p[ii]->type == 'q') {
      /* Nuclear quadrupole coupling constant. */
      sh->pro_data[sh->pd_map[1]]->coupling = x[i];
      coeff[p[ii]->index] = x[i];
    }
  }
  /* Set parameters unique to spin Hamiltonian. */
  for (ii = n_zx-n_ushx; ii < n_zx; ii++) {
    if (p[ii]->type == 'h') {
      /* Nuclear dipole coupling constant. */
      sh->pro_data[sh->pd_map[0]]->coupling = x[i];
    }
    else if (p[ii]->type == 'q') {
      /* Nuclear quadrupole coupling constant. */
      sh->pro_data[sh->pd_map[1]]->coupling = x[i];
    }
  }

}


/* Objective function for fit to energy levels only. */
double efit_obj(size_t n, double *x, double *grad, void *data) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->h->coeff, x);
  zhd('N', d->eval, NULL, d->h, d->hd_w);

  return d->echisq_weight * echisq(d->eval, d->ex);
}

/* Objective function for multi-eigenvalue vector fit. */
double mhfit_obj(size_t n, double *x, double *grad, void *data) {
  int i, hi;
  double chisq;
  mhfit_data *d = data;

  chisq = 0;
#pragma omp parallel for private(i, hi) reduction(+:chisq) schedule(static)
  for (i = 0; i < d->n; i++) {
    hi = d->hi[i];
    parse_param_data(d->n_zx, d->p[i], d->ha[i]->coeff, x);
    zhd('N', d->h_eval[hi], NULL, d->ha[i], d->hd_w[hi]);
    chisq += d->weights[i]*mhchisq(d->h_eval[hi], d->exa[i], d->bc_blockdim[i]);
  }

  return chisq;
}

/*  Objective function for fit to both energy levels and spin Hamiltonians in
 *  case the complete Hamiltonian is the same as the projection Hamiltonian. */
double eshfit_obj(size_t n, double *x, double *grad, void *data) {
  int i;
  double chisq;
  eshfit_data *d = data;

  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh, x);
  /* Calculate the energy level chi^2. */
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);
  chisq = d->echisq_weight * echisq(d->h_eval, d->ex);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], d->h_evect, i, d->sh, d->shp_w);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }

  return chisq;
}

/*  Objective function for fit to both energy levels and spin Hamiltonians. */
double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) {
  int i, j, sh_index;
  double chisq;
  eshfit_data *d = data;

  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh, x);
  
  /* Calculate the energy level chi^2. */
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);
  chisq = d->echisq_weight * echisq(d->h_eval, d->ex);

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], d->h_evect, i, d->sh, d->shp_w);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }

  return chisq;
}

/*  Function used to get an initial estimate of chi^2 values, for energy level
 *  fit only. */
void efit_chi2(double *x, void *data, double *chi2) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->h->coeff, x);
  zhd('N', d->eval, NULL, d->h, d->hd_w);
  *chi2 = echisq(d->eval, d->ex);
  d->echisq_weight = 1; //CFL_MIN_START_CHI2/(*chi2);
}

/*  Function used to get an initial estimate of chi^2 values, for
 *  multi-eigenvalue vector fit. */
void mhfit_chi2(double *x, void *data, double *chi2) {
  int i, hi;
  double chisq;
  mhfit_data *d = data;
  
  *chi2 = 0;
  for (i = 0; i < d->n; i++) {
    hi = d->hi[i];
    parse_param_data(d->n_zx, d->p[i], d->ha[i]->coeff, x);
    zhd('N', d->h_eval[hi], NULL, d->ha[i], d->hd_w[hi]);
    *chi2 += d->weights[i]*mhchisq(d->h_eval[hi], d->exa[i], d->bc_blockdim[i]);
  }
  d->echisq_weight = 1; //CFL_MIN_START_CHI2/(*chi2);
}

/*  Function used to get an initial estimate of chi^2 values, in scenario where
 *  the complete Hamiltonian is the same as the projection Hamiltonian. */
void eshfit_chi2(double *x, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh, x);
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);
  *chi2 = echisq(d->h_eval, d->ex);
  d->echisq_weight = CFL_MIN_START_CHI2/(*chi2);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], d->h_evect, i, d->sh, d->shp_w);
    chi2[i+1] = shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Function used to get an initial estimate of chi^2 values, in scenario where
 * the complete Hamiltonian is not the same as the projection Hamiltonian. */
void eshfit_hpro_chi2(double *x, void *data, double *chi2) {
  int i, j, sh_index;
  eshfit_data *d = data;

  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh, x);
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);
  chi2[0] = echisq(d->h_eval, d->ex);
  d->echisq_weight = CFL_MIN_START_CHI2/chi2[0];

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], d->h_evect, i, d->sh, d->shp_w);
    chi2[i+1] = shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Function for evaluating the covariance matrix for an energy level fit.
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are choosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double efit_cov_df(double x, void *data) {
  cov_data *cov_d = (cov_data *)data;
  efit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p, d->h->coeff, cov_d->df_x);
  zhd('N', d->eval, NULL, d->h, d->hd_w);

  /* Return the value of the specified energy level. */
  return d->eval[d->ex->li[cov_d->obs_index]];
}

/* Function for evaluating the covariance matrix for a multi-eigenvalue vector
 * fit. 
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are choosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double mhfit_cov_df(double x, void *data) {
  int i, hi, ex_n;
  double chisq;
  cov_data *cov_d = (cov_data *)data;
  mhfit_data *d = cov_d->obj_f_data;

  i = 0;
  ex_n = 0;
  do {
    ex_n += d->exa[i]->n;
    i++;
  } while (cov_d->obs_index > ex_n);
  i--;
  ex_n -= d->exa[i]->n;

  hi = d->hi[i];

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p[i], d->ha[i]->coeff, cov_d->df_x);
  zhd('N', d->h_eval[hi], NULL, d->ha[i], d->hd_w[hi]);

  return d->h_eval[hi][d->exa[i]->li[cov_d->obs_index-ex_n]];
}

/* Function for evaluating the covariance matrix for an energy level and spin
 * Hamiltonian fit in the scenario where the complete Hamiltonian is the same as
 * the projection Hamiltonian. 
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are choosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double eshfit_cov_df(double x, void *data) {
  int i, shi, shel;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh,
      cov_d->df_x);
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);

  if (cov_d->obs_index >= d->ex->n) {
    /* obs_index corresponds to an observable from the spin Hamiltonian. */

    /* The current spin Hamiltonian index. */
    shi = cov_d->shi_index[cov_d->obs_index - d->ex->n];
    /* The current spin Hamiltonian element. */
    shel = cov_d->shel_index[cov_d->obs_index - d->ex->n];

    zshp(d->sh_pa[shi], d->h_evect, shi, d->sh, d->shp_w);

    /* Return the upper diagonal entries of the parameter matrix. */
    if (shel < 3) {
      /* The first row. */
      return d->sh_pa[i][shel];
    }
    else if (shel < 5) {
      /* Second row; diagonal starts at 1. */
      return d->sh_pa[i][shel+1];
    }
    else {
      /* Last row and column. */
      return d->sh_pa[i][8];
    }
  }
  else {
    /* obs_index corresponds to an energy level observable; return the value of
     * the specified level.*/
    return d->h_eval[d->ex->li[cov_d->obs_index]];
  }
}

/* Function for evaluating the covariance matrix for an energy level and spin
 * Hamiltonian fit in the scenario where the complete Hamiltonian is not the
 * same as the projection Hamiltonian. 
 *
 * Returns the value of a single observable given the parameter x, which
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are choosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double eshfit_hpro_cov_df(double x, void *data) {
  int i, shi, shel;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->n_ushx, d->p, d->h->coeff, d->sh,
      cov_d->df_x);
  zhd('V', d->h_eval, d->h_evect, d->h, d->hd_w);

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  if (cov_d->obs_index >= d->ex->n) {
    /* obs_index corresponds to an observable from the spin Hamiltonian. */

    /* The current spin Hamiltonian index. */
    shi = cov_d->shi_index[cov_d->obs_index - d->ex->n];
    /* The current spin Hamiltonian element. */
    shel = cov_d->shel_index[cov_d->obs_index - d->ex->n];

    zshp(d->sh_pa[shi], d->h_evect, shi, d->sh, d->shp_w);

    /* Return the upper diagonal entries of the parameter matrix. */
    if (shel < 3) {
      /* The first row. */
      return d->sh_pa[i][shel];
    }
    else if (shel < 5) {
      /* Second row; diagonal starts at 1. */
      return d->sh_pa[i][shel+1];
    }
    else {
      /* Last row and column. */
      return d->sh_pa[i][8];
    }
  }
  else {
    /* obs_index corresponds to an energy level observable; return the value of
     * the specified level.*/
    return d->h_eval[d->ex->li[cov_d->obs_index]];
  }
}

/* Common steps for covariance matrix estimation for *fit_cov functions. */
inline void covariance_helper(size_t m, size_t n, size_t *shi_index, size_t
    *shel_index, gsl_function F, double *x0, void *data, double sigma, double
    *cov_inv) {
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

  cov_d->shi_index = shi_index;
  cov_d->shel_index = shel_index;
  cov_d->obj_f_data = data;

  /* Create copy of x0, since the derivative function modifies the parameter
   * value w.r.t. which we're differentiating. */
  memcpy(cov_d->df_x, x0, n*sizeof(double));
  F.params = cov_d;
  int status;
  for (i = 0; i < n; i++) {
    cov_d->par_index = i;
    for (j = 0; j < m; j++) {
      cov_d->obs_index = j;
      status = gsl_deriv_central(&F, x0[i], COV_DERIV_H, &result, &abserr);
      if (status) {
        CFL_ERROR_VOID("Derivative failure during covariance matrix estimation.\
            Disable covariance estimation matrix estimation, or attempt to \
            change COV_DERIV_H in cfl_conf.h and recompiling.");
      }
      a[i*n+j] = result/sigma;
      /* Restore original value of the modified df_x element. */
      cov_d->df_x[i] = x0[i];
    }
  }

  /* Calculate a^T a. */
  for (k = 0; k < n; k++) {
    for (i = 0; i < n; i++) {
      cov_inv[i*n+k] = 0;
      for (j = 0; j < m; j++) {
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

  covariance_helper(m, n, NULL, NULL, F, x0, d, sigma, cov_inv);
}


/* Estimate the covariance matrix for a for a multi-eigenvalue vector fit. 
 *
 * Parameters
 * ----------
 *  x0      The parameters found by the minimization.
 *  cov_inv Pointer to space that will be overwritten with the inverse
 *          covariance matrix. 
 *  obj     The cfl_min_obj for which the minimization was run.
 */
void mhfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) {
  int i;
  size_t m, n;
  double sigma, chisq;
  gsl_function F;
  mhfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables. */
  m = 0;
  for (i = 0; i < d->n; i++) {
    m += d->exa[i]->n;
  }

  F.function = &mhfit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * energy levels (pg. 780, Press et al. 3rd edition). */
  mhfit_chi2(x0, d, &chisq);
  sigma = sqrt(chisq/(m-n));

  covariance_helper(m, n, NULL, NULL, F, x0, d, sigma, cov_inv);
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
  int i, j, obs_i;
  size_t m, n, *shi_index, *shel_index;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables.  We count 6 observables per spin Hamiltonian
   * term, except for the quadrupole term which is traceless and thus only
   * contributes 5 observables. */
  m = d->ex->n;
  for (i = 0; i < d->sh->ninter; i++) {
    if (!strcmp("quadrupole", d->sh->inter[i])) {
      m += 5;
    }
    else {
      m += 6;
    }
  }

  /* Create index arrays that map the obs_index, minus the number of energy
   * level observables, to the corresponding spin Hamiltonian interaction and
   * spin Hamiltonian elements. */
  shi_index = (size_t *) malloc((m-d->ex->n)*sizeof(size_t));
  if (shi_index == 0) {
    CFL_ERROR_VOID("malloc failed for cov_d->shi_index");
  }
  shel_index = (size_t *)malloc((m-d->ex->n)*sizeof(size_t));
  if (shel_index == 0) {
    free(shi_index);
    CFL_ERROR_VOID("malloc failed for cov_d->shel_index");
  }
  obs_i = 0;
  for (i = 0; i < d->sh->ninter; i++) {
    if (!strcmp("quadrupole", d->sh->inter[i])) {
      for (j = 0; j < 5; j++) {
        shi_index[obs_i] = i;
        shel_index[obs_i] = j;
        obs_i++;
      }
    }
    else {
      for (j = 0; j < 6; j++) {
        shi_index[obs_i] = i;
        shel_index[obs_i] = j;
        obs_i++;
      }
    }    
  }

  F.function = &eshfit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  eshfit_chi2(x0, d, chisq);
  sigma = sqrt((chisq[0] + chisq[1])/(m-n));

  covariance_helper(m, n, shi_index, shel_index, F, x0, d, sigma, cov_inv);

  free(shi_index);
  free(shel_index);
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
  int i, j, obs_i;
  size_t m, n, *shi_index, *shel_index;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables.  We count 6 observables per spin Hamiltonian
   * term, except for the quadrupole term which is traceless and thus only
   * contributes 5 observables. */
  m = d->ex->n;
  for (i = 0; i < d->sh->ninter; i++) {
    if (!strcmp("quadrupole", d->sh->inter[i])) {
      m += 5;
    }
    else {
      m += 6;
    }
  }

  /* Create index arrays that map the obs_index, minus the number of energy
   * level observables, to the corresponding spin Hamiltonian interaction and
   * spin Hamiltonian elements. */
  shi_index = (size_t *) malloc((m-d->ex->n)*sizeof(size_t));
  if (shi_index == 0) {
    CFL_ERROR_VOID("malloc failed for cov_d->shi_index");
  }
  shel_index = (size_t *)malloc((m-d->ex->n)*sizeof(size_t));
  if (shel_index == 0) {
    free(shi_index);
    CFL_ERROR_VOID("malloc failed for cov_d->shel_index");
  }
  obs_i = 0;
  for (i = 0; i < d->sh->ninter; i++) {
    if (!strcmp("quadrupole", d->sh->inter[i])) {
      for (j = 0; j < 5; j++) {
        shi_index[obs_i] = i;
        shel_index[obs_i] = j;
        obs_i++;
      }
    }
    else {
      for (j = 0; j < 6; j++) {
        shi_index[obs_i] = i;
        shel_index[obs_i] = j;
        obs_i++;
      }
    }    
  }

  F.function = &eshfit_hpro_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  eshfit_chi2(x0, d, chisq);
  sigma = sqrt((chisq[0] + chisq[1])/(m-n));

  covariance_helper(m, n, shi_index, shel_index, F, x0, d, sigma, cov_inv);

  free(shi_index);
  free(shel_index);
}
