/*
   Copyright (C) 2014-2016 Sebastian Horvath (sebastian.horvath@gmail.com)

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
 * Fitting is performed by a weighted chi^2 method.  Weights for both energy
 * level data and spin Hamiltonian data can be specified in the ex and shx
 * structs, respectively.  There's a global weighting factor for all energy
 * level data pertaining to a single CF Hamiltonian, whereas spin Hamiltonian
 * interactions can be weighted individually.  Furthermore, a the *fit_chi2
 * functions can be employed to get initial and final values for the chi2
 * contribution of each fit component.
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
 * Alloc data for fitting to energy levels.  Since weighting for energy levels
 * is a global factor, any value set in ex will not be referenced by efit.
 *
 * Parameters
 * ----------
 *  job     Set to 'N' if state label sorting is not required for h, in which
 *          case ex->lah, ex->ildh, and ex->fldh will not be referenced.  If
 *          state label sorting is required for h, set to 'S', in which case all
 *          elements of ex must be alloced. 
 *  h       Pointer to the Hamiltonian.  
 *  ex      Experimental energy level data. 
 *  n_zx    The number of complex valued parameters to be fit to the Hamiltonian
 *  p       Array of pointers to parameters to be fit.
 */
efit_data *efit_data_alloc(char job, zh *h, ex_data *ex, int n_zx,
    param_type **p) {
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
  if (job == 'S') {
    data->hd_w = zhd_w_alloc('V', h);
  }
  else {
    data->hd_w = zhd_w_alloc('N', h);
  }
  if (data->hd_w == 0) {
    free(data->eval);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }
  if (job == 'S') {
    data->evect = (complex double *) calloc(h->n*h->n,sizeof(complex double));
    if (data->evect == 0) {
      free(data->eval);
      zhd_w_free(data->hd_w);
      free(data);
      CFL_ERROR_NULL("calloc failed for data->evect");
    }
  }
  else {
    data->evect = NULL;
  }

  data->job = job;
  data->h = h;
  data->ex = ex;
  data->n_zx = n_zx;
  data->p = p;

  return data;
}

void efit_data_free(efit_data *data) {
  zhd_w_free(data->hd_w);
  free(data->eval);
  if (data->job = 'S') {
    free(data->evect);
  }
  free(data);
}

/* Alloc data for fitting to multiple Hamiltonians. 
 *
 * Parameters
 * ----------
 *  job         For each h in ha, set to 'N' if state label sorting is not
 *              required, in which case ex->lah, ex->ildh, and ex->fldh will not
 *              be referenced for the respective h.  If state label sorting is
 *              required, set to 'S', in which case all elements of the
 *              corresponding ex must be alloced. 
 *  n           The number of Hamiltonians. 
 *  input_data  Array of eigenvalue data structs. 
 *  ha          Array of length n containing pointers to Hamiltonians.
 *  exa         Array of pointers to experimental energy level data. 
 *  n_zx        The number of complex valued parameters to be fit to each of the
 *              Hamiltonians.
 *  p           Array of length n to arrays of pointers to parameter type
 *              structs. 
 */
mhfit_data *mhfit_data_alloc(char *job, int n, zh **ha, ex_data **exa, 
    int *n_zx, param_type ***p) {
  int i, j, nhd_w;
  int num_procs;
  mhfit_data *data;
  long *lwork;
  int *iwork;

  data = (mhfit_data *) malloc(sizeof(mhfit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for mhfit_data");
  }
  data->job = (char *) malloc(n*sizeof(char));
  if (data->job == 0) {
    free(data);
    CFL_ERROR_NULL("malloc failed for data->job");
  }
  memcpy(data->job, job, n*sizeof(char));
  data->sl_sort = 0;
  
  for (i = 0; i < n; i++) {
    if (job[i] == 'S') {
      data->sl_sort = 1;
      break;
    }
  }

  iwork = (int *) calloc(n,sizeof(int));
  if (iwork == 0) {
    free(data->job);
    free(data);
    CFL_ERROR_NULL("calloc failed for iwork");
  }

  lwork = (long *) calloc(n,sizeof(long));
  if (lwork == 0) {
    free(data->job);
    free(iwork);
    free(data);
    CFL_ERROR_NULL("calloc failed for lwork");
  }
  data->hi = (int *) calloc(n,sizeof(int));
  if (data->hi == 0) {
    free(data->job);
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
      /* A new tensor; record j index. */
      if (j >= nhd_w) {
        data->hi[i] = j;
        lwork[nhd_w] = ha[i]->hh;
        iwork[j] = i;
        nhd_w++;
        break;
      }
      /* Check whether Hamiltonian hash matches. */
      else if (ha[i]->hh == lwork[j]) {
        data->hi[i] = j;
        break;
      }
    }
  }
#endif /* _OPENMP */

  data->hd_w = (zhd_w **) malloc(nhd_w*sizeof(zhd_w *));
  if (data->hd_w == 0) {
    free(data->job);
    free(data->hi);
    free(data);
    free(iwork);
    free(lwork);
    CFL_ERROR_NULL("malloc failed for data->hd_w");
  }
  data->eval = (double **) malloc(nhd_w*sizeof(double *));
  if (data->eval == 0) {
    free(data->job);
    free(data->hi);
    free(data->hd_w);
    free(data);
    free(iwork);
    free(lwork);
    CFL_ERROR_NULL("malloc failed for data->eval");
  }

  for (i = 0; i < nhd_w; i++) {
    data->eval[i] = (double *) calloc(ha[iwork[i]]->n,sizeof(double));
    if (data->eval[i] == 0) {
      for (j = 0; j < i; j++) {
        free(data->eval[j]);
        free(data->hd_w[j]);
      }
      free(data->job);
      free(data->hi);
      free(data);
      free(iwork);
      free(lwork);
      CFL_ERROR_NULL("calloc failed for data->eval[i]");
    }
    
    if (data->sl_sort) {
      data->hd_w[i] = (zhd_w *) zhd_w_alloc('V', ha[iwork[i]]);
    }
    else {
      data->hd_w[i] = (zhd_w *) zhd_w_alloc('N', ha[iwork[i]]);
    }
    if (data->hd_w[i] == 0) {
      for (j = 0; j < i; j++) {
        free(data->eval[j]);
        free(data->hd_w[j]);
      }
      free(data->eval[i]);
      free(data->job);
      free(data->hi);
      free(data);
      free(iwork);
      free(lwork);
      CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w[i]");
    }
  }

  if (data->sl_sort) {
    data->evect = (complex double **) malloc(nhd_w*sizeof(complex double *));
    if (data->evect == 0) {
      for (i = 0; i < nhd_w; i++) {
        free(data->eval[i]);
        free(data->hd_w[i]);
      }
      free(data->job);
      free(data->hi);
      free(data);
      free(iwork);
      free(lwork);
      CFL_ERROR_NULL("malloc failed for data->evect");
    }
    for (i = 0; i < nhd_w; i++) {
      data->evect[i] = (complex double *) calloc(ha[iwork[i]]->n*ha[iwork[i]]->n,\
          sizeof(complex double));
      if (data->evect[i] == 0) {
        for (j = 0; j < i; j++) {
          free(data->evect[j]);
        }
        for (j = 0; j < nhd_w; j++) {
          free(data->eval[j]);
          free(data->hd_w[j]);
        }
        free(data->job);
        free(data->hi);
        free(data);
        free(iwork);
        free(lwork);
        CFL_ERROR_NULL("calloc failed for data->evect[i]");
      }
    }
  }
  else {
    data->evect = NULL;
  }

  free(iwork);
  free(lwork);

  data->n = n;
  data->ha = ha;
  data->exa = exa;
  data->nhd_w = nhd_w;
  data->n_zx = n_zx;
  data->p = p;

  return data;
}

void mhfit_data_free(mhfit_data *data) {
  int i;

  free(data->job);
  free(data->hi);
  for (i = 0; i < data->nhd_w; i++) {
    free(data->eval[i]);
    zhd_w_free(data->hd_w[i]);
  }
  if (data->sl_sort) {
    for (i = 0; i < data->nhd_w; i++) {
      free(data->evect[i]);
    }
    free(data->evect);
  }
  free(data->eval);
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
 *  job     Set to 'N' if state label sorting is not required for h, in which
 *          case ex->lah, ex->ildh, and ex->fldh will not be referenced.  If
 *          state label sorting is required for h, set to 'S', in which case all
 *          elements of ex must be alloced. 
 *  inv_job Set to 'S' if spin Hamiltonian parameter tensor symmeterization by
 *          SVD is required, otherwise set to 'N'.
 *  h       Pointer to the complete Hamiltonian.  
 *  hpro    Pointer to the projection Hamiltonian; can be NULL if identical to
 *          h.  The tensor order of hpro must match the tensor order of h, since
 *          they share the same coefficient array; hpro will ignore any
 *          coefficients that are solely required by h.  Furthermore, if the
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
 *  p       Array of pointers to parameters to be fit.
 */
eshfit_data *eshfit_data_alloc(char job, char inv_job, zh *h, zh *hpro, ex_data *ex, zsh *sh, 
    shx_data **shx, int n_zx, param_type **p) {
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
  data->evect = (complex double *) calloc(h->n*h->n,sizeof(complex double));
  if (data->evect == 0) {
    free(data);
    CFL_ERROR_NULL("calloc failed for data->evect");
  }
  data->eval = (double *) calloc(h->n,sizeof(double));
  if (data->eval == 0) {
    free(data->evect);
    free(data);
    CFL_ERROR_NULL("calloc failed for data->eval");
  }
  data->hd_w = zhd_w_alloc('V', h);
  if (data->hd_w == 0) {
    free(data->evect);
    free(data->eval);
    free(data);
    CFL_ERROR_NULL("zhd_w_alloc failed for data->hd_w");
  }
  data->shp_w = zshp_w_alloc(inv_job, sh);
  if (data->shp_w == 0) {
    free(data->evect);
    free(data->eval);
    zhd_w_free(data->hd_w);
    free(data);
    CFL_ERROR_NULL("zshp_w_alloc failed for data->shp_w");
  }
  data->sh_pa = (double **) malloc(sh->ninter*sizeof(double *));
  if (data->sh_pa == 0) {
    free(data->evect);
    free(data->eval);
    zhd_w_free(data->hd_w);
    free(data->shp_w);
    free(data);
    CFL_ERROR_NULL("malloc failed for data->sh_pa");
  }
  for (i = 0; i < sh->ninter; i++) {
    data->sh_pa[i] = (double *) calloc(9,sizeof(double));
    if (data->sh_pa[i] == 0) {
      free(data->evect);
      free(data->eval);
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
      free(data->evect);
      free(data->eval);
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
      free(data->evect);
      free(data->eval);
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
      free(data->evect);
      free(data->eval);
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

  data->job = job;
  data->h = h;
  data->hpro = hpro;
  data->ex = ex;
  data->sh = sh;
  data->shx = shx;
  data->n_zx = n_zx;
  data->p = p;
  data->shi_index = NULL;
  data->shel_index = NULL;

  return data;
}

void eshfit_data_free(eshfit_data *data) {
  int i;

  free(data->evect);
  free(data->eval);
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
  if (data->shi_index != NULL) {
    free(data->shi_index);
    free(data->shel_index);
  }
  free(data);
}

/*
 * Alloc data for fitting to multiple CF Hamiltonians and spin Hamiltonians.  
 *
 * Calls eshfit_obj (or eshfit_hpro_obj where appropriate) for multiple CF
 * Hamiltonian/spin Hamiltonian pairs. 
 *
 * Parameters
 * ----------
 * n          The number of CF Hamiltonian/spin Hamiltonian pairs.
 * eshfit_d   Array of pointers to previously allocated eshfit_data structs. 
 */
meshfit_data *meshfit_data_alloc(int n, eshfit_data **eshfit_d) {
  meshfit_data *data;
  data = (meshfit_data *) malloc(sizeof(meshfit_data));
  if (data == 0) {
    CFL_ERROR_NULL("malloc failed for meshfit_data");
  }

  data->n = n;
  data->data = eshfit_d;

  return data;
}

void meshfit_data_free(meshfit_data *data) {
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
  /* The chisq contribution due to absolute energy level data. */
  for (i = 0; i < d->n_a; i++) {
    chisq += pow(e[d->la[i]] - d->e[i], 2);
  }
  /* The chisq contribution due to difference energy level data. */ 
  for (i = 0; i < d->n_d; i++) {
    chisq += pow(fabs(e[d->fld[i]] - e[d->ild[i]]) - d->e[i+d->n_a], 2);
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
inline double shchisq(double *pa, double *xpa) {
  int i;
  double chisq;

  chisq = 0;
  for (i = 0; i < 9; i++) {
    chisq += pow(fabs(pa[i]) - fabs(xpa[i]), 2);
  }
  
  return chisq;
}

/* Determine the indices that match energy levels of a Hamiltonian according to
 * a set of specified LS coupled state labels. 
 *
 * Parameters
 * ----------
 *  ex      The experimental data struct. 
 *  h       The Hamiltonian for which to sort perform the state label sort. 
 *  evect   The eigenvectors of the h. 
 */
inline void find_sort_indices(ex_data *ex, zh *h, complex double *evect) {
  int i, j, pj, obs_cnt;
  double pv;

  obs_cnt = 0;
  for (i = 0; i < h->n; i++) {
    pv = 0;      /* The abs principal value of this evect. */
    for (j = 0; j < h->n; j++) {
      if (cabs(evect[i*h->n+j]) > pv) {
        pv = cabs(evect[i*h->n+j]);
        pj = j;
      }
    }
    for (j = 0; j < ex->n_a; j++) {
      if (h->slabels->lh[pj] == ex->lah[j]) {
        ex->la[j] = i;
        obs_cnt++;
      }
    }
    for (j = 0; j < ex->n_d; j++) {
      if (h->slabels->lh[pj] == ex->ildh[j]) {
        ex->ild[j] = i;
        obs_cnt++;
      }
      else if (h->slabels->lh[pj] == ex->fldh[j]) {
        ex->fld[j] = i;
        obs_cnt++;
      }
    }
    /* Two observables counted per single difference data point. */
    if (obs_cnt >= ex->n_a + 2 * ex->n_d) {
      break;
    }
  }
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
  int i;
  
  /* This allows for a single x array to be mapped
   * into coeff arrays corresponding to Hamiltonians with different matrix
   * elements.*/
  for(i = 0; i < n_zx; i++) {
    if (p[i]->type == 'c') {
      /* Parameter is a complex number. */
      coeff[p[i]->ci] = x[p[i]->xi]+x[p[i]->xi+1]*I;
    }
    else if (p[i]->type == 'i') {
      /* Parameter is a purely imaginary number. */
      coeff[p[i]->ci] = x[p[i]->xi]*I;
    }
    else {
      /* Parameter is a purely real number. */
      coeff[p[i]->ci] = x[p[i]->xi];
    }
  }
}

/* Parse an array of doubles into an array of complex doubles using param_type
 * data for a Hamiltonian coeff array.  Furthermore, we parse the nuclear dipole
 * and quadrupole coupling constants to sh->proj_data, if these interactions are
 * present.  
 *
 * Parameters
 * ----------
 *  n_zx      The number of complex parameters.
 *  p         Array of param_type data.
 *  coeff     Complex array which will be overwritten with the parsed data.
 *  sh        Pointer to spin Hamiltonian.    
 *  x         Source of data. 
 */
inline void sh_parse_param_data(int n_zx, param_type **p, 
    complex double *coeff, zsh *sh, double *x) {
  int i;

  for (i=0; i<n_zx; i++) {
    if (p[i]->type == 'c') {
      /* Parameter is a complex number. */
      coeff[p[i]->ci] = x[p[i]->xi]+x[p[i]->xi+1]*I;
    }
    else if (p[i]->type == 'i') {
      /* Parameter is a purely imaginary number. */
      coeff[p[i]->ci] = x[p[i]->xi];
    }
    else if (p[i]->type == 'r') {
      /* Parameter is a purely real number. */
      coeff[p[i]->ci] = x[p[i]->xi];
    }
    else if (p[i]->type == 'h') {
      /* Nuclear dipole coupling constant. */
      sh->pro_data[sh->pd_map[0]]->coupling = x[p[i]->xi];
      coeff[p[i]->ci] = x[p[i]->xi];
    }
    else if (p[i]->type == 'q') {
      /* Nuclear quadrupole coupling constant. */
      sh->pro_data[sh->pd_map[1]]->coupling = x[p[i]->xi];
      coeff[p[i]->ci] = x[p[i]->xi];
    }
  }
}


/* Objective function for fit to energy levels only. */
double efit_obj(size_t n, double *x, double *grad, void *data) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->h->coeff, x);
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }

  return echisq(d->eval, d->ex);
}


/* Objective function for multi-eigenvalue vector fit. */
double mhfit_obj(size_t n, double *x, double *grad, void *data) {
  int i, hi;
  double chisq;
  mhfit_data *d = data;

  chisq = 0;
  if (d->sl_sort) {
#pragma omp parallel for private(i, hi) reduction(+:chisq) schedule(static)
    for (i = 0; i < d->n; i++) {
      hi = d->hi[i];
      parse_param_data(d->n_zx[i], d->p[i], d->ha[i]->coeff, x);
      if (d->job[i] == 'S') {
        zhd('V', d->eval[hi], d->evect[hi], d->ha[i], d->hd_w[hi]);
        find_sort_indices(d->exa[i], d->ha[i], d->evect[hi]);
      }
      else {
        zhd('N', d->eval[hi], NULL, d->ha[i], d->hd_w[hi]);
      }
      chisq += d->exa[i]->chisq_weight*echisq(d->eval[hi], d->exa[i]);
    }
  }
  else {
#pragma omp parallel for private(i, hi) reduction(+:chisq) schedule(static)
    for (i = 0; i < d->n; i++) {
      hi = d->hi[i];
      parse_param_data(d->n_zx[i], d->p[i], d->ha[i]->coeff, x);
      zhd('N', d->eval[hi], NULL, d->ha[i], d->hd_w[hi]);
      chisq += d->exa[i]->chisq_weight*echisq(d->eval[hi], d->exa[i]);
    }
  }

  return chisq;
}

/*  Objective function for fit to both energy levels and spin Hamiltonians in
 *  case the complete Hamiltonian is the same as the projection Hamiltonian. */
double eshfit_obj(size_t n, double *x, double *grad, void *data) {
  int i;
  double chisq;
  eshfit_data *d = data;

  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh, x);
  /* Calculate the energy level chi^2. */ 
  zhd('V', d->eval, d->evect, d->h, d->hd_w);

  if (d->job == 'S') {
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  /* Check whether energy level fitting is disabled; i.e., a pure sh fit. */
  if (d->ex->n_obs == 0) {
    chisq = 0;
  } 
  else {
    chisq = d->ex->chisq_weight * echisq(d->eval, d->ex);
  }

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], NULL, d->evect, i, d->sh, d->shp_w);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
  
  return chisq;
}

/*  Objective function for fit to both energy levels and spin Hamiltonians. */
double eshfit_hpro_obj(size_t n, double *x, double *grad, void *data) {
  int i;
  double chisq;
  eshfit_data *d = data;

  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh, x);

  /* Calculate the energy level chi^2. */
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }
  /* Check whether energy level fitting is disabled; i.e., a pure sh fit. */
  if (d->ex->n_obs == 0) {
    chisq = 0;
  } 
  else {
    chisq = d->ex->chisq_weight * echisq(d->eval, d->ex);
  }

  /* Diagonalize the projection Hamiltonian. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], NULL, d->hpro_evect, i, d->sh, d->shp_w);
    chisq += d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }

  return chisq;
}

/*  Objective function for fit to energy levels and multiple spin Hamiltonians.
 */
double meshfit_obj(size_t n, double *x, double *grad, void *data) {
  int i;
  double chisq = 0;
  meshfit_data *d = data;

#pragma omp parallel for private(i) reduction(+:chisq) schedule(static)
  for (i=0; i<d->n; i++) {
    if (d->data[i]->hpro == NULL) {
      chisq += eshfit_obj(n, x, grad, d->data[i]);
    }
    else {
      chisq += eshfit_hpro_obj(n, x, grad, d->data[i]);
    }
  }
  
  return chisq;
}


/*  Function used to get an initial estimate of chi^2 values, for energy level
 *  fit only. */
void efit_chi2(double *x, void *data, double *chi2) {
  efit_data *d = data;

  parse_param_data(d->n_zx, d->p, d->h->coeff, x);
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }

  *chi2 = echisq(d->eval, d->ex);
}

/*  Function used to get an initial estimate of chi^2 values, for
 *  multi-eigenvalue vector fit.  chi2 must be of length d->n, that is, the
 *  number of crystal field Hamiltonians. */
void mhfit_chi2(double *x, void *data, double *chi2) {
  int i, hi;
  double chisq;
  mhfit_data *d = data;
  
  if (d->sl_sort) {
    for (i = 0; i < d->n; i++) {
      hi = d->hi[i];
      parse_param_data(d->n_zx[i], d->p[i], d->ha[i]->coeff, x);
      if (d->job[i] == 'S') {
        zhd('V', d->eval[hi], d->evect[hi], d->ha[i], d->hd_w[hi]);
        find_sort_indices(d->exa[i], d->ha[i], d->evect[hi]);
      }
      else {
        zhd('N', d->eval[hi], NULL, d->ha[i], d->hd_w[hi]);
      }
      chi2[i] = d->exa[i]->chisq_weight*echisq(d->eval[hi], d->exa[i]);
    }
  }
  else {
    for (i = 0; i < d->n; i++) {
      hi = d->hi[i];
      parse_param_data(d->n_zx[i], d->p[i], d->ha[i]->coeff, x);
      zhd('N', d->eval[hi], NULL, d->ha[i], d->hd_w[hi]);
      chi2[i] = d->exa[i]->chisq_weight*echisq(d->eval[hi], d->exa[i]);
    }
  }
}

/*  Function used to get an initial estimate of chi^2 values, in scenario where
 *  the complete Hamiltonian is the same as the projection Hamiltonian. */
void eshfit_chi2(double *x, void *data, double *chi2) {
  int i;
  eshfit_data *d = data;
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh, x);
  
  zhd('V', d->eval, d->evect, d->h, d->hd_w);
  if (d->job == 'S') {
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  
  /* Get chi2 provided energy-level fitting isn't disabled. */
  if (d->ex->n_obs == 0) {
    chi2[0] = 0;
  } 
  else {
    chi2[0] = d->ex->chisq_weight * echisq(d->eval, d->ex);
  }
  
  /* Project out the spin Hamiltonian, and invert the result to obtain the spin
   * Hamiltonian parameters. */
  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], NULL, d->evect, i, d->sh, d->shp_w);
    chi2[i+1] = d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/* Function used to get an initial estimate of chi^2 values, in scenario where
 * the complete Hamiltonian is not the same as the projection Hamiltonian. */
void eshfit_hpro_chi2(double *x, void *data, double *chi2) {
  int i;
  eshfit_data *d = data;

  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh, x);
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }
  
  /* Get chi2 provided energy-level fitting isn't disabled. */
  if (d->ex->n_obs == 0) {
    chi2[0] = 0;
  } 
  else {
    chi2[0] = d->ex->chisq_weight * echisq(d->eval, d->ex);
  }

  /* Diagonalize the projection Hamiltonian, project out the spin Hamiltonian,
   * and invert the result to obtain the spin Hamiltonian parameters. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  for (i = 0; i < d->sh->ninter; i++) {
    zshp(d->sh_pa[i], NULL, d->hpro_evect, i, d->sh, d->shp_w);
    chi2[i+1] = d->shx[i]->chisq_weight * shchisq(d->sh_pa[i], d->shx[i]->pa);
  }
}

/*  Objective function for fit to energy levels and multiple spin Hamiltonians.
 */
void meshfit_chi2(double *x, void *data, double *chi2) {
  int i, chi2_index;
  meshfit_data *d = data;
  
  chi2_index = 0;
  for (i=0; i<d->n; i++) {
    if (d->data[i]->hpro == NULL) {
      eshfit_chi2(x, d->data[i], &chi2[chi2_index]);
    }
    else {
      eshfit_hpro_chi2(x, d->data[i], &chi2[chi2_index]);
    }
    /* The chi2 array contains an entry for each sh interaction plus 1 for the
     * total energy level chi2. */
    chi2_index += d->data[i]->sh->ninter + 1;
  }
}

/* Function for evaluating the covariance matrix for an energy level fit.
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are chosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double efit_cov_df(double x, void *data) {
  cov_data *cov_d = (cov_data *)data;
  efit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx, d->p, d->h->coeff, cov_d->df_x);
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }

  /* Return the value of the specified energy level. */
  if (cov_d->obs_index < d->ex->n_a) {
    /* Absolute energy level. */
    return d->eval[d->ex->la[cov_d->obs_index]];
  }
  else {
    /* Energy level difference. */
    return fabs(d->eval[d->ex->fld[cov_d->obs_index - d->ex->n_a]] \
        - d->eval[d->ex->ild[cov_d->obs_index - d->ex->n_a]]);
  }
}

/* Function for evaluating the covariance matrix for a multi-eigenvalue vector
 * fit. 
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are chosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double mhfit_cov_df(double x, void *data) {
  int i, hi, ex_n, n_a, lobs_index;
  cov_data *cov_d = (cov_data *)data;
  mhfit_data *d = cov_d->obj_f_data;

  /* Determine the index i of the hamiltonian which corresponds to the current
   * observable. */
  i = 0;
  ex_n = 0;
  do {
    ex_n += d->exa[i]->n_obs;
    i++;
  } while (ex_n < cov_d->obs_index + 1);
  i--;
  ex_n -= d->exa[i]->n_obs;
  
  /* Local version of observable index. */
  lobs_index = cov_d->obs_index - ex_n;

  hi = d->hi[i];
  n_a = d->exa[hi]->n_a;

  cov_d->df_x[cov_d->par_index] = x;
  parse_param_data(d->n_zx[i], d->p[i], d->ha[i]->coeff, cov_d->df_x);
  
  if (d->job[i] == 'S') {
    zhd('V', d->eval[hi], d->evect[hi], d->ha[i], d->hd_w[hi]);
    find_sort_indices(d->exa[i], d->ha[i], d->evect[hi]);
  }
  else {
    zhd('N', d->eval[hi], NULL, d->ha[i], d->hd_w[hi]);
  }

  if (lobs_index < n_a) {
    /* Absolute energy level. */
    return d->eval[hi][d->exa[hi]->la[lobs_index]];
  }
  else {
    /* Energy level difference. */
    return fabs(d->eval[hi][d->exa[hi]->fld[lobs_index - n_a]] \
        - d->eval[hi][d->exa[hi]->ild[lobs_index - n_a]]);
  }
}

/* Function for evaluating the covariance matrix for an energy level and spin
 * Hamiltonian fit in the scenario where the complete Hamiltonian is the same as
 * the projection Hamiltonian. 
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are chosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double eshfit_cov_df(double x, void *data) {
  int i, shi, shel;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh,
      cov_d->df_x);
  zhd('V', d->eval, d->evect, d->h, d->hd_w);
  if (d->job == 'S') {
    find_sort_indices(d->ex, d->h, d->evect); 
  }

  if (cov_d->obs_index >= d->ex->n_obs) {
    /* obs_index corresponds to an observable from the spin Hamiltonian. */

    /* The current spin Hamiltonian index. */
    shi = d->shi_index[cov_d->obs_index - d->ex->n_obs];
    /* The current spin Hamiltonian element. */
    shel = d->shel_index[cov_d->obs_index - d->ex->n_obs];

    zshp(d->sh_pa[shi], NULL, d->evect, shi, d->sh, d->shp_w);

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
    if (cov_d->obs_index < d->ex->n_a) {
      /* Absolute energy level. */
      return d->eval[d->ex->la[cov_d->obs_index]];
    }
    else {
      /* Energy level difference. */
      return fabs(d->eval[d->ex->fld[cov_d->obs_index - d->ex->n_a]] \
          - d->eval[d->ex->ild[cov_d->obs_index - d->ex->n_a]]);
    }
  }
}

/* Function for evaluating the covariance matrix for an energy level and spin
 * Hamiltonian fit in the scenario where the complete Hamiltonian is not the
 * same as the projection Hamiltonian. 
 *
 * Returns the value of a single observable given the parameter x, which
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are chosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double eshfit_hpro_cov_df(double x, void *data) {
  int i, shi, shel;
  cov_data *cov_d = (cov_data *)data;
  eshfit_data *d = cov_d->obj_f_data;

  cov_d->df_x[cov_d->par_index] = x;
  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh,
      cov_d->df_x);

  /* Diagonalize the primary Hamiltonian. */
  if (d->job == 'S') {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    find_sort_indices(d->ex, d->h, d->evect); 
  }
  else {
    zhd('N', d->eval, NULL, d->h, d->hd_w);
  }

  /* Diagonalize the projection Hamiltonian. */
  zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);

  if (cov_d->obs_index >= d->ex->n_obs) {
    /* obs_index corresponds to an observable from the spin Hamiltonian. */

    /* The current spin Hamiltonian index. */
    shi = d->shi_index[cov_d->obs_index - d->ex->n_obs];
    /* The current spin Hamiltonian element. */
    shel = d->shel_index[cov_d->obs_index - d->ex->n_obs];

    zshp(d->sh_pa[shi], NULL, d->hpro_evect, shi, d->sh, d->shp_w);

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
    if (cov_d->obs_index < d->ex->n_a) {
      /* Absolute energy level. */
      return d->eval[d->ex->la[cov_d->obs_index]];
    }
    else {
      /* Energy level difference. */
      return fabs(d->eval[d->ex->fld[cov_d->obs_index - d->ex->n_a]] \
          - d->eval[d->ex->ild[cov_d->obs_index - d->ex->n_a]]);
    }
  }
}


/* Function for evaluating the covariance matrix for a multi-eigenvalue vector
 * fit for meshfit. 
 *
 * Returns the value of a single observable given the parameter x, where x
 * corresponds to the value of the par_index entry of the real valued tensor
 * coefficient array.  The observable is specified using data->cov_d->obs_index.
 * The function arguments are chosen s.t. the function can be directly passed
 * to the gsl derivative routines, allowing one to calculate the derivative of
 * each observable w.r.t. each parameter, thus yielding the covariance matrix.
 */
double meshfit_cov_df(double x, void *data) {
  int i, hi, ex_n, n_a, lobs_index, shi, shel;
  eshfit_data *d;
  cov_data *cov_d = (cov_data *)data;
  meshfit_data *mesh_d = cov_d->obj_f_data;

  /* Determine the index i of the hamiltonian which corresponds to the current
   * observable. */
  i = 0;
  ex_n = 0;
  do {
    ex_n += mesh_d->data[i]->n_obs;
    i++;
  } while (ex_n < cov_d->obs_index + 1);
  i--;
  ex_n -= mesh_d->data[i]->n_obs;
  

  d = mesh_d->data[i];

  /* Local version of observable index. */
  lobs_index = cov_d->obs_index - ex_n;

  cov_d->df_x[cov_d->par_index] = x;

  /* Sets both h and hpro coeffs, since ptrs are aliased. */
  sh_parse_param_data(d->n_zx, d->p, d->h->coeff, d->sh, cov_d->df_x);
  
  /* Check whether current h/sh pair requires a dedicated proj hamiltonian. */
  if (d->hpro == NULL) {
    zhd('V', d->eval, d->evect, d->h, d->hd_w);
    if (d->job == 'S') {
      find_sort_indices(d->ex, d->h, d->evect); 
    }
  } 
  else {
    if (d->job == 'S') {
      zhd('V', d->eval, d->evect, d->h, d->hd_w);
      find_sort_indices(d->ex, d->h, d->evect); 
    }
    else {
      zhd('N', d->eval, NULL, d->h, d->hd_w);
    }
    /* Diagonalize the projection Hamiltonian. */
    zhd('V', d->hpro_eval, d->hpro_evect, d->hpro, d->hprod_w);
  }

  if (lobs_index >= d->ex->n_obs) {
    /* lobs_index corresponds to an observable from the spin Hamiltonian. */

    /* The current spin Hamiltonian index. */
    shi = d->shi_index[lobs_index - d->ex->n_obs];
    /* The current spin Hamiltonian element. */
    shel = d->shel_index[lobs_index - d->ex->n_obs];
    
    if (d->hpro == NULL) {
      zshp(d->sh_pa[shi], NULL, d->evect, shi, d->sh, d->shp_w);
    }
    else {
      zshp(d->sh_pa[shi], NULL, d->hpro_evect, shi, d->sh, d->shp_w);
    }

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
    /* lobs_index corresponds to an energy level observable; return the value of
     * the specified level.*/
    if (lobs_index < d->ex->n_a) {
      /* Absolute energy level. */
      return d->eval[d->ex->la[lobs_index]];
    }
    else {
      /* Energy level difference. */
      return fabs(d->eval[d->ex->fld[lobs_index - d->ex->n_a]] \
          - d->eval[d->ex->ild[lobs_index - d->ex->n_a]]);
    }
  }
}


/* Common steps for covariance matrix estimation for *fit_cov functions. */
inline void covariance_helper(int m, int n, gsl_function F, double *x0,
    void *data, double sigma, double *cov_inv) {
  int i, j, k, status;
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

  /* Create copy of x0, since the derivative function modifies the parameter
   * value w.r.t. which we're differentiating. */
  memcpy(cov_d->df_x, x0, n*sizeof(double));
  F.params = cov_d;
  
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
  int m, n;
  double sigma, chisq;
  gsl_function F;
  efit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables. */
  m = d->ex->n_obs;

  F.function = &efit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * energy levels (pg. 780, Press et al. 3rd edition). */
  efit_chi2(x0, d, &chisq);
  sigma = sqrt(chisq/(m-n));

  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
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
  int i, m, n;
  double sigma, chisq;
  gsl_function F;
  mhfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;
  /* The number of observables. */
  m = 0;
  for (i = 0; i < d->n; i++) {
    m += d->exa[i]->n_obs;
  }

  F.function = &mhfit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * energy levels (pg. 780, Press et al. 3rd edition). */
  mhfit_chi2(x0, d, &chisq);
  sigma = sqrt(chisq/(m-n));

  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}

/* Common procedures for all eshfit cov functions. Allocs and populates index
 * arrays and returns the number of observables of the provided eshfit_data
 * object. 
 *
 * Parameters
 * ----------
 * d      The eshfit object for which to alloc and calculate cov index arrays,
 *        and determine the total number of observables for.
 */
inline int sh_cov_helper(eshfit_data *d) {
  int i, j, m, obs_i;

  /* The number of observables.  We count 6 observables per spin Hamiltonian
   * term, except for the quadrupole term which is traceless and thus only
   * contributes 5 observables. */
  m = d->ex->n_obs;
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
  d->shi_index = (int *) malloc((m - d->ex->n_obs)*sizeof(int));
  if (d->shi_index == 0) {
    CFL_ERROR_VAL("malloc failed for d->shi_index", -1);
  }
  d->shel_index = (int *) malloc((m - d->ex->n_obs)*sizeof(int));
  if (d->shel_index == 0) {
    free(d->shi_index);
    CFL_ERROR_VAL("malloc failed for d->shel_index", -1);
  }
  obs_i = 0;
  for (i = 0; i < d->sh->ninter; i++) {
    if (!strcmp("quadrupole", d->sh->inter[i])) {
      for (j = 0; j < 5; j++) {
        d->shi_index[obs_i] = i;
        d->shel_index[obs_i] = j;
        obs_i++;
      }
    }
    else {
      for (j = 0; j < 6; j++) {
        d->shi_index[obs_i] = i;
        d->shel_index[obs_i] = j;
        obs_i++;
      }
    }    
  }
 
  /* Retain a copy of the number of observables for this h/sh pair (used by
   * mesh cov). */
  d->n_obs = m;

  return m;
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
  int m, n;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;

  /* The number of observables.*/
  m = sh_cov_helper(d);
  
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
  int m, n;
  double sigma;
  double chisq[2] = {0, 0};
  gsl_function F;
  eshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;

  /* The number of observables.*/
  m = sh_cov_helper(d);

  F.function = &eshfit_hpro_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  eshfit_hpro_chi2(x0, d, chisq);
  sigma = sqrt((chisq[0] + chisq[1])/(m-n));

  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}

/* Estimate the covariance matrix for an multi energy level and spin Hamiltonian
 * fit. 
 *
 * Parameters
 * ----------
 *  x0      The parameters found by the minimization.
 *  cov_inv Pointer to space that will be overwritten with the inverse
 *          covariance matrix. 
 *  obj     The cfl_min_obj for which the minimization was run.
 */
void meshfit_cov(double *x0, double *cov_inv, cfl_min_obj *obj) {
  int i, m, n, nchi2;
  double sigma, chi2_sum, *chi2;
  gsl_function F;
  meshfit_data *d = obj->obj_f_data;

  /* The number of parameters. */
  n = obj->n;

  /* The number of observables.*/
  m = 0;
  nchi2 = 0;
  for (i=0; i<d->n; i++) {
    m += sh_cov_helper(d->data[i]);
    nchi2 += d->data[i]->sh->ninter + 1;
  }
  
  F.function = &meshfit_cov_df;

  /* Estimate the uncertainty, assuming model fit and the same sigma for all
   * observables (energy and sh) (pg. 780, Press et al. 3rd edition). */
  chi2 = (double *) calloc(nchi2, sizeof(double));
  if (chi2 == 0) {
    CFL_ERROR_VOID("calloc failed for chisq");
  } 
  meshfit_chi2(x0, d, chi2);
  chi2_sum = 0;
  for (i=0; i<nchi2; i++) {
    chi2_sum += chi2[i];
  }
  free(chi2);
  sigma = sqrt(chi2_sum/(m-n));
  
  covariance_helper(m, n, F, x0, d, sigma, cov_inv);
}

