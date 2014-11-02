#ifndef _CFL_CONFIG_H_ 
#define _CFL_CONFIG_H_


#if !defined(USE_MKL)
#define USE_MKL       FALSE
#define MKL_Complex16 double complex
#endif

/* set MKL_NUM_THREADS -- see mkl user guide. */

/* gsl multimin defaults. */
/* Absolute tolerance used for stopping criteria. */
#define GSL_EPSABS            1e-2
/* Absolute tolerance used for stopping criteria in derivative algorithms. */
#define GSL_DERIV_EPSABS      1e-3
/* Step size for derivative based algorithms. */
#define GSL_SS                0.01
/* Line minimization accuracy. */
#define GSL_TOL               1e-4
/* Numerical derivative step-size. */
#define GSL_DERIV_H           1e-9

/* basinhopping defaults. */
/* Static temperature for the Metropolis criterion. */
#define BH_T                  1
/* Multiplicative factor whereby the stepsize is updated if the target rate is
 * not being met. */
#define BH_STEP_FACTOR        0.9
/* The default stepsize. */
#define BH_DEF_STEP           2
/* The default target acceptance rate for adaptive stepsize. */
#define BH_DEF_TARGET_ACCEPT  0.5
/* The default number of interations between adaptive stepsize updates. */
#define BH_DEF_ADAPT_INT      20

#endif /* _TEST_DATA_H_ */
