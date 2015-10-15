#ifndef _CFL_CONFIG_H_ 
#define _CFL_CONFIG_H_

#define TRUE 1 
#define FALSE 0

#if !defined(USE_MKL)
#define USE_MKL       FALSE
#endif
#define MKL_Complex16 double complex

/* gsl multimin defaults. */
/* Absolute tolerance used for stopping criteria. */
#define GSL_MIN_EPSABS          1e-2
/* Absolute tolerance used for stopping criteria in derivative algorithms. */
#define GSL_MIN_DERIV_EPSABS    1e-3
/* Step size for derivative based algorithms. */
#define GSL_MIN_SS              0.01
/* Line minimization accuracy. */
#define GSL_MIN_TOL             1e-4
/* Numerical derivative step-size for gradient based minimization. */
#define GSL_MIN_DERIV_H         1e-9

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

/* Numerical derivative step-size for covariance matrix estimation. */
#define COV_DERIV_H           1e-10

/* The absolute error tolerance for the eigenvalues of Hamiltonian
 * diagonalization (see LAPACK zheevr for detailed description). */
#define ZHEEVR_ABSTOL         -1.0

/* The maximum number of blocks per Hamiltonian. */
#define CFL_MAX_BLOCK_NUM     20

/* The starting value of the chi^2 fit that each Hamiltonian/Spin Hamiltonian
 * contribution to the total chi^2 is normalized to. */
#define CFL_MIN_START_CHI2    1e5

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#endif /* _CFL_CONFIG_H_ */
