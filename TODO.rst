TODO
====

  * ImportSLJM should parse and create crs tensors one at a time (at least the
    step that populate the dense matrix), to avoid unreasonable memory use for
    large matrices. 
  * Think about a complex tensor-prefactor... since a standard multiplication by
    a complex number will not preserve the hermiticity of a matrix. Presumably,
    we multiply by the complex conjugate on the lower-diagonal?
  * When passing a void * type argument, does the function call have to cast to
    void? Currently, bh_min call in opt_test does not, yet gsl_min calls do.
  * In explanation section, discuss the fortran vs c array mess. Everything is
    expected to be a fortran array at the moment? At least the inversion matrix
    for the inversion tests needs to be... the rest don't. 
  * Fit to multiple spin Hamiltonians
  * Try adaptive chi^2 weighting using annealing.
  * Consolidate unit testing. 
  * There is a disagreement between the state admixtures between my calculation
    and Mikes for Ce:LiYF4. Most probably this is since Mike's program operates
    on block diagonals and, consequently, in effect chooses a linear combination
    of eigenvectors. Verify that there exists a linear combination of states 1
    and 2 that reproduce Mike's state admixtures. 
  * There is an issue with the number of observables of quadrupole terms in the
    covariance matrix estimation in cfl_h_fit.c. Set the number of observables
    in cfl.pyx, since the python SpinHamiltonian class knows the true number of
    observables.
  * Add weighting to spin Hamiltonian log.
  * Make sure there is no duplicate information in spin Hamiltonian type given
    the new state labeling data structure. 

Distributed memory parallelization
----------------------------------

  * Parallel creation of dense matrix for diag.  Create zero matrices once, then
    copy them and fill in non-zero entries (or some other fast means of creating
    the matrices)
  * Use methods pzgels and pzheevd from ScalaPACK and pzgemm and pzhemm from
    PBLAS.  Link to ScalaPACK doc about matrix size per core guideline 
    n_core = ~ (m by n)/10^6
