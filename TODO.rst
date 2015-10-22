TODO
====
  * Add macro that sets the d->echisq_weight factor... probably should be
    something at the upper limit possible given precision, since we'll
    necessarily minimize... 
  * ImportSLJM should parse and create crs tensors one at a time (at least the
    step that populate the dense matrix), to avoid unreasonable memory use for
    large matrices. 
  * Remove principal component state-label sorting from sh projection... since
    with al=0 all states are degenerate in I, there's no point in finding the
    principal component.
  * Extra "seniority equivalent" label is currently added with label key T when
    imported, but there's no mechanism to print this upon return... so, sort out
    the handling of the seniority label in the printing of results
    large matrices. Also, import as scipy.sparse then convert to cfl_sparse by
    throwing out lower diag.  
  * Think about a complex tensor-prefactor... since a standard multiplication by
    a complex number will not preserve the hermiticity of a matrix. Presumably,
    we multiply by the complex conjugate on the lower-diagonal?
  * When passing a void * type argument, does the function call have to cast to
    void? Currently, bh_min call in opt_test does not, yet gsl_min calls do.
  * Fit to multiple spin Hamiltonians
  * Try adaptive chi^2 weighting using annealing.
  * Consolidate unit testing. 
  * Add weighting to spin Hamiltonian log.
  * Tidy up summary printing.
  * Make sure there is no duplicate information in spin Hamiltonian type given
    the new state labeling data structure.
  * Change all small MAGZ values to a macro def.
  * Change ImportSLJM to return a dictionary of tensors... that would make it
    more consistent, and easier to call from a loop?
  * make sure that the spin hamiltonian level, l, passed by cython starts at 0.
  * cov_inv in CFLMin is currently a 2 dimensional c type array. Make sure this
    is correct, rather than a fortran style array. 

Notes on col vs row major arrays
--------------------------------
  * All lapack/blas calls are ROW major, that is, fortran style arrays, since
    they are passed as 1 by n dimensional contiguous blocks of memory (see
    examples in http://www.netlib.org/lapack/lapacke.html).  However, we store
    sparse matrix elements using the compressed row storage format (CRS).  Since
    we don't directly pass CRS matrices to LAPACK routines, we apply an exact
    inverse to the CRS parsing transformation to revert to dense matrices.  This
    ensures that the matrix is again stored in a 1 by n dimensional array.
    Matrices should be passed from cython as fortran contigious arrays, although
    for tensor matrix elements such inconsistencies don't always show up due to
    their hermiticity.  Furthermore, some sometimes 1 dimensional arrays are
    passed as c style arrays, since for these cases they are identical to
    fortran style arrays. 

Notes to be included in documentation
-------------------------------------
  * We use the "relatively robust representation" algorithm to find the eigenvalues
    of the Hamlitonian. This returns unitary eigenvectors, even for degenerate
    eigenvalues. This is not true for the divide-and-conquer algorithms. For
    notes on the LAPACK RRR routine, see
    http://www.netlib.org/lapack/lug/node30.html#subsecdriveeigSEP.

Distributed memory parallelization
----------------------------------

  * Parallel creation of dense matrix for diag.  Create zero matrices once, then
    copy them and fill in non-zero entries (or some other fast means of creating
    the matrices)
  * Use methods pzgels and pzheevd from ScalaPACK and pzgemm and pzhemm from
    PBLAS.  Link to ScalaPACK doc about matrix size per core guideline 
    n_core = ~ (m by n)/10^6
