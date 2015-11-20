TODO
====
  * Current F->D transition labels are very limited... should store the complete
    label somehow... Only works for (2F) labels presently. 
  * Think about the efit/eshfit->chisq weight factor... should all (spin)
    hamiltonians be normalized, or only the principal one?
  * Fit to multiple spin Hamiltonians
  * Try adaptive chi^2 weighting using annealing.
  * Consolidate unit testing. 
  * Add weighting to spin Hamiltonian log.
  * Print all parameters, even static ones, in a fit summary... Maybe add a
    generic input section? 
  * Add support to energy difference printing.
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
    Matrices should be passed from cython as fortran contiguous arrays, although
    for tensor matrix elements such inconsistencies don't always show up due to
    their hermiticity.  Furthermore, some sometimes 1 dimensional arrays are
    passed as c style arrays, since for these cases they are identical to
    fortran style arrays.

Turning your code into a pretzel
--------------------------------
  * In order to assign names to Tensor objects created by arithmetic (__add__,
    __sub__, and __mul__ methods) we use the Tensor.get_names() method.  If the
    name is not explicitly set, this method looks up the name of the variable to
    which the Tensor object is assigned to in the interpreter namespace. Since
    namespaces are confined to modules, we have to perform the lookup in the
    interpreter namespaces, called __global__.__dict__.  This has potential for
    self referential issues, but proved to be the only way to conveniently use
    python duck typing with Tensor arithmetic and avoid explicit naming of
    Tensors created this way.  Tensors can still be explicitly named when they
    are instantiated, and by setting the Tensor.name attribute.  Furthermore,
    Tensors created with arithmetic also have a arith_name attribute, which
    details composition in terms of absolute (explicitly named) tensor names.
    For a multi Hamiltonian fit with different matrix elements, one still has to
    explicitly set the name for any tensors created by arithmetic that are to
    share the same coefficient, since they will obviously have a different name
    in the global namespace.  
  * The obvious danger of looking up the variable name in this form is that
    multiple variable names can point to the same Tensor object, and since the
    lookup is from a dictionary, it is not possible to predict which name is
    returned.  Consequently, whenever one relies on this automatic name
    assignment, it is important that only a single copy of each tensor is
    assigned to a variable.  If this is unclear, the safest method is to always
    explicitly set Tensor.name attributes, which, if defined, will always take
    precedence.  The current get_name() method checks for uniqueness, and if
    more than one variable name point to the same Tensor object, it raises a
    RuntimeError.

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

Debian/Ubuntu dependencies
--------------------------
libgsl0-dev
build-essential
gfortran
liblapacke-dev
liblapack-dev
libnlopt-dev
python-numpy
python-scipy
python-matplotlib
cython

