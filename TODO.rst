CFL
===

  * Update makefile to explicitly list rules; then add targets for shared objects
  * Consolidate #ifdef statements for mkl version, and add mkl linking to
    setup.py. 
  * Change nomenclature to avoid first/second order for spin Hamiltonian
    interactions...
  * ImportSLJM should parse and create crs tensors one at a time (at least the
    step that populate the dense matrix), to avoid unreasonable memory use for
    large matrices. 
  * Change appropriate struct attributes and function arguments from int to
    size_t
  * Update doc strings to function return values
  * Think about a complex tensor-prefactor... since a standard multiplication by
    a complex number will not preserve the hermiticity of a matrix. Presumably,
    we multiply by the complex conjugate on the lower-diagonal?
  * When passing a void * type argument, does the function call have to cast to
    void? Currently, bh_min call in opt_test does not, yet gsl_min calls do.
  * In explaination section, discuss the fortran vs c array mess. Everything is
    expected to be a fortran array at the moment? At least the inversion matrix
    for the inversion tests needs to be... the rest don't.
  * Explain the coefficient array use in h_fit.  In particular, explain why only
    a single coefficient array is required for both h and hfo, i.e., because
    both h and hfo have a given number of tensors (of which they are aware), and
    so if one passes an array that contains additional coefficients to hfo, it
    will never look at those components.  A further point that is necessary for
    this to work is that the set_coeff function, and the therein called zsham
    function, do not modify the coeff array... if this were the case, there
    would be other undesired side-effects, such as the inability to reuse the
    same coefficient array from iteration to iteration. 
  * sigmas, covariance matrix
