CFL
===

  * Consolidate #ifdef statements for mkl version, and add mkl linking to
    setup.py. 
  * ImportSLJM should parse and create crs tensors one at a time (at least the
    step that populate the dense matrix), to avoid unreasonable memory use for
    large matrices. 
  * Change appropriate struct attributes and function arguments from int to
    size_t
  * Think about a complex tensor-prefactor... since a standard multiplication by
    a complex number will not preserve the hermiticity of a matrix. Presumably,
    we multiply by the complex conjugate on the lower-diagonal?
  * When passing a void * type argument, does the function call have to cast to
    void? Currently, bh_min call in opt_test does not, yet gsl_min calls do.
  * In explaination section, discuss the fortran vs c array mess. Everything is
    expected to be a fortran array at the moment? At least the inversion matrix
    for the inversion tests needs to be... the rest don't. 
  * covariance matrix
