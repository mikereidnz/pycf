CFL
===

  * Update makefile to explicitly list rules; then add targets for shared objects
  * Create python jmcalc parser
  * Change spectrum labels of zh objects to copy the string and store internally
  * Calculate hash of state labels, and provide a state comparison function. 
  * Merge zh and state label functions to a common source file, since state
    lables will also be useful for spin Hamiltonians. 
  * Change appropriate struct attributes and function arguments from int to
    size_t
  * Update doc strings to function return values
  * Think about a complex tensor-prefactor... since a standard multiplication by
    a complex number will not preserve the hermiticity of a matrix. Presumably,
    we multiply by the complex conjugate on the lower-diagonal?
  * Move GSL tolerances to macros. 

Spin Hamiltonian
----------------


