TODO
====

  * Add error checking to SpinHamiltonian.add_H_term to raise an exception if
    the dimensions of val do not match the expected dimensions.
  * Possibly add a fast add_H_term method that also inverts terms for use in
    least squares fitting; this would allow one to do error checks in the
    calling function and remove exceptions overhead, python attribute data
    storage etc. 
