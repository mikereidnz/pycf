pycf overview
=============

Introduction
------------
pycf is a collection of python classes and functions used for crystal field
theory and spin Hamiltonian calculations.  The crystal field theory calculations
are done using the :mod:`pyemp` module which is a collection of classes that
wrap Michael F. Reid's crystal field theory routines and facilitate easy
plotting of the results.  The spin Hamiltonian module :mod:`spinh` is a
collection of classes and functions that allow both the calculation of spin
Hamiltonians from spin Hamiltonian parameter matrices and the inversion of spin
Hamiltonians to recover the parameter matrices.  By employing both :mod:`pyemp`
and :mod:`spinh` one can least squares fit crystal field theory parameters using
spin Hamiltonian data.

Additionally, out of requirement for :mod:`spinh`, there is also a Wigner
nj-symbol implementation :mod:`njsymbols` and a module for calculating matrix
elements for rank one tensors, :mod:`matel`.  For details see their respective
api references. 

Document outline
----------------
The document is divided into four parts.  The first part outlines how to install
and compile pycf. The second part contains tutorial-style explanations of the
provided example calculations.  The third part contains the api reference
generated from docstrings.  While the examples aim to give a good overview of
the basic functions, the api reference should be consulted for exhaustive
implementation details.  The fourth part contains miscellaneous notes.
