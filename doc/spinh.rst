=========================
Spin Hamiltonian tutorial
=========================

.. currentmodule:: spinh

.. |Er3Y2SiO5| replace:: Er\ :sup:`3+`\:Y\ :sub:`2`\SiO\ :sub:`5`

The :mod:`spinh` module is a collection of classes and functions that allow both
the derivation of the spin Hamiltonian given the spin Hamiltonian parameter
matrices, such as the dipole or quadrupole hyperfine matrices, and the inversion
of a given spin Hamiltonian to recover the parameter matrices.  In this tutorial
we will begin by calculating the eigenvalue spectrum of |Er3Y2SiO5| using the
magnetic dipole and electric quadrupole parameters from Guillot-Noel et al.
(Journal of Alloys and Compounds 451 (2008) 62-64).

Eigenvalues of Er3+:Y2SiO5
===========================

While the :mod:`spinh` module interface exposes a number of low level functions,
the most convenient way to calculate a spin Hamiltonian is by creating an object
of type :class:`SpinHamiltonian`.  The constructor requires a list indicating
which terms the spin Hamiltonian contains; possible values are ``bgs``, ``ias``
and ``iqi``.  Additional keyword arguments specify the magnetic field strength
`B`, the spin projection `S_z` and the nuclear spin projection `I_z`; see the
:class:`SpinHamiltonian` reference for further details.

The following code will create a :class:`SpinHamiltonian` object::
  
  #!/usr/bin/env python

  from __future__ import division
  import numpy as np
  from numpy import linalg as LA
  from matplotlib import pyplot as plt
  from pyspectrum.spinh import *
  
  # Create the SpinHamiltonian object 
  sh = SpinHamiltonian(['ias', 'iqi'], S = 1/2, I = 7/2)

We can now add the experimental data for the `IAS` and `IQI` terms using the
:func:`SpinHamiltonian.add_term` method::

  # Define the A and Q parameter matrices. 
  A = np.array([[69.35, -580.73, -248.83], [-580.73, 696.30, 682.49], [-248.83, 682.49, 495.54]])
  Q = np.array([[21.40, -8.18, -15.27], [-8.18, 3.79, 0.60], [-15.27, 0.60, -25.20]])

  # Add values for the dipole and quadrupole terms.
  sh.add_term('ias', A)
  sh.add_term('iqi', Q)

Now the total spin Hamiltonian can be returned using the ``H`` attribute of
``sh``::

  # The 'H' attribute returns the full spin Hamiltonian, then we use numpy to
  # calculate the eigenvalues. 
  w, v = LA.eig(sh['H'])
  E = w.real
  E = np.sort(E - min(E))

  print("Energy spectrum:\n{}".format(E))

To plot the energy levels we can use matplotlib::
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.hlines(E, [0], [1])
  ax.set_ylabel('HFS (MHz)')
  plt.show()

Which should yield the following plot:

.. image:: figures/spin_hamiltonian.pdf

Spin Hamiltonian inversion
==========================

To demonstrate the inversion of a spin Hamiltonian we will first adapt the above
example to recover the `A` and `Q` matrices from the full spin Hamiltonian.
Inversions involving Zeeman interactions are a little more difficult, for the
inversion is under-determined for a single spin Hamiltonian with a given
magnetic field strength.  For a complete inversion, one requires a minimum of
three spin Hamiltonian terms for linearly independent magnetic field strengths.
A mock example of such an inversion is given in the section `Zeeman interaction
spin Hamiltonian inversion`_.  

Dipole and quadrupole hyperfine spin Hamiltonian inversion
----------------------------------------------------------

Again, the easiest method is to create a :class:`SpinHamiltonian` object, but
with the ``inv`` argument specified as ``True``::

  sh_inv = SpinHamiltonian(['ias', 'iqi'], S = 1/2, I = 7/2, inv = True)

We can now add the spin Hamiltonian terms using the
:func:`SpinHamiltonian.add_H_terms` method::

  sh_inv.add_H_term('ias', sh['ias_H'])
  sh_inv.add_H_term('iqi', sh['iqi_H'])

We have used the ``ias_H`` and ``iqi_H`` attributes of the previously calculated
spin Hamiltonian ``sh`` to obtain a copy of the full dimensional term data, in
this case `16` by `16` matrices.  In a real calculation this data would either
be read in or calculated using :mod:`pyemp`.  To recover the `A` and `Q`
matrices we use the :func:`SpinHamiltonian.inv_term` method, which returns a `9`
by `1` array and consequently is reshaped; this shape is returned since it
allows for convenient least squares fitting using the :func:`sh_lsq_func`
function.  To finish the example we invert both terms and print the original and
the calculated parameter matrices for comparison::
  
  A_c = np.reshape(sh_inv.inv_term('ias'), (3, 3))
  Q_c = np.reshape(sh_inv.inv_term('iqi'), (3, 3))
  
  print("Calculated A:\n{}".format(np.reshape(A_c, (3, 3))))
  print("Original A:\n{}".format(A))
  print("Calculated Q:\n{}".format(np.reshape(Q_c, (3, 3))))
  print("Original Q:\n{}".format(Q))

Zeeman interaction spin Hamiltonian inversion
---------------------------------------------

In order calculate the Zeeman spin Hamiltonian parameter matrix `g` we require
three linearly independent Zeeman spin Hamiltonians.  We generate these using::
  
  B_m = np.eye(3)
  bgs_list = []
  for i in range(3):
      sh = SpinHamiltonian(['bgs'], B = B_m[:, i], S = 1/2)
      sh.add_term('bgs', g)
      bgs_list += [sh['bgs_H']]

When a :class:`SpinHamiltonian` object is instantiated for inversion and the
term list includes ``bgs`` the constructor expects a list of numpy arrays for
the magnetic field keyword argument.  In the below example we do this using
`List Comprehensions`_::
  
  sh_inv = SpinHamiltonian(['bgs'], B = [B_m[:, i] for i in range(3)], S = 1/2 inv = True)
  sh_inv.add_H_term('bgs', bgs_list)
  
The result can now be calculated just as in the dipole and quadrupole hyperfine
spin Hamiltonian example::
  
  g_c = np.reshape(sh_inv.inv_term('bgs'), (3, 3))
  print("Calculated g:\n{}".format(np.reshape(g_c, (3, 3))))
  print("Original g:\n{}".format(g)) 

.. _List Comprehensions: http://docs.python.org/2/tutorial/datastructures.html#list-comprehensions


Notes for spin Hamiltonian fitting and interaction with pyemp
=============================================================

  * The state label order for inversion is important for terms with ambiguity,
    at the moment only the 'ias' term.  Spin Hamiltonian's created using cfit
    and pyemp ensure states are in the order that the SpinHamiltonian inv_term
    method expects; that is, states are first sorted by the nuclear spin label,
    then by the electronic spin label.  For example, the upper-left-hand 2 by 2
    block of the spin Hamiltonian corresponds to S = 1/2 and S = -1/2 for a spin
    half system.  Sorting is from largest to smallest label. 


  


  
  



