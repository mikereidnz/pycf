==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations. 

pycfl
=====

pycfl is a crystal field fitting program that supports a variety of fitting
modes, including fitting to spin Hamiltonian data and multiple crystal field
Hamiltonians simultaneously.  It is divided into two parts: cfl, an independent
library written in C99 which handles all core calculations, and a python
wrapper, called pycf, which takes care of any input data preparation.  

The source for the c library is located in ``/pycf/cfl`` and can be complied
independently with gnu make.  The python/cython wrapper is called ``cfl.pyx``
(with associated .pxd file) and can be found in ``/pycf/pycf/``.  It is compiled
with using pythons dist-utils.  There are also a number of pure python modules
that can be used for spin Hamiltonian calculations.


pyemp
=====
A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Facilitates automatic generation of emp input files and parsing their
output.  Currently wrapped erun applications are 'cfit', 'inten', 'vtrans', and
'spectrum'.  This wrapper is written in pure python and is called
``pycf/pycf/pyemp.py``.  
