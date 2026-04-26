pycf overview
=============

Directory layout and program names
----------------------------------

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The primary module is cfl (for crystal field
library), but there's also several other modules which might be useful and are
consequently included here.

cfl
~~~
cfl is a reimplementation of 'cfit' with support for spin Hamiltonian fitting.
It is divided into two parts. All of the heavy lifting is performed by a library
written in c99, located in ``pycf/cfl``.  This enables parallelization for
shared memory machines using OpenMP, as well as optimization with
high-performance compilers such as icc. For ease of use, the library is called
via a python interface written in cython (``/pycf/pycf/cfl.pyx`` and associated
``cfl.pxd`` header).  


This wrapper, and supporting modules located in ``/pycf/pycf/``, take care of
any input data preparation, such as crystal-field matrix element loading and
spin Hamiltonian matrix element evaluations, as well as pretty-printing
calculation results.  While direct calls to cfl without python are certainly
possible, manually inputting data quickly becomes intractable for realistic
problems.  In principle, it would also be possible to create bindings for cfl in
other languages, such as Matlab. 

spinh
~~~~~
spinh is a module written in pure python which is primarily intended as a set of
helper routines for calculations performed with pycf. It can be used to
calculate the spin Hamiltonian matrix elements given a set of parameter
matrices; this can be useful when used in conjunction with the multi-Hamiltonian
fit of pycf.  Furthermore, it allows one to perform an inversion that
calculates the spin Hamiltonian parameter matrices from complete spin
Hamiltonian.  Supported interactions are Zeeman, hyperfine, and quadrupole.

pyemp
~~~~~
A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Facilitates automatic generation of emp input files and parsing their
output.  Currently wrapped erun applications are 'cfit', 'inten', 'vtrans', and
'spectrum'.  This wrapper is written in pure python and is called
``pycf/pycf/pyemp.py``.  While pycf works as a replacement and extension of
'cfit', this wrapper is still very useful for automating intensity calculations. 


