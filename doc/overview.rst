pycf overview
=============

Introduction
------------

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The two primary modules are, pyemp and cfl. 

pyemp
~~~~~
A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Facilitates automatic generation of emp input files and parsing their
output.  Currently wrapped erun applications are 'cfit', 'inten', 'vtrans', and
'spectrum'.  This wrapper is written in pure python and is called
``pycf/pycf/pyemp.py``.  


pycfl
~~~~~

A is a reimplementation of 'cfit' with support for spin Hamiltonian fitting.  It
is divided into two parts, a small library called cfl which is written in c99
and handles all core calculations.  It is located in ``/pycf/cfl`` and is
intended to be an independent component that can easily be reused in other
applications requiring crystal-field calculations.  The second part is a python
wrapper written in cython, which is called ``/pycf/pycf/cfl.pxd``.  This
wrapper, and supporting modules located in ``/pycf/pycf/``, take care of any
input data preparation, such as CF matrix element loading and spin Hamiltonian
matrix element evaluations, as well as pretty-printing calculation results.
While direct calls to cfl without python are certainly possible, the manual
input data entry quickly becomes intractable for realistic problems.  It would
also be possible to create bindings for cfl in other languages, such as Matlab,
etc.   
