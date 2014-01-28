==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The primary module for crystal field theory is pyemp,
which wraps Michael F. Reid's F-shell empirical crystal field theory routines.
Spin Hamiltonian calculations are done with the spinh module.

Installation
============

This package uses the standard python distribution utilities (distutils).  For a
basic system wide installation fetch the binary (x86_64 Debian testing) from
`downloads <https://bitbucket.org/sebastianhorvath/pycf/downloads/>`_, unpack
the archive, and run::

  $ python setup.py install

For further installation options see::

  $ python setup.py --help

For example, to install to a non-standard directory use::

  $ python setup.py install --prefix=/path/to/dir

Dependencies
------------

pycf requires the following dependencies to run:

  * numpy
  * scipy
  * matplotlib (for plotting in example calculations)

These should be available via the package manager on most modern linux
distributions.  Furthermore, if you want to build the optimized branch of pycf
you require:

  * `cython <http://cython.org/>`_ - C extensions for Python
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * gcc 

Compilation
-----------

The ``master`` branch does not contain any optimizations and can be run without
compilation.  To build the ``opt`` branch run::

  $ python setup.py build

which creates a directory called build containing the package.  For testing it
is useful to build in-place using::

  $ python setup.py build_ext -i

which places the extension module files into the ``pycf`` directory.

PYTHONPATH
----------

If you install to a non-standard location you need to ensure that the python
site-packages directory is part of the ``PYTHONPATH`` environment variable.


Running
=======

To get an idea of how to do various calculations with pycf have a look at the
``examples`` directory. 


