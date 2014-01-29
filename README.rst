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
distributions.  Furthermore, if you want to modify cython modules, such as
``spinh_c``, you need:

  * `cython <http://cython.org/>`_ - C extensions for Python
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * gcc 

Compilation
-----------

If you want to modify cython modules, or want to compile the existing modules
for a new target, use::

  $ python setup.py sdist

which creates a ``tar.gz`` file in the ``dist`` directory.  For testing it
is useful to build in-place using::

  $ python setup.py build_ext -i

which places the extension module files into the package source directory such
that it can be directly imported by other modules.

PYTHONPATH
----------

If you install to a non-standard location you need to ensure that the python
site-packages directory is part of the ``PYTHONPATH`` environment variable.


Running
=======

To get an idea of how to do various calculations with pycf have a look at the
``examples`` directory. 


