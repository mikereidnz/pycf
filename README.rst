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
basic system wide installation fetch the binary from `downloads
<https://bitbucket.org/sebastianhorvath/pycf/downloads/>`_ and run::

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

These should be available via the package manager on most modern linux distributions.  Furthermore, if you want to build pycf you require:

  * `cython <http://cython.org/>`_
  * gcc 

  PYTHONPATH
----------

If you install to a non-standard location you need to ensure that the python
dist-packages (some distributions call this site-packages) directory is part of
the ``PYTHONPATH`` environment variable.


Running
=======

To get an idea of how to do various calculations with pycf have a look at the
``examples`` directory. 


