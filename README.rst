About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The primary module for crystal field theory is pyemp,
which wraps Michael F. Reid's F-shell empirical crystal field theory routines.
Spin Hamiltonian calculations are done with the spinh module.

Installation
============

Dependencies
------------

pycf has the following dependencies:

  * numpy
  * scipy
  * matplotlib (for plotting in example calculations)

These should be available via the package manager on most modern linux distributions. 


Installing pycf
---------------

This package uses the standard python distribution utilities (distutils).  For
installation options see::

  $ python setup.py --help

To install pycf in the standard location for third party packages,
``/usr/local/lib/``, simply run::

  $ python setup.py install

or to install to a non-standard directory run::

  $ python setup.py install --prefix=/path/to/dir


PYTHONPATH
----------

If you install to a non-standard location you need to ensure that the install
directory is part of the ``PYTHONPATH`` environment variable.


Running
=======

To get an idea of how to do various calculations with pycf have a look at the
``examples`` directory. 


