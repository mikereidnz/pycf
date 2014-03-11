==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The primary module for crystal field theory is pyemp,
which wraps Michael F. Reid's F-shell empirical crystal field theory routines.
Spin Hamiltonian calculations are done with the spinh module.

Installation
============

This package uses the standard python distribution utilities (distutils).
Before installation, ensure that all the listed `Dependencies`_ are satisfied.
Then fetch the archive from  `downloads
<https://bitbucket.org/sebastianhorvath/pycf/downloads/>`_, extract it, and
run::

  $ python setup.py install

This will automatically build all cython modules and install pycf in your
``dist-packages`` directory.  The provided ``setup.py`` file should work without
modification on Debian testing, but may require adaptation for different
distributions. 

To manually specify the installation prefix use::

  $ python setup.py install --prefix=/path/to/dir

If you install to a non-standard location you need to ensure that the python
``dist-packages`` directory is part of the ``PYTHONPATH`` environment variable.

For further installation options see::

  $ python setup.py --help


Dependencies
------------

pycf has the following dependencies:

  * numpy
  * scipy (>= 0.12.0)
  * matplotlib (for plotting in example calculations)
  * `cython <http://cython.org/>`_ - C extensions for Python
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * gcc 

These should be available via the package manager on most linux distributions.
If you compile LAPACKE from source you must ensure that liblapack, libblas, and
libgfortran are available to the linker when pycf is installed.  The location of
these can be specified manually by editing the ``setup.py`` file.  Additionally,
since cython compiles c modules as shared objects, all linked objects must be
compiled as position independent code. 


Development
-----------

If you want to modify pycf and redistribute it you can create a new archive
using the command::

  $ python setup.py sdist

which creates a ``tar.gz`` file in the ``dist`` directory.  For testing it
is useful to build in-place using::

  $ python setup.py build_ext -i

This places the extension module files into the package source directory such
that it can be directly imported by other modules.


Running
=======

To get an idea of how to do various calculations with pycf have a look at the
`manual <https://bitbucket.org/sebastianhorvath/pycf/downloads/pycf.pdf>`_,
along with the files provided in the ``pycf/examples`` directory.


