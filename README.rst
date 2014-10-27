==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The two primary modules are, pyemp and cfl. 

pyemp

A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Supports easy scripting of emp routines and plotting of intensity
spectra. 

cfl

A reimplementation of 'cfit' in c99, with python bindings.  Primarily intended
for fitting crystal field parameters to spin Hamiltonians.


Installation
============

The cfl library is presently on a separate development branch from some of the
original pycf scripts.  To get an up-to-date copy::

  git clone https://bitbucket.org/sebastianhorvath/pycf/ -b cfl

The c library for now uses a separate build system from the python modules.  To
compile it, ensure all of the listed `Dependencies`_ are satisfied, then a
simple ``make`` in the ``cfl`` directory should suffice.  To compile using icc
and mkl, use the target ``make mkl`` instead.  If icc and mkl are installed in a
non-standard location, you must edit the ``INTEL_PATH`` variable in the
makefile. 

The python modules use the standard python distribution utilities (distutils)
for installation. To compile and install them, navigate to the root directory
``pycf`` and run::

  $ python setup.py install

This will automatically build all cython modules and install pycf in your
``dist-packages`` (also called ``site_packages`` on some operating systems)
directory.  

The provided ``setup.py`` file should work without modification provided the
dependency libraries are installed in standard system locations. 

To manually specify the installation prefix for pycf use::

  $ python setup.py install --prefix=/path/to/dir

If you install to a non-standard location you need to ensure that the python
``dist-packages`` directory is part of the ``PYTHONPATH`` environment variable.


Dependencies
------------

To build cfl you will need to satisfy the following dependencies:
 
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * `gsl <https://www.gnu.org/software/gsl/>`_ - the GNU scientific library
  * `nlopt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ - nonlinear
    optimization library
  * gcc 
  * build-essential package or your distributions equivalent

cfl also builds with Intel's icc and mkl, but you will still require gcc to
build the python extension. 

To build the cfl python extension and pyemp the following dependencies have to
be satisfied:
  
  * python
  * numpy 
  * scipy 
  * matplotlib
  * `cython <http://cython.org/>`_ - C extensions for Python

All of the above should be available via the package manager on most linux
distributions.  If you compile any of the cfl dependencies from source you
either have to specify the runtime libraries to the linker (gcc option
``-Wl,-rpath``) or add them as ``extra_objects`` in ``setup.py``.  Additionally,
since cython compiles c modules as shared objects, all linked objects must be
compiled as position independent code (``-fPIC``). 


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


