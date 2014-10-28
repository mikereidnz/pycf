==========
About pycf
==========

Overview
========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The two primary modules are, pyemp and cfl. 

pyemp
-----
A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Supports easy scripting of emp routines and plotting of intensity
spectra. 

cfl
---

A reimplementation of 'cfit' in c99, with python bindings.  Primarily intended
for fitting crystal field parameters to spin Hamiltonians.


Installation
============

To build cfl and it's python bindings, get a copy of the source by running::

  git clone https://bitbucket.org/sebastianhorvath/pycf/ -b cfl

Then, in the package root directory::

  python setup.py install --prefix=/path/to/dir

Provided you have all of the dependencies satisfied, this should build both the
c library and the python bindings.  Typically you will want to specify the
installation prefix to somewhere other than the system default.  In this case,
you need to explicitly add the location (``--prefix``) of the python
``dist-packages`` (called ``site_packages`` on some linux distributions)
directory to the ``PYTHONPATH`` environment variable.




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

Before building you will need to satisfy the following dependencies:
 
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * `gsl <https://www.gnu.org/software/gsl/>`_ - the GNU scientific library
  * `nlopt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ - nonlinear
    optimization library
  * gcc 
  * build-essential package or your distributions equivalent
  * python
  * numpy 
  * scipy 
  * matplotlib
  * `cython <http://cython.org/>`_ - C extensions for Python

All of the above should be available via the package manager on most linux
distributions.

Note that if any of the dependencies are installed in a non-standard location
(not listed in ``/etc/ld.so.conf``) you need to specify any include and lib
directories using the following environment variables::

  export CFL_CFLAGS='-I/path/to/include1 -I/path/to/include2'
  export CFL_LDLIBS='-L/path/to/lib1 -L/path/to/lib2'

Additionally, since cython compiles c extensions as shared objects, all linked
objects must be compiled as position independent code (``-fPIC``).  If you are
getting ``undefined symbol`` errors at runtime, even though ldd claims
``cfl.so`` is fully linked, this suggests that perhaps one of the statically
linked libraries was not position independent.

Intel mkl
---------

cfl also builds with Intel's icc compiler and math kernel library (instead of
LAPACK and ATLAS/BLAS).  Note that you will still require gcc to build the
python extension, unless you also rebuild your python distribution and
supporting libraries with icc.

To build with cfl with icc set the following environment variable::

  export CFL_CC=icc
  export INTEL_PATH=/path/to/inteldir

where ``inteldir`` should contain both icc and mkl. 


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


License
=======

The cfl and the cfl python extension are licensed under the GNU General Public
License, while pyemp and related components are licensed under the X11/MIT
license.  

The use of GPLv3 is mandated by the linking of cfl against GSL, which by the
FSF's interpretation of copyright law makes cfl a derivative work of GSL.  Since
GSL is only used for random number generation and some optional minimization
routines, it would be easy to find more permissively licensed replacements (such
as the Mersenne Twister), and I'm happy to relicense all non-GSL components
under the X11 or the modified BSD license.  The choice to use GSL is primarily
convenience; it is widely available on most linux distributions and reduces the
number of obscure dependencies.

