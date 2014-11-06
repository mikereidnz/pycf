==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The two primary modules are, pyemp and cfl. 

pyemp
=====
A python wrapper for Michael F. Reid's F-shell empirical crystal field theory
routines.  Supports easy scripting of emp routines and plotting of intensity
spectra. 

cfl
===

A reimplementation of 'cfit' in c99, with python bindings.  Primarily intended
for fitting crystal field parameters to spin Hamiltonians.


Installation
============

To install pyemp, get a copy of the source by running::

  git clone https://bitbucket.org/sebastianhorvath/pycf/ -b cfl

Then, in the package root directory::

  python setup.py install --prefix=/path/to/dir

Provided you have all of the dependencies satisfied, this first builds both the
c library and the python bindings, and then installs both the cfl python
bindings and pyemp in the python ``dist-packages`` (called ``site-packages`` on
some linux distributions) directory.  Typically it is a good idea to specify
prefix to something other than the default (``/usr/lib``), in which case you
need to add the location of the resulting ``dist-packages`` (``site-packages``)
to the ``PYTHONPATH`` environment variable.

The c library uses GNU make, so for development of cfl it is easiest to directly
execute make.  Running ``make`` in the ``cfl`` directory should suffice.  It may
also be useful to ``make debug`` to compile with ``-O1``.  


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
(not listed in ``/etc/ld.so.conf``) you need to specify the path to any include
and lib directories prior to running ``python setup.py install``.  This is done
by setting the following environment variables::

  export CFL_CFLAGS='-I/path/to/include1 -I/path/to/include2'
  export CFL_LDLIBS='-L/path/to/lib1 -L/path/to/lib2'

Additionally, since cython compiles c extensions as shared objects, all linked
objects must be compiled as position independent code (``-fPIC``).  If you are
getting ``undefined symbol`` errors at runtime, even though ldd claims
``cfl.so`` is fully linked, this suggests that perhaps one of the statically
linked libraries was not position independent.

Redhat based systems
--------------------

Redhat based systems provide the c++ version of ``nlopt`` via the package
manager.  This means the application has to be linked with the g++ linker, which
unfortunately fails for the cython extension.  

The easiest solution to this on a Redhat based system is to compile the library
from source.  The nlopt installation page has detailed `instructions
<http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation>`_ on how to do
this. Then, by setting ``CFL_CFLAGS`` and ``CFL_LDLIBS`` variables to wherever
you installed nlopt,  you should be able to compile pyemp.  Note that since your
object files need to be position independent code (or cython will not be able to
create a shared object), you need to compile nlopt as a shared library (or set
the ``-fPIC`` compiler option). See the nlopt `page
<http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries>`_
for details on how to do this.

Intel mkl
---------

cfl also builds with Intel's icc compiler and math kernel library (instead of
LAPACK and ATLAS/BLAS).  Provided the bin directory containing icc is part of
your system ``$PATH``, building with icc and linking against mkl is done by::
  
  python setup.py install --compiler=intel

where any additional arguments, such as prefix or inplace can also be added.
  
To build only cfl with icc set the following environment variables prior to
running make::

  export CFL_CC=icc
  export MKLROOT=/path/to/inteldir

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
License version three, while pyemp and related components are licensed under the
X11/MIT license.  

The use of GPLv3 is mandated by cfl dynamically linking against GSL, which by
the FSF's interpretation of copyright law makes cfl a derivative work of GSL.
Since GSL is only used for random number generation and some optional
minimization routines, it would be easy to find more permissively licensed
replacements (such as the Mersenne Twister), and I'm happy to relicense all GSL
independent components under the X11 or the modified BSD license.  
