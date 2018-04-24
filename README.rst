==========
About pycf
==========

pycf is a collection of python modules for crystal field theory and spin
Hamiltonian calculations.  The primary use case is to enable fitting to Stark
level data in addition to high-resolution magnetic data. 

It is divided into two parts: an independent library written in C99 which
handles all core calculations, and a python wrapper, which takes care of any
input data preparation.  

The source for the c component is located in ``/pycf/cfl`` and can be complied
independently with gnu make.  The python/cython wrapper is called ``cfl.pyx``
(with associated .pxd file) and can be found in ``/pycf/pycf/``.  It is compiled
with python's dist-utils.  There are also a number of pure python modules
that can be used for spin Hamiltonian calculations.


pyemp
=====
There's also a python wrapper for Michael F. Reid's F-shell empirical crystal
field theory routines.  It facilitates automatic generation of emp input files
and parsing their output.  Currently wrapped erun applications are 'cfit',
'inten', 'vtrans', and 'spectrum'.  This wrapper can be found in the file
``pycf/pycf/pyemp.py``.  


Installation
============

Note: I've only tested this on linux, and the instructions below have been
tested for Debian (incl. Ubuntu) and Redhat derived distributions only. 


Basic installation
------------------
First, grab a copy of the source using::

  git clone https://bitbucket.org/sebastianhorvath/pycf/ 

This clones the master branch into a local directory called pycf.  I tend to
merge updates into master somewhat sporadically, but generally will only include
well tested commits.  Specific versions of pycf (printed in log files)
correspond to the git revisions, so it should always be possible to figure out
what source was used to produce a specific output file.  For a branch other than
master, for example devel, use the ``-b branchname`` argument of git. 

Before building, there's a number of dependencies that have to be satisfied. See
the subsection below for details, but if you're on Debian or a derivative
system, the following packages should take care of it:: 

  sudo apt-get install build-essential gfortran liblapacke-dev liblapack-dev \
  libgsl0-dev libnlopt-dev python-numpy python-scipy python-matplotlib cython

Then, in the package root directory, execute::

  python setup.py install --prefix=/path/to/dir

This first builds both the cfl library and the python bindings, and then
installs them to the specified ``prefix`` directory.  It is typically a good
idea to specify ``prefix`` to something other than the default (``/usr/local``).
For a non-default ``prefix`` you need to tell the python interpreter where the
modules are installed.  This is achieved using the ``PYTHONPATH`` environment
variable, which has to be set to include the path to the ``site-packages``
directory into which the module was installed. 

Here is a quick example which installs pycf to ``opt`` in the ``$HOME``
directory.  Assuming you're in the pycf root directory::

  python setup.py install --prefix $HOME/opt

which (for python 2.7) will install all the modules into::

  $HOME/opt/lib/python2.7/site-packages 

To add this directory to the ``PYTHONPATH`` environment variable, run::

  export PYTHONPATH=$PYTHONPATH:$HOME/opt/lib/python2.7/site-packages

This environment variable change can be made persistent for future terminal
sessions by adding the above line to the ``~/.bashrc`` file::
  
  echo 'export PYTHONPATH=$PYTHONPATH:$HOME/opt/lib/python2.7/site-packages' >> ~/.bashrc 

Note that Debian (and derivative distributions) install to a directory called
``dist-packages`` for system wide installations.  Consequently, if python can't
find pycf (``import cfl`` fails in the interpreter), explicitly check the path
to make sure you're specifying the correct directory.  For further details on
this convention, have a look at the Debian Python `wiki
<https://wiki.debian.org/Python#Deviations_from_upstream>`_.


Dependencies
------------

Before building you will need to satisfy the following dependencies:
 
  * GCC and gfortran
  * build-essential package or your distributions equivalent
  * `LAPACKE <http://www.netlib.org/lapack/lapacke.html>`_ - C interface to
    LAPACK
  * `gsl <https://www.gnu.org/software/gsl/>`_ - the GNU scientific library
  * `nlopt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ - nonlinear
    optimization library
  * python (tested with version 2.7)
  * `cython <http://cython.org/>`_ (version >=0.20.1) - C extensions for Python
  * numpy (version >= 1.7) 
  * scipy 
  * matplotlib (pyemp plotting only; can be omitted for pycfl)

GCC, gfortran, and LAPACK can be substituded for their Intel MKL equivalent; see
the Intel mkl section below for details.

Note that if any of the dependencies are not installed using your distributions
package manager, and they are in a non-standard location (not listed in
``/etc/ld.so.conf``), then you need to specify the path to any include and lib
directories prior to running ``python setup.py install``.  This is done by
setting the environment variables ``CFL_CFLAGS`` and ``CFL_LDLIBS``, where the
former should be a space separated list of required include paths, and the
latter should be a space separated list of required link-time and run-time
library paths.  For example, say some additional libraries are installed in
``$HOME/opt``, then one would set the include and link/run -time paths using::

  export CFL_CFLAGS="-I$HOME/opt/include"
  export CFL_LDLIBS="-L$HOME/opt/lib -Wl,-rpath,$HOME/opt/lib"

Additionally, since cython compiles c extensions as shared objects, all linked
objects must be compiled as position independent code (``-fPIC``).  If you are
getting ``undefined symbol`` errors at runtime, even though ldd claims
``cfl.so`` is fully linked, this suggests that perhaps one of the statically
linked libraries was not position independent.


Build cfl withe GNU make
------------------------

The cfl library uses GNU make and can be built independently from the python
modules.  Running ``make`` in the ``cfl`` directory should suffice provided the
dependencies are satisfied.  There is also a ``debug`` target which builds with
``-O1``. 


Redhat based systems
--------------------

Redhat based systems provide the c++ version of ``nlopt`` via the package
manager.  This means the application has to be linked with the g++ linker, which
unfortunately fails for the cython extension.  

The easiest solution to this on a Redhat based system is to compile the library
from source.  The nlopt installation page has detailed `instructions
<http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation>`_ on how to do
this.  Then, by setting ``CFL_CFLAGS`` and ``CFL_LDLIBS`` variables to wherever
you installed nlopt,  you should be able to compile pyemp.  Note that since your
object files need to be position independent code (or cython will not be able to
create a shared object), you need to compile nlopt as a shared library.  See the
nlopt `page
<http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation#Shared_libraries>`_
for details on how to do this.


Intel mkl
---------

cfl also builds with Intel's icc compiler and math kernel library (instead of
LAPACK and ATLAS/BLAS).  If you don't have a working lapacke installation, you
need to change the ``USE_MKL`` macro in ``pycf/cfl/include/cfl_config.h`` to
``TRUE``.  This is not strictly necessary if you also have lapacke, since the
mkl_lapacke and standard lapacke headers seem to be interchangable, and the
linker will use the mkl library even if ``USE_MKL`` is false. Then, provided the
bin directory containing icc is part of your system ``$PATH``, building with icc
and linking against mkl is done by::
  
  python setup.py build_ext --compiler=intel

The module can then be installed the usual way::

  python setup.py install --prefix=/path/to/dir

Note that this defaults to linking against the lp64 interface, and OpenMP
support.  To adjust these settings, modify the ``MKL_CFLAGS`` variable in
``pycf/cfl/makefile``.  Furthermore, it is assumed that core libraries are in
``intel/lib/intel64`` and mkl libraries are in ``intel/mkl/lib/intel64/``, where
the location of the ``intel`` directory is inferred from the location of icc.
  
To build only cfl with icc set the following environment variables prior to
running make::

  export CFL_CC=icc
  export INTEL_PATH=/path/to/inteldir

where ``inteldir`` is again assumed to follow the standard intel installation
directory layout. 
