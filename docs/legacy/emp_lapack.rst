================================
Using LAPACK linked emp routines
================================

Compiling 32-bit LAPACK
=======================

Since emp routines only work with the 19990118-1.i386 version of gpc we require
a 32-bit version of LAPACK to link against.  Unfortunatly Debian testing does
not have a multi-lib version of LAPACK so it is necessary to compile LAPACK.
Download and unpack the LAPACK `package
<http://www.netlib.org/lapack/lapack.tgz>`_ then copy
``INSTALL/make.inc.gfortran`` to ``make.inc`` in the lapack source directory.
Furthermore, edit ``make.inc`` and add the ``-m32`` and ``-fPIC`` flags to the
``FORTRAN`` and the ``LOADER`` variables.  To build the reference BLAS library
and LAPACK, execute::

  make blaslib
  make

This will build 32-bit versions of ``librefblas.a`` and ``liblapack.a``.  These
can then be moved or linked to a directory of your choice.  If you link to a
non-standard location for libraries, ensure that the relevant directory is added
to the ``LIBRARY_PATH`` environment variable or specified explicitly when gpc is
called.

Compiling LAPACK linked emp routines
====================================

The distributed version of ``gnu_pco_lapack.csh`` needs to be adapted to work
with versions of gcc newer than 4.0.  In particular, ``g2c`` has been replaced
with ``gfortran`` and we need to specify a 32-bit version of BLAS to link --
while not optimized the reference BLAS library compiled during the LAPACK
compilation is a convenient choice.  My gpc call in ``gnu_pco_lapack.csh`` is::

  gpc -mcpu=i386 -Wa,-32 -I$EMPSRC -I$EMPSRC/gnu -O3 -o $EMPBIN/$1.exe $1.p \
    -L/usr/lib/gcc/x86_64-linux-gnu/4.7/32/ -llapack -lgfortran -lrefblas

where the path to the gfortran library may need to be updated for your version
of gcc.
