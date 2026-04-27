=========================================================================
Installation of gpc-19990118 for compilation of Mike's programs on Debian
=========================================================================

Installing the compiler
=======================
The only compiler that I have found that compiles fully-functional versions of
Mike's emp programs on Linux is the GNU pascal compiler version 19990118. Newer
versions seem to compile the programs fine, but the resulting exectuables are
unreliable and often fail to execute, yielding hard to debug errors such as a
``CANNOT FIND MOMENT`` warning by inten.

You need to install both the pascal compiler,
gpc-19990118-1.i386-pc-linux-gnu.rpm, and a compatible version of gcc contained
in gpc-extras-19990118-1.i386-pc-linux-gnu.rpm. I installed the compiler in my
home directory, so the easiest method on a dpkg based system is to install
rpm2cpio and extract the archieves using::

  rpm2cpio gpc-19990118-1.i386-pc-linux-gnu.rpm | cpio -idmv

You only need the ``bin`` and the ``lib`` folder. Do the same for the extras
package and move the resulting ``libgcc.a`` archieve to the
``lib/gcc-lib/i386-redhat-linux/2.8.1/`` directory.

The ``install-gpc-binary.sh`` script provided on the gpc website is too new for
this version of gpc and will always fail since it searches for gpcpp, which in
our version is called gpc-cpp. Furthermore the ``GPC_EXEC_PREFIX`` environment
variable does not seem to be used by this version of gpc. Consequently, I ended
up creating symbolic links in the ``bin`` directory targeted at the ``gpc-cpp``
binary and the ``gpc1`` binary, both in the ``2.8.1`` directory. After adding
the ``bin`` directory to the PATH, gpc seemed to execute fine.

Specifying the linker search path
=================================
Since gpc (or rather the version of gcc that gpc calls) passes a default search
path to the linker, the current installation will fail to assemble compiled
programs correctly. This can be solved by adding the::

  lib/gcc-lib/i386-redhat-linux/2.8.1/

directory to the ``LIBRARY_PATH`` environment variable.

Compiling in an amd64 environment
=================================
Note: below instructions fail with Debian Stretch (2018-01-09), but I will
retain the text just in case. For Stretch, after trying to install a bunch of
i386 packages manually (such as ``libc6:i386``), I finally got the necessary
dependencies by installing ``gcc-multilib``. I is a meta package which installs
all kinds of libraries, and I assume it made all my previous, manual, installs
redundant.  Old instructions for Jessie follow below.

If you have a 64-bit enviroment, you will need to install the 32-bit c
libraries, called ``libc6:i386`` and ``libglib2.0-0:i386`` on Debian Jessie.
Furthermore you will need to explicitly specify the path to shared 32-bit
library objects for the linker, using the ``LIBRARY_PATH`` environment variable.
On Debian multilib (since Wheezy), the 32-bit libraries are in ``/bin/lib32``.

Additionally, you need to explicitly pass the ``-32`` flag to the linker using
the ``-Wa`` argument of gcc. My modified ``gnu_pco.csh`` now has the following
line ::

  time gpc -mcpu=i386 -Wa,-32 -I$EMPSRC -I$EMPSRC/gnu -O3 -o $EMPBIN/$1.exe $1.p
