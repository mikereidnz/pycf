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
