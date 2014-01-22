pyemp tutorial 
==============

.. currentmodule:: pyemp

pyemp is a python wrapper for Michael F. Reid's emperical crystal field theory
routines, henceforth abbreviated as emp.  The principal application is to
automate the calling of emp processes in python scripts.  A secondary outcome of
this automation is that, given iPython's excellent interactive support, these
modules are also useful for interactive use of emp routines.  

This tutorial will automate the fitting of crystal field levels in |Ho3KY3F10|.
Initially, we will fit the crystal field levels and plot the resulting spectrum.
This example will then be adapted to vary the magnetic dipole interaction
parameter to produce a 3D plot showing transition intensities with respect to
energy and the magnetic dipole parameter. 

Energy level fitting for Ho3+:KY3F10
------------------------------------

This example requires a working installation of Mike's crystal field theory
routines.  Additionally, :mod:`pyemp` presently only wraps the cfit,
vtrans, inten and spectrum programs; consequently the below code has to be
executed in a directory that contains states and tensors previously generated
with jmcalc and sljcalc.  For the purpose of this tutorial, the necessary files
are provided in the ``ky3f10`` examples directory. 

The first step is to create a :class:`Spectrum` object.  This servers as a
useful way of abstracting all the information relevant to a given spectrum.
Since this requires a large number of parameters, such as crystal field
parameters, initial and final energy levels, dipole tensors, etc., which must be
passed as keyword arguments, it is convenient to create a function that returns
a dictionary that contains this information.  An additional advantage of this
approach is that this same function can later be used to vary a selection of the
parameters, making it easy to script the creation of :class:`Spectrum` objects. 

For the example of |Ho3KY3F10| we have::
  
  from __future__ import division
  import numpy as np 
  import matplotlib.pyplot as plt
  from pycf.pyemp import *
  
  def spec_params(iLevel, fLevel, filename, x, al, eqhyp):
      spectrumData = {
              'emproot': linuxemp,
              'states': filename,
              'tensors': filename,
              'matel': 'hoc4vint_i',
              'addTensors':
              """addten MTOT M0 1 M2 0.56 M4 0.31
                 addten PTOT P2 1 P4 0.5 P6 0.1 
                 addten m111 mag10 0.2695 mag11 -0.190595+i0.190595
                 addten m001 mag10 0.466860
                 addten m100 mag11 0.330120
                 addten al ahyp 1 bhyp -3.16227766""",
              'addAssign':
              """assign F2 80063 F4 66361 F6 51637
                 assign ALPHA 17.15 BETA -607.9 GAMMA 1800
                 assign t2 400 t3 37 t4 105 t6 -264 t7 316 t8 336 
                 assign ZETA 2142
                 assign MTOT 2.54 PTOT 605 
                 assign c20  -669
                 assign c40 -1269
                 assign c44   344 
                 assign c60   525 
                 assign c64     9
                 assign al {0}
                 assign eqhyp {1}""".format(al, eqhyp),
              'expParams':
              """exptval unweighted.exp weight
                 delta c20 1 c40 1 c44 1 c60 1 c64 1
                 delta f2 0.2 f4 1 f6 1
                 delta zeta 1
                 lsq 10""",
              'levels': [iLevel[0], iLevel[1], fLevel[0], fLevel[1], iLevel[2],
                  fLevel[2]],
              'intenParams':
              """EDCONSTRUCT 9
                 Ho ky3f10 hyperfine transitions
                 A210  2 1 0 
                 A230  2 3 0 
                 A430  4 3 0 
                 A450  4 5 0 
                 A454  4 5 4 
                 A650  6 5 0 
                 A654  6 5 4 
                 A670  6 7 0
                 A674  6 7 4""",
              'edipoleTensor':
              """A210 2.978914463e-10 % 
                 A230 6.624105316e-11 % 
                 A430 -5.492288570e-11 % 
                 A450 -2.252749050e-11 % 
                 A454 4.052097980e-12 % 
                 A650 2.529080178e-10 % 
                 A654 -4.549144380e-11 % 
                 A670 4.912922087e-15 % 
                 A674 1.750735430e-13 \n""",
              'edipole': '1',
              'mdipole': '1',
              'temp' : 10,
              'xrange' : x,
              'plotArgs': {'polon': 'isotropic', 'lines': 'lines', 'linewidth': '0.05'}}
  
      return(spectrumData)
  
When called, this function will return a dictionary that can be directly passed
to the :class:`Spectrum` constructor.  For details of the individual dictionary
keys, see the :class:`Spectrum` reference.  

We now specify a few global variables::

  # set emproot path
  linuxemp = '/home/sph/local/linuxemp'
  
  # Specify an initial and some final levels.
  # 5I8 -> 3K8
  iLevel = ['1', '16', '5I8']
  fLevel = ['537', '552', '5F5']
  x = [15472.6, 15473.3]

In this case the path ``linuxemp`` must be updated to reflect the installation
directory of emp programs on your machine. 

We can now create a :class:`Spectrum` object::
  
 ky3f10 = Spectrum(name = 'ky3f10', plt = 'ky3f10', **spec_params(iLevel,
    fLevel, 'hoc4v_i', x, 0.037, 0.06)) 

The double asterisk unpacks the dictionary and passes the values as keyword
arguments; see `Unpacking Argument Lists`_ for details.

To proceed, we instantiate objects of type :class:`Cfit`, :class:`Vtrans`,
:class:`Inten` and :class:`SpectrumData`.  These must be instantiated in this
order, but if parameters are varied for ``ky3f10`` which only affect later
calculations, one does not have to re-instantiate earlier objects.  The first
three returned objects all have a ``log`` attribute that returns the appropriate
emp log file; this is useful for interactive sessions.  :class:`SpectrumData`
does not execute the emp spectrum program and consequently does not have a
``log`` attribute -- see `SpectrumData vs SpectrumErun`_ for details. 

Creating the four objects::
  
  ky3f10Cfit = Cfit(ky3f10)
  ky3f10Vtrans = Vtrans(ky3f10)
  ky3f10Inten = Inten(ky3f10)
  ky3f10Spectrum = SpectrumData(ky3f10)

To plot the resulting spectrum data ``pyemp`` provides a new matplotlib
projection which automatically parses data from a :class:`Spectrum` object, in
our case ``ky3f10``.  To use this functionality we have to specify the keyword
argument ``projection = 'spectrum'`` when instantiating the axis object::
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='spectrum')
  ax.spectrumplot(ky3f10, transitionlabels='true')
  ax.set_xlabel('Wavenumbers (cm${}^{-1}$)') 
  ax.grid(True)
  plt.show()

For details of :func:`spectrumplot` see the :class:`SpectrumAxes` reference.

Scripting pyemp
---------------

We will now extend the above example to vary the magnetic hyperfine parameter
``al`` and re-execute the emp routines to generate a 3D plot.  In anticipation
of this, the ``spec_params`` function provided in the `Energy level fitting for
Ho3+:KY3F10`_ example already has arguments for ``al`` and ``eqhyp``.  Since the
:class:`SpectrumAxes` projection does not support 3D plots, we will be using the
standard matplotlib 3D projection and manually get the data from the
:class:`Spectrum` object::
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  for i in range(10):
      ky3f10 = Spectrum(name = 'ky3f10', **spec_params(iLevel, fLevel, 'hoc4v_i',
          x, '10', 0.025 + i/500.0, eqhyp))

      ky3f10Cfit = CfitData(ky3f10)
      ky3f10Vtrans = VtransData(ky3f10)
      ky3f10Inten = IntenData(ky3f10)
      ky3f10Spectrum = SpectrumData(ky3f10, action = 'exec')

      # Generate numerical values for al, and create plot.
      al = 0.025 + i/500.0 * np.ones(len(ky3f10['curveEnergies']))
      ax.plot(ky3f10['curveEnergies'], al, ky3f10['curveInten'])
  ax.set_xlabel('Energy')
  ax.set_ylabel('Magnetic dipole moment')
  ax.set_zlabel('Isotropic')
 
  plt.show() 

Transition labels
-----------------

One useful feature made possible by the reading of the inten program log file is
the labeling of transitions by their initial and final states.  To enable this,
the plotting section from the `Energy level fitting for Ho3+:KY3F10`_ example
should be adapted to read::
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='spectrum')
  ax.spectrumplot(ky3f10, transitionlabels='true')
  ax.set_xlabel('Wavenumbers (cm${}^{-1}$)') 
  ax.grid(True)
  plt.show()

This yields a plot such as:

.. image:: figures/transition_labels.pdf

For further details and options, see the :func:`SpectrumAxes.spectrumplot`
method reference. 

SpectrumData vs SpectrumErun
----------------------------

There are two different ways for generating spectrum data that can be used with
:class:`SpectrumAxes`, in particular, :class:`SpectrumData` and
:class:`SpectrumErun`.  :class:`SpectrumData` mines data from the inten log file
and natively generates the line and curve data.  This is the preferred method,
since it provides additional information such as the initial and final state
labels of transitions.  If you wish to use the transitions label feature of
:class:`SpectrumAxes` you must generate the lines and curves data with
:class:`SpectrumData`.  :class:`SpectrumErun` executes the c spectrum program
and then loads the resulting ``lines.gp_`` and ``curves.gp_`` files.  It is
mostly included for legacy reasons, but is still useful for vibronics
calculations (which :class:`SpectrumData` does not support) or loading existing
``lines.gp_`` and ``curves.gp_`` files. 

Gotchas
-------

If you write a script that you edit often and re-execute then it is easy to run
into execution order problems with emp classes.  Say, for example, one initially
has a script that instantiates all four :class:`Cfit`, :class:`Vtrans`,
:class:`Inten` and :class:`SpectrumData` objects, and then one wants to manually
change the initial and final energy levels and re-execute the script.  Since
only :class:`Inten` and :class:`SpectrumData` depend on the initial and final
energy level variables, it is tempting to comment out the cfit and vtrans lines;
however, the dependency checking of :class:`GenericErun` will fail, since the
:class:`Spectrum` object does not have a ``tvals`` or ``trans`` attribute.  A
convenient workaround is to pass these to the :class:`Inten` and
:class:`SpectrumData` constructors; so, for the example in `Energy level fitting
for Ho3+:KY3F10`_, one would use::
  
  ky3f10Inten = Inten(ky3f10, tvals='ky3f10', trans='ky3f10')
  ky3f10Spectrum = SpectrumData(ky3f10)


.. |Ho3KY3F10| replace:: Ho\ :sup:`3+`\:KY\ :sub:`3`\F\ :sub:`10`
.. _Unpacking Argument Lists: http://docs.python.org/2/tutorial/controlflow.html#tut-unpacking-arguments



