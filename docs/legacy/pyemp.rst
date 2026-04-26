pyemp tutorial 
==============

.. currentmodule:: pyemp

pyemp is a python wrapper for Michael F. Reid's emperical crystal field theory
routines, henceforth abbreviated as emp.  It is primarily intended to ease
automated calling of emp processes in python scripts.  It can also be useful for
interactive plotting sessions using IPython. 

The examples provided for pyemp are all an adaptation of crystal field
calculations of |Ho3KY3F10|.  Initially, we will fit the crystal field levels,
calculate the transition intensities, and plot the resulting spectrum.  The
example is then adapted to vary the magnetic dipole interaction parameter to
produce a 3D plot showing transition intensities with respect to energy and the
magnetic dipole parameter. 

Energy level fitting for Ho3+:KY3F10
------------------------------------

Since :mod:`pyemp` presently only supports the cfit, vtrans, inten and spectrum
programs the below code has to be executed in a directory that contains states
and tensors previously generated with jmcalc and sljcalc.  The directory
``pycf/examples/ho_ky3f10`` contains all the required matrix element and state
input files.


The first step is to create a :class:`Spectrum` object.  The :class:`Spectrum`
class is used to abstracting all the information required to run the various emp
programs.  Consequently we have a large number of parameters that must be passed
to the :class:`Spectrum` constructor as keyword arguments.  A convenient way of
handling this is  create a function that returns a dictionary with the
appropriate key value pairs.  This has the advantage that if one wants to later
vary one of these parameters one can simply add the parameter as an argument to
this function and automate the creation of :class:`Spectrum` instances.  For the
example of |Ho3KY3F10| (see the ``ho_ky3f10.py`` example file) we use the
following function::
  
  def spec_params(levels, x, al):
      spectrum_data = {
              'emproot': '/home/sph/local/linuxemp',
              'states': 'hoc4v_i',
              'tensors': 'hoc4v_i',
              'matel': 'hoc4vint_i',
              'addtensors':
              """addten MTOT M0 1 M2 0.56 M4 0.31
                 addten PTOT P2 1 P4 0.5 P6 0.1 
                 addten m111 mag10 0.2695 mag11 -0.190595+i0.190595
                 addten m001 mag10 0.466860
                 addten m100 mag11 0.330120
                 addten magx mag11   0.707106781186547
                 addten magy mag11 -i0.707106781186547
                 addten magz mag10 1
                 addten al ahyp 1 bhyp -3.16227766""",
              'addassign':
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
                 assign eqhyp 0.06""".format(al),
              'expparams':
              """exptval unweighted.exp weight
                 delta c20 1 c40 1 c44 1 c60 1 c64 1
                 delta f2 0.2 f4 1 f6 1
                 delta zeta 1
                 lsq 10""",
              'levels': levels,
              'edconstruct':
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
              'edipoletensor':
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
              'plotargs': {'polarization': 'isotropic', 'lines': 'lines', 'linewidth':
                      '0.12', 'temp': 10, 'xrange': x}}
  
      return spectrum_data

When called, ``spec_params`` will return a dictionary that can be directly
passed to the :class:`Spectrum` constructor.  For details of the individual
dictionary keys, see the :class:`Spectrum` reference.  Note that the path
``linuxemp`` must be updated to reflect the installation directory of emp
programs on your machine.

The next section of ``ho_ky3f10.py`` specifies the values of a few parameters
that we have chosen to explicitly pass to ``spec_params``::

  # Specifying transition energy and level range for 5F5 -> 5I8.
  x = [15472.6, 15473.3]
  levels = ['1', '16', '537', '552', '5F5', '5I8']
  
  # Nuclear magnetic dipole parameter.
  al = 0.038


We can now create a :class:`Spectrum` object::
  
  ho = Spectrum(name = 'ho', **spec_params(levels, x, al))

The double asterisk unpacks the dictionary and passes the values as keyword
arguments; see `Unpacking Argument Lists`_ for details.

To execute the emp programs we run the methods :func:`Spectrum.cfit`,
:func:`Spectrum.vtrans`, :func:`Spectrum.inten` and
:func:`Spectrum.spectrum_data`.  These must be called in this order, just like
their corresponding erun programs, but if parameters are varied for ``ho`` which
only affect later calculations, one does not have to re-run earlier programs.   

Calling the four methods::

  cfit_obj = ho.cfit() 
  vtrans_obj = ho.vtrans()
  inten_obj = ho.inten()
  spectrum_data_obj = ho.spectrum_data()

To plot the resulting spectrum data ``pyemp`` provides a new matplotlib
projection which automatically parses data from a :class:`Spectrum` object.  To
use this functionality we have to specify the keyword argument ``projection =
'spectrum'`` when instantiating the axis object::
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='spectrum')
  ax.spectrumplot(ho)
  ax.set_xlabel('Wavenumbers (cm${}^{-1}$)') 
  ax.grid(True)

  plt.show()

For details of :func:`spectrumplot` see the :class:`SpectrumAxes` reference.

Scripting pyemp
---------------

We will now extend the above example to vary the magnetic hyperfine parameter
``al`` and execute the emp routines to generate a 3D plot of transition
intensities.  In anticipation
of this, the ``spec_params`` function provided in the `Energy level fitting for
Ho3+:KY3F10`_ example already has an argument for ``al``.  Since the
:class:`SpectrumAxes` projection does not support 3D plots, we will need to use
the standard matplotlib 3D projection.  We can obtain the data from the spectrum
object using the ``curve_energies`` and ``curve_inten`` attributes::

  for i in range(10):
      ho = Spectrum(name = 'ho', **spec_params(levels, x, 0.025 + i/500.0))
  
      cfit_obj = ho.cfit() 
      vtrans_obj = ho.vtrans()
      inten_obj = ho.inten()
      spectrum_data_obj = ho.spectrum_data()
      al = 0.025 + i/500.0 * np.ones(len(ho.curve_energies))
      ax.plot(ho.curve_energies, al, ho.curve_inten)
  
  ax.set_xlabel('Energy')
  ax.set_ylabel('Magnetic dipole moment')
  ax.set_zlabel('Isotropic')
  
  plt.show() 

For a complete version of this example, see the ``ho_ky3f10_al.py`` file.

Transition labels
-----------------

One useful feature made possible by the reading of the inten program log file is
the labeling of transitions by their initial and final states.  To enable this,
the plotting section from the `Energy level fitting for Ho3+:KY3F10`_ example
should be adapted to read::
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='spectrum')
  ax.spectrumplot(ho, transitionlabels='true')
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
and natively generates the line and curve data.  This method provides additional
information such as the initial and final state labels of transitions.  If you
wish to use the transitions label feature of :class:`SpectrumAxes` you must
generate the lines and curves data with :class:`SpectrumData`.
:class:`SpectrumErun` executes the c spectrum program and then loads the
resulting ``lines.gp_`` and ``curves.gp_`` files.  This allows one to take
advantage of several options that the c spectrum program supports, such as
vibronics calculations.  Additionally, :class:`SpectrumErun` can be used for
loading existing ``lines.gp_`` and ``curves.gp_`` files.

Gotchas
-------

If you write a script that you edit often and re-execute then it is easy to run
into execution order problems with emp methods.  Say, for example, one initially
has a script that instantiates all four :class:`Cfit`, :class:`Vtrans`,
:class:`Inten` and :class:`SpectrumData` objects, and then one wants to manually
change the initial and final energy levels and re-execute the script.  Since
only :class:`Inten` and :class:`SpectrumData` depend on the initial and final
energy level variables, it is tempting to comment out the cfit and vtrans lines;
however, the dependency checking of :class:`GenericErun` will fail, since the
:class:`Spectrum` object does not have a ``tvals`` or ``trans`` attribute.  A
convenient workaround is to pass these to the :class:`Inten` and
:class:`SpectrumData` constructors or their :class:`Spectrum` wrapper methods,
which have the same name specified as a keyword argument when the
:class:`Spectrum` method was created.  For example, in `Energy level fitting for
Ho3+:KY3F10`_, one would use::

  inten_obj = ho.inten(tvals='ho', trans='ho')
  spectrum_data_obj = ho.spectrum_data() 


.. |Ho3KY3F10| replace:: Ho\ :sup:`3+`\:KY\ :sub:`3`\F\ :sub:`10`
.. _Unpacking Argument Lists: http://docs.python.org/2/tutorial/controlflow.html#tut-unpacking-arguments



