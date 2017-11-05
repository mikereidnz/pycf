#!/usr/bin/env python

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from pycf.pyemp import *

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


# Specifying transition energy and level range for 5F5 -> 5I8.
x = [15472.6, 15473.3]
levels = ['1', '16', '537', '552', '5F5', '5I8']

# Nuclear magnetic dipole parameter.
al = 0.038
# Instantiate a spectrum object. 
ho = Spectrum(name = 'ho', **spec_params(levels, x, al))

# Run emp methods.
cfit_obj = ho.cfit() 
vtrans_obj = ho.vtrans()
inten_obj = ho.inten()
spectrum_data_obj = ho.spectrum_data()

# Plotting the spectrum.
fig = plt.figure()
ax = fig.add_subplot(111, projection='spectrum')
ax.spectrumplot(ho)
ax.set_xlabel('Wavenumbers (cm${}^{-1}$)') 
ax.grid(True)

plt.show()
