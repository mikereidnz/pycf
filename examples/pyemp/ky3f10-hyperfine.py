#!/usr/bin/env python

"""
Copyright (C) 2012-2013
Sebastian Horvath (sebastian.horvath@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser Public License as published by
the Free Software Foundation, either version 2.1 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser Public License
along with This program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np 
import matplotlib.pyplot as plt
from pyemp import *


def hoSpectrum(iLevel, fLevel, filename, x, t):
    """Spectrum parameters. Returns a dictionary that can be conveniently
    unpacked and passed as kwargs to the Spectrum class. Function arguments
    are arbitrary and can be used to vary any of the below defined
    parameters."""  
    #eqhyp = 0.06 originally.
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
               assign al 0.038
               assign eqhyp 0.06""",
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
            'plotargs': {'polarization': 'isotropic', 'lines': 'lines', 'linewidth':
                    '0.12', 'temp': t, 'xrange': x}}

    return spectrumData

# Load and manipulate experimental data.
#data05 = np.loadtxt("data/31-05-13_Z1-D3_excitation_0.05Ho.txt", delimiter=' ',
#        skiprows=0)
#wavelength05 = np.linspace(646.279, 646.303, len(data05[:,0]))
#wavenumbers05 = 1/wavelength05 * 10**7
#index05 = np.where(wavelength05>646.034)
#data = np.loadtxt("data/10-05-13_Z1-D3_excitation.txt", delimiter=' ',
#        skiprows=0)
#wavelength = np.linspace(646.279, 646.303, len(data[:,0]))
#wavenumbers = 1/wavelength * 10**7
#index = np.where(wavelength>646.034)

# set emproot path
linuxemp = '/home/sph/local/linuxemp'

# Specify an  initial and some final levels.
iLevel = ['1', '16', '5I8']

# 5I8 -> 5I7
#x = [5092, 5102]
#fLevel = ['137', '256', '5I7']

# 5I8 -> 5I6
#x = [8623, 8625]
#fLevel = ['257', '360', '5I6']

# 5I8 -> 3K8
x = [15472.6, 15473.3]
#x = [21300, 21400]
fLevel = ['537', '552', '5F5']

# Instantiate a spectrum object for some initial and final level. 
ky3f10 = Spectrum(name = 'ky3f10', plt = 'ky3f10', **hoSpectrum(iLevel, fLevel,
    'hoc4v_i', x, 10))

# Instantiate cfit, vtrans, inten and spectrum objects. These must be
# instantiated in this order, but if parameters are varied for ky3f10 which
# only affect later calculations, one does not have re-instantiate earlier
# objects. They all have the attribute object['log'] which should prove
# useful in an interactive session. 
#ky3f10Cfit = Cfit(ky3f10)
#ky3f10Vtrans = Vtrans(ky3f10, matel='hoc4vint_i', tvals='ky3f10')
ky3f10Inten = Inten(ky3f10, tvals='ky3f10', trans='ky3f10')
ky3f10Spectrum = SpectrumData(ky3f10)
#ky3f10Spectrum = SpectrumErun(ky3f10, action='load')


# When we instantiate the axis object we must pass the
# projection='spectrum' kwarg. This then allows one to use the spectrum
# method of ax which has the usual *args and **kwargs of plot, but requires
# as the first argument an object of type SpectrumData.
# plot experimental spectrum
fig = plt.figure()

#ax = fig.add_subplot(311)
#ax.plot( wavenumbers[index] ,-data[index,1][0],)
#ax.set_ylabel('Intensity (arb. units)')
#ax.grid(True)
#ax1 = fig.add_subplot(211)
#ax1.plot( wavenumbers05[index05] ,-data05[index05,1][0],)
#plt.setp(ax1.get_xticklabels(), visible=False)
#ax1.set_ylabel('Intensity (arb. units)')
#ax1.grid(True)

# plot fitted spectrum
ax2 = fig.add_subplot(111, projection='spectrum')
ax2.spectrumplot(ky3f10, transitionlabels='true')
#ax2.spectrumplot(ky3f10)
ax2.set_xlabel('Wavenumbers (cm${}^{-1}$)') 
ax2.grid(True)
#plt.xlim([15350, 15650])
#plt.ylim([0, 0.010])


plt.show()
#plt.savefig('ky3f10_spectrum_2.pdf',format='pdf')

