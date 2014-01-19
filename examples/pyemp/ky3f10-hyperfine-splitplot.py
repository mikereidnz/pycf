#!/usr/bin/python2


import numpy as np 
import matplotlib.pyplot as plt
from pyspectrum import *


def hoTransition(iLevel, fLevel, filename, x, t):
    """Transition parameters. Returns a dictionary that can be conveniently unpacked
    and passed as kwargs to the Transition class. Function arguments are arbitrary 
    and can be used to vary any of the below defined parameters."""  
    transitionData = {
            'emproot': linuxemp,
            'states': filename,
            'tensors': filename,
            'vintInput': 'hoc4vint_i',
            'addTensors':
            """addten MTOT M0 1 M2 0.56 M4 0.31
               addten PTOT P2 1 P4 0.5 P6 0.1 
               addten m111 mag10 0.2695 mag11 -0.190595+i0.190595
               addten m001 mag10 0.466860
               addten m100 mag11 0.330120
               addten al ahyp 1 bhyp -3.16227766""",
            'addAssign':
            """assign F2 94063 F4 66361 F6 51637
               assign ALPHA 17.15 BETA -607.9 GAMMA 1800
               assign t2 400 t3 37 t4 105 t6 -264 t7 316 t8 336 
               assign ZETA 2142
               assign MTOT 2.54 PTOT 605 
               assign c20  -669
               assign c40 -1269
               assign c44   344 
               assign c60   525 
               assign c64     9
               assign al 0.037
               assign eqhyp 0.06""",
            'levels': [iLevel[0], iLevel[1], fLevel[0], fLevel[1], iLevel[2], fLevel[2]],
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
            'temp': t,
            'xrange': x,
            'plotArgs': {'polon': 'isotropic', 'lines': 'lines', 'linewidth': '0.05'}}

    return transitionData


if __name__ == '__main__':
    # set emproot path
    linuxemp = '/home/sph/local/linuxemp'
    
    # Specify an  initial and some final levels.
    iLevel = ['1', '56', '5I8']
    
    # 5I8 -> 5I7
    x = [5092, 5102]
    fLevel = ['137', '256', '5I7']
    
    # 5I8 -> 5I6
    #x = [8623, 8625]
    #fLevel = ['257', '360', '5I6']
    
    # 5I8 -> 3K8
    #x = [21342, 21352]
    #x = [21300, 21400]
    #fLevel = ['833', '952', '3K8']
    
    # splitplot dict
    
    splitDict = {'energy': [0, 26], 'var': 'ZETA', 'range': [2100., 2200.]}  
    
    ky3f10 = Transition(name = 'ky3f10', splitplot = splitDict, **hoTransition(iLevel, fLevel, 'hoc4v_i', x, '10'))
    
    ky3f10Cfit = CfitData(ky3f10)
    #ky3f10Vtrans = VtransData(ky3f10)
    #ky3f10Inten = IntenData(ky3f10)
    #ky3f10Spectrum = SpectrumData(ky3f10, action='execSpectrum')
    ##ky3f10Spectrum = SpectrumData('./', action='load')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='spectrum')
    #ax.spectrum(ky3f10Spectrum)
    ax.splitplot(ky3f10Cfit)
    
    ax.grid(True)
    #plt.xlim(x)
    #plt.ylim([0, 0.001*10**(-10)])
    
    
    plt.show()
    ##plt.savefig('ky3f10Spectrum.pdf',format='pdf')
    
