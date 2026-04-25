#!/usr/bin/env python

from pathlib import Path
import numpy as np
from numpy import linalg as LA

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.cfl_util import *

t = ImportSLJM(str(Path(__file__).parent / "matel" / "smcf_trunc"))

MTOT = t.M0 + 0.56*t.M2 + 0.31*t.M4
MTOT.name = 'MTOT'
PTOT =  t.P2 + 0.5*t.P4 + 0.1*t.P6  
PTOT.name = 'PTOT'
c4 = -0.25*t.C40 + 0.790569415042095*t.C42 + 0.448210728500398*t.C44
c4.name = 'c4'
c6 = -1.625*t.C60 - 0.640434422872475*t.C62 + 1.16926793336686*t.C64 - 0.949917759598166*t.C66
c6.name = 'c6'


t_list = [t.EAVG, t.F2, t.F4, t.F6, t.ALPHA, t.BETA, t.GAMMA, t.T2,
        t.T3, t.T4, t.T6, t.T7, t.T8, t.ZETA, t.C20, t.C40, t.C60, t.C22,
        t.C42, t.C62, t.C44, t.C64, t.C66, c4, c6, MTOT, PTOT, t.MAGZ, t.HYP]

coeff = {
'ALPHA':    20.1600,    
'BETA' :  -566.0000,    
'GAMMA':  1500.0000,    
'T2'   :   300.0000,    
'T3'   :    36.0000,    
'T4'   :    56.0000,    
'T6'   :  -347.0000,    
'T7'   :   373.0000,    
'T8'   :   348.0000,    
'MTOT' :     2.6000,    
'PTOT' :   357.0000,    
'MAGZ' : 0.00, 
# PRB 61 13593
'c4'   : -2112,
'c6'   :   945,
'EAVG'       :            47486.46,
'F2'         :            79010.34,
'F4'         :            57172.54,
'F6'         :            39566.03,
'ZETA'       :             1174.29,
'C20'        :              127.27,
'C22'        :               -9.46,
'C40'        :               38.29,
'C60'        :             -340.80,
'C42'        :              132.23,
'C62'        :               -3.53,
'C44'        :              -43.15,
'C64'        :             -228.56,
'C66'        :              203.19,
'HYP'        :              0.0049,
}

h = cfl.Hamiltonian(t_list)
h.set_coeff(coeff)
(w, z) = h.diag()
w = w - np.min(w)

sh = cfl.SpinHamiltonian(['zeeman', 'hyperfine'], S=1/2, I=7/2, level=1)
sh.set_pro_data([t.MAGX, t.MAGY, t.MAGZ, t.HYP], {'HYP': 0.0048})

ex = np.loadtxt('smnacaf2_exp_hyp.txt')

#print(h.gen_summary(ex=ex))
g_tensor = np.diag([0.606, 0.30, 0.477])
A_tensor = MHz2cm1(np.diag([644, 607, 220]))

# Optimization bounds and stepsize. 
bounds = bal_bounds(coeff, {
'EAVG'  :   500,
'F2'    :  2500,
'F4'    :  2500,
'F6'    :  1500,
'ZETA'  :    10,
'C20'   :   200, 
'C22'   :   200,
'C40'   :   200,
'C60'   :   200,
'C42'   :   300,
'C62'   :   200,
'C44'   :   200,
'C64'   :   200,
'C66'   :   200,
})
bounds['HYP'] = (0.001, 0.01)

#cfl_min = cfl.CFLMin('basinhopping', niter=200, lmin='nlopt_bobyqa', xtol=1e-5,
#    bounds=bounds, stepsize=stepsize, step_adapt_int=25, cov=False)
cfl_min = cfl.CFLMin('nlopt_sbplx', xtol=1e-5, bounds=bounds, dry_run=False)

shx = {'zeeman': g_tensor, 'hyperfine': A_tensor}
weights = {'zeeman': 5000000.0, 'hyperfine': 50000.0}
res = cfl.esh_fit(['EAVG','F2', 'F4', 'F6', 'ZETA', 'C20', 'C22', 'C40', 'C60',
    'C42', 'C62', 'C44', 'C64', 'C66', 'HYP'], h, sh, ex, shx, weights, cfl_min)


print(res['summary'])

#with open("2018-10-16_smnacaf2.txt", "w") as summary_file:
#with open("2018-10-17_smnacaf2.txt", "w") as summary_file:
#    summary_file.write(res['summary'])

