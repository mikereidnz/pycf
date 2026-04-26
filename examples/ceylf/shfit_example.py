#!/usr/bin/env python3
"""
Spin Hamiltonian extraction and fitting from crystal field Hamiltonian.

This example demonstrates:
1. Extracting spin Hamiltonian parameters from full crystal field Hamiltonian
2. Calculating effective g-tensors and zero-field splitting parameters
3. Fitting spin Hamiltonian parameters to match CF energy levels
4. Comparing CF predictions with spin Hamiltonian effective models

The example shows Ce:YLF Kramers doublet modeling using spin S=1/2 with
Zeeman interaction projections (MAGX, MAGY, MAGZ tensors).

Data source: 10.1016/j.optmat.2015.06.046

Prerequisites:
- Ce:YLF matrix element data (crystal field and magnetic) in matel/f1cf/
- numpy, pycf
"""

from pathlib import Path
import numpy as np


import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.cfl_util import *

#### Example of Spin Hamiltonian calculations for Ce:YLF, w/ data from
# 10.1016/j.optmat.2015.06.046

t = ImportSLJM(str(Path(__file__).parent / "matel" / "f1cf"))
coeff = {
'EAVG'  :    1035.1277,                
'ZETA'  :     625.6990,                
'C20'   :     297.8906,                
'C40'   :   -1328.1522,                
'C44'   :   -1282.4766,                
'C60'   :    -191.5100,                
'C64'   :   -1743.1424+692.8662j
}
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64])
h.set_coeff(coeff)


# Calculate the Spin Hamiltonian from the CF Hamiltonian. 
sh = cfl.SpinHamiltonian('zeeman', S=1/2, level=1)
sh.set_pro_data([t.MAGX, t.MAGY, t.MAGZ])
h.set_coeff(coeff)
print("Spin Hamiltonian w/ input CF parameters:")
print(np.real(sh.calc_param(h)))


# Spin Hamiltonian and energy level fitting.  
ex = np.array([
        [2,       0],
        [3,     216],
        [8,    2216],
        [9,  2312.8],
        [12, 2428.8],
        [14, 3157.8]])

g_tensor = np.array([[1.473, 0, 0], [0, 1.473, 0], [0, 0, 2.765]])

shx = {'zeeman': g_tensor}
weights = {'energy': 1, 'zeeman': 1e7}

cfl_min = cfl.CFLMin('nlopt_bobyqa', xtol=1e-6)
res = cfl.esh_fit(['EAVG', 'ZETA', 'C20', 'C40', 'C44', 'C60', 'C64'], h, sh, ex, shx, weights, cfl_min, svd_sym=True)

print(res['summary'])
