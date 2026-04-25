#!/usr/bin/env python3

from pathlib import Path
import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM

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

exdata = np.array([
    [2,       0],
    [3,     216],
    [8,    2216],
    [9,  2312.8],
    [12, 2428.8],
    [14, 3157.8]])


cfl_min = cfl.CFLMin('gsl_nls', niter=1)
param = ['EAVG', 'C20', 'C40', 'C44']

# Specify the list of Hamiltonians, and associated information, for the fitting.
h_list = [h, h]
weights_list = [1, 0.5]
exdata_list = [exdata, exdata]

res = cfl.mh_fit(param, h_list, weights_list, exdata_list, cfl_min)
print(res['summary'])
