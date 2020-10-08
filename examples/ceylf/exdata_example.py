#!/usr/bin/env python3

from __future__ import division
import numpy as np


import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.cfl_util import *

#### Example for Ce:YLF, w/ data from 10.1016/j.optmat.2015.06.046

# Select what kind of experimental energy level data should be used
# Options:  abs - absolute energy level values 
#           abs_diff - absolute energy level data incl. level differences
#           sl_diff - state label energy level data incl. level differences
data_sel = 'abs'

t = ImportSLJM("matel/f1cf")
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
(w, z) = h.diag()
w = w - np.min(w)
#print(h.gen_summary())


if data_sel == 'abs':
    ex = np.array([
        [2,       0],
        [3,     216],
        [8,    2216],
        [9,  2312.8],
        [12, 2428.8],
        [14, 3157.8]])
    weights = np.ones(len(ex))
    exdata = cfl.ExData(ex, 'A', weights=weights)

elif data_sel == 'abs_diff':
    # Mixture of absolute and difference energy level data
    ex_abs = np.array([
        [2,       0],
        [3,     216],
        [8,    2216],
        [9,  2312.6]])
    w_abs = np.array([1, 2, 1, 2])
    ex_diff = np.array([
        [9,  12, 116.0], 
        [12, 14, 729.0]])
    w_diff = np.array([2, 1]) 
    exdata = cfl.ExData((ex_abs, ex_diff), ('A', 'D'), weights=(w_abs, w_diff))

elif data_sel == 'sl_diff':
    # State label energy level data, both absolute and difference. State labels
    # are specified with the quantum numbers of the principal component using
    # the ordering given by the Label key. In this example, the label key is
    # SLJM. If you are unsure about the label key for the states you are using,
    # it is printed at the bottom of the output from h.gen_summary(). 
    
    # The first four elements for each energy are the state label values (SLJM),
    # the last is the energy.
    ex_asl = np.array([
        [2, 3, 5, 5, 0], 
        [2, 3, 5, 1, 216], 
        [2, 3, 7, 7, 2216.10], 
        [2, 3, 7, 3, 2312.80]])
    # The first eight elements are state label values for the initial and final
    # states, and the last entry is the energy level difference. 
    ex_dsl = np.array([
        [2, 3, 7, 3, 2, 3, 7, 1, 116.0], 
        [2, 3, 7, 1, 2, 3, 7, 5, 729.0]])
    
    exdata = cfl.ExData((ex_asl, ex_dsl), ('AS', 'DS'), label_key='SLJM')

else:
    raise ValueError("Invalid sel_data selection")


cfl_min = cfl.CFLMin('nlopt_bobyqa', xtol=1e-6)
param = ['EAVG', 'C20', 'C40', 'C44']

res = cfl.e_fit(param, h, exdata, cfl_min)
print(res['summary'])
