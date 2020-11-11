#!/usr/bin/env python

from __future__ import division
import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.cfl_util import *
from pycf.spinh import SpinH
from numpy import linalg as LA

I = np.complex(0,1)
# Import the matrix elements.
t = ImportSLJM("matel/f11cf")
thfs = ImportSLJM("matel/erhfs")

#t.print_names()

MTOT = t.M0 + 0.56*t.M2 + 0.31*t.M4
PTOT = t.P2 + 0.5*t.P4 + 0.1*t.P6

hfsMTOT = thfs.M0 + 0.56*thfs.M2 + 0.31*thfs.M4
hfsMTOT.name = 'MTOT'
hfsPTOT =  thfs.P2 + 0.5*thfs.P4 + 0.1*thfs.P6
hfsPTOT.name = 'PTOT'

# Z, X, and Y magnetic fields times Bohr magnetion in cm-1/T.
mu_b = 0.466860
MZ = mu_b * t.MAGZ
MX = mu_b * t.MAGX
MY = mu_b * t.MAGY
hfsMZ = mu_b * thfs.MAGZ
hfsMZ.name = 'MZ'
hfsMX = mu_b * thfs.MAGX
hfsMX.name = 'MX'
hfsMY = mu_b * thfs.MAGY
hfsMY.name = 'MY'

t_list = [t.EAVG, t.F2, t.F4, t.F6, t.ALPHA, t.BETA, t.GAMMA, t.T2, t.T3, t.T4,
        t.T6, t.T7, t.T8, t.ZETA, t.C20, t.C21, t.C22, t.C40, t.C41, t.C42,
        t.C43, t.C44, t.C60, t.C61, t.C62, t.C63, t.C64, t.C65, t.C66, MTOT, PTOT, MX, MY, MZ]

thfs_list = [thfs.EAVG, thfs.F2, thfs.F4, thfs.F6, thfs.ALPHA, thfs.BETA,
        thfs.GAMMA, thfs.T2, thfs.T3, thfs.T4, thfs.T6, thfs.T7, thfs.T8,
        thfs.ZETA, thfs.C20, thfs.C21, thfs.C22, thfs.C40, thfs.C41, thfs.C42,
        thfs.C43, thfs.C44, thfs.C60, thfs.C61, thfs.C62, thfs.C63, thfs.C64,
        thfs.C65, thfs.C66, hfsMTOT, hfsPTOT, thfs.HYP, thfs.EQHYP, hfsMX, hfsMY, hfsMZ]

coeff = {
'ALPHA':                      17.79,
'BETA' :                    -582.10,
'GAMMA':                    1800.00,
'T2'   :                     400.00,
'T3'   :                      43.00,    
'T4'   :                      73.00,    
'T6'   :                    -271.00,    
'T7'   :                     308.00,    
'T8'   :                     299.00,    
'ZETA' :                    2376.00,
'MTOT' :                       3.86,    
'PTOT' :                     594.00,
'MZ'   :                       0.00,
'MX'   :                       0.00,
'MY'   :                       0.00,
# 2018-07-24_eryso_site1_sbplx.txt 
'EAVG'       :            35503.50,
'ZETA'       :             2362.91,
'F2'         :            96029.59,
'F4'         :            67670.64,
'F6'         :            53167.09,
'C20'        :             -149.75,
'C21'        :      420.65+396.02j,
'C22'        :      -228.53+27.55j,
'C40'        :             1131.25,
'C41'        :       985.73+34.21j,
'C42'        :      296.81+144.97j,
'C43'        :     -402.27-381.72j,
'C44'        :    -282.28+1114.28j,
'C60'        :             -263.23,
'C61'        :      111.94+222.91j,
'C62'        :      124.70+195.87j,
'C63'        :      -97.90+139.66j,
'C64'        :      -93.71-144.97j,
'C65'        :       13.95+109.55j,
'C66'        :        3.01-108.62j,
'HYP'        :   0.005466387150634,
'EQHYP'      :  0.0715677485869964,
}


def gen_b_spiral(B0, n):
    t = np.linspace(-1,1,n)
    B = np.array([[B0*np.sqrt(1-t**2)*np.cos(6*np.pi*t)], [B0*np.sqrt(1-t**2)*np.sin(6*np.pi*t)], [B0*t]]).reshape(3,n)

    return B.transpose()


# Set weighting factors
e_w = 3e-3               # Electronic levels
rh_w = 1e8              # 80 MHz RH data 
g_hfs_w = 0.5e5          # Ground state g, hfs
ge_w = 50e5             # Excited state g

# B spiral for ground and excited state
n = 12      # Number of field data points to be used
B0 = 0.05   # Max field strength in Tesla
B_list = gen_b_spiral(B0,n)

## Stephen's parameters; includes superconducting cavity zero-field data
gg = np.array([[2.8976, -2.9451, -3.5568], [-2.9451, 8.9003, 5.5683], [-3.5568, 5.5683, 5.1208]])
Ag = np.array([[274.2878, -202.5226, -350.8188], [-202.5226, 827.5043, 635.1453], [-350.8188, 635.1453, 706.1526]])
Qg = np.array([[10.3950, -9.1166, -9.9576], [-9.1166, -5.9530, -14.3238], [-9.9576, -14.3238, -4.4420]])

# Sun et al. excited state g
ge = np.array([[1.950, -2.212, -3.584], [-2.212, 4.232, 4.986], [-3.584, 4.986, 7.888]])

# Zero field Hamiltonian and energies.
h = cfl.Hamiltonian(t_list)
h.set_coeff(coeff)
h_list = [h]
ex = np.loadtxt('eryso_site1_energy.txt', skiprows=1)
exdata_list = [cfl.ExData(ex)]
weights_list = [e_w]

# Low freq (80 MHz) RH data, ac line - need  unique field values because of
# sample orientation. 

# Generate field values in mT (0, then 0.5 in each x, y, and z magnet
# coordinates)
lf_bvals = [np.zeros(3)]
v = 0.5
for i in range(3):
    B = np.zeros(3)
    B[i] = v 
    lf_bvals += [np.copy(B)]

# Curvature tensor and offset (MHz and MHz/mT^2)
f0 = 85.096939
M = np.array([[15.7, 30.1, -40. ], [ 30.1,  67.6, -82.5], [-40., -82.5, 104.3]])

for B in lf_bvals:
    fcalc = f0 + np.transpose(B).dot(M).dot(B)
    fcalc = fcalc/29.9702547e3  # Convert from MHz to cm-1
    exdata_list += [cfl.ExData(np.array([[137, 138, fcalc]]), 'D')]
    
    B = B*1e-3   # Convert to T
    h = cfl.Hamiltonian(thfs_list)
    coeff['MX'] = B[0]
    coeff['MY'] = B[1]
    coeff['MZ'] = B[2]
    h.set_coeff(coeff)
    h_list += [h]
    weights_list += [rh_w]

## High freq. (800 MHz) RH data (in polynomial form). Sweeps of xcoil,
## corresponds to MY in cryst coords. 
rh_ls = ['n', 'i', 't', 'l', 'p', 'ab', 'a', 'm', 'h', 'w', 'e', 'd']

hf_pd = {'a': np.array([ -66.15555132,    6.39162656,  930.61283071]),
        'ab': np.array([ -69.04862725,    5.41291306,  952.53418828]),
        'e': np.array([ -7.37903518e-01,  -7.54025848e-02,   8.79316067e+02]),
        'd': np.array([  7.24218971e-01,   4.02538296e-02,   8.79375239e+02]),
        'g': np.array([ -2.40721395e+00,  -1.07639045e-01,   8.80549189e+02]),
        'f': np.array([  1.79185958e+00,   1.30468394e-01,   8.79264214e+02]),
        'i': np.array([  -16.77488309,    -1.95603484,  1010.3429402 ]),
        'h': np.array([ -1.19913355e+00,  -1.91767706e-01,   8.23918680e+02]),
        'k': np.array([ -45.15444527,   -2.78151756,  818.58787975]),
        'j': np.array([  2.18620204e+01,   4.77922935e-01,   8.25555479e+02]),
        'm': np.array([  1.78788342e+00,   1.45801605e-01,   7.74219752e+02]),
        'l': np.array([  19.61210058,    2.27686832,  752.61764749]),
        'n': np.array([   34.30221452,     5.33369394,  1096.58093532]),
        'p': np.array([   28.76980869,     2.75856858,  1167.87303762]),
        'b': np.array([  1.13813883e-01,   8.96736961e+02]),
        'c': np.array([  1.13813883e-01,   8.96736961e+02]),
        't': np.array([ -33.78216636,   -3.17673515,  667.08738712]),
        'w': np.array([  -2.45056251,   -0.75119208,  726.13596956]),
        'v': np.array([-303.52686482,   14.53612923,  612.24778816]),
        'x': np.array([  54.99637342,    3.57944151,  717.38335428]),
        'z': np.array([ -55.69957211,    4.80461517,  802.89630244])}

# High frequency RH data level indices.
hf_li = {
        'n': [135, 138],
        'i': [135, 137],
        't': [136, 137],
        'p': [7, 10],
        'ab': [6, 8],
        'a': [9, 11],
        'w': [142, 144],
        'm': [7, 9],
        'h': [8, 10],
        't': [136, 137],
        'l': [136, 138],
        'e': [1, 3], # only valid at 0.5 mT
        'd': [2, 4], # only valid at 0.5 mT
        'v': [6, 7],
        'k': [138,139] # only valid at 0.5mT
        }

hf_bvals = np.zeros([3,3])
hf_bvals[1,1] = 0.5
hf_bvals[2,1] = 0.3

for ii,B in enumerate(hf_bvals):
    rh_ex = []
    for i,k in enumerate(rh_ls):
        if (k in ['z', 'x', 'b', 'c', 'v']) & (ii != 0):
            # only valid at 0.0 mT
            continue
        elif (k in ['f', 'g', 'e', 'd', 'k']) & (ii != 1):
            # only valid at 0.5 mT
            continue
        else:
            p = np.poly1d(hf_pd[k])
            fcalc = p(B[1])/29.9702547e3  # Convert from MHz to cm-1
            rh_ex += [hf_li[k]+[fcalc]]

    exdata_list += [cfl.ExData(np.array(rh_ex), 'D')]
    
    B = B*1e-3   # Convert to T
    h = cfl.Hamiltonian(thfs_list)
    coeff['MX'] = B[0]
    coeff['MY'] = B[1]
    coeff['MZ'] = B[2]
    h.set_coeff(coeff)
    h_list += [h]
    weights_list += [rh_w]



# Ground state hfs & g
for B in B_list:
    sh = SpinH(['bgs', 'ias', 'iqi', 'bi'], B = B, S = 1/2, I = 7/2)
    sh.add_term('bgs', gg)
    sh.add_term('ias', Ag)
    sh.add_term('iqi', Qg)
    sh.add_term('bi', 0.1618)

    w, v = LA.eig(sh.get_H())
    E = w.real
    E = np.sort(E - min(E))

    E = E/29.9702547e3
    # Ground state energy differences, w.r.t. first level.
    exdata_list += [cfl.ExData(np.array([[1,i+2, e] for i,e in enumerate(E[1:])]), 'D')]
    h = cfl.Hamiltonian(thfs_list)
    coeff['MX'] = B[0]
    coeff['MY'] = B[1]
    coeff['MZ'] = B[2]
    h.set_coeff(coeff)
    h_list += [h]
    weights_list += [g_hfs_w]

# Excited state g
for B in B_list:
    sh = SpinH(['bgs'], B = B, S = 1/2)
    sh.add_term('bgs', ge)

    w, v = LA.eig(sh.get_H())
    E = w.real
    E = np.sort(E - min(E))
    E = E/29.9702547e3

    # Excited state energy differences. First 4I13/2 level is 17 (1 based
    # indexing, no hyperfine).
    exdata_list += [cfl.ExData(np.array([[17, 18, E[1]]]), 'D')]

    h = cfl.Hamiltonian(t_list)
    coeff['MX'] = B[0]
    coeff['MY'] = B[1]
    coeff['MZ'] = B[2]
    h.set_coeff(coeff)
    h_list += [h]
    weights_list += [ge_w]





# Optimization bounds and stepsize for the basinhopping algorithm.
bounds = bal_bounds(coeff, {
    'EAVG' :  100,
    'F2'   :  100,
    'F4'   :  200,
    'F6'   :  100,
    'ZETA' :  1,
    'C20'  :  105,
    'C21'  :  105 + 105*I,
    'C22'  :  105 + 105*I,
    'C40'  :  200,
    'C41'  :  105 + 105*I,
    'C42'  :  105 + 105*I,
    'C43'  :  105 + 105*I,
    'C44'  :  100 + 100*I,
    'C60'  :  105,
    'C61'  :  100 + 100*I,
    'C62'  :  100 + 100*I,
    'C63'  :  100 + 100*I,
    'C64'  :  100 + 100*I,
    'C65'  :  100 + 100*I,
    'C66'  :  100 + 100*I,
    'HYP'  :  0.0005,
    'EQHYP':  0.01,
    'ALPHA':  20,   
    'BETA' :  20,
    'GAMMA':  200,
    'T2'   :  20,
    'T3'   :  20,
    'T4'   :  20,
    'T6'   :  20,
    'T7'   :  20,
    'T8'   :  20,
    'MTOT' :  10,
    'PTOT' : 300,
    })

stepsize = {
    'EAVG' :  50,
    'F2'   :  50,
    'F4'   :  50,
    'F6'   :  50,
    'ZETA' :  0.1,
    'C20'  :  20,
    'C21'  :  20 + 20*I,
    'C22'  :  20 + 20*I,
    'C40'  :  20,
    'C41'  :  20 + 20*I,
    'C42'  :  20 + 20*I,
    'C43'  :  20 + 20*I,
    'C44'  :  20 + 20*I,
    'C60'  :  20,
    'C61'  :  20 + 20*I,
    'C62'  :  20 + 20*I,
    'C63'  :  20 + 20*I,
    'C64'  :  20 + 20*I,
    'C65'  :  20 + 20*I,
    'C66'  :  20 + 20*I,
    'HYP'  :  0.0005,
    'EQHYP':  0.005,
    }
param = ['EAVG', 'ZETA','F2', 'F4', 'F6', 'C20', 'C21', 'C22', 'C40', 'C41', 'C42',
	'C43', 'C44', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'HYP','EQHYP'] 

#cfl_min = cfl.CFLMin('siman', niter=1e7, Tstart=1.0, Tmin=1.0, dry_run=False, stepsize=stepsize, maxtime=800000)

# Polishing fit.
cfl_min = cfl.CFLMin('nlopt_sbplx', xtol=1e-6, bounds=bounds, cov=False, dry_run=False, maxtime=86400)

res = cfl.mh_fit(param, h_list, weights_list, exdata_list, cfl_min)

#print("Number of Hamiltonians: {}".format(len(h_list)))
print(res['summary'])

with open("2018-08-28_eryso_site1_sbplx.txt", "w") as summary_file:
    summary_file.write(res['summary'])

#np.save('2018-08-26_eryso_site1_xaccept.npy', res['xaccept'])
#np.save('2018-08-26_eryso_site1_chi2accept.npy', res['chi2accept'])

#print(res['xaccept'])
#print(res['chi2accept'])
#print(res['retval'])
