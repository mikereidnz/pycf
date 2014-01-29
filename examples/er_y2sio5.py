#!/usr/bin/env python

from __future__ import division
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from pycf.spinh import *

# Define the g, A and Q parameter matrices.
g = np.array([[2.92, -3.08, -3.68], [-3.08, 8.19, 5.96], [-3.68, 5.96, 5.52]])
A = np.array([[69.35, -580.73, -248.83], [-580.73, 696.30, 682.49], [-248.83,
    682.49, 495.54]])
Q = np.array([[21.40, -8.18, -15.27], [-8.18, 3.79, 0.60], [-15.27, 0.60,
    -25.20]])

# Create the SpinHamiltonian object and add values for the dipole and quadrupole
# terms.
sh = SpinHamiltonian(['ias', 'iqi'], S = 1/2, I = 7/2)

sh.add_term('ias', A)
sh.add_term('iqi', Q)

# The get_H method returns the full spin Hamiltonian, then we use numpy to
# calculate the eigenvalues. 
w, v = LA.eig(sh.get_H())
E = w.real
E = np.sort(E - min(E))

print("Energy spectrum:\n{}".format(E))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hlines(E, [0], [1])
ax.set_ylabel('HFS (MHz)')
plt.show()

sh_inv = SpinHamiltonian(['ias', 'iqi'], S = 1/2, I = 7/2, inv = True)
sh_inv.add_H_term('ias', sh.terms['ias'])
sh_inv.add_H_term('iqi', sh.terms['iqi'])

A_c = np.reshape(sh_inv.inv_term('ias'), (3, 3))
Q_c = np.reshape(sh_inv.inv_term('iqi'), (3, 3))

print("Calculated A:\n{}".format(np.reshape(A_c, (3, 3))))
print("Original A:\n{}".format(A))
print("Calculated Q:\n{}".format(np.reshape(Q_c, (3, 3))))
print("Original Q:\n{}".format(Q))

# Zeeman spin Hamiltonian inversion.

# Create a list of spin Hamiltonian data for three orthogonal magnetic fields.
B_m = np.eye(3)
bgs = []
for i in range(3):
    sh = SpinHamiltonian(['bgs'], B = B_m[:, i], S = 1/2, I = 7/2)
    sh.add_term('bgs', g)
    bgs += [sh.terms['bgs']]

sh_inv = SpinHamiltonian(['bgs'], B = [B_m[:, i] for i in range(3)], S = 1/2, I
        = 7/2, inv = True)
sh_inv.add_H_term('bgs', bgs)

g_c = np.reshape(sh_inv.inv_term('bgs'), (3, 3))
print("Calculated g:\n{}".format(np.reshape(g_c, (3, 3))))
print("Original g:\n{}".format(g))
