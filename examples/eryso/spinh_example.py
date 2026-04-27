#!/usr/bin/env python
"""
Spin Hamiltonian construction and diagonalization example.

This example demonstrates:
1. Building spin Hamiltonian parameter matrices (g-tensor, hyperfine, quadrupole)
2. Using the SpinH auxiliary class for spin Hamiltonian calculations
3. Constructing full spin Hamiltonian matrix from terms
4. Diagonalizing to find eigenvalues and eigenvectors
5. Analyzing energy level splittings

The example uses Er^3+ spin S=1/2, nuclear I=7/2 with:
- g-tensor (Zeeman coupling)
- Hyperfine coupling (A tensor)
- Quadrupole coupling (Q tensor)

Data source: Guillot-Noël et al., PhysRevB.74.214409

Prerequisites:
- numpy, matplotlib, pycf.spinh module
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

from pycf.spinh import *

# This file shows some of the functionallity of the auxiliary python scipts for
# spin Hamiltonian calculations.

# g, A and Q parameter matrices, Guillot-Noël et al., PhysRevB.74.214409
g = np.array([[2.92, -3.08, -3.68], [-3.08, 8.19, 5.96], [-3.68, 5.96, 5.52]])
A = np.array([[69.35, -580.73, -248.83], [-580.73, 696.30, 682.49], [-248.83, 682.49, 495.54]])
Q = np.array([[21.40, -8.18, -15.27], [-8.18, 3.79, 0.60], [-15.27, 0.60, -25.20]])

# Create the SpinH object and add values for the dipole and quadrupole
# terms.
sh = SpinH(["ias", "iqi"], S=1 / 2, I=7 / 2)

sh.add_term("ias", A)
sh.add_term("iqi", Q)

# The get_H method returns the full spin Hamiltonian, then we use numpy to
# calculate the eigenvalues.
w, v = LA.eig(sh.get_H())
E = w.real
E = np.sort(E - min(E))

print("Energy spectrum:\n{}".format(E))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hlines(E, [0], [1])
ax.set_ylabel("HFS (MHz)")
plt.show()

sh_inv = SpinH(["ias", "iqi"], S=1 / 2, I=7 / 2, inv=True)
sh_inv.add_H_term("ias", sh.terms["ias"])
sh_inv.add_H_term("iqi", sh.terms["iqi"])

A_c = np.reshape(sh_inv.inv_term("ias"), (3, 3))
Q_c = np.reshape(sh_inv.inv_term("iqi"), (3, 3))

print("Calculated A:\n{}".format(np.reshape(A_c, (3, 3))))
print("Original A:\n{}".format(A))
print("Calculated Q:\n{}".format(np.reshape(Q_c, (3, 3))))
print("Original Q:\n{}".format(Q))

# Zeeman spin Hamiltonian inversion.

# Create a list of spin Hamiltonian data for three orthogonal magnetic fields.
B_m = np.eye(3)
bgs = []
for i in range(3):
    sh = SpinH(["bgs"], B=B_m[:, i], S=1 / 2, I=7 / 2)
    sh.add_term("bgs", g)
    bgs += [sh.terms["bgs"]]

sh_inv = SpinH(["bgs"], B=[B_m[:, i] for i in range(3)], S=1 / 2, I=7 / 2, inv=True)
sh_inv.add_H_term("bgs", bgs)

g_c = np.reshape(sh_inv.inv_term("bgs"), (3, 3))
print("Calculated g:\n{}".format(np.reshape(g_c, (3, 3))))
print("Original g:\n{}".format(g))
