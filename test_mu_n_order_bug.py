#!/usr/bin/env python3
"""Test case to expose mu/n ordering bug"""

import numpy as np
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
import os

# Load Ce:YLF system
os.chdir('/home/users/mfr24/dev/pycf/examples/ceylf')
t = ImportSLJM('matel/f1cf')

# Create Hamiltonian with just a few key tensors
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20])

# Try with minimum_q=4 to get multiple mu values  
h.minimum_q = 4
h.half_integer_states = True

# Set some coefficients
h.set_coeff({"EAVG": 100, "ZETA": -500, "C20": 300})
h.diag()

print("\n=== With minimum_q=4, eigenstate mu/n mapping ===")
print("Checking which mu groups exist in the Hamiltonian...")

from pycf.cfl_util import get_eigenstate_mu_n

mu_values_seen = set()
for i in range(min(h.z.shape[1], 10)):
    mu, n = get_eigenstate_mu_n(i, h.z, t.EAVG.states.labels, h.w, h.minimum_q, h.half_integer_states)
    mu_values_seen.add(mu)
    if i < 5:
        print(f"  Eigenstate {i}: mu={mu}, n={n}, energy={h.w[i]:.2f}")

print(f"\nMu values found: {sorted(mu_values_seen)}")

print("\n=== Testing with non-ascending mu/n order ===")

# Create test data with mu/n pairs PROVIDED IN NON-ASCENDING EIGENSTATE ORDER
# This should expose the sorting bug
ex_amu = np.array([
    [3, 1, 1000.0],   # First eigenstate in mu=3 group (eigenstate 2)
    [1, 1, 500.0],    # First eigenstate in mu=1 group (eigenstate 0)
], dtype=np.float32)

print(f"User provides (mu, n) pairs in order: (3,1) then (1,1)")
print(f"With energies: 1000.0, 500.0")
print(f"Expected eigenstate indices: 2, 0")
print(f"But after sorting by eigenstate index: 0, 2")
print(f"So energies should become: 500.0, 1000.0 (swapped!)")

exdata = cfl.ExData((ex_amu), ("AMu"), label_key="MuN")

cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True)
param = []

print("\nFitting with non-ascending mu/n order...")

try:
    res = cfl.e_fit(param, h, exdata, cfl_min, suppress_input=False)
    print("\nFitting completed")
except Exception as e:
    print(f"\nFitting failed: {e}")

print("\n=== Now test with ascending mu/n order ===")

ex_amu2 = np.array([
    [1, 1, 500.0],    # First eigenstate in mu=1 group (eigenstate 0)
    [3, 1, 1000.0],   # First eigenstate in mu=3 group (eigenstate 2)
], dtype=np.float32)

print(f"User provides (mu, n) pairs in order: (1,1) then (3,1)")
print(f"With energies: 500.0, 1000.0")
print(f"Expected eigenstate indices: 0, 2")
print(f"Already in ascending order!")

exdata2 = cfl.ExData((ex_amu2), ("AMu"), label_key="MuN")

print("\nFitting with ascending mu/n order...")

try:
    res2 = cfl.e_fit(param, h, exdata2, cfl_min, suppress_input=False)
    print("\nFitting completed")
except Exception as e:
    print(f"\nFitting failed: {e}")
