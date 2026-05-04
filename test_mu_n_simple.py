#!/usr/bin/env python3
"""Simplified test showing mu/n energy mapping fix"""

import numpy as np
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
import os

# Load Ce:YLF system
os.chdir('/home/users/mfr24/dev/pycf/examples/ceylf')
t = ImportSLJM('matel/f1cf')

# Create Hamiltonian 
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20])
h.minimum_q = 4
h.half_integer_states = True

# Set some coefficients
h.set_coeff({"EAVG": 100, "ZETA": -500, "C20": 300})
h.diag()

# Show eigenstate mapping
print("\n=== Eigenstate -> (mu, n) mapping ===")
from pycf.cfl_util import get_eigenstate_mu_n
for i in range(min(4, h.z.shape[1])):
    mu, n = get_eigenstate_mu_n(i, h.z, t.EAVG.states.labels, h.w, h.minimum_q, h.half_integer_states)
    print(f"Eigenstate {i}: (mu={mu}, n={n}) energy={h.w[i]:.2f}")

# Test: User provides data in mixed order
print("\n=== User provides (mu,n) in mixed order ===")

ex_amu = np.array([
    [1, 1, 100.0],   # First entry: (mu=1, n=1) with experiment=100
    [1, 2, 200.0],   # Second entry: (mu=1, n=2) with experiment=200
], dtype=np.float32)

print(f"User provides:")
print(f"  Row 0: (mu=1, n=1) with experiment energy 100.0")
print(f"  Row 1: (mu=1, n=2) with experiment energy 200.0")

exdata = cfl.ExData((ex_amu), ("AMu"), label_key="MuN")

print(f"\nExData created:")
print(f"  ex.e (energies):   {exdata.e[:2]}")
print(f"  ex.mu_n_abs (mu, n pairs):")
for i, (mu, n) in enumerate(exdata.mu_n_abs):
    print(f"    [{i}] mu={int(mu)}, n={int(n)}")

# Now do the fit
cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True)
param = ["EAVG"]

print(f"\nRunning fit with param={param}...")

try:
    res = cfl.e_fit(param, h, exdata, cfl_min, suppress_input=False)
    
    # Now check what was stored internally
    print(f"\nAfter EFit initialization:")
    print(f"  ex.la (eigenstate indices, 0-based): {res.ex.la[:2]}")
    print(f"  ex.e (energies after sorting):        {res.ex.e[:2]}")
    
    # Recreate the (mu, n) mapping to show what gets displayed
    print(f"\nDisplay will show:")
    for i in range(2):
        eigidx = int(res.ex.la[i])
        mu, n = get_eigenstate_mu_n(eigidx, h.z, t.EAVG.states.labels, h.w, h.minimum_q, h.half_integer_states)
        energy_theory = h.w[eigidx]
        energy_exp = res.ex.e[i]
        print(f"  Eigenstate {eigidx}: (mu={mu}, n={n}) Theory={energy_theory:.2f}, Exp={energy_exp:.1f}")
    
    print("\n✓ Fit completed successfully")
    
except Exception as e:
    import traceback
    print(f"\n✗ Fit failed: {e}")
    traceback.print_exc()
