#!/usr/bin/env python3
"""
Example demonstrating the folded magnetic quantum number (mu) feature for
crystal-field Hamiltonians.

The mu quantum number provides a robust way to identify degenerate and
near-degenerate states by folding the principal component's m value into
a fundamental domain determined by the smallest non-zero q in the C_kq
tensor expansion.

This is more robust than using (S, L, J, M) quantum numbers for identifying
states in realistic crystal-field problems with heavily mixed eigenstates.
"""

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM

# Load crystal-field tensor data
t = ImportSLJM("matel/f1cf")

# Create Hamiltonian
h = cfl.Hamiltonian([t.EAVG, t.C20, t.C22, t.C40, t.C42, t.C44, t.C60, t.C62, 
                     t.C64, t.ZETA], 
                    label="Eu3+ in LiYF4")

# Set crystal-field coefficients (example values)
# Note: q=0 tensors (C20, C40, C60) are set to zero to isolate the mu/n effect
coeff = {
    'EAVG': 64186,
    'C20': -551,
    'C22': 0,
    'C40': 1360,
    'C42': 0,
    'C44': 345,
    'C60': -850,
    'C62': 0,
    'C64': 1250,
    'ZETA': 1340,
}
h.set_coeff(coeff)

# Diagonalize
h.diag()

# Demonstrate the feature
print("=" * 80)
print("Crystal-Field Energy Levels for Eu3+ in LiYF4")
print("=" * 80)

print("\n1. DEFAULT: Energy summary without mu/n quantum numbers")
print("-" * 80)
print(h.gen_summary(nstates=1))

print("\n" + "=" * 80)
print("2. WITH mu/n QUANTUM NUMBERS: Set minimum_q on Hamiltonian")
print("-" * 80)
print("The mu quantum number wraps m values into a fundamental domain.")
print("For f-electrons with half-integer m values (stored as doubled integers)")
print("and minimum_q=4:")
print("  - Effective minimum_q = 4 × 2 = 8 (to account for doubled storage)")
print("  - mu folds m modulo 8, with folding back at the midpoint (4)")
print("  - States with same mu are grouped together by energy")
print()

# Set the folded magnetic quantum number parameters on the Hamiltonian
h.minimum_q = 4  # Smallest non-zero q in the expansion (C20, C22, ...)
h.half_integer_states = True  # m values are stored as doubled integers (half-integers)

# Print with mu/n columns
print(h.gen_summary(nstates=1))

print("\nLabel key: S = total spin, L = orbital angular momentum, J = total,")
print("           M = magnetic quantum number (stored value)")
print("\nNote: mu column shows the folded magnetic quantum number.")
print("      n column shows the ordinal index within each mu group.")
