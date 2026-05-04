#!/usr/bin/env python3
"""
Test example for mu/n-based experimental data (ExData).

This example demonstrates:
1. Loading and displaying mu/n quantum numbers from energy summaries
2. Creating ExData with AMu (absolute mu/n energy) format
3. Creating ExData with DMu (difference mu/n energy) format
4. Verifying that ExData correctly stores and retrieves (mu, n) data
5. Converting (mu, n) pairs to level indices using mu_n_to_level()

This is a stepping stone before integrating mu/n into actual fitting.
"""

from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.cfl_util import mu_n_to_level
from pycf.import_sljm import ImportSLJM

# Load Ce:YLF crystal field matrix elements
t = ImportSLJM(str(Path(__file__).parent / "matel" / "f1cf"))

# Build Hamiltonian with known Ce:YLF parameters
coeff = {
    "EAVG": 1035.1277,
    "ZETA": 625.6990,
    "C20": 297.8906,
    "C40": -1328.1522,
    "C44": -1282.4766,
    "C60": -191.5100,
    "C64": -1743.1424 + 692.8662j,
}
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64])
h.set_coeff(coeff)
(w, z) = h.diag()
w = w - np.min(w)

print("=" * 80)
print("Ce:YLF Crystal-Field Energy Levels with mu/n Quantum Numbers")
print("=" * 80)

# Set mu/n parameters for Ce (f-electron: J=5/2, so m stored as doubled integers)
h.minimum_q = 2  # Smallest non-zero q in Ce expansion (C20, C22, ...)
h.half_integer_states = False  # m values are stored as doubled integers

print("\nStep 1: Display energy levels with mu/n information")
print("-" * 80)
print("Parameters: minimum_q=2, half_integer_states=False")
print("(m values are doubled integers for f-electrons)")
print()
summary = h.gen_summary(nstates=1)
print(summary)

# Extract the energy levels for reference
print("\n" + "=" * 80)
print("Step 2: Create ExData with AMu (absolute mu/n energy) format")
print("-" * 80)
print("Format: [mu, n, energy]")
print()

# Create some example experimental data using mu/n format
# Based on the summary above, ALL states have mu=1, so we can only use (1, n) pairs
# Use actual calculated energies to verify the conversion
ex_amu = np.array([
    [1, 1, 0.0],        # mu=1, n=1: level 1, energy 0.0 cm⁻¹
    [1, 3, 212.323869], # mu=1, n=3: level 3, energy 212.3239 cm⁻¹
    [1, 5, 412.918602], # mu=1, n=5: level 5, energy 412.9186 cm⁻¹
])

print("Example AMu data:")
print(ex_amu)
print()

# Create ExData with AMu format
try:
    exdata_amu = cfl.ExData(ex_amu, "AMu", label_key="MuN")
    print("✓ ExData created successfully with AMu format")
    print(f"  - has_mu_n: {exdata_amu.has_mu_n}")
    print(f"  - mu_n_abs shape: {exdata_amu.mu_n_abs.shape}")
    print(f"  - mu_n_abs content:\n{exdata_amu.mu_n_abs}")
except Exception as e:
    print(f"✗ Error creating AMu ExData: {e}")

print()

# ============================================================================
print("=" * 80)
print("Step 3: Create ExData with DMu (difference mu/n energy) format")
print("-" * 80)
print("Format: [mu_i, n_i, mu_f, n_f, energy_diff]")
print()

# Create difference format data: transitions between states
# All states have mu=1, so transitions are between states with the same mu
ex_dmu = np.array([
    [1, 1, 1, 3, 212.323869],   # Transition from (mu=1,n=1) to (mu=1,n=3): 212.3239 cm⁻¹
    [1, 3, 1, 5, 200.594733],   # Transition from (mu=1,n=3) to (mu=1,n=5): 200.5947 cm⁻¹
])

print("Example DMu data:")
print(ex_dmu)
print()

try:
    exdata_dmu = cfl.ExData(ex_dmu, "DMu", label_key="MuN")
    print("✓ ExData created successfully with DMu format")
    print(f"  - has_mu_n: {exdata_dmu.has_mu_n}")
    print(f"  - mu_n_diff shape: {exdata_dmu.mu_n_diff.shape}")
    print(f"  - mu_n_diff content:\n{exdata_dmu.mu_n_diff}")
except Exception as e:
    print(f"✗ Error creating DMu ExData: {e}")

print()

# ============================================================================
print("=" * 80)
print("Step 4: Create ExData with mixed AMu and DMu data")
print("-" * 80)
print()

try:
    exdata_mixed = cfl.ExData(
        (ex_amu, ex_dmu),
        ("AMu", "DMu"),
        label_key="MuN"
    )
    print("✓ ExData created successfully with mixed AMu and DMu")
    print(f"  - has_mu_n: {exdata_mixed.has_mu_n}")
    print(f"  - mu_n_abs shape: {exdata_mixed.mu_n_abs.shape}")
    print(f"  - mu_n_diff shape: {exdata_mixed.mu_n_diff.shape}")
except Exception as e:
    print(f"✗ Error creating mixed ExData: {e}")

print()

# ============================================================================
print("=" * 80)
print("Step 5: Verify data storage and retrieval")
print("-" * 80)
print()

print("AMu ExData attributes:")
print(f"  exdata_amu.has_mu_n = {exdata_amu.has_mu_n}")
print(f"  exdata_amu.mu_n_abs[0] = {exdata_amu.mu_n_abs[0]}")  # First row
print()

print("DMu ExData attributes:")
print(f"  exdata_dmu.has_mu_n = {exdata_dmu.has_mu_n}")
print(f"  exdata_dmu.mu_n_diff[0] = {exdata_dmu.mu_n_diff[0]}")  # First row
print()

# ============================================================================
print("=" * 80)
print("Step 6: Convert (mu, n) pairs to level indices using mu_n_to_level()")
print("-" * 80)
print()

# Convert mu/n pairs to level indices
try:
    # For AMu data: extract the (mu, n) pairs from ExData
    mu_n_pairs_amu = exdata_amu.mu_n_abs
    
    # Convert to level indices (1-based)
    level_indices_amu = mu_n_to_level(
        h,
        mu_n_pairs_amu,
        h.minimum_q,
        h.half_integer_states
    )
    
    print("AMu conversion:")
    print(f"  Input (mu, n) pairs:\n{mu_n_pairs_amu}")
    print(f"  Output level indices: {level_indices_amu}")
    
    # Verify that the level indices match the energy values in our test data
    energies_from_levels = w[level_indices_amu - 1]  # Convert to 0-based
    print(f"  Energies from level indices: {energies_from_levels}")
    print(f"  Expected energies (from AMu): {ex_amu[:, 2]}")
    
    # Check if conversion is correct (within tolerance)
    tol = 1e-3
    if np.allclose(energies_from_levels, ex_amu[:, 2], atol=tol):
        print(f"  ✓ Conversion verified! Energies match within tolerance {tol}")
    else:
        print(f"  ⚠ Energies don't match exactly:")
        print(f"    Differences: {energies_from_levels - ex_amu[:, 2]}")

except Exception as e:
    import traceback
    print(f"✗ Error converting AMu to level indices: {e}")
    traceback.print_exc()

print()

try:
    # For DMu data: we need to convert pairs of (mu, n) indices
    mu_n_pairs_dmu_initial = exdata_dmu.mu_n_diff[:, :2]
    mu_n_pairs_dmu_final = exdata_dmu.mu_n_diff[:, 2:4]
    
    # Convert to level indices
    initial_levels = mu_n_to_level(
        h,
        mu_n_pairs_dmu_initial,
        h.minimum_q,
        h.half_integer_states
    )
    final_levels = mu_n_to_level(
        h,
        mu_n_pairs_dmu_final,
        h.minimum_q,
        h.half_integer_states
    )
    
    print("DMu conversion:")
    print(f"  Initial states (mu, n):\n{mu_n_pairs_dmu_initial}")
    print(f"  Initial level indices: {initial_levels}")
    print(f"  Final states (mu, n):\n{mu_n_pairs_dmu_final}")
    print(f"  Final level indices: {final_levels}")
    
    # Verify energy differences
    energy_diffs_computed = w[final_levels - 1] - w[initial_levels - 1]
    print(f"  Computed energy differences: {energy_diffs_computed}")
    print(f"  Expected energy differences (from DMu): {ex_dmu[:, 4]}")
    
    tol = 1e-3
    if np.allclose(energy_diffs_computed, ex_dmu[:, 4], atol=tol):
        print(f"  ✓ Conversion verified! Energy differences match within tolerance {tol}")
    else:
        print(f"  ⚠ Energy differences don't match exactly:")
        print(f"    Differences: {energy_diffs_computed - ex_dmu[:, 4]}")
        
except Exception as e:
    import traceback
    print(f"✗ Error converting DMu to level indices: {e}")
    traceback.print_exc()

print()

print("=" * 80)
print("✓ Test example complete. Ready to integrate mu_n_to_level() into fitting.")
print("=" * 80)
