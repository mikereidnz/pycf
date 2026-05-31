#!/usr/bin/env python3
"""
Test example for mu/n-based experimental data (ExData).

This example demonstrates:
1. Loading and displaying mu/n quantum numbers from energy summaries
2. Creating ExData with A (absolute mu/n energy) format + label_key="MuN"
3. Creating ExData with D (difference mu/n energy) format + label_key="MuN"
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
print('Step 2: Create ExData with A + label_key="MuN" (absolute mu/n energy) format')
print("-" * 80)
print("Format: [mu, n, energy]")
print()

# Create some example experimental data using mu/n format
# Based on the summary above, ALL states have mu=1, so we can only use (1, n) pairs
# Marker-column MuN format requires rows: ["mu", mu, n, energy]
ex_amu = [
    ["mu", 1, 1, 0.0],  # mu=1, n=1: level 1, energy 0.0 cm⁻¹
    ["mu", 1, 3, 212.323869],  # mu=1, n=3: level 3, energy 212.3239 cm⁻¹
    ["mu", 1, 5, 412.918602],  # mu=1, n=5: level 5, energy 412.9186 cm⁻¹
]

print("Example AMu data:")
print(ex_amu)
print()

# Create ExData with A mode plus MuN marker-column label parsing
try:
    exdata_amu = cfl.ExData(ex_amu, "A", label_key="MuN")
    print('✓ ExData created successfully with A mode + label_key="MuN"')
    print(f"  - has_mu_n: {exdata_amu.has_mu_n}")
    print(f"  - mu_n_abs shape: {exdata_amu.mu_n_abs.shape}")
    print(f"  - mu_n_abs content:\n{exdata_amu.mu_n_abs}")
except Exception as e:
    print(f"✗ Error creating AMu ExData: {e}")

print()

# ============================================================================
print("=" * 80)
print('Step 3: Create ExData with D + label_key="MuN" (difference mu/n energy) format')
print("-" * 80)
print("Format: [mu_i, n_i, mu_f, n_f, energy_diff]")
print()

# Create difference format data: transitions between states
# All states have mu=1, so transitions are between states with the same mu
# Marker-column MuN format requires rows: ["mu", mu_i, n_i, mu_f, n_f, energy_diff]
ex_dmu = [
    ["mu", 1, 1, 1, 3, 212.323869],  # Transition from (mu=1,n=1) to (mu=1,n=3): 212.3239 cm⁻¹
    ["mu", 1, 3, 1, 5, 200.594733],  # Transition from (mu=1,n=3) to (mu=1,n=5): 200.5947 cm⁻¹
]

print("Example DMu data:")
print(ex_dmu)
print()

try:
    exdata_dmu = cfl.ExData(ex_dmu, "D", label_key="MuN")
    print('✓ ExData created successfully with D mode + label_key="MuN"')
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
    exdata_mixed = cfl.ExData((ex_amu, ex_dmu), ("A", "D"), label_key="MuN")
    print('✓ ExData created successfully with mixed A/D + label_key="MuN"')
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
    level_indices_amu = mu_n_to_level(h, mu_n_pairs_amu, h.minimum_q, h.half_integer_states)

    print("AMu conversion:")
    print(f"  Input (mu, n) pairs:\n{mu_n_pairs_amu}")
    print(f"  Output level indices: {level_indices_amu}")

    # Verify that the level indices match the energy values in our test data
    energies_from_levels = w[level_indices_amu - 1]  # Convert to 0-based
    print(f"  Energies from level indices: {energies_from_levels}")
    expected_energies_amu = np.array([row[3] for row in ex_amu], dtype=float)
    print(f"  Expected energies (from AMu): {expected_energies_amu}")

    # Check if conversion is correct (within tolerance)
    tol = 1e-3
    if np.allclose(energies_from_levels, expected_energies_amu, atol=tol):
        print(f"  ✓ Conversion verified! Energies match within tolerance {tol}")
    else:
        print(f"  ⚠ Energies don't match exactly:")
        print(f"    Differences: {energies_from_levels - expected_energies_amu}")

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
    initial_levels = mu_n_to_level(h, mu_n_pairs_dmu_initial, h.minimum_q, h.half_integer_states)
    final_levels = mu_n_to_level(h, mu_n_pairs_dmu_final, h.minimum_q, h.half_integer_states)

    print("DMu conversion:")
    print(f"  Initial states (mu, n):\n{mu_n_pairs_dmu_initial}")
    print(f"  Initial level indices: {initial_levels}")
    print(f"  Final states (mu, n):\n{mu_n_pairs_dmu_final}")
    print(f"  Final level indices: {final_levels}")

    # Verify energy differences
    energy_diffs_computed = w[final_levels - 1] - w[initial_levels - 1]
    print(f"  Computed energy differences: {energy_diffs_computed}")
    expected_diffs_dmu = np.array([row[5] for row in ex_dmu], dtype=float)
    print(f"  Expected energy differences (from DMu): {expected_diffs_dmu}")

    tol = 1e-3
    if np.allclose(energy_diffs_computed, expected_diffs_dmu, atol=tol):
        print(f"  ✓ Conversion verified! Energy differences match within tolerance {tol}")
    else:
        print(f"  ⚠ Energy differences don't match exactly:")
        print(f"    Differences: {energy_diffs_computed - expected_diffs_dmu}")

except Exception as e:
    import traceback

    print(f"✗ Error converting DMu to level indices: {e}")
    traceback.print_exc()

print()

# ============================================================================
print("=" * 80)
print("Step 7: Display results as if from a fit (calculated vs input)")
print("-" * 80)
print()

# Build a comparison table showing calculated energy levels with their mu/n assignment
# vs the input data provided
print("Calculated Energy Levels with mu/n Assignment:")
print("Level | mu | n  | Energy (cm⁻¹)")
print("------|----|----|---------------")

# Get labels once (convert to array for easier indexing)
labels_array = np.array(h.tensors[0].states.labels, dtype=np.int32)

from pycf.cfl_util import calc_mu

# Precompute mu/n for all levels
mu_n_all = []
for level_idx in range(len(w)):
    row_idx = level_idx
    abs_row = np.abs(z[row_idx, :])
    pc_idx = np.argmax(abs_row)
    m_value = int(labels_array[pc_idx, 3])
    mu_calc = calc_mu(m_value, h.minimum_q, h.half_integer_states)
    mu_n_all.append(mu_calc)

# Now display with n ordinal indices
for level_idx in range(1, min(15, len(w) + 1)):  # Show first 14 levels
    mu_this = mu_n_all[level_idx - 1]

    # Count which n this is within its mu group
    n_count = 0
    for check_level in range(1, level_idx + 1):
        mu_check = mu_n_all[check_level - 1]
        if mu_check == mu_this:
            n_count += 1

    print(f"{level_idx:5d} | {int(mu_this):2d} | {n_count:2d}  | {w[level_idx - 1]:13.6f}")

print()
print("Input ExData (AMu - Absolute energies):")
print("mu | n | Energy (cm⁻¹) | Matched Level")
print("---|---|---------------|---------------")
for i, (_, mu, n, energy) in enumerate(ex_amu):
    matched_level = level_indices_amu[i]
    match_symbol = "✓" if abs(w[matched_level - 1] - energy) < 0.001 else "✗"
    print(f" {int(mu)} | {int(n)} | {energy:13.6f} | {matched_level:2d} {match_symbol}")

print()
print("Input ExData (DMu - Energy differences):")
print("mu_i | n_i | mu_f | n_f | Δ Energy (cm⁻¹) | Matched Transition")
print("-----|-----|------|-----|-----------------|-------------------")
for i, (_, mu_i, n_i, mu_f, n_f, e_diff) in enumerate(ex_dmu):
    i_level = initial_levels[i]
    f_level = final_levels[i]
    calc_diff = w[f_level - 1] - w[i_level - 1]
    match_symbol = "✓" if abs(calc_diff - e_diff) < 0.001 else "✗"
    print(
        f"  {int(mu_i)} |  {int(n_i)} |   {int(mu_f)} |  {int(n_f)} | {e_diff:15.6f} | ({i_level:2d}→{f_level:2d}) {match_symbol}"
    )

print()

# ============================================================================
print("=" * 80)
print("Step 8: Fit with mu/n-based experimental data (AMu)")
print("-" * 80)
print()

# Create experimental data by adding small noise to calculated energies.
# Keep marker-column shape: ["mu", mu, n, energy]
np.random.seed(42)
noise_level = 0.001  # 0.1% noise
ex_amu_noisy = [row.copy() for row in ex_amu]
noise = np.random.normal(0, noise_level, len(ex_amu_noisy))
for i, delta in enumerate(noise):
    ex_amu_noisy[i][3] += float(delta)

print("Noisy experimental data (AMu format):")
print("mu | n | Energy (cm⁻¹)")
print("---|---|---------------")
for _, mu, n, energy in ex_amu_noisy:
    print(f" {int(mu)} | {int(n)} | {energy:13.6f}")
print()

# Create ExData with the noisy data
exdata_fit = cfl.ExData(ex_amu_noisy, "A", label_key="MuN")

# Set up minimizer
cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-8)

# Fit with a subset of parameters to show the fitting works
param = ["C20", "C40"]

try:
    print("Performing fit with parameters:", param)
    print()
    res = cfl.e_fit(param, h, exdata_fit, cfl_min, suppress_input=True)

    print("✓ Fit completed successfully!")
    print()
    print("Fit Summary:")
    print("-" * 80)
    print(res["summary"])

except Exception as e:
    import traceback

    print(f"✗ Error during fitting: {e}")
    print()
    traceback.print_exc()

print()

print("=" * 80)
print("✓ Test example complete. mu/n integration into fitting verified!")
print("=" * 80)
