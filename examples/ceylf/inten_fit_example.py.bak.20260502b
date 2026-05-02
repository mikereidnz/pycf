#!/usr/bin/env python3
"""
Example: Fit Altp parameters to intensity data (Ce3+ C3 symmetry with B-field).

Demonstrates fitting of Altp (electric dipole coupling) parameters to match
target intensity data by:
1. Computing intensities with known Altp parameters (target data)
2. Computing intensities with perturbed initial guess
3. Fitting parameters to recover the target values
4. Comparing fitted vs target intensities

This validates the intensity calculation and Altp fitting infrastructure.
"""

from pathlib import Path
import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, fit_altp


def print_intensities(label, group_data):
    """Print intensity data for a set of groups."""
    print(f"\n{label}")
    print("-" * 70)
    print(f"{'Group':<8} {'Initial State':<20} {'Final State':<20} {'f value':<15}")
    print("-" * 70)
    for group_idx, group in enumerate(group_data, start=1):
        i_state = group.get('i_state', 'N/A')
        f_state = group.get('f_state', 'N/A')
        f_value = group.get('f', 0.0)
        print(f"{group_idx:<8} {i_state:<20} {f_state:<20} {f_value:.6e}")
    print("-" * 70)


def main():
    """Main Altp fitting example."""
    
    print("\n" + "=" * 70)
    print("ALTP PARAMETER FITTING EXAMPLE")
    print("Ce3+ in C3 Symmetry with Magnetic Field")
    print("=" * 70)
    
    # Load SLJM data
    test_inten_dir = Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
    MATEL_CF = test_inten_dir / "f1cf"
    MATEL_INT = test_inten_dir / "f1int"
    
    t_cf = ImportSLJM(str(MATEL_CF))
    t_int = ImportSLJM(str(MATEL_INT), sl_name=str(MATEL_CF))
    
    # Build Hamiltonian with B-field
    coeff = {
        "EAVG": 1402.6553816211135,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": (200 + 100j),
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 1e-10,
        "MY": 0,
        "MZ": 1,
    }
    
    mu_b = 4.669202e-5
    MX = mu_b * t_cf.MAGX
    MY = mu_b * t_cf.MAGY
    MZ = mu_b * t_cf.MAGZ
    MX.name, MY.name, MZ.name = "MX", "MY", "MZ"
    
    h = cfl.Hamiltonian([t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, 
                         t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ])
    h.set_coeff(coeff)
    w, z = h.diag()
    
    # ======================================================================
    # STEP 1: Known Altp parameters and their target intensities
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 1: TARGET INTENSITIES (Known Altp Parameters)")
    print("=" * 70)
    
    known_altp = [
        ["A210", 1e-10],
        ["A230", -1e-10],
        ["A233", 1e-10 + 2e-10j],
    ]
    
    print("\nKnown Altp parameters:")
    for name, value in known_altp:
        print(f"  {name}: {value}")
    
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    
    spectrum_config = {
        "hamiltonian": h,
        "name": "Target spectrum",
        "i_range": [1],
        "f_range": [7, 8, 9, 10, 11, 12, 13, 14],
        "intensity_tensors": intensity_tensors,
        "altp": known_altp,
        "group_tol": 1e-3,
        "nrefractive": 1.0,
        "md": True,
        "ed": True,
    }
    
    spec_target = Spectrum(**spectrum_config)
    spec_target.calculate_intensities(polarization='isotropic')
    
    target_intensities = {}
    print("\nTarget transition groups:")
    for group_idx, group in enumerate(spec_target.groups, start=1):
        f_value = group.get('f', 0.0)
        target_intensities[group_idx] = f_value
        i_label = group.get('i_label', 'N/A')
        f_label = group.get('f_label', 'N/A')
        print(f"  Group {group_idx}: {i_label} → {f_label}: f = {f_value:.6e}")
    
    # ======================================================================
    # STEP 2: Perturbed initial parameters (50% of known values)
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 2: INITIAL PARAMETERS (Perturbed, 50% of Known Values)")
    print("=" * 70)
    
    initial_altp = [
        ["A210", 5e-11],
        ["A230", -5e-11],
        ["A233", 5e-11 + 1e-10j],
    ]
    
    print("\nInitial Altp parameters:")
    for name, value in initial_altp:
        print(f"  {name}: {value}")
    
    spectrum_config["altp"] = initial_altp
    spec_initial = Spectrum(**spectrum_config)
    spec_initial.calculate_intensities(polarization='isotropic')
    
    print("\nInitial transition groups (with perturbed parameters):")
    for group_idx, group in enumerate(spec_initial.groups, start=1):
        f_value = group.get('f', 0.0)
        i_label = group.get('i_label', 'N/A')
        f_label = group.get('f_label', 'N/A')
        rel_error = abs(f_value - target_intensities[group_idx]) / target_intensities[group_idx] if target_intensities[group_idx] != 0 else 0
        print(f"  Group {group_idx}: {i_label} → {f_label}: f = {f_value:.6e} (error: {rel_error*100:.2f}%)")
    
    # ======================================================================
    # STEP 3: Fit Altp parameters
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FIT ALTP PARAMETERS")
    print("=" * 70)
    
    print("\nFitting parameters: A210, A230, A233")
    print("Using Nelder-Mead optimization...")
    
    result = fit_altp(
        ["A210", "A230", "A233"],
        h,
        spectrum_config,
        target_intensities,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    print(f"\nFit converged with χ² = {result['chi2']:.6e}")
    
    fitted_altp = [[name, result['fitted_params'][name]] for name in ["A210", "A230", "A233"]]
    
    print("\nFitted Altp parameters:")
    for name, value in fitted_altp:
        print(f"  {name}: {value}")
    
    # ======================================================================
    # STEP 4: Final verification
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 4: VERIFICATION (Fitted Parameters)")
    print("=" * 70)
    
    spectrum_config["altp"] = fitted_altp
    spec_final = Spectrum(**spectrum_config)
    spec_final.calculate_intensities(polarization='isotropic')
    
    print("\nFinal transition groups (with fitted parameters):")
    print(f"{'Group':<8} {'Target f':<18} {'Fitted f':<18} {'Rel. Error':<15}")
    print("-" * 60)
    
    max_error = 0.0
    for group_idx, group in enumerate(spec_final.groups, start=1):
        fitted_f = group.get('f', 0.0)
        target_f = target_intensities[group_idx]
        rel_error = abs(fitted_f - target_f) / target_f if target_f != 0 else 0
        max_error = max(max_error, rel_error)
        print(f"{group_idx:<8} {target_f:<18.6e} {fitted_f:<18.6e} {rel_error*100:<14.4f}%")
    
    print("-" * 60)
    print(f"\nMaximum relative error: {max_error*100:.4f}%")
    
    if max_error < 1e-4:
        print("\n✓ Fit converged successfully!")
    else:
        print("\n⚠ Fit did not converge well")


if __name__ == "__main__":
    main()
