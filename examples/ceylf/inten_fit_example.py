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
from pycf.inten import Spectrum, fit_altp, gen_inten_summary, inten_plot


def print_transitions_with_energy(label, spec):
    """Print transition data with energies."""
    print(f"\n{label}")
    print("-" * 100)
    print(f"{'Group':<8} {'Trans. Energy (cm⁻¹)':<22} {'Initial Level':<18} {'Final Level':<18} {'f value':<20}")
    print("-" * 100)
    for group_idx, group in enumerate(spec.groups, start=1):
        trans_energy = group.get('Energy', 0.0)
        e_i = group.get('e_i', 0.0)
        e_f = group.get('e_f', 0.0)
        f_value = group.get('f', 0.0)
        print(f"{group_idx:<8} {trans_energy:<22.4f} {e_i:<18.4f} {e_f:<18.4f} {f_value:<20.6e}")
    print("-" * 100)


def print_parameter_uncertainties(label, result):
    """Print fitted parameters with uncertainties."""
    print(f"\n{label}")
    print("-" * 80)
    print(f"{'Parameter':<15} {'Fitted Value':<30} {'Uncertainty (1σ)':<30}")
    print("-" * 80)
    for pname in ["A210", "A230", "A233"]:
        if pname in result['fitted_params']:
            value = result['fitted_params'][pname]
            unc = result['uncertainties'].get(pname)
            if unc is not None:
                if isinstance(unc, tuple):
                    # Complex parameter: (real_unc, imag_unc)
                    real_unc, imag_unc = unc
                    print(f"{pname:<15} {str(value):<30} ({real_unc:.3e}, {imag_unc:.3e})")
                else:
                    # Real parameter
                    print(f"{pname:<15} {str(value):<30} {unc:.3e}")
            else:
                print(f"{pname:<15} {str(value):<30} [unable to estimate]")
    print("-" * 80)


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
    
    # Build Hamiltonian with B-field (using spectroscopic Bohr magneton units)
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
    
    mu_b = 0.466860  # Bohr magneton in spectroscopic units (cm^-1/T)
    MX = mu_b * t_cf.MAGX
    MY = mu_b * t_cf.MAGY
    MZ = mu_b * t_cf.MAGZ
    MX.name, MY.name, MZ.name = "MX", "MY", "MZ"
    
    h = cfl.Hamiltonian([t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, 
                         t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ])
    h.set_coeff(coeff)
    w, z = h.diag()
    
    # Print Hamiltonian parameters
    print("\nHamiltonian parameters:")
    print("-" * 60)
    print("Crystal field coefficients:")
    for key in ["EAVG", "ZETA", "C20", "C40", "C43", "C60", "C63", "C66"]:
        val = coeff.get(key, 0)
        print(f"  {key:<10s}: {val:>20.6f}")
    print("\nMagnetic field (Zeeman):")
    print(f"  μ_B (Bohr magneton): {mu_b:.6e}")
    for key in ["MX", "MY", "MZ"]:
        val = coeff.get(key, 0)
        print(f"  {key:<10s} coefficient: {val:>20.6e} → field component: {val * mu_b:>20.6e}")
    print("-" * 60)
    
    # Print energy level structure
    print("\nEnergy levels:")
    print("-" * 40)
    for i in range(1, 15):
        print(f"  Level {i:2d}: {w[i-1]:10.4f} cm⁻¹")
    print("-" * 40)
    
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
        "group_tol": 1e-8,
        "nrefractive": 1.0,
        "md": True,
        "ed": True,
    }
    
    spec_target = Spectrum(**spectrum_config)
    spec_target.calculate_intensities(polarization='isotropic')
    
    target_intensities = {}
    print_transitions_with_energy("Target transition groups:", spec_target)
    
    for group_idx, group in enumerate(spec_target.groups, start=1):
        f_value = group.get('f', 0.0)
        target_intensities[group_idx] = f_value
    
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
    
    print_transitions_with_energy("Initial transition groups (with perturbed parameters):", spec_initial)
    
    # ======================================================================
    # STEP 3: Fit Altp parameters
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FIT ALTP PARAMETERS")
    print("=" * 70)
    
    print("\nFitting parameters: A210, A230, A233")
    print("Using Nelder-Mead optimization...")
    print(f"Number of observables (transition groups): {len(spec_target.groups)}")
    print(f"Number of parameters: 4")
    
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
    
    print_parameter_uncertainties("Fitted Altp parameters with uncertainties:", result)
    
    # ======================================================================
    # STEP 4: Final verification
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 4: VERIFICATION (Fitted Parameters)")
    print("=" * 70)
    
    spectrum_config["altp"] = fitted_altp
    spec_final = Spectrum(**spectrum_config)
    spec_final.calculate_intensities(polarization='isotropic')
    
    print_transitions_with_energy("Final transition groups (with fitted parameters):", spec_final)
    
    print("\nComparison of target vs fitted intensities:")
    print("-" * 100)
    print(f"{'Group':<8} {'Transition (cm⁻¹)':<20} {'Fitted f':<20} {'Target f':<20} {'Rel. Error':<15}")
    print("-" * 100)
    
    max_error = 0.0
    for group_idx, group in enumerate(spec_final.groups, start=1):
        fitted_f = group.get('f', 0.0)
        target_f = target_intensities[group_idx]
        trans_energy = group.get('Energy', 0.0)
        rel_error = abs(fitted_f - target_f) / target_f if target_f != 0 else 0
        max_error = max(max_error, rel_error)
        print(f"{group_idx:<8} {trans_energy:<20.4f} {fitted_f:<20.6e} {target_f:<20.6e} {rel_error*100:<14.4f}%")
    
    print("-" * 100)
    print(f"\nMaximum relative error: {max_error*100:.4f}%")
    
    if max_error < 1e-3:
        print("\n✓ Fit converged successfully!")
    else:
        print("\n⚠ Fit did not converge well")
    
    # ======================================================================
    # Create experimental data for STEP 5
    # ======================================================================
    np.random.seed(42)
    expt_data = []
    for group_idx in sorted(target_intensities.keys()):
        f_target = target_intensities[group_idx]
        # For BEFORE FIT, use target intensities as "experimental" values
        # (in real scenarios, these would be measured values)
        expt_data.append([group_idx, f_target])
    
    # --- BRIEF FORMAT BEFORE FIT (fitted parameters from STEP 4 vs experimental data) ---
    print("\n" + "=" * 160)
    print("BRIEF FORMAT: BEFORE FIT (fitted params from STEP 4 vs experimental data)")
    print("=" * 160)
    display_config = spectrum_config.copy()
    display_config["name"] = "Before Fit"
    display_config["altp"] = fitted_altp
    display_config["expt_data"] = expt_data  # Show experimental data comparison
    spec_before_fit = Spectrum(**display_config)
    spec_before_fit.calculate_intensities(polarization='isotropic')
    print(gen_inten_summary(spec_before_fit, h, format='brief'))
    
    # ======================================================================
    # STEP 5: Demonstrate experimental data integration
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EXPERIMENTAL DATA FITTING")
    print("=" * 70)
    
    # Generate experimental data with 5% noise for fitting
    np.random.seed(42)
    expt_data_noisy = []
    expt_target_intensities = {}
    for group_idx in sorted(target_intensities.keys()):
        f_target = target_intensities[group_idx]
        # Add 5% random noise to create synthetic experimental data
        f_expt = f_target * (1.0 + 0.05 * (np.random.random() - 0.5))
        expt_data_noisy.append([group_idx, f_target, f_expt])
        expt_target_intensities[group_idx] = f_expt
    
    print("\nExperimental data (with 5% noise):")
    print(expt_data_noisy)
    
    # --- Fit to experimental data ---
    print("\n" + "=" * 70)
    print("FITTING TO EXPERIMENTAL DATA")
    print("=" * 70)
    
    print(f"Fitting to {len(expt_target_intensities)} experimental data points...")
    result_expt = fit_altp(
        ["A210", "A230", "A233"],
        h,
        spectrum_config,
        expt_target_intensities,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    print(f"\nFit converged with χ² = {result_expt['chi2']:.6e}")
    fitted_altp_expt = [[name, result_expt['fitted_params'][name]] for name in ["A210", "A230", "A233"]]
    print("\nFitted Altp parameters from experimental data:")
    for name, value in fitted_altp_expt:
        print(f"  {name}: {value}")
    
    # --- BRIEF FORMAT AFTER FIT (new fitted parameters with experimental data) ---
    print("\n" + "=" * 160)
    print("BRIEF FORMAT: AFTER FIT (refitted params vs experimental data)")
    print("=" * 160)
    display_config["name"] = "After Fit"
    display_config["altp"] = fitted_altp_expt
    display_config["expt_data"] = expt_data  # Include experimental data for comparison
    spec_after_fit = Spectrum(**display_config)
    spec_after_fit.calculate_intensities(polarization='isotropic')
    print(gen_inten_summary(spec_after_fit, h, format='brief'))
    
    # ======================================================================
    # Plot the intensity spectrum
    # ======================================================================
    print("\n" + "=" * 70)
    print("PLOTTING INTENSITY SPECTRUM")
    print("=" * 70)
    
    fig, ax = inten_plot(spec_after_fit, fwhm=0.5, npoints=10000)
    plot_file = Path(__file__).parent / "inten_fit_example_plot.pdf"
    fig.savefig(plot_file, dpi=150, bbox_inches='tight', format='pdf')
    print(f"\n✓ Plot saved to: {plot_file}")
    
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception:
        pass  # Matplotlib display not available in headless environment


if __name__ == "__main__":
    main()
