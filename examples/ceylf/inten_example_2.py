#!/usr/bin/env python3
"""
Example: Compute and display intensity spectra with magnetic field (Ce3+ C3 symmetry).

Demonstrates the Spectrum class with a magnetic field applied along Z.
This example includes a magnetic field term (MZ with coefficient 1) to show
how external fields affect the spectrum. This setup is comparable to the
C1 magnetic field test case for verification against reference calculations.

This script template shows how to:
1. Load crystal-field and intensity tensors from SLJM output
2. Build and diagonalize a Hamiltonian with magnetic field terms
3. Create Spectrum objects with 1-based level ranges (Z1, Z2, ... convention)
4. Call calculate_intensities() to compute oscillator strengths and Einstein A coefficients
5. Print text and CSV summaries
6. Compare results against the zero-field case

NOTE: This example requires both crystal-field (*cf) and intensity (*int) SLJM data.
The ceylf/matel/ directory only contains CF data; to run this example with real data,
use the test data in tests/integration/inten/matel/ (f1cf/ and f1int/ subdirectories).

For a complete working example, see test_inten_c3.py in the test suite.
"""

from pathlib import Path

import numpy as np

import pycf
import pycf.cfl as cfl
from pycf.cfl_util import gen_e_summary
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, gen_inten_summary


def main():
    """Main intensity calculation example."""

    print("\nIntensity Spectrum Example (Ce3+ C3 Symmetry)")
    print("=" * 80)

    # Paths to SLJM data
    # NOTE: You need both CF (*cf) and intensity (*int) tensor sets
    script_dir = Path(__file__).resolve().parent

    # For demonstration, we'll use the test data location
    test_inten_dir = (
        Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
    )
    # In the test matel directory, f1cf.txt, f1cf.mi_, f1cf.st_ are CF tensors
    # and f1int.txt, f1int.mi_, f1int.st_ are intensity tensors
    # ImportSLJM takes the base name without suffix
    MATEL_CF = test_inten_dir / "f1cf"
    MATEL_INT = test_inten_dir / "f1int"

    # Check if base files exist (ImportSLJM expects .txt, .mi_, .st_ suffix)
    cf_files_exist = any(
        [(test_inten_dir / f"f1cf{ext}").exists() for ext in [".txt", ".mi_", ".st_"]]
    )
    int_files_exist = any(
        [(test_inten_dir / f"f1int{ext}").exists() for ext in [".txt", ".mi_", ".st_"]]
    )

    if not cf_files_exist:
        print(f"Error: CF tensor data not found in {test_inten_dir}")
        print(
            "Please ensure SLJM matrix element files (f1cf.txt, f1cf.mi_, f1cf.st_) are available."
        )
        return

    if not int_files_exist:
        print(f"Error: Intensity tensor data not found in {test_inten_dir}")
        print("Please ensure intensity SLJM files (f1int.txt, f1int.mi_, f1int.st_) are available.")
        return

    print(f"Loading SLJM data from: {test_inten_dir}")

    # Load crystal-field tensors (for Hamiltonian)
    t_cf = ImportSLJM(str(MATEL_CF))

    # Load intensity tensors (electric and magnetic dipole operators)
    t_int = ImportSLJM(str(MATEL_INT), sl_name=str(MATEL_CF))

    # Define crystal-field parameters (exact same as test_inten_c3.py)
    coeff = {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 1e-10,
        "MY": 0,
        "MZ": 1,
    }

    # Scale magnetic tensor by Bohr magneton (cm-1/T)
    mu_b = 0.466860
    MX = mu_b * t_cf.MAGX
    MY = mu_b * t_cf.MAGY
    MZ = mu_b * t_cf.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"

    # Build and diagonalize Hamiltonian
    h = cfl.Hamiltonian(
        [
            t_cf.EAVG,
            t_cf.ZETA,
            t_cf.C20,
            t_cf.C40,
            t_cf.C43,
            t_cf.C60,
            t_cf.C63,
            t_cf.C66,
            MX,
            MY,
            MZ,
        ]
    )
    h.set_coeff(coeff)

    # Fit EAVG to match test (fitting is a no-op in dry_run mode)
    ex = np.array([[2, 0], [4, 1], [6, 2]])
    weights = np.ones(len(ex))
    exdata = cfl.ExData(ex, "A", weights=weights)
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True)
    param = ["EAVG"]
    res = cfl.e_fit(param, h, exdata, cfl_min)
    fitcoeff = res["coeff"]
    for p in fitcoeff:
        coeff[p] = fitcoeff[p]
    h.set_coeff(coeff)

    print(f"\nCrystal-field parameters (from test_inten_c3.py):")
    for p, v in coeff.items():
        if v != 0:
            print(f"  {p:6s} = {v}")

    print(f"\nEnergy levels:")
    w, z = h.diag()
    print(gen_e_summary(w, z, h.tensors[0].states.labels, h.tensors[0].states.label_key))

    # Prepare list of intensity tensors
    # Must match test_inten_c3.py: M11, M10, U20, U21, U22
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]

    # Electric dipole coupling parameters (exact same as test_inten_c3.py)
    altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]

    print(f"\nAltp (electric dipole coupling) parameters:")
    for name, value in altp:
        print(f"  {name}: {value}")

    # =========================================================================
    # Define Spectrum 1: Ground state absorption (Z1 -> Y1+Y2) with MD+ED
    # i_range and f_range use 1-based indexing (Z1=1, Z2=2, ..., Y1=7, Y2=8, ...)
    # =========================================================================

    spec_abs = Spectrum(
        hamiltonian=h,
        name="Ground state absorption (Z1 -> Y1 + Y2) with B-field||Z",
        i_range=[1, 2],  # Z1 Kramers doublet (1-based)
        f_range=[7, 8, 9, 10],  # Y1+Y2 multiplet (1-based)
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        nrefractive=1.0,
        md=True,  # Magnetic dipole (default in test)
        ed=True,  # Electric dipole with Altp
    )

    # =========================================================================
    # Define Spectrum 2: Emission from Y1+Y2 to Z1 (MD+ED, same as absorption)
    # =========================================================================

    spec_em = Spectrum(
        hamiltonian=h,
        name="Emission from Y1 + Y2 -> Z1 with B-field||Z",
        i_range=[7, 8],  # Y1+Y2 (1-based)
        f_range=[1, 2, 3, 4, 5, 6],  # Z1 and all intermediate levels (1-based)
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        nrefractive=1.0,
        md=True,  # Magnetic dipole (default in test)
        ed=True,  # Electric dipole with Altp (same as absorption)
    )

    # =========================================================================
    # Generate intensity data for both spectra (new API: call calculate_intensities)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Computing intensity spectra...")
    print("=" * 80)

    spec_abs.calculate_intensities(polarization="isotropic")
    spec_em.calculate_intensities(polarization="isotropic")

    print(
        f"\nAbsorption: {len(spec_abs.groups)} transition groups, total f = {spec_abs.total_f:.6e}"
    )
    print(f"Emission:   {len(spec_em.groups)} transition groups, total A = {spec_em.total_A:.6e}")

    # Print absorption summary (brief format - compact tabular)
    print("\n" + "=" * 80)
    print("Absorption - Brief format (compact tabular):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_abs, h, format="brief"))

    # Print absorption summary (verbose format - BRIEF + individual transitions)
    print("\n" + "=" * 80)
    print("Absorption - Verbose format (BRIEF + individual transitions):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_abs, h, format="detailed"))

    # Print absorption summary (ultra format - VERBOSE + dipole moments)
    print("\n" + "=" * 80)
    print("Absorption - Ultra format (VERBOSE + dipole moments):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_abs, h, format="moments"))

    # Print emission summary (brief format - compact tabular)
    print("\n" + "=" * 80)
    print("Emission - Brief format (compact tabular):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_em, h, format="brief"))

    # Print emission summary (verbose format - BRIEF + individual transitions)
    print("\n" + "=" * 80)
    print("Emission - Verbose format (BRIEF + individual transitions):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_em, h, format="detailed"))

    # Print emission summary (ultra format - VERBOSE + dipole moments)
    print("\n" + "=" * 80)
    print("Emission - Ultra format (VERBOSE + dipole moments):")
    print("=" * 80)
    print("\n" + gen_inten_summary(spec_em, h, format="moments"))

    # =========================================================================
    # Export to CSV for spreadsheet import/analysis
    # =========================================================================
    abs_csv_path = script_dir / "inten_absorption.csv"
    em_csv_path = script_dir / "inten_emission.csv"

    with open(abs_csv_path, "w") as f:
        f.write(gen_inten_summary(spec_abs, h, format="csv"))

    with open(em_csv_path, "w") as f:
        f.write(gen_inten_summary(spec_em, h, format="csv"))

    print(f"\nCSV files written:")
    print(f"  {abs_csv_path}")
    print(f"  {em_csv_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    pycf.pycf_info()
    main()
