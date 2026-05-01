#!/usr/bin/env python3
"""
Example: Compute and display intensity spectra (template).

Demonstrates the new Spectrum class and gen_intensity() API for computing
and summarizing electric- and magnetic-dipole intensity data.

This script template shows how to:
1. Load crystal-field and intensity tensors from SLJM output
2. Build and diagonalize a Hamiltonian
3. Define spectra: absorption (ground state) and emission (excited state)
4. Generate oscillator strengths and Einstein A coefficients
5. Print text and CSV summaries

NOTE: This example requires both crystal-field (*cf) and intensity (*int) SLJM data.
The ceylf/matel/ directory only contains CF data; to run this example with real data,
use the test data in tests/integration/inten/matel/ (f1cf/ and f1int/ subdirectories).

For a complete working example, see test_inten_c3.py in the test suite.
"""

from pathlib import Path
import numpy as np

import pycf
from pycf.inten import Spectrum, gen_intensity, gen_inten_summary
from pycf.import_sljm import ImportSLJM
import pycf.cfl as cfl


def main():
    """Main intensity calculation example."""
    
    print("\nIntensity Spectrum Example (Ce3+ C3 Symmetry)")
    print("=" * 80)
    
    # Paths to SLJM data
    # NOTE: You need both CF (*cf) and intensity (*int) tensor sets
    script_dir = Path(__file__).resolve().parent
    
    # For demonstration, we'll use the test data location
    test_inten_dir = Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
    # In the test matel directory, f1cf.txt, f1cf.mi_, f1cf.st_ are CF tensors
    # and f1int.txt, f1int.mi_, f1int.st_ are intensity tensors
    # ImportSLJM takes the base name without suffix
    MATEL_CF = test_inten_dir / "f1cf"
    MATEL_INT = test_inten_dir / "f1int"
    
    # Check if base files exist (ImportSLJM expects .txt, .mi_, .st_ suffix)
    cf_files_exist = any([
        (test_inten_dir / f"f1cf{ext}").exists() for ext in [".txt", ".mi_", ".st_"]
    ])
    int_files_exist = any([
        (test_inten_dir / f"f1int{ext}").exists() for ext in [".txt", ".mi_", ".st_"]
    ])
    
    if not cf_files_exist:
        print(f"Error: CF tensor data not found in {test_inten_dir}")
        print("Please ensure SLJM matrix element files (f1cf.txt, f1cf.mi_, f1cf.st_) are available.")
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

    # Define crystal-field parameters
    # Using exact same parameters as test_inten_c3.py for easy verification
    coeff = {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 0,
        "MY": 0,
        "MZ": 0,
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
    h = cfl.Hamiltonian([
        t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43,
        t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ
    ])
    h.set_coeff(coeff)

    print(f"\nCrystal-field parameters (from test_inten_c3.py):")
    for p, v in coeff.items():
        if v != 0:
            print(f"  {p:6s} = {v}")

    # Prepare list of intensity tensors
    # Typically includes electric dipole (U2q, U4q, ...) and magnetic dipole (M1q) tensors
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]

    # =========================================================================
    # Define Spectrum 1: Ground state absorption (Z1 -> Y1+Y2)
    # =========================================================================
    # lrange specifies which levels participate in the transitions:
    # - Initial levels: [0, 1] (Z1 Kramers doublet)
    # - Final levels: [6, 7, 8, 9] (Y1+Y2 multiplet in C3 symmetry)
    
    spec_abs = Spectrum(
        name="Ground state absorption (Z1 -> Y1 + Y2)",
        lrange=[[0, 1], [6, 7, 8, 9]],
        intensity_tensors=intensity_tensors,
        group_tol=1e-3,      # Tolerance for grouping transitions by level pair
        nrefractive=1.0,     # Refractive index (1.0 = vacuum)
        md=True,             # Include magnetic dipole
        ed=False,            # No electric dipole (altp=None)
    )

    # =========================================================================
    # Define Spectrum 2: Emission from Y1+Y2 to Z1
    # =========================================================================
    spec_em = Spectrum(
        name="Emission from Y1 + Y2 -> Z1",
        lrange=[[6, 7], [0, 1, 2, 3, 4, 5]],
        intensity_tensors=intensity_tensors,
        group_tol=1e-3,
        nrefractive=1.0,
    )

    # =========================================================================
    # Generate intensity data for both spectra
    # =========================================================================
    print("\n" + "=" * 80)
    print("Computing intensity spectra...")
    print("=" * 80)
    
    gen_intensity(h, spec_abs, polarization='isotropic')
    gen_intensity(h, spec_em, polarization='isotropic')

    # Print absorption summary (text format)
    print("\n" + gen_inten_summary(spec_abs, h, format='text'))

    # Print emission summary (text format)
    print("\n" + gen_inten_summary(spec_em, h, format='text'))

    # =========================================================================
    # Export to CSV for spreadsheet import/analysis
    # =========================================================================
    abs_csv_path = script_dir / "inten_absorption.csv"
    em_csv_path = script_dir / "inten_emission.csv"

    with open(abs_csv_path, "w") as f:
        f.write(gen_inten_summary(spec_abs, h, format='csv'))

    with open(em_csv_path, "w") as f:
        f.write(gen_inten_summary(spec_em, h, format='csv'))

    print(f"\nCSV files written:")
    print(f"  {abs_csv_path}")
    print(f"  {em_csv_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    pycf.pycf_info()
    main()
