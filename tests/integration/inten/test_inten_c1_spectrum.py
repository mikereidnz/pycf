#!/usr/bin/env python3
"""
New-style intensity tests using the Spectrum class API.

This file demonstrates the modern Spectrum-based workflow for computing
intensities, which replaces the lower-level dipole_str/group_transitions API.
"""

from pathlib import Path
import pytest
import numpy as np
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, gen_inten_summary
from pycf.cfl_util import update_coeff


def test_inten_c1_spectrum_absorption() -> None:
    """Test absorption spectrum for Ce3+ in C1 symmetry using Spectrum class."""
    # Load crystal-field and intensity tensors
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent / "matel" / "f1int"
    
    t = ImportSLJM(MATEL_BASE)
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)
    
    # Set up Hamiltonian with C1 crystal-field parameters
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
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    h = cfl.Hamiltonian([
        t.EAVG, t.ZETA, t.C20, t.C40, t.C43,
        t.C60, t.C63, t.C66, MX, MY, MZ
    ])
    h.set_coeff(coeff)
    
    # Define absorption spectrum: ground state (Z1) to excited state (Y1+Y2)
    # Using 1-based indexing (matching user convention)
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    
    spectrum = Spectrum(
        hamiltonian=h,
        name="C1 absorption (Z1 -> Y1+Y2)",
        i_range=[1, 2],           # Z1 Kramers doublet (1-based)
        f_range=[7, 8, 9, 10],    # Y1+Y2 multiplet (1-based)
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    # Calculate intensities
    groups = spectrum.calculate_intensities(polarization='isotropic')
    
    # Verify we got groups back
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
    assert spectrum.total_f > 0, "Total oscillator strength should be positive"
    
    # Verify values match C3 test expectations (same parameters)
    pascal_f = [4.482614e-08, 4.148602e-08]
    for i, group in enumerate(groups):
        assert group["f"] == pytest.approx(pascal_f[i], rel=1e-6), \
            f"Group {i} f-value {group['f']} doesn't match Pascal {pascal_f[i]}"


def test_inten_c1_spectrum_update_altp() -> None:
    """Test updating Altp parameters without recreating Spectrum."""
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent / "matel" / "f1int"
    
    t = ImportSLJM(MATEL_BASE)
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)
    
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
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    h = cfl.Hamiltonian([
        t.EAVG, t.ZETA, t.C20, t.C40, t.C43,
        t.C60, t.C63, t.C66, MX, MY, MZ
    ])
    h.set_coeff(coeff)
    
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    initial_altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    
    spectrum = Spectrum(
        hamiltonian=h,
        name="Test Altp update",
        i_range=[1, 2],
        f_range=[7, 8, 9, 10],
        intensity_tensors=intensity_tensors,
        altp=initial_altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    # Calculate with initial Altp
    groups1 = spectrum.calculate_intensities()
    f1 = spectrum.total_f
    
    # Update Altp and recalculate
    new_altp = [["A210", 2e-10], ["A230", -2e-10], ["A233", 2e-10 + 4e-10j]]
    spectrum.set_altp(new_altp)
    groups2 = spectrum.calculate_intensities()
    f2 = spectrum.total_f
    
    # With doubled Altp parameters, intensities should scale
    # (approximately 4x for electric dipole since it's squared in intensity formula)
    assert f2 != f1, "Altp update should change intensities"
    assert spectrum.altp == new_altp, "set_altp() should update altp field"


def test_inten_c1_spectrum_summary() -> None:
    """Test gen_inten_summary with new Spectrum class."""
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent / "matel" / "f1int"
    
    t = ImportSLJM(MATEL_BASE)
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)
    
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
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    h = cfl.Hamiltonian([
        t.EAVG, t.ZETA, t.C20, t.C40, t.C43,
        t.C60, t.C63, t.C66, MX, MY, MZ
    ])
    h.set_coeff(coeff)
    
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    
    spectrum = Spectrum(
        hamiltonian=h,
        name="C1 absorption",
        i_range=[1, 2],
        f_range=[7, 8, 9, 10],
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    # Calculate intensities
    spectrum.calculate_intensities()
    
    # Generate summary (uses cached eigenvalues and principal components)
    summary_text = gen_inten_summary(spectrum, h, format='text')
    assert "C1 absorption" in summary_text
    assert "Total oscillator strength" in summary_text
    assert "f:" in summary_text
    
    # Generate CSV
    summary_csv = gen_inten_summary(spectrum, h, format='csv')
    assert "initial_level" in summary_csv
    assert "initial_label" in summary_csv


def test_inten_c1_spectrum_update_coeff_helper() -> None:
    """Test update_coeff helper function for post-fit coefficient updates."""
    # Example: after fitting, merge fit results into full coefficient dict
    base_coeff = {
        "EAVG": 1000,
        "ZETA": 600,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,
        "C60": 0,
        "C63": 0,
        "C66": 0,
    }
    
    # Fit result (only some parameters fitted)
    fit_result = {
        "EAVG": 1010,
        "C20": 480,
    }
    
    # Update coefficients using helper
    updated = update_coeff(base_coeff, fit_result)
    
    # Check updated values
    assert updated["EAVG"] == 1010, "EAVG should be updated"
    assert updated["C20"] == 480, "C20 should be updated"
    assert updated["ZETA"] == 600, "ZETA should be unchanged"
    assert updated["C40"] == 0, "C40 should be unchanged"
    
    # Original should be unchanged
    assert base_coeff["EAVG"] == 1000, "Original dict should not be modified"
    
    # With updated coefficients, can now set on Hamiltonian
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    t = ImportSLJM(MATEL_BASE)
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    
    # Add the magnetic tensor coefficients that Hamiltonian expects
    updated["MX"] = 0
    updated["MY"] = 0
    updated["MZ"] = 0
    
    h = cfl.Hamiltonian([
        t.EAVG, t.ZETA, t.C20, t.C40, t.C43,
        t.C60, t.C63, t.C66, MX, MY, MZ
    ])
    h.set_coeff(updated)  # Should work without error
    w, z = h.diag()
    assert len(w) == 14  # Ce3+ has 14 levels for f^1
