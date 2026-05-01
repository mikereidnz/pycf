#!/usr/bin/env python3
"""
New-style intensity tests using the Spectrum class API for C3 symmetry.

This file demonstrates the modern Spectrum-based workflow for computing
absorption and emission intensities with the Spectrum class.
"""

from pathlib import Path
import pytest
import numpy as np
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, gen_inten_summary


def test_inten_c3_spectrum_absorption() -> None:
    """Test absorption spectrum for Ce3+ in C3 symmetry using Spectrum class."""
    # Load crystal-field and intensity tensors
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent / "matel" / "f1int"
    
    t = ImportSLJM(MATEL_BASE)
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)
    
    # Set up Hamiltonian with C3 crystal-field parameters
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
    
    # Define absorption spectrum using Spectrum class
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    
    spectrum = Spectrum(
        hamiltonian=h,
        name="C3 absorption (Z1 -> Y1+Y2)",
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
    
    # Verify results
    assert len(groups) == 2, f"Expected 2 absorption groups, got {len(groups)}"
    assert spectrum.total_f > 0, "Total f should be positive for absorption"
    assert spectrum.total_A >= 0, "Total A should be non-negative"
    
    # Compare to Pascal calculation
    pascal_f = [4.482614e-08, 4.148602e-08]
    tolerance = 1e-6
    for i, group in enumerate(groups):
        assert group["f"] == pytest.approx(pascal_f[i], rel=tolerance), \
            f"Group {i} f={group['f']} doesn't match Pascal {pascal_f[i]}"


def test_inten_c3_spectrum_emission() -> None:
    """Test emission spectrum for Ce3+ in C3 symmetry using Spectrum class."""
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
    
    # Define emission spectrum: excited state (Y1+Y2) to lower levels
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    
    spectrum = Spectrum(
        hamiltonian=h,
        name="C3 emission (Y1+Y2 -> lower levels)",
        i_range=[7, 8],           # Y1+Y2 (1-based)
        f_range=[1, 2, 3, 4, 5, 6],  # Z1 and intermediate levels (1-based)
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    # Calculate intensities
    groups = spectrum.calculate_intensities(polarization='isotropic')
    
    # Verify results
    assert len(groups) == 3, f"Expected 3 emission groups, got {len(groups)}"
    assert spectrum.total_A > 0, "Total A should be positive for emission"
    assert spectrum.total_f >= 0, "Total f should be non-negative"
    
    # Compare to Pascal calculation (from test_inten_c3.py)
    pascal_A = [0.1407653, 0.1747824, 0.0038221]
    tolerance = 1e-5
    for i, group in enumerate(groups):
        assert group["A"] == pytest.approx(pascal_A[i], rel=tolerance), \
            f"Group {i} A={group['A']} doesn't match Pascal {pascal_A[i]}"


def test_inten_c3_spectrum_two_spectra() -> None:
    """Test creating and managing both absorption and emission spectra."""
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
    
    # Create both spectra sharing the same Hamiltonian
    abs_spectrum = Spectrum(
        hamiltonian=h,
        name="Absorption",
        i_range=[1, 2],
        f_range=[7, 8, 9, 10],
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    em_spectrum = Spectrum(
        hamiltonian=h,
        name="Emission",
        i_range=[7, 8],
        f_range=[1, 2, 3, 4, 5, 6],
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )
    
    # Calculate intensities for both
    abs_groups = abs_spectrum.calculate_intensities()
    em_groups = em_spectrum.calculate_intensities()
    
    assert len(abs_groups) == 2
    assert len(em_groups) == 3
    
    # Both share the same cached Hamiltonian eigenvectors
    assert abs_spectrum.hamiltonian is em_spectrum.hamiltonian
    assert abs_spectrum.hamiltonian is h
    
    # Generate summaries
    abs_text = gen_inten_summary(abs_spectrum, h, format='text')
    em_text = gen_inten_summary(em_spectrum, h, format='text')
    
    assert "Absorption" in abs_text
    assert "Emission" in em_text
    assert "Total oscillator strength" in abs_text
    assert "Total A coefficient" in em_text
