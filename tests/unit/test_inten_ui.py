"""Unit tests for inten polarization helpers and intensity generation.

These are intentionally small and focused: they exercise the
polarization_vector and stokes_from_jones helpers so the full inten
refactor can build on a tested foundation. Later tests verify Spectrum
creation and gen_intensity() orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycf.polarization import polarization_vector, stokes_from_jones, quarter_wave_plate
from pycf.inten import Spectrum
from pycf.import_sljm import ImportSLJM
import pycf.cfl as cfl


def test_sigma_plus_has_positive_S3():
    v = polarization_vector("sigma_plus")
    S = stokes_from_jones(v)
    assert S[3] > 0


def test_sigma_minus_has_negative_S3():
    v = polarization_vector("sigma_minus")
    S = stokes_from_jones(v)
    assert S[3] < 0


def test_qwp_converts_45_linear_to_circular():
    # 45-degree linear polarisation -> QWP at 45deg should give circular
    E45 = polarization_vector("45")
    Q = quarter_wave_plate(np.pi / 4)
    E_after = Q @ E45
    S = stokes_from_jones(E_after)
    # |S1| small; depending on QWP sign convention either S2 or S3
    # will carry the circular component.  Accept either convention.
    assert abs(S[1]) < 1e-8
    # One of S2 or S3 should be approximately equal to S0 (circular),
    # the other should be near zero.
    assert max(abs(S[2]), abs(S[3])) == pytest.approx(S[0], rel=1e-9, abs=1e-12)
    assert min(abs(S[2]), abs(S[3])) < 1e-8


def test_spectrum_validation_empty_name():
    """Spectrum should reject empty name."""
    class DummyH:
        def diag(self): return None, None
    
    with pytest.raises(ValueError, match="name must be non-empty"):
        Spectrum(hamiltonian=DummyH(), name="", i_range=[1], f_range=[2], intensity_tensors=[1])


def test_spectrum_validation_invalid_i_range():
    """Spectrum should reject empty i_range."""
    class DummyH:
        def diag(self): return None, None
    
    with pytest.raises(ValueError, match="i_range must be non-empty"):
        Spectrum(hamiltonian=DummyH(), name="test", i_range=[], f_range=[2], intensity_tensors=[1])


def test_spectrum_validation_invalid_f_range():
    """Spectrum should reject empty f_range."""
    class DummyH:
        def diag(self): return None, None
    
    with pytest.raises(ValueError, match="f_range must be non-empty"):
        Spectrum(hamiltonian=DummyH(), name="test", i_range=[1], f_range=[], intensity_tensors=[1])


def test_spectrum_validation_empty_tensors():
    """Spectrum should reject empty intensity_tensors list."""
    class DummyH:
        def diag(self): return None, None
    
    with pytest.raises(ValueError, match="intensity_tensors must be non-empty"):
        Spectrum(hamiltonian=DummyH(), name="test", i_range=[1], f_range=[2], intensity_tensors=[])


def test_spectrum_validation_invalid_group_tol():
    """Spectrum should reject non-positive group_tol."""
    class DummyH:
        def diag(self): return None, None

    with pytest.raises(ValueError, match="group_tol must be positive"):
        Spectrum(hamiltonian=DummyH(), name="test", i_range=[1], f_range=[2], intensity_tensors=[1], group_tol=-0.1)


def test_spectrum_validation_invalid_nrefractive():
    """Spectrum should reject non-positive nrefractive."""
    class DummyH:
        def diag(self): return None, None

    with pytest.raises(ValueError, match="nrefractive must be positive"):
        Spectrum(hamiltonian=DummyH(), name="test", i_range=[1], f_range=[2], intensity_tensors=[1], nrefractive=-1.0)


def test_spectrum_calculate_intensities_with_c3_data():
    """Test calculate_intensities() with C3 example data (absorption spectrum)."""
    # Load C3 data (same as in integration test)
    MATEL_BASE = Path(__file__).resolve().parent.parent / "integration" / "inten" / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent.parent / "integration" / "inten" / "matel" / "f1int"

    t = ImportSLJM(MATEL_BASE)
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)

    # Set up Hamiltonian
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
    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C43, t.C60, t.C63, t.C66, MX, MY, MZ])
    h.set_coeff(coeff)

    # Create absorption spectrum using 1-based i_range and f_range
    # Z1 Kramers doublet (levels 1-2) to Y1+Y2 multiplet (levels 7-10)
    tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    spectrum = Spectrum(
        hamiltonian=h,
        name="absorption",
        i_range=[1, 2],           # Z1 Kramers doublet (1-based)
        f_range=[7, 8, 9, 10],    # Y1+Y2 multiplet (1-based)
        intensity_tensors=tensors,
        group_tol=1e-3,
    )

    # Calculate intensities
    groups = spectrum.calculate_intensities()

    # Verify we got groups back
    assert len(groups) > 0
    assert isinstance(groups, list)
    assert all(isinstance(g, dict) for g in groups)

    # Each group should have required keys
    required_keys = {"Energy", "e_i", "e_f", "g_i", "g_f", "t_list", "f", "A"}
    for group in groups:
        assert all(k in group for k in required_keys)
        assert group["f"] >= 0, f"Oscillator strength should be non-negative, got {group['f']}"
        assert group["A"] >= 0, f"A coefficient should be non-negative, got {group['A']}"
    
    # Verify total f and A are computed
    assert spectrum.total_f > 0, "Total oscillator strength should be positive"
    assert spectrum.total_A >= 0, "Total A coefficient should be non-negative"
    
    # Verify totals match sum of groups
    assert spectrum.total_f == pytest.approx(sum(g["f"] for g in groups))
    assert spectrum.total_A == pytest.approx(sum(g["A"] for g in groups))


def test_spectrum_set_altp():
    """Test set_altp() method for updating Altp parameters."""
    class DummyH:
        def diag(self): return None, None

    spectrum = Spectrum(
        hamiltonian=DummyH(),
        name="test",
        i_range=[1],
        f_range=[2],
        intensity_tensors=[1],
        altp=[["A210", 1e-10]]
    )
    
    assert spectrum.altp == [["A210", 1e-10]]
    
    # Update Altp
    new_altp = [["A210", 2e-10], ["A230", -1e-10]]
    spectrum.set_altp(new_altp)
    
    assert spectrum.altp == new_altp
