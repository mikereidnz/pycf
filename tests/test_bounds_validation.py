#!/usr/bin/env python3
"""
Tests for physical bounds validation.
Ensures that critical functions properly validate input parameters
for physical plausibility (e.g., positive temperatures, positive linewidths).
"""
import numpy as np
import pytest

from pycf.inten import boltzmann_factor, inten, lorentzian
from pycf.paramcalc import Ckq, RInt4f, Xi_val


class TestLorentzianBounds:
    """Test Lorentzian line shape parameter validation."""
    def test_lorentzian_positive_fwhm(self):
        """Lorentzian should accept positive FWHM."""
        result = lorentzian(0.0, 0.0, 1.0)
        assert result > 0
    def test_lorentzian_negative_fwhm_raises(self):
        """Lorentzian should reject negative FWHM."""
        with pytest.raises(ValueError, match="fwhm must be positive"):
            lorentzian(0.0, 0.0, -1.0)
    def test_lorentzian_zero_fwhm_raises(self):
        """Lorentzian should reject zero FWHM."""
        with pytest.raises(ValueError, match="fwhm must be positive"):
            lorentzian(0.0, 0.0, 0.0)
class TestIntenBounds:
    """Test inten function parameter validation."""
    @pytest.fixture
    def simple_transition(self):
        """Simple single transition for testing."""
        return [
            {
                "e": 100.0,  # transition energy
                "ei": 0.0,  # initial state energy
                "pi": 1.0,  # pi polarization strength
                "sigma": 0.5,  # sigma polarization strength
            }
        ]
    def test_inten_positive_linewidth(self, simple_transition):
        """inten should accept positive linewidth."""
        result = inten(simple_transition, "pi", 1.0, 300.0)
        assert (
            len(result) == 4
        )  # Returns (line_energies, line_inten, curve_energies, curve_inten)
    def test_inten_negative_linewidth_raises(self, simple_transition):
        """inten should reject negative linewidth."""
        with pytest.raises(ValueError, match="linewidth must be positive"):
            inten(simple_transition, "pi", -1.0, 300.0)
    def test_inten_zero_linewidth_raises(self, simple_transition):
        """inten should reject zero linewidth."""
        with pytest.raises(ValueError, match="linewidth must be positive"):
            inten(simple_transition, "pi", 0.0, 300.0)
    def test_inten_negative_temperature_raises(self, simple_transition):
        """inten should reject negative temperature."""
        with pytest.raises(ValueError, match="Temperature T must be non-negative"):
            inten(simple_transition, "pi", 1.0, -10.0)
    def test_inten_zero_temperature_allowed(self, simple_transition):
        """inten should accept zero temperature (ground state only)."""
        result = inten(simple_transition, "pi", 1.0, 0.0)
        assert len(result) == 4
    def test_inten_zero_npoints_raises(self, simple_transition):
        """inten should reject npoints < 1."""
        with pytest.raises(ValueError, match="npoints must be >= 1"):
            inten(simple_transition, "pi", 1.0, 300.0, npoints=0)
    def test_inten_empty_transitions_raises(self):
        """inten should reject empty transition list."""
        with pytest.raises(ValueError, match="inten requires at least one transition"):
            inten([], "pi", 1.0, 300.0)
class TestXiValBounds:
    """Test Xi_val parameter validation."""
    def test_xi_val_valid_parameters(self):
        """Xi_val should accept valid parameters."""
        result = Xi_val(1, 2, "Pr")
        assert isinstance(result, (float, np.floating))
    def test_xi_val_invalid_t_raises(self):
        """Xi_val should reject invalid t parameter."""
        with pytest.raises(ValueError, match="t must be in"):
            Xi_val(2, 2, "Pr")  # t must be in {1, 3, 5, 7}
    def test_xi_val_invalid_l_raises(self):
        """Xi_val should reject invalid l parameter."""
        with pytest.raises(ValueError, match="l must be in"):
            Xi_val(1, 3, "Pr")  # l must be in {2, 4, 6}
    def test_xi_val_invalid_lanthanide_raises(self):
        """Xi_val should reject invalid lanthanide."""
        with pytest.raises(ValueError, match="Invalid lanthanide"):
            Xi_val(1, 2, "Ce")  # Ce not in Xi_val's list
class TestRInt4fBounds:
    """Test RInt4f parameter validation."""
    def test_rint4f_valid_parameters(self):
        """RInt4f should accept valid parameters."""
        result = RInt4f(2, "Ce")
        assert isinstance(result, (float, np.floating))
        assert result > 0
    def test_rint4f_invalid_l_raises(self):
        """RInt4f should reject invalid l parameter."""
        with pytest.raises(ValueError, match="l must be in"):
            RInt4f(3, "Ce")  # l must be in {2, 4, 6}
    def test_rint4f_invalid_lanthanide_raises(self):
        """RInt4f should reject invalid lanthanide."""
        with pytest.raises(ValueError, match="Invalid lanthanide"):
            RInt4f(2, "Gd")  # Gd not in RInt4f's list
class TestCkqBounds:
    """Test Ckq spherical harmonic parameter validation."""
    def test_ckq_valid_parameters(self):
        """Ckq should accept valid parameters."""
        result = Ckq(2, 0, np.pi / 4, 0.0)
        assert isinstance(result, (complex, np.complexfloating))
    def test_ckq_negative_k_raises(self):
        """Ckq should reject negative k."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            Ckq(-1, 0, np.pi / 4, 0.0)
    def test_ckq_q_exceeds_k_positive_raises(self):
        """Ckq should reject q > k."""
        with pytest.raises(ValueError, match="q must satisfy"):
            Ckq(2, 3, np.pi / 4, 0.0)
    def test_ckq_q_exceeds_k_negative_raises(self):
        """Ckq should reject q < -k."""
        with pytest.raises(ValueError, match="q must satisfy"):
            Ckq(2, -3, np.pi / 4, 0.0)
    def test_ckq_q_at_boundary(self):
        """Ckq should accept q at the boundary |q| = k."""
        result1 = Ckq(2, 2, np.pi / 4, 0.0)
        result2 = Ckq(2, -2, np.pi / 4, 0.0)
        assert isinstance(result1, (complex, np.complexfloating))
        assert isinstance(result2, (complex, np.complexfloating))
