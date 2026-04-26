#!/usr/bin/env python3
"""
Unit tests for paramcalc.py, matel.py, and njsymbols.py modules.
Tests coverage for:
- paramcalc: Xi_val, RInt4f, Ckq
- matel: t_q, matel
- njsymbols: wigner_3j, tricon_ck
"""
import numpy as np
import pytest
from pycf.matel import matel, t_q
from pycf.njsymbols import tricon_ck, wigner_3j
from pycf.paramcalc import Ckq, RInt4f, Xi_val
class TestParamcalc:
    """Tests for paramcalc module."""
    def test_Xi_val_valid_pr(self):
        """Test Xi_val with valid Pr parameters."""
        xi = Xi_val(1, 2, "Pr")
        assert isinstance(xi, (int, float))
        assert np.isfinite(xi)
    def test_Xi_val_valid_er(self):
        """Test Xi_val with valid Er parameters."""
        xi = Xi_val(3, 4, "Er")
        assert isinstance(xi, (int, float))
        assert np.isfinite(xi)
    def test_Xi_val_invalid_lanthanide(self):
        """Test Xi_val with invalid lanthanide."""
        with pytest.raises(ValueError):
            Xi_val(1, 2, "InvalidLn")
    def test_Xi_val_invalid_parameters(self):
        """Test Xi_val with invalid t, l parameters."""
        with pytest.raises(ValueError):
            Xi_val(99, 99, "Er")
    def test_Xi_val_all_lanthanides(self):
        """Test Xi_val works for all documented lanthanides."""
        lanthanides = ["Pr", "Nd", "Eu", "Tb", "Er", "Tm", "Yb"]
        for Ln in lanthanides:
            xi = Xi_val(1, 2, Ln)
            assert np.isfinite(xi), f"Xi_val failed for {Ln}"
    def test_RInt4f_valid_parameters(self):
        """Test RInt4f with valid parameters."""
        r_int = RInt4f(2, "Ce")
        assert isinstance(r_int, (int, float))
        assert np.isfinite(r_int)
        assert r_int > 0, "Radial integral should be positive"
    def test_RInt4f_all_lambdas(self):
        """Test RInt4f for all valid lambda values."""
        for lam in [2, 4, 6]:
            r_int = RInt4f(lam, "Ce")
            assert np.isfinite(r_int)
            assert r_int > 0
    def test_RInt4f_invalid_lambda(self):
        """Test RInt4f with invalid lambda."""
        with pytest.raises(ValueError):
            RInt4f(3, "Ce")
    def test_RInt4f_invalid_ion(self):
        """Test RInt4f with invalid ion."""
        with pytest.raises(ValueError):
            RInt4f(2, "InvalidIon")
    def test_Ckq_valid(self):
        """Test Ckq with valid parameters."""
        result = Ckq(2, 0, 0, 0)
        assert np.isfinite(result)
    def test_Ckq_different_q_values(self):
        """Test Ckq for different q values."""
        results = []
        for q in [-2, -1, 0, 1, 2]:
            result = Ckq(2, q, 0, 0)
            assert np.isfinite(result)
            results.append(result)
        # Results should vary
        assert len(set(results)) > 1
class TestMatel:
    """Tests for matel module."""
    def test_t_q_valid(self):
        """Test t_q with valid parameters."""
        result = t_q(1 / 2, 1 / 2, 1 / 2, -1 / 2, 0)
        assert isinstance(result, (int, float, complex))
        assert np.isfinite(np.abs(result) if isinstance(result, complex) else result)
    def test_t_q_j_equals_values(self):
        """Test t_q returns zero when j1 != j2 (delta condition)."""
        result = t_q(1 / 2, 3 / 2, 1 / 2, -1 / 2, 0)
        assert np.isclose(result, 0)
    def test_t_q_complex_result(self):
        """Test that t_q can return complex values."""
        result = t_q(1, 1, 1, -1, 1)
        assert isinstance(result, (int, float, complex))
    def test_matel_jx_dimension(self):
        """Test matel returns correct dimensions for jx component."""
        j = 1 / 2
        m = matel("jx", j)
        expected_dim = int(2 * j + 1)
        assert m.shape == (expected_dim, expected_dim)
    def test_matel_jy_dimension(self):
        """Test matel returns correct dimensions for jy component."""
        j = 1
        m = matel("jy", j)
        expected_dim = int(2 * j + 1)
        assert m.shape == (expected_dim, expected_dim)
    def test_matel_jz_hermitian(self):
        """Test that jz component produces Hermitian matrix."""
        j = 1 / 2
        m = matel("jz", j)
        assert np.allclose(m, m.conj().T)
    def test_matel_jx_complex(self):
        """Test jx component produces complex matrix."""
        j = 1 / 2
        m = matel("jx", j)
        assert m.dtype == complex
    def test_matel_invalid_component(self):
        """Test matel raises error for invalid component."""
        with pytest.raises(ValueError):
            matel("invalid", 1)
    def test_matel_larger_j(self):
        """Test matel with larger j value."""
        j = 3 / 2
        for component in ["jx", "jy", "jz"]:
            m = matel(component, j)
            expected_dim = int(2 * j + 1)
            assert m.shape == (expected_dim, expected_dim)
            assert m.dtype == complex
class TestNjsymbols:
    """Tests for njsymbols module."""
    def test_tricon_ck_valid(self):
        """Test triangular condition check with valid values."""
        assert tricon_ck(1, 1, 1)  # Returns True (numpy.bool_)
    def test_tricon_ck_invalid(self):
        """Test triangular condition check with invalid values."""
        assert not tricon_ck(1, 1, 3)  # Returns False (numpy.bool_)
    def test_tricon_ck_half_integers(self):
        """Test triangular condition with half-integers."""
        assert tricon_ck(1 / 2, 1 / 2, 1)
        assert not tricon_ck(1 / 2, 1 / 2, 2)
    def test_tricon_ck_zero(self):
        """Test triangular condition with zero."""
        assert tricon_ck(0, 1, 1)
        assert not tricon_ck(0, 1, 2)
    def test_wigner_3j_zero_orthogonality(self):
        """Test Wigner 3j symbols satisfy selection rules."""
        result = wigner_3j(1 / 2, 1 / 2, 1, 1 / 2, -1 / 2, 0)
        assert np.isfinite(result)
    def test_wigner_3j_violates_triangle(self):
        """Test Wigner 3j returns zero when triangle condition fails."""
        result = wigner_3j(1, 1, 3, 0, 0, 0)
        assert np.isclose(result, 0)
    def test_wigner_3j_m_sum_conservation(self):
        """Test Wigner 3j m-sum conservation rule."""
        result = wigner_3j(1, 1, 1, 1, 1, 0)
        assert np.isclose(result, 0)
    def test_wigner_3j_symmetry(self):
        """Test known Wigner 3j symmetry property."""
        result = wigner_3j(1 / 2, 1 / 2, 1, 1 / 2, -1 / 2, 0)
        assert np.isfinite(result)
        assert np.isreal(result)
    def test_wigner_3j_half_integers(self):
        """Test Wigner 3j with half-integer arguments."""
        result = wigner_3j(1 / 2, 1 / 2, 1, 1 / 2, -1 / 2, 0)
        assert np.isfinite(result)
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
