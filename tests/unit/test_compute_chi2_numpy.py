#!/usr/bin/env python3
"""Unit tests for compute_chi2_numpy() function.

Tests the vectorized chi² computation from eigenvalues and experimental data.
Covers:
- Basic chi² computation with known values
- Invalid level indices (negative)
- Empty experimental data
- Weighted vs unweighted residuals
- Edge cases (single point, all invalid, zero-weight data)
"""

from pathlib import Path

import numpy as np
import pytest

import pycf.cfl as cfl
from pycf.cfl_util import compute_chi2_numpy
from pycf.import_sljm import ImportSLJM

MATEL_BASE = Path(__file__).resolve().parent.parent / "integration" / "ceylf" / "matel" / "f1cf"


class MockEFit:
    """Mock EFit-like object for testing compute_chi2_numpy()."""
    
    def __init__(self, eigenvalues: np.ndarray, la: np.ndarray, energies: np.ndarray, weights: np.ndarray):
        """Create mock EFit with eigenvalues and experimental data.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Fitted eigenvalues (1D array)
        la : np.ndarray
            Level indices for each experimental point
        energies : np.ndarray
            Experimental energies
        weights : np.ndarray
            Weight factors (sigma or inverse variance)
        """
        self.h = type('obj', (object,), {'w': eigenvalues})()
        self.ex = type('obj', (object,), {
            'la': la,
            'e': energies,
            'w': weights
        })()


def test_compute_chi2_basic_computation():
    """Test basic chi² computation with known values."""
    # Simple case: 2 eigenvalues, 2 experimental points
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([0, 2])  # Indices to levels 0 and 2
    e_exp = np.array([0.0, 200.0])
    sigma = np.array([1.0, 1.0])  # Unit weights
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Expected: sum((fitted_e - e_exp)² / sigma²) = (0-0)²/1² + (200-200)²/1² = 0
    assert chi2 == pytest.approx(0.0, abs=1e-10)


def test_compute_chi2_with_residuals():
    """Test chi² computation with non-zero residuals."""
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([1, 3])  # Indices to levels 1 and 3
    e_exp = np.array([110.0, 290.0])  # Experimental values different from eigenvalues
    sigma = np.array([1.0, 1.0])  # Unit weights
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Expected: (100 - 110)²/1² + (300 - 290)²/1² = 100 + 100 = 200
    assert chi2 == pytest.approx(200.0, abs=1e-10)


def test_compute_chi2_weighted_residuals():
    """Test chi² with weighted residuals (matching C echisq formula)."""
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([1, 2])
    e_exp = np.array([110.0, 210.0])
    weights = np.array([2.0, 5.0])  # Weight factors (not 1/sigma²)
    
    mock_efit = MockEFit(evals, la, e_exp, weights)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Expected: w[0] * (e[1]-e_exp[0])² + w[1] * (e[2]-e_exp[1])²
    #         = 2.0 * (100-110)² + 5.0 * (200-210)²
    #         = 2.0 * 100 + 5.0 * 100 = 200 + 500 = 700
    assert chi2 == pytest.approx(700.0, abs=1e-10)


def test_compute_chi2_with_negative_indices():
    """Test that negative indices (invalid levels) are excluded."""
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([-1, 1, -2, 2])  # Mix of valid and invalid indices
    e_exp = np.array([0.0, 110.0, 50.0, 210.0])
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Only indices 1 and 2 are valid (non-negative)
    # (100 - 110)² / 1² + (200 - 210)² / 1² = 100 + 100 = 200
    assert chi2 == pytest.approx(200.0, abs=1e-10)


def test_compute_chi2_all_invalid_indices():
    """Test with all negative indices returns 0."""
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([-1, -2, -3])
    e_exp = np.array([10.0, 20.0, 30.0])
    sigma = np.array([1.0, 1.0, 1.0])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    assert chi2 == pytest.approx(0.0, abs=1e-10)


def test_compute_chi2_empty_data():
    """Test with empty experimental data."""
    evals = np.array([0.0, 100.0, 200.0])
    la = np.array([], dtype=int)
    e_exp = np.array([])
    sigma = np.array([])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    assert chi2 == pytest.approx(0.0, abs=1e-10)


def test_compute_chi2_single_point():
    """Test with single experimental point."""
    evals = np.array([0.0, 100.0, 200.0])
    la = np.array([1])
    e_exp = np.array([105.0])
    weights = np.array([2.0])
    
    mock_efit = MockEFit(evals, la, e_exp, weights)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # w * (e_calc - e_exp)² = 2.0 * (100 - 105)² = 2.0 * 25 = 50.0
    assert chi2 == pytest.approx(50.0, abs=1e-10)


def test_compute_chi2_large_eigenvalue_set():
    """Test with large eigenvalue set."""
    evals = np.linspace(0, 10000, 1000)  # 1000 eigenvalues
    la = np.array([0, 500, 999])
    e_exp = np.array([evals[0], evals[500] + 5.0, evals[999] - 3.0])
    sigma = np.array([1.0, 1.0, 1.0])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # (0 - 0)² + (e[500] - e[500] - 5)² + (e[999] - e[999] + 3)² = 0 + 25 + 9 = 34
    assert chi2 == pytest.approx(34.0, abs=1e-8)


def test_compute_chi2_with_real_hamiltonian():
    """Test with real Hamiltonian and ExData."""
    # Use Ce:YLF test data
    t = ImportSLJM(str(MATEL_BASE))
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
    h.minimum_q = 2
    h.half_integer_states = True
    
    w, z = h.diag()
    w = w - np.min(w)
    
    # Create EFit with simple absolute energy data
    ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8]])
    exdata = cfl.ExData(ex, "A", weights=np.ones(len(ex)))
    
    efit = cfl.EFit(["EAVG", "C20"], h, exdata)
    chi2 = compute_chi2_numpy(efit)
    
    # Should return a positive float
    assert isinstance(chi2, float)
    assert chi2 >= 0.0


def test_compute_chi2_matches_manual_calculation():
    """Verify chi² computation matches manual calculation using weighted formula."""
    evals = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    la = np.array([1, 3])
    e_exp = np.array([45.0, 160.0])
    weights = np.array([2.0, 4.0])
    
    mock_efit = MockEFit(evals, la, e_exp, weights)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Manual calculation using chi² = sum(w * residual²):
    # fitted_e = [50.0, 150.0]
    # residuals = [(50-45), (150-160)] = [5, -10]
    # chi² = 2.0 * 5² + 4.0 * (-10)² = 2.0 * 25 + 4.0 * 100 = 50 + 400 = 450
    assert chi2 == pytest.approx(450.0, abs=1e-10)


def test_compute_chi2_out_of_bounds_index():
    """Test with level index beyond eigenvalue array."""
    evals = np.array([0.0, 100.0, 200.0])
    la = np.array([0, 10])  # Index 10 is out of bounds
    e_exp = np.array([0.0, 1000.0])
    sigma = np.array([1.0, 1.0])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    
    # This should raise an IndexError when accessing evals[10]
    with pytest.raises(IndexError):
        compute_chi2_numpy(mock_efit)


def test_compute_chi2_mixed_valid_invalid():
    """Test with mix of valid indices, invalid (negative), and out of bounds."""
    evals = np.array([0.0, 100.0, 200.0, 300.0])
    la = np.array([-1, 0, -2, 2, -3])  # Only 0 and 2 are valid
    e_exp = np.array([10.0, 0.0, 30.0, 200.0, 50.0])
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    mock_efit = MockEFit(evals, la, e_exp, sigma)
    chi2 = compute_chi2_numpy(mock_efit)
    
    # Only valid indices are 0 and 2
    # (0 - 0)² / 1² + (200 - 200)² / 1² = 0
    assert chi2 == pytest.approx(0.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
