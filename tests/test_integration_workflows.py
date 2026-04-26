#!/usr/bin/env python
"""
Integration tests for complete pycf workflows.
Tests verify end-to-end scenarios:
- Hamiltonian diagonalization
- Physical parameter bounds
- Wigner symbols and matrix elements
"""

import numpy as np
import pytest

from pycf import cfl


class TestHamiltonianIntegration:
    """Integration tests for Hamiltonian workflow."""

    def test_hamiltonian_diagonalization_complete(self, ceylf_diagonalized):
        """Verify Hamiltonian diagonalization produces valid eigenvalues/eigenvectors."""
        E, V = ceylf_diagonalized
        # Eigenvalues should be sorted
        assert np.all(np.diff(E) >= -1e-10)
        # Eigenvectors should be orthonormal
        VtV = V.T.conj() @ V
        assert np.allclose(VtV, np.eye(len(E)), atol=1e-10)
        # Should have many eigenvalues for Ce:YLF
        assert len(E) >= 5


class TestParameterValidation:
    """Integration tests for parameter validation."""

    def test_crystal_field_parameter_bounds(self):
        """Verify CF parameter bounds are enforced."""
        from pycf.paramcalc import Ckq, RInt4f, Xi_val

        # Valid parameters work
        xi = Xi_val(1, 2, "Er")
        assert xi != 0
        rint = RInt4f(2, "Er")
        assert rint > 0
        # Invalid t should fail
        with pytest.raises(ValueError):
            Xi_val(2, 2, "Er")  # t must be in {1,3,5,7}
        # Invalid l should fail
        with pytest.raises(ValueError):
            RInt4f(3, "Er")  # l must be in {2,4,6}

    def test_spherical_harmonic_bounds(self):
        """Verify spherical harmonic bounds are enforced."""
        from pycf.paramcalc import Ckq

        # Valid parameters
        c_valid = Ckq(2, 1, 0.5, 1.0)
        assert isinstance(c_valid, (complex, np.complexfloating))
        # Invalid k
        with pytest.raises(ValueError):
            Ckq(-1, 0, 0.5, 1.0)
        # Invalid q
        with pytest.raises(ValueError):
            Ckq(2, 3, 0.5, 1.0)


class TestIntensityBounds:
    """Integration tests for intensity calculation bounds."""

    def test_linewidth_lorentzian_bounds(self):
        """Verify Lorentzian bounds are enforced."""
        from pycf.inten import lorentzian

        x = np.linspace(0, 100, 100)
        # Valid linewidth works
        y = lorentzian(x, 50, 1.0)
        assert np.all(y > 0)
        # Zero linewidth fails
        with pytest.raises(ValueError):
            lorentzian(x, 50, 0)
        # Negative linewidth fails
        with pytest.raises(ValueError):
            lorentzian(x, 50, -1)


class TestWignerSymbols:
    """Integration tests for Wigner symbol calculations."""

    def test_wigner_3j_calculation(self):
        """Verify Wigner 3j calculations work correctly."""
        from pycf.njsymbols import tricon_ck, wigner_3j

        # Triangular condition should be satisfied
        assert tricon_ck(2, 3, 4) == True
        assert tricon_ck(1, 1, 3) == False  # Invalid triangle
        # Wigner 3j should calculate
        w3j = wigner_3j(1, 1, 1, 0, 0, 0)
        assert isinstance(w3j, (float, int))

    def test_wigner_6j_9j_calculation(self):
        """Verify Wigner 6j and 9j calculations work."""
        from pycf.njsymbols import wigner_6j, wigner_9j

        # Should calculate without error
        w6j = wigner_6j(2, 2, 3, 2, 3, 2)
        assert isinstance(w6j, (float, int))
        # 9j should also calculate
        w9j = wigner_9j(2, 2, 3, 2, 2, 3, 3, 3, 1)
        assert isinstance(w9j, (float, int))


class TestMatrixElementCalculations:
    """Integration tests for matrix element calculations."""

    def test_angular_momentum_matrix_elements(self):
        """Verify angular momentum matrix element calculations."""
        from pycf.matel import matel

        # For j=1
        j = 1
        jx = matel("jx", j)
        # Should be a 3x3 matrix (2*j+1 states)
        assert jx.shape == (3, 3)
        # Should be Hermitian
        assert np.allclose(jx, jx.T.conj())

    def test_tensor_matrix_elements(self):
        """Verify tensor matrix element calculations."""
        from pycf.matel import t_q

        # Rank-1 tensor between j1=1 and j2=1
        for q in [-1, 0, 1]:
            for m1 in [-1, 0, 1]:
                for m2 in [-1, 0, 1]:
                    elem = t_q(1, 1, m1, m2, q)
                    # Should be complex
                    assert isinstance(elem, complex)


class TestConstantDefinitions:
    """Integration tests for physical constants."""

    def test_physical_constants_defined(self):
        """Verify physical constants are properly defined."""
        from pycf.constants import (
            BOHR_RADIUS,
            BOLTZMANN_CM_INVERSE,
            ELECTRON_MASS,
            ELEMENTARY_CHARGE,
            EPSILON_0,
            HBAR,
            SPEED_OF_LIGHT,
        )

        # All should be positive
        assert ELECTRON_MASS > 0
        assert ELEMENTARY_CHARGE > 0
        assert EPSILON_0 > 0
        assert HBAR > 0
        assert SPEED_OF_LIGHT > 0
        assert BOLTZMANN_CM_INVERSE > 0
        assert BOHR_RADIUS > 0

    def test_boltzmann_factor_calculation(self):
        """Verify Boltzmann factor uses correct constants."""
        from pycf.inten import boltzmann_factor

        # Valid calculations
        bf_100_300 = boltzmann_factor(100, 300)
        bf_200_300 = boltzmann_factor(200, 300)
        # Higher energy should give lower factor
        assert bf_200_300 < bf_100_300
        # Both should be in [0,1]
        assert 0 <= bf_100_300 <= 1
        assert 0 <= bf_200_300 <= 1
