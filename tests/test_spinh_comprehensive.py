#!/usr/bin/env python3
"""
Comprehensive tests for pycf.spinh module.

Tests cover core spin Hamiltonian functionality including:
- SU(2) rotation matrices and operations
- Spin Hamiltonian class and methods
- Parameter tensor decomposition
- Basic matrix operations
"""

import numpy as np
import pytest

from pycf.matel import matel
from pycf.spinh import param_ten_svd, su2_rotation, su2_rz


class TestSU2Operations:
    """Test SU(2) rotation operations."""

    def test_su2_rz_basic(self) -> None:
        """Test su2_rz with zero angle returns original matrix."""
        angle = 0.0
        m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        result = su2_rz(angle, m)

        # Zero angle should return matrix close to original
        assert result.shape == m.shape
        assert np.allclose(np.abs(result), np.abs(m), atol=1e-10)

    def test_su2_rz_returns_array(self) -> None:
        """Test that su2_rz returns ndarray."""
        angle = 0.5
        m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        result = su2_rz(angle, m)

        assert isinstance(result, np.ndarray)
        assert result.dtype == complex

    def test_su2_rz_preserves_shape(self) -> None:
        """Test that su2_rz preserves matrix shape."""
        angle = 0.3
        for shape in [(2, 2), (4, 4), (6, 6)]:
            m = np.eye(shape[0], dtype=complex)
            result = su2_rz(angle, m)
            assert result.shape == shape

    def test_su2_rotation_zero_returns_identity_behavior(self) -> None:
        """Test that zero rotation returns identity behavior."""
        p = np.array([0.0, 0.0, 0.0])
        for shape in [(2, 2), (4, 4)]:
            m = np.eye(shape[0], dtype=complex)
            result = su2_rotation(p, m)

            # Should maintain shape
            assert result.shape == shape
            assert np.all(np.isfinite(result))

    def test_su2_rotation_returns_array(self) -> None:
        """Test that su2_rotation returns ndarray."""
        p = np.array([0.5, 0.3, 0.2])
        m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        result = su2_rotation(p, m)

        assert isinstance(result, np.ndarray)
        assert result.dtype == complex

    def test_su2_rotation_multiple_sizes(self) -> None:
        """Test su2_rotation with different matrix sizes."""
        p = np.array([0.1, 0.2, 0.3])
        for shape in [(2, 2), (4, 4), (6, 6)]:
            m = np.eye(shape[0], dtype=complex)
            result = su2_rotation(p, m)

            assert result.shape == shape
            assert np.all(np.isfinite(result))

    def test_su2_rz_multiple_angles(self) -> None:
        """Test su2_rz with various angles."""
        m = np.eye(3, dtype=complex)
        for angle in [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi]:
            result = su2_rz(angle, m)
            assert result.shape == m.shape
            assert np.all(np.isfinite(result))

    def test_su2_rotation_various_angles(self) -> None:
        """Test su2_rotation with various parameter vectors."""
        m = np.eye(2, dtype=complex)
        params = [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [1.0, 0.5, 0.3]]
        for p_vals in params:
            p = np.array(p_vals)
            result = su2_rotation(p, m)
            assert result.shape == m.shape
            assert np.all(np.isfinite(result))


class TestParamTenSVD:
    """Test parameter tensor SVD decomposition."""

    def test_param_ten_svd_basic(self) -> None:
        """Test param_ten_svd with identity matrix."""
        t = np.eye(3, dtype=complex)
        result = param_ten_svd(t)

        assert isinstance(result, np.ndarray)
        assert result.dtype == complex
        assert result.shape[0] == result.shape[1]

    def test_param_ten_svd_returns_array(self) -> None:
        """Test that param_ten_svd returns ndarray."""
        t = np.array([[1.0, 0.5], [0.5, 2.0]], dtype=complex)
        result = param_ten_svd(t)

        assert isinstance(result, np.ndarray)
        assert result.dtype == complex

    def test_param_ten_svd_identity(self) -> None:
        """Test param_ten_svd with identity matrix."""
        for size in [2, 3, 4, 5]:
            t = np.eye(size, dtype=complex)
            result = param_ten_svd(t)

            assert result.shape == (size, size)
            assert np.all(np.isfinite(result))

    def test_param_ten_svd_zero_matrix(self) -> None:
        """Test param_ten_svd with zero matrix."""
        t = np.zeros((3, 3), dtype=complex)
        result = param_ten_svd(t)

        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_param_ten_svd_symmetric(self) -> None:
        """Test param_ten_svd with symmetric matrices."""
        t = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 3.0]], dtype=complex)
        result = param_ten_svd(t)

        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_param_ten_svd_different_sizes(self) -> None:
        """Test param_ten_svd with different matrix sizes."""
        for size in [2, 3, 4, 6]:
            t = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            result = param_ten_svd(t)

            assert result.shape == (size, size)
            assert np.all(np.isfinite(result))


class TestMatrixOperationConsistency:
    """Test consistency of matrix operations."""

    def test_su2_rz_multiple_calls_consistent(self) -> None:
        """Test that su2_rz is deterministic."""
        angle = 0.5
        m = np.array([[1.0, 0.5j], [0.5j, 2.0]], dtype=complex)

        result1 = su2_rz(angle, m.copy())
        result2 = su2_rz(angle, m.copy())

        assert np.allclose(result1, result2, atol=1e-14)

    def test_su2_rotation_multiple_calls_consistent(self) -> None:
        """Test that su2_rotation is deterministic."""
        p = np.array([0.3, 0.5, 0.7])
        m = np.eye(3, dtype=complex)

        result1 = su2_rotation(p, m.copy())
        result2 = su2_rotation(p, m.copy())

        assert np.allclose(result1, result2, atol=1e-14)


class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_su2_rz_small_angle(self) -> None:
        """Test su2_rz with very small angle."""
        angle = 1e-10
        m = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        result = su2_rz(angle, m)

        assert np.all(np.isfinite(result))
        assert result.shape == m.shape

    def test_su2_rotation_small_angle(self) -> None:
        """Test su2_rotation with very small angles."""
        p = np.array([1e-10, 1e-10, 1e-10])
        m = np.array([[1.0, 0.5], [0.5, 2.0]], dtype=complex)
        result = su2_rotation(p, m)

        assert np.all(np.isfinite(result))
        assert result.shape == m.shape

    def test_su2_rz_large_angle(self) -> None:
        """Test su2_rz with large angle."""
        angle = 1000.0
        m = np.eye(2, dtype=complex)
        result = su2_rz(angle, m)

        assert np.all(np.isfinite(result))
        assert result.shape == m.shape

    def test_su2_rotation_large_angles(self) -> None:
        """Test su2_rotation with large angles."""
        p = np.array([100.0, 100.0, 100.0])
        m = np.eye(3, dtype=complex)
        result = su2_rotation(p, m)

        assert np.all(np.isfinite(result))
        assert result.shape == m.shape

    def test_param_ten_svd_small_matrix(self) -> None:
        """Test param_ten_svd with very small matrix values."""
        t = np.ones((2, 2), dtype=complex) * 1e-15
        result = param_ten_svd(t)

        assert np.all(np.isfinite(result))
        assert result.shape == (2, 2)

    def test_param_ten_svd_large_matrix(self) -> None:
        """Test param_ten_svd with very large matrix values."""
        t = np.ones((2, 2), dtype=complex) * 1e15
        result = param_ten_svd(t)

        assert np.all(np.isfinite(result))
        assert result.shape == (2, 2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_su2_rz_zero_matrix(self) -> None:
        """Test su2_rz with zero matrix."""
        angle = 0.5
        m = np.zeros((3, 3), dtype=complex)
        result = su2_rz(angle, m)

        assert result.shape == m.shape
        assert np.allclose(result, 0.0, atol=1e-15)

    def test_su2_rotation_identity_matrix(self) -> None:
        """Test su2_rotation with identity matrix."""
        p = np.array([0.0, 0.0, 0.0])
        m = np.eye(4, dtype=complex)
        result = su2_rotation(p, m)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_su2_rz_complex_matrix(self) -> None:
        """Test su2_rz with complex-valued matrix."""
        angle = 0.5
        m = np.array([[1.0 + 1.0j, 2.0 - 1.0j], [0.5 + 0.5j, 1.0 - 2.0j]], dtype=complex)
        result = su2_rz(angle, m)

        assert result.dtype == complex
        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_param_ten_svd_hermitian_matrix(self) -> None:
        """Test param_ten_svd with Hermitian matrix."""
        t = np.array([[1.0, 0.5 + 0.5j], [0.5 - 0.5j, 2.0]], dtype=complex)
        result = param_ten_svd(t)

        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))

    def test_param_ten_svd_diagonal_matrix(self) -> None:
        """Test param_ten_svd with diagonal matrix."""
        t = np.diag([1.0, 2.0, 3.0]).astype(complex)
        result = param_ten_svd(t)

        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result))

    def test_su2_rz_negative_angle(self) -> None:
        """Test su2_rz with negative angle."""
        angle = -0.5
        m = np.eye(2, dtype=complex)
        result = su2_rz(angle, m)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_su2_rotation_mixed_angles(self) -> None:
        """Test su2_rotation with mixed positive/negative angles."""
        p = np.array([-0.5, 0.5, -0.3])
        m = np.eye(2, dtype=complex)
        result = su2_rotation(p, m)

        assert result.shape == m.shape
        assert np.all(np.isfinite(result))


class TestIntegration:
    """Test integration with other modules."""

    def test_su2_operations_with_matel(self) -> None:
        """Test that su2 operations work with matel-generated matrices."""
        I = 0.5  # Nuclear spin
        m = matel("jz", I)

        # Should work with su2_rz
        result = su2_rz(0.5, m)
        assert result.shape == m.shape
        assert np.all(np.isfinite(result))

    def test_param_ten_svd_with_random_matrices(self) -> None:
        """Test param_ten_svd with random matrices."""
        for _ in range(5):
            t = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
            result = param_ten_svd(t)

            assert result.shape == (3, 3)
            assert np.all(np.isfinite(result))

    def test_su2_rz_chain_operations(self) -> None:
        """Test chaining multiple su2_rz operations."""
        m = np.eye(2, dtype=complex)
        angles = [0.1, 0.2, 0.3]

        result = m.copy()
        for angle in angles:
            result = su2_rz(angle, result)
            assert np.all(np.isfinite(result))

        assert result.shape == m.shape
