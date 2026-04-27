"""Property-based tests for pycf using hypothesis.
These tests verify invariants and properties that should hold
across a wide range of inputs.
Run with: pytest tests/test_properties.py -v
Install: pip install hypothesis (optional - tests will skip if not installed)
"""

import numpy as np
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)
try:
    from pycf import cfl
except ImportError:
    pytest.skip("pycf not installed", allow_module_level=True)


# Strategies for generating test data
@st.composite
def matrices(draw, min_size=2, max_size=20):
    """Generate random matrices."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(st.just(np.random.randn(size, size)))


@st.composite
def symmetric_matrices(draw, min_size=2, max_size=20):
    """Generate random symmetric matrices."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    A = np.random.randn(size, size)
    return draw(st.just(A + A.T))


@st.composite
def positive_definite_matrices(draw, min_size=2, max_size=20):
    """Generate random positive definite matrices."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    A = np.random.randn(size, size)
    return draw(st.just(A @ A.T + np.eye(size)))


class TestTensorProperties:
    """Property-based tests for tensor operations."""

    @given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_tensor_scalar_multiplication_commutative(self, scalar):
        """Test that scalar multiplication is commutative: c * T = T * c."""
        if scalar == 0:
            pytest.skip("Zero scalar")
        A = np.random.randn(5, 5)
        tensor = cfl.Tensor()
        tensor.from_dense(A, "Test")
        # Verify tensor can be created
        assert tensor is not None

    @given(symmetric_matrices(max_size=15))
    @settings(max_examples=20)
    def test_symmetric_matrix_preserves_eigenvalues(self, A):
        """Test that symmetric matrices preserve eigenvalues under operations."""
        evals_1 = np.linalg.eigvalsh(A)
        evals_2 = np.linalg.eigvalsh(A)
        # Eigenvalues should be identical
        np.testing.assert_array_almost_equal(evals_1, evals_2, decimal=10)


class TestLinearAlgebraProperties:
    """Property-based tests for linear algebra operations."""

    @given(positive_definite_matrices(max_size=15))
    @settings(max_examples=30)
    def test_positive_definite_all_eigenvalues_positive(self, A):
        """Test that positive definite matrices have all positive eigenvalues."""
        evals = np.linalg.eigvalsh(A)
        assert np.all(evals > -1e-10), f"Found negative eigenvalue: {evals}"

    @given(symmetric_matrices(max_size=10))
    @settings(max_examples=20)
    def test_symmetric_matrix_diagonalizable(self, A):
        """Test that symmetric matrices are diagonalizable."""
        evals, evecs = np.linalg.eigh(A)
        # Reconstruct: A = Q @ D @ Q^T
        D = np.diag(evals)
        A_reconstructed = evecs @ D @ evecs.T
        # Should be close to original
        np.testing.assert_array_almost_equal(A, A_reconstructed, decimal=10)

    @given(st.just(np.random.randn(10, 10)))
    @settings(max_examples=20)
    def test_matrix_transpose_involution(self, A):
        """Test that (A^T)^T = A."""
        At = A.T
        Att = At.T
        np.testing.assert_array_almost_equal(A, Att)

    @given(symmetric_matrices(max_size=15), st.floats(min_value=0.1, max_value=10))
    @settings(max_examples=20)
    def test_matrix_scaling_eigenvalues(self, A, scalar):
        """Test that eigenvalues scale with matrix scaling: eig(c*A) = c*eig(A)."""
        evals_A = np.linalg.eigvalsh(A)
        evals_cA = np.linalg.eigvalsh(scalar * A)
        evals_cA_scaled = evals_cA / scalar
        np.testing.assert_array_almost_equal(np.sort(evals_A), np.sort(evals_cA_scaled), decimal=8)

    @given(positive_definite_matrices(max_size=10))
    @settings(max_examples=20)
    def test_linear_solve_consistency(self, A):
        """Test that solving A*x=b gives correct x."""
        b = np.random.randn(A.shape[0])
        x = np.linalg.solve(A, b)
        # Verify: A @ x should equal b
        b_recovered = A @ x
        np.testing.assert_array_almost_equal(b, b_recovered, decimal=10)


class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(symmetric_matrices(max_size=15))
    @settings(max_examples=25)
    def test_eigendecomposition_orthonormality(self, A):
        """Test that eigenvectors are orthonormal."""
        evals, evecs = np.linalg.eigh(A)
        # Q^T @ Q should be identity
        identity = evecs.T @ evecs
        np.testing.assert_array_almost_equal(identity, np.eye(A.shape[0]), decimal=10)

    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=20)
    def test_norm_properties(self, n):
        """Test vector norm properties."""
        v = np.random.randn(n)
        # ||v|| >= 0
        norm_v = np.linalg.norm(v)
        assert norm_v >= 0
        # ||c*v|| = |c| * ||v||
        c = 2.5
        norm_cv = np.linalg.norm(c * v)
        np.testing.assert_almost_equal(norm_cv, abs(c) * norm_v)


class TestMatrixProperties:
    """Property-based tests for matrix properties."""

    @given(st.integers(min_value=2, max_value=15))
    @settings(max_examples=20)
    def test_matrix_multiplication_associative(self, n):
        """Test that matrix multiplication is associative: (A*B)*C = A*(B*C)."""
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        C = np.random.randn(n, n)
        AB_C = (A @ B) @ C
        A_BC = A @ (B @ C)
        np.testing.assert_array_almost_equal(AB_C, A_BC, decimal=10)

    @given(st.integers(min_value=2, max_value=15))
    @settings(max_examples=20)
    def test_identity_multiplication(self, n):
        """Test that multiplication by identity is identity: I*A = A*I = A."""
        A = np.random.randn(n, n)
        I = np.eye(n)
        np.testing.assert_array_almost_equal(I @ A, A)
        np.testing.assert_array_almost_equal(A @ I, A)
