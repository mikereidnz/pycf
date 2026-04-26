"""Performance benchmarks for pycf using pytest-benchmark.
These benchmarks track performance of critical operations:
- Array operations
- Linear algebra (eigenvalues, solving)
- NumPy performance baseline
Run with: pytest tests/test_benchmarks.py --benchmark-only
Install: pip install pytest-benchmark (optional - tests will skip if not installed)
"""

import numpy as np
import pytest

try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False
    pytest.skip("pytest-benchmark not installed", allow_module_level=True)
try:
    from pycf import cfl
except ImportError:
    pytest.skip("pycf not installed", allow_module_level=True)


class TestArrayBenchmarks:
    """Benchmark NumPy and core array operations."""

    def test_matrix_multiply_small(self, benchmark):
        """Benchmark small matrix multiplication."""
        A = np.random.randn(50, 50)
        B = np.random.randn(50, 50)

        def matmul():
            return np.dot(A, B)

        result = benchmark(matmul)
        assert result.shape == (50, 50)

    def test_matrix_multiply_medium(self, benchmark):
        """Benchmark medium matrix multiplication."""
        A = np.random.randn(200, 200)
        B = np.random.randn(200, 200)

        def matmul():
            return np.dot(A, B)

        result = benchmark(matmul)
        assert result.shape == (200, 200)

    def test_linear_solve_small(self, benchmark):
        """Benchmark solving small linear system."""
        A = np.random.randn(30, 30)
        A = A @ A.T  # Make positive definite
        b = np.random.randn(30)

        def solve():
            return np.linalg.solve(A, b)

        result = benchmark(solve)
        assert result.shape == (30,)

    def test_linear_solve_medium(self, benchmark):
        """Benchmark solving medium linear system."""
        A = np.random.randn(100, 100)
        A = A @ A.T  # Make positive definite
        b = np.random.randn(100)

        def solve():
            return np.linalg.solve(A, b)

        result = benchmark(solve)
        assert result.shape == (100,)

    def test_eigh_small(self, benchmark):
        """Benchmark symmetric eigendecomposition - small."""
        A = np.random.randn(50, 50)
        A = A + A.T  # Make symmetric

        def eigh():
            evals, evecs = np.linalg.eigh(A)
            return evals, evecs

        evals, evecs = benchmark(eigh)
        assert len(evals) == 50
        assert evecs.shape == (50, 50)

    def test_eigh_medium(self, benchmark):
        """Benchmark symmetric eigendecomposition - medium."""
        A = np.random.randn(150, 150)
        A = A + A.T  # Make symmetric

        def eigh():
            evals, evecs = np.linalg.eigh(A)
            return evals, evecs

        evals, evecs = benchmark(eigh)
        assert len(evals) == 150
        assert evecs.shape == (150, 150)

    def test_svd_small(self, benchmark):
        """Benchmark singular value decomposition - small."""
        A = np.random.randn(50, 40)

        def svd():
            U, S, Vt = np.linalg.svd(A)
            return U, S, Vt

        U, S, Vt = benchmark(svd)
        assert U.shape[0] == 50
        assert len(S) == 40

    def test_norm_computation(self, benchmark):
        """Benchmark Frobenius norm computation."""
        A = np.random.randn(200, 200)

        def norm():
            return np.linalg.norm(A, "fro")

        result = benchmark(norm)
        assert result > 0


class TestArrayOperations:
    """Benchmark various array operations."""

    def test_element_wise_multiply(self, benchmark):
        """Benchmark element-wise multiplication."""
        A = np.random.randn(500, 500)
        B = np.random.randn(500, 500)

        def multiply():
            return A * B

        result = benchmark(multiply)
        assert result.shape == (500, 500)

    def test_matrix_transpose(self, benchmark):
        """Benchmark matrix transpose."""
        A = np.random.randn(500, 500)

        def transpose():
            return A.T

        result = benchmark(transpose)
        assert result.shape == (500, 500)

    def test_array_sum(self, benchmark):
        """Benchmark sum reduction."""
        A = np.random.randn(1000, 1000)

        def sum_all():
            return np.sum(A)

        result = benchmark(sum_all)
        assert isinstance(result, (float, np.floating))

    def test_reshape_operation(self, benchmark):
        """Benchmark reshape operation."""
        A = np.random.randn(1000, 1000)

        def reshape():
            return A.reshape(1000000)

        result = benchmark(reshape)
        assert result.shape == (1000000,)


class TestCFLIntegration:
    """Benchmark pycf integration with NumPy."""

    def test_cfl_import_time(self, benchmark):
        """Benchmark pycf import (cached)."""

        def import_cfl():
            import pycf

            return pycf

        result = benchmark(import_cfl)
        assert result is not None
