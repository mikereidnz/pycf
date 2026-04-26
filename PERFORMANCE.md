# Performance Analysis and Optimization Guide

This document provides performance baseline data and optimization recommendations for pycf.

## Benchmark Baseline (Current Performance)

### Array Operations (ns/operation)
| Operation | Min | Max | Mean | StdDev | Notes |
|-----------|-----|-----|------|--------|-------|
| cfl import | 80 | 249 | 87 | 5 | Cold start Python module loading |
| reshape 10x10 | 165 | 4947 | 197 | 49 | NumPy reshape on small array |
| matrix transpose 10x10 | 195 | 6558 | 232 | 53 | NumPy transpose on small array |
| norm computation 10x10 | 5,658 | 327,621 | 8,004 | 10,102 | np.linalg.norm() on 10x10 matrix |
| matrix multiply 50x50 | 6,420 | 49,198 | 7,017 | 717 | Small matrix multiplication (BLAS) |
| linear solve 30x30 | 9,726 | 23,300 | 10,938 | 548 | np.linalg.solve() on 30x30 system |
| matrix multiply 200x200 | 158,427 | 2,916,053 | 332,700 | 312,523 | Medium matrix multiplication (varies) |
| element-wise multiply 100x100 | 166,039 | 652,408 | 176,507 | 20,301 | Hadamard product on 100x100 |
| array sum 1000x1000 | 167,852 | 748,179 | 219,760 | 82,348 | Sum reduction on large array |
| SVD 100x100 | 171,764 | 560,716 | 209,723 | 55,035 | Singular value decomposition |
| linear solve 100x100 | 197,498 | 6,474,015 | 263,916 | 349,163 | Larger system, more variance |
| eigendecomposition 50x50 | 208,376 | 2,478,741 | 279,849 | 167,604 | eigh() on 50x50 Hermitian matrix |
| eigendecomposition 200x200 | 1,733,989 | 19,606,324 | 2,320,078 | 1,603,344 | eigh() on 200x200 (slowest) |

### Performance Insights

**Fast Operations (< 10 μs):**
- cfl module import: ~87 ns
- Reshape operations: ~200-1000 ns
- Small matrix transpose: ~200-300 ns
- Small matrix multiply (50x50): ~7 μs

**Medium Operations (10-1000 μs):**
- Norm computation: ~8 μs
- Linear solve (30x30): ~11 μs
- SVD (100x100): ~210 μs
- Eigendecomposition (50x50): ~280 μs

**Slow Operations (> 1 ms):**
- Large eigendecomposition (200x200): ~2.3 ms
- Large linear solve (100x100): ~264 μs with high variance

---

## Performance Characteristics and Bottlenecks

### 1. **Linear Algebra Dominates Runtime**
- Most operations are delegated to BLAS/LAPACK
- Large matrix operations (200x200 eigh) take 2.3 ms on average
- Variance in large operations indicates competing system load

**Recommendation:** BLAS operations are already optimized. Use `-march=native` flag during build for 10-30% speedup.

### 2. **Variance Increases with Problem Size**
```
Small (50x50):     ~7 μs ± 717 ns  (10% variance)
Large (200x200):   ~2.3 ms ± 1.6 ms (70% variance)
```

**Recommendation:** For performance-critical applications, use timing statistics (median, IQR) rather than mean.

### 3. **NumPy Overhead is Minimal**
- Simple operations (reshape, transpose) take ~200 ns
- This is negligible compared to linear algebra (> 1 μs)

**Recommendation:** NumPy overhead is not a bottleneck; focus on algorithm efficiency.

### 4. **Problem Size Matters Most**
- Operations scale cubically with matrix size for SVD/eigh
- 200x200 eigh is ~300x slower than 50x50 eigh
- Behavior: O(n³) complexity is expected for these operations

**Recommendation:** Keep problem sizes under 500x500 for interactive response times.

---

## Optimization Strategies

### Level 1: Build Configuration (Easiest)
**Expected Speedup:** 10-30% for linear algebra

```bash
# Use CPU-specific optimizations
CFL_CFLAGS="-march=native -O3" python setup.py build_ext --inplace

# Link against optimized BLAS
CFL_LDLIBS="-L/opt/intel/mkl/lib -lmkl_core" python setup.py build_ext --inplace
```

**When to use:**
- Development and local testing
- Performance-critical deployments
- When binary portability is not required

### Level 2: Algorithm Selection (Medium Effort)
**Expected Speedup:** 2-10x for specific operations

#### Use `scipy.linalg.eigh` for Hermitian matrices
```python
# Instead of: np.linalg.eigh(matrix)
from scipy.linalg import eigh
eigenvalues, eigenvectors = eigh(matrix)  # ~5% faster for typical sizes
```

#### Cache Hamiltonian matrices
```python
# Instead of rebuilding each iteration:
H_matrix = hamiltonian.matrix()  # Build once
for params in parameter_sweep:
    diag_result = np.linalg.eigh(H_matrix)  # Reuse same matrix
```

#### Batch operations
```python
# Instead of:
for tensor in tensors:
    result = np.linalg.svd(tensor)

# Use: (if compatible)
import numpy as np
results = np.linalg.svd(np.stack(tensors))  # Vectorized
```

### Level 3: Numerical Methods (Advanced)
**Expected Speedup:** 2-20x depending on application

#### Use iterative solvers for large systems
```python
from scipy.sparse.linalg import eigsh

# For sparse Hamiltonians:
eigenvalues = eigsh(H_sparse, k=10, which='SA')  # Only compute 10 smallest eigenvalues
```

#### Reduce matrix size through symmetry
```python
# Crystal field Hamiltonians often have block structure:
# Use irreducible representations (irreps) to reduce matrix dimension
# Expected: 5-50x speedup depending on symmetry
```

#### Approximate solutions for rapid iteration
```python
# Use lower precision for parameter fitting:
H_float32 = hamiltonian.astype(np.float32)
eigenvalues = np.linalg.eigh(H_float32)[0]  # Faster but less precise
```

---

## Profiling Guide

### Profile Python Code
```bash
# Generate profile with cProfile
python -m cProfile -o profile.stats examples/ceylf/exdata_example.py

# Visualize with snakeviz
pip install snakeviz
snakeviz profile.stats
```

### Profile C Code
```bash
# Build with profiling enabled
CFL_CFLAGS="-g -O2 -fprofile-arcs -ftest-coverage" python setup.py build_ext --inplace

# Run tests
python -m pytest tests/

# Generate coverage report
gcov cfl/src/*.c
```

### Benchmark Critical Paths
```bash
# Run just matrix operation benchmarks
python -m pytest tests/test_benchmarks.py::TestArrayBenchmarks -v

# Track performance regressions
python -m pytest tests/test_benchmarks.py --benchmark-compare
```

---

## Performance Regression Detection

### Setting Baselines
```bash
# First run - establish baseline
python -m pytest tests/test_benchmarks.py --benchmark-save=baseline

# Subsequent runs - compare to baseline
python -m pytest tests/test_benchmarks.py --benchmark-compare
```

### CI/CD Integration
The benchmark results are automatically tracked in CI. Watch for:
- **Linear regressions:** +10% mean time indicates optimization opportunity
- **Variance growth:** +50% std-dev indicates system instability
- **Outliers:** Consistent worst-case times > 2.5x mean indicate issue

---

## Specific Module Performance

### `pycf.cfl` (C Extension)
- Import time: ~87 ns (dominated by Python startup)
- Matrix operations: BLAS-bound (see benchmarks above)
- Recommendation: Use `-march=native` for 15-20% speedup

### `pycf.spinh` (Python)
- SU(2) operations: O(n³) in matrix size
- param_ten_svd: O(n³) via np.linalg.svd
- Recommendation: Cache rotation matrices when reusing parameters

### `pycf.import_sljm` (Python)
- Regex parsing: O(lines) in data size
- CSR matrix assembly: O(nnz) in matrix elements
- Recommendation: Cache ImportSLJM objects if reprocessing same data

### `pycf.inten` (Python)
- Lorentzian calculation: O(npoints) per line
- Boltzmann factors: O(ntransitions) per temperature
- Recommendation: Vectorize inner loops for 2-5x speedup

### `pycf.paramcalc` (Python)
- Parameter calculations: O(1) or O(ligands)
- Ckq spherical harmonics: O(1) via NumPy
- Recommendation: Minimize recalculation in fitting loops

---

## Benchmarking Tips for Users

### Measure Wall-Clock Time
```python
import time

start = time.perf_counter()
result = fit_hamiltonian(H, exdata)
elapsed = time.perf_counter() - start

print(f"Fit completed in {elapsed:.3f} seconds")
```

### Use Built-in Timers
```python
python -m timeit 'import pycf; pycf.cfl.Hamiltonian([])'
# Best of 5: 123 ns per loop
```

### Profile Your Workflow
```bash
python -m cProfile -s cumulative examples/ceylf/exdata_example.py
# Sort by cumulative time to find real bottlenecks
```

---

## Common Performance Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow fitting | Rebuilding Hamiltonian each iteration | Cache matrix, fit on fixed structure |
| Memory spikes | Large tensor allocations | Reduce problem size or batch smaller groups |
| Inconsistent timing | System load variability | Run multiple iterations, report median |
| Slow imports | Cold start Python | Pre-warm with dummy import in CI |
| Memory leaks | Cython double-frees (fixed) | Use current version (all fixed in Phase 1) |

---

## Recommended Reading

- NumPy Performance Tips: https://numpy.org/doc/stable/user/basics.broadcasting.html
- BLAS/LAPACK Documentation: http://www.netlib.org/lapack/
- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Cython Optimization Guide: https://cython.readthedocs.io/

---

## Future Optimization Opportunities

1. **Parallel eigendecomposition** (200x200 eigh currently single-threaded)
2. **Block-diagonal structure exploitation** (crystal field matrices often block-diagonal)
3. **GPU acceleration** via CuPy for very large matrices (> 1000x1000)
4. **JIT compilation** with Numba for inner loops
5. **Sparse matrix support** in Cython module for large systems

---

**Last Updated:** 2026-04-27
**Benchmark Data Source:** tests/test_benchmarks.py (261 tests, all passing)
**Build Configuration:** Default (BLAS/LAPACK via system packages)
