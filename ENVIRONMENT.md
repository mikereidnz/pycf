# Environment Variables and Build Configuration

This document describes environment variables that control pycf build and runtime behavior.

## Build Configuration Variables

### C Compiler and Flags

#### `CFL_CC`
**Description:** Override the C compiler for building the core `cfl` library.

**Default:** Detected automatically (gcc, clang, or icc)

**Example:**
```bash
CFL_CC=clang python setup.py build_ext --inplace
CFL_CC=icc python setup.py build_ext --inplace  # Use Intel C compiler
```

**Use Cases:**
- Force specific compiler (Intel ICC for optimized builds)
- Use alternative compiler if default is not available
- Performance tuning for specific hardware

---

#### `CFL_CFLAGS`
**Description:** Additional C compiler flags to pass to the `cfl` library build.

**Default:** Debug/optimization flags set by `cfl/Makefile`

**Example:**
```bash
# Add sanitizers for debugging
CFL_CFLAGS="-fsanitize=address -fsanitize=undefined" python setup.py build_ext --inplace

# Add march=native for CPU-specific optimization
CFL_CFLAGS="-march=native" python setup.py build_ext --inplace

# Add debugging symbols
CFL_CFLAGS="-g -O0" python setup.py build_ext --inplace
```

**Common Flags:**
- `-march=native` - CPU-specific optimizations (up to 50% speedup on some systems)
- `-O3` - Aggressive optimization (default is -O2)
- `-O0` - Disable optimization (for debugging)
- `-g` - Include debugging symbols
- `-fsanitize=address` - AddressSanitizer (detect memory leaks/corruption)
- `-fsanitize=undefined` - UndefinedBehaviorSanitizer (detect undefined behavior)
- `-fprofile-arcs -ftest-coverage` - Code coverage instrumentation

**Performance Notes:**
- Thread count: The build may auto-detect CPU core count. Set `CFL_CFLAGS="-march=native"` for single-threaded builds if needed.
- Sanitizers significantly slow down code; use only for debugging or CI.

---

#### `CFL_LDLIBS`
**Description:** Additional linker libraries to link against `cfl`.

**Default:** Standard system libraries

**Example:**
```bash
# Link against MKL for high-performance linear algebra
CFL_LDLIBS="-lmkl_core" python setup.py build_ext --inplace

# Link against OpenBLAS
CFL_LDLIBS="-lopenblas" python setup.py build_ext --inplace
```

**Use Cases:**
- Link against optimized BLAS libraries (Intel MKL, OpenBLAS, ATLAS)
- Support platform-specific libraries
- Custom build environments

---

### Intel Compiler Configuration

#### `INTEL_PATH`
**Description:** Path to Intel compiler and MKL installation directory.

**Default:** Auto-detected if Intel compilers are in PATH

**Example:**
```bash
# For Intel oneAPI installation (2021+)
export INTEL_PATH=/opt/intel/oneapi
CFL_CC=icc python setup.py build_ext --inplace

# For older Intel Parallel Studio
export INTEL_PATH=/opt/intel/composerxe
CFL_CC=icc python setup.py build_ext --inplace
```

**Use Cases:**
- Explicitly point to Intel compiler installation
- Support multiple Intel installations on same system
- Enable MKL linking in custom environments

**Prerequisites:**
- Intel C compiler (icc or icc from oneAPI) must be installed
- oneAPI environment may need sourcing: `source /opt/intel/oneapi/setvars.sh`

---

## Runtime Configuration Variables

### Python and Cython

#### `PYTHONPATH`
**Description:** Python module search path.

**Default:** System Python paths

**Example:**
```bash
# Add pycf development directory to search path
export PYTHONPATH=/home/users/mfr24/dev/pycf:$PYTHONPATH

# For editable installation via pip
pip install -e .  # Automatically manages PYTHONPATH
```

**Use Cases:**
- Development: Run pycf without full installation
- Multiple Python versions: Keep separate PYTHONPATH for each version
- Testing: Isolate development code from system packages

---

#### `PYTHONDONTWRITEBYTECODE`
**Description:** Prevent Python from creating `.pyc` bytecode files.

**Example:**
```bash
export PYTHONDONTWRITEBYTECODE=1
python -m pytest tests/
```

**Use Cases:**
- Keep git repository clean (avoid `.pyc` in diffs)
- Debug: Ensure fresh imports on each run
- CI/CD: Prevent stale cached bytecode

---

#### `CYTHON_TRACE`
**Description:** Enable Cython line tracing for profiling.

**Default:** Disabled

**Example:**
```bash
CYTHON_TRACE=1 python setup.py build_ext --inplace
```

**Use Cases:**
- Performance profiling of Cython code
- Debug: Get accurate line numbers in error traces
- Note: Line tracing adds 5-10% runtime overhead

---

## Debugging and Testing

### Build Debugging

#### Quick Build Verification
```bash
# Build without installation
python setup.py build_ext --inplace

# Verify core C library
make -C cfl test

# Verify Python tests
python -m pytest tests/ -q
```

#### Memory Safety (GitHub Actions)
```bash
# Run with AddressSanitizer (GCC/Clang)
CFL_CFLAGS="-fsanitize=address" python -m pytest tests/

# Run with UndefinedBehaviorSanitizer
CFL_CFLAGS="-fsanitize=undefined" python -m pytest tests/
```

**Note:** Sanitized builds are slow (10-50x slower) but catch memory errors.

---

### Coverage and Profiling

#### Code Coverage
```bash
# Generate coverage report
python -m pytest tests/ --cov=pycf --cov-report=html

# View report
open htmlcov/index.html
```

#### Performance Profiling
```bash
# Run benchmarks
python -m pytest tests/test_benchmarks.py --benchmark-only

# Profile specific test
python -m pytest tests/test_benchmarks.py::test_matrix_multiply_50x50 -v
```

---

## Pre-commit and Linting

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

### Code Formatting (Black)

```bash
# Black uses 100-char line width (configured in pyproject.toml)
black pycf/ tests/

# Check formatting without modifying
black --check pycf/ tests/
```

### Type Checking (mypy)

```bash
# Run type checker
mypy pycf/

# Run with strict mode
mypy --strict pycf/
```

### Import Sorting (isort)

```bash
# Sort imports
isort pycf/ tests/

# Check imports without modifying
isort --check pycf/ tests/
```

### Linting (flake8)

```bash
# Run linter (configured for 100-char line width)
flake8 pycf/ tests/
```

---

## System Requirements and Optimization

### Thread Count Handling

The `cfl` library build may use multiple threads for compilation.

**Autodetection:**
- Default: Build system auto-detects CPU cores (e.g., 2 threads on dual-core systems)
- Result: `top` shows ~200%, `htop` shows 2 threads

**Control Thread Count:**
```bash
# Single-threaded build (adds -march=native for optimization)
CFL_CFLAGS="-march=native -O2" make -C cfl clean && make -C cfl

# For CI with thread limit
MAKEFLAGS=-j1 python setup.py build_ext --inplace
```

### CPU-Specific Optimization

#### `-march=native`
**Description:** Compile for current CPU, using all supported instruction sets.

**Performance Impact:** +20% to +50% on modern CPUs (varies by architecture and algorithm)

**Example:**
```bash
CFL_CFLAGS="-march=native" python setup.py build_ext --inplace
```

**Requirements:**
- Binary is only portable to same CPU generation
- Use default flags (no `-march=`) for binary distribution
- Recommended for local development and performance-critical deployments

**Affected Functions:**
- Linear algebra operations (matrix multiply, SVD, eigendecomposition)
- Tensor contraction and summation
- Fits involving large matrices (100x100+)

---

## Example Workflows

### Development Build (with debugging)
```bash
export PYTHONDONTWRITEBYTECODE=1
CFL_CFLAGS="-g -O0" python setup.py build_ext --inplace
python -m pytest tests/ -v
```

### Optimized Build (single machine)
```bash
CFL_CFLAGS="-march=native -O3" python setup.py build_ext --inplace
python -m pytest tests/test_benchmarks.py --benchmark-only
```

### Portable Build (for distribution)
```bash
# No special flags - uses defaults in cfl/Makefile
python setup.py build_ext --inplace
python -m pytest tests/ -q
```

### CI/CD Build (with sanitizers)
```bash
export PYTHONDONTWRITEBYTECODE=1
CFL_CFLAGS="-fsanitize=address -fsanitize=undefined" python setup.py build_ext --inplace
python -m pytest tests/ --cov=pycf
```

---

## Troubleshooting

### Build Failures

**Symptom:** Compiler error with unrecognized flag
```
cc1: error: unrecognized command line option '-march=native'
```
**Solution:** Remove `-march=native` or update compiler
```bash
CFL_CFLAGS="-O2" python setup.py build_ext --inplace
```

---

**Symptom:** "Cannot find mex.h" or MKL linking errors
```
ld: cannot find -lmkl_core
```
**Solution:** Set `CFL_LDLIBS` with correct MKL path
```bash
CFL_LDLIBS="-L${INTEL_PATH}/mkl/lib/intel64 -lmkl_core" \
  python setup.py build_ext --inplace
```

---

**Symptom:** High memory usage during build
```
make[1]: warning: -j does not specify a number of jobs.  Will infer from system.
```
**Solution:** Limit thread count
```bash
MAKEFLAGS=-j2 python setup.py build_ext --inplace
```

---

## Related Documentation

- **INSTALL.md** - Installation instructions
- **setup.py** - Build configuration
- **cfl/Makefile** - C library build rules
- **.pre-commit-config.yaml** - Linting and formatting standards
- **pyproject.toml** - Python project configuration
- **.github/workflows/ci.yml** - Continuous integration setup
