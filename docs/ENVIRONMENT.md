# Environment Variables for pycf

This document describes environment variables that can be used to customize the build and runtime behavior of pycf.

## Build Configuration

### CFL_CC

**Type:** String (compiler command)  
**Default:** Autodetected (usually `gcc`)  
**Example:** `CFL_CC=icc`

Specify the C compiler to use for building the core CFL library. Common options:
- `gcc`: GNU C Compiler (default)
- `icc`: Intel C Compiler
- `clang`: Clang/LLVM compiler

### CFL_CFLAGS

**Type:** String (compiler flags)  
**Default:** Empty (uses setup.py defaults)  
**Example:** `CFL_CFLAGS="-O3 -march=native -fno-strict-aliasing"`

Custom compiler optimization flags. These are appended to the default flags.
Common optimization options:
- `-O3`: Aggressive optimization
- `-march=native`: Optimize for local CPU architecture
- `-ffast-math`: Relax floating-point standards (may affect numerical accuracy)
- `-g`: Include debug symbols

### CFL_LDLIBS

**Type:** String (linker libraries)  
**Default:** Empty (uses autodetected libraries)  
**Example:** `CFL_LDLIBS="-L/opt/intel/mkl/lib -lmkl_core"`

Additional libraries to link against. Useful for:
- Custom LAPACK/BLAS implementations
- MKL (Intel Math Kernel Library) optimizations
- GSL alternatives or custom builds

### INTEL_PATH

**Type:** String (directory path)  
**Default:** Not set (uses system paths)  
**Example:** `INTEL_PATH=/opt/intel/oneapi`

Root directory for Intel compiler and MKL installation. When set:
- Enables Intel C compiler (icc) detection
- Links against Intel MKL libraries
- Requires both `CFL_CC=icc` and this variable to be set

Related: `CFL_CC`, `CFL_LDLIBS`

## Runtime Configuration

### PYTHONPATH

**Type:** String (colon-separated directory list)  
**Default:** Not set (standard Python search paths)  
**Example:** `PYTHONPATH=/home/user/pycf/pycf:/home/user/pycf/env/lib/python3.13/site-packages`

Add directories to Python's module search path. Useful for:
- Using development versions of pycf
- Adding custom modules before installed versions
- Avoiding conflicts with multiple installations

**Note:** pycf handles this automatically when using virtual environments with
`pip install -e .` (editable install).

## Examples

### Build with Intel Compiler and MKL

```bash
export CFL_CC=icc
export INTEL_PATH=/opt/intel/oneapi
export CFL_CFLAGS="-O3 -march=native"
export CFL_LDLIBS="-lmkl_core"
python setup.py build_ext --inplace
```

### Build with Custom GSL Installation

```bash
export CFL_LDLIBS="-L/opt/gsl/lib -lgsl -lgslcblas"
export CFL_CFLAGS="-I/opt/gsl/include"
python setup.py build_ext --inplace
```

### Use Development Version with System Installation

```bash
# Development work in ~/dev/pycf
export PYTHONPATH=~/dev/pycf:$PYTHONPATH
python -c "import pycf; print(pycf.__file__)"
# Output: ~/dev/pycf/pycf/__init__.py (uses development version)
```

### Build for Specific CPU Architecture

```bash
# Optimize for AMD Ryzen
export CFL_CFLAGS="-O3 -march=znver3 -fno-strict-aliasing"
python setup.py build_ext --inplace
```

## Troubleshooting

### Build fails with MKL linker errors

**Problem:** Linker cannot find MKL libraries (e.g., `libmkl_core.so.so`)

**Solution:**
```bash
export INTEL_PATH=/opt/intel/oneapi
export CFL_LDLIBS="-L${INTEL_PATH}/mkl/latest/lib -lmkl_core -lmkl_sequential"
python setup.py build_ext --inplace
```

### Wrong pycf version is imported

**Problem:** Python imports system pycf instead of development version

**Solution:**
```bash
# Check which pycf is imported
python -c "import pycf; print(pycf.__file__)"

# Fix by setting PYTHONPATH
export PYTHONPATH=/home/user/dev/pycf:$PYTHONPATH
python -c "import pycf; print(pycf.__file__)"
```

### GSL library not found at runtime

**Problem:** Error like "libgsl.so.X: cannot open shared object file"

**Solution:**
```bash
# Set library path for runtime
export LD_LIBRARY_PATH=/opt/gsl/lib:$LD_LIBRARY_PATH
python -c "import pycf"  # Should work now
```

## See Also

- `INSTALL.md`: Installation instructions
- `README.rst`: Project overview
- `setup.py`: Build script with CFL configuration
- `cfl/Makefile`: C library build configuration
