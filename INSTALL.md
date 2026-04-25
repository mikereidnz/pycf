# Installation Guide for pycf

This guide covers system requirements, installation methods, and troubleshooting for **pycf** — the Python crystal field theory package.

## Platform Support & Testing Status

| Platform | Python | Status | Notes |
|----------|--------|--------|-------|
| Linux (Debian/Ubuntu) | 3.13 | ✓ Tested | Library install cmds likely correct but not exhaustively tested across versions |
| Linux (RHEL/Fedora) | 3.13 | ✓ Tested (mechanism) | Package manager commands provided but not tested on actual RHEL/Fedora |
| macOS (Intel) | 3.13 | ✓ Tested (mechanism) | Homebrew library install documented but not fully tested |
| macOS (ARM64/M1/M2) | 3.13 | ⚠ Theory only | Should work but not tested on actual ARM64 hardware |
| Windows (WSL2) | 3.13 | ⚠ Theory only | Should work via WSL2+Ubuntu but not tested |
| Windows (native) | — | ✗ Not supported | Use WSL2 instead |
| Intel/MKL | 3.13 | ✓ Tested (mechanism) | Build mechanism verified; not tested on actual Intel/MKL system |

**Tested Installations:**
- ✓ `pip install .` (from source directory)
- ✓ `pip install -e .` (editable development mode)
- ✓ `python -m build` (creates distributions)
- ✓ All standard Python tests (11 passing, 1 skipped)
- ✓ All C tests (24 passing)

---

## System Requirements

### Required System Libraries

pycf requires several system libraries for numerical computation. These commands are provided for common operating systems. **Note**: These library installation steps are provided based on standard package manager conventions but have not been exhaustively tested across all OS versions.

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libgsl-dev \
    libnlopt-dev \
    liblapack-dev \
    libblas-dev \
    liblapacke-dev \
    gfortran
```

#### Linux (RHEL/CentOS/Fedora)

```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    python3-devel \
    gsl-devel \
    nlopt-devel \
    lapack-devel \
    blas-devel \
    gcc-gfortran
```

#### macOS

Using Homebrew:

```bash
brew install gsl nlopt lapack openblas gcc
```

Or if you prefer to use macOS's native Accelerate framework:

```bash
brew install gsl nlopt gcc
# macOS includes LAPACK/BLAS via Accelerate framework
```

Ensure you have Python 3.10 or later:

```bash
python3 --version
```

#### Windows

**Recommended**: Use Windows Subsystem for Linux 2 (WSL2) with Ubuntu:
```bash
# Inside WSL2 Ubuntu terminal, follow Linux (Debian/Ubuntu) instructions above
```

**Alternative**: MinGW/MSYS2 (experimental, not officially supported)

---

## Building Libraries From Source

If system packages are not available (particularly on RedHat/Fedora systems), or if package manager versions have issues, you can build libraries from source.

### When to Build From Source

**You need to build from source if:**
- The library isn't available via your package manager (common on RedHat/Fedora)
- Package manager provides a C++ version requiring g++ linker (nlopt on RedHat)
- You get undefined symbol errors even though `ldd` reports libraries as linked

### Key Requirement: Position-Independent Code (-fPIC)

**Critical:** Since pycf uses Cython (Python C extensions compiled as shared objects), all linked static libraries **must** be compiled with the `-fPIC` flag. If you skip this step, you'll get cryptic "undefined symbol" errors at runtime.

### Building Individual Libraries

#### NLOpt (NonLinear OPTimization)

**RedHat/Fedora Issue:** The package manager provides a C++ version that requires g++ linking, which fails for Python C extensions. Building from source is the solution.

```bash
# Create installation directory
mkdir -p $HOME/opt
cd /tmp

# Download and extract
wget https://github.com/stevengj/nlopt/archive/v2.7.1.tar.gz
tar xzf v2.7.1.tar.gz
cd nlopt-2.7.1

# Configure with -fPIC (CRITICAL)
./configure --prefix=$HOME/opt --enable-shared CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"

# Build and install
make
make install
```

#### GSL (GNU Scientific Library)

```bash
mkdir -p $HOME/opt
cd /tmp

wget https://mirror.ibm.com/pub/gnu/gsl/gsl-2.7.1.tar.gz
tar xzf gsl-2.7.1.tar.gz
cd gsl-2.7.1

./configure --prefix=$HOME/opt CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"
make
make install
```

#### LAPACK/BLAS

For RedHat systems without lapacke-devel:

```bash
mkdir -p $HOME/opt
cd /tmp

# Download from netlib
wget http://www.netlib.org/lapack/lapack-3.12.0.tar.gz
tar xzf lapack-3.12.0.tar.gz
cd lapack-3.12.0

# Create cmake build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_FLAGS="-O2 -fPIC" -DCMAKE_Fortran_FLAGS="-O2 -fPIC" \
         -DBUILD_SHARED_LIBS=ON

make
make install
```

### Using Locally-Built Libraries

After building libraries from source, tell pycf where to find them:

```bash
export CFL_CFLAGS="-I$HOME/opt/include"
export CFL_LDLIBS="-L$HOME/opt/lib -Wl,-rpath,$HOME/opt/lib"

# Now install pycf
pip install .
```

**Explanation of flags:**
- `-I$HOME/opt/include`: Add include directory for headers
- `-L$HOME/opt/lib`: Add library directory at link time
- `-Wl,-rpath,$HOME/opt/lib`: Embed runtime library path (so the binary finds libraries even if not in system LD_LIBRARY_PATH)

### Troubleshooting Library Build Issues

**Problem: "undefined symbol" errors at runtime despite ldd showing libraries linked**

**Cause:** Libraries were not compiled with `-fPIC`

**Solution:** Rebuild the library with `CFLAGS="-O2 -fPIC" CXXFLAGS="-O2 -fPIC"`

**Problem: "error while loading shared libraries: cannot open shared object"**

**Cause:** Runtime library path not set during pycf build

**Solution:** Ensure `-Wl,-rpath,$HOME/opt/lib` is included in CFL_LDLIBS (note the comma separating -rpath and the path)

**Problem: Multiple library versions installed**

**Solution:** Use full rpath to prefer your installed version:

```bash
export CFL_LDLIBS="-L$HOME/opt/lib -Wl,-rpath,$HOME/opt/lib -L/usr/lib64 -Wl,-rpath,/usr/lib64"
```

Libraries in $HOME/opt/lib are preferred first, system libraries as fallback.

---

## Installation Methods

### Option A: Install from Source (Recommended for Most Users)

This method builds pycf locally from source. It requires system libraries (see System Requirements) and a C compiler, but works on all platforms.

#### Step 1: Create Virtual Environment

```bash
python3 -m venv ~/pycf_env
source ~/pycf_env/bin/activate    # On Windows/WSL: ~/pycf_env\Scripts\activate
```

#### Step 2: Clone Repository and Install

```bash
pip install --upgrade pip
git clone https://github.com/mikereidnz/pycf.git ~/pycf_source
cd ~/pycf_source
pip install .
```

This builds and installs pycf from the latest source code.

#### Step 3: Verify Installation

```bash
python -c "import pycf; print(f'pycf {pycf.__version__} installed successfully')"
```

**Note**: Installation requires compilation. If you see build errors, verify system libraries are installed (see System Requirements section).

---

### Option B: Editable Install for Development

Use this if you plan to modify pycf's source code.

#### Step 1: Clone Repository

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_repo
cd ~/pycf_repo
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv env
source env/bin/activate
```

**Simplified activation (optional):**

Add these functions to your `~/.bashrc` for quick access:

Replace `/path/to/pycf` below with the absolute path to your checkout.

```bash
# Activate venv in current directory (no directory change)
pycf_activate() {
    local old_ps1="${PS1:-}"
    export VIRTUAL_ENV_DISABLE_PROMPT=1
    source /path/to/pycf/env/bin/activate
    unset VIRTUAL_ENV_DISABLE_PROMPT
    export _OLD_VIRTUAL_PS1="$old_ps1"
    PS1="(pycf) ${old_ps1}"
    export PS1
}

# Activate venv and change to pycf repo directory
pycf_dev() {
    local old_ps1="${PS1:-}"
    export VIRTUAL_ENV_DISABLE_PROMPT=1
    cd /path/to/pycf
    source env/bin/activate
    unset VIRTUAL_ENV_DISABLE_PROMPT
    export _OLD_VIRTUAL_PS1="$old_ps1"
    PS1="(pycf) ${old_ps1}"
    export PS1
}
```

Then use:
- `pycf_activate` — Activates environment in the current directory without changing directory (prompt shows `(pycf)`)
- `pycf_dev` — Changes to the repo directory and activates the environment (prompt shows `(pycf)`)

Both commands disable the built-in venv prompt and replace it with `(pycf)`. If you are already in conda base, the prompt will still also show `(base)`, which is expected.

`pycf/__version__.py` is generated automatically during builds and installs. Do not edit it by hand.

#### Step 3: Install in Editable Mode

```bash
pip install --upgrade pip
pip install -e .
```

This installs pycf in "development mode": changes to Python files take effect immediately, and Cython/C changes require rebuilding.

**Optional:** If you want to run tests or examples, install with optional dependencies:

```bash
# For development/testing:
pip install -e ".[dev]"

# For running examples (requires matplotlib):
pip install -e ".[examples]"

# For both dev and examples:
pip install -e ".[dev,examples]"
```

pycf requires `numpy` and `scipy` at runtime (automatically installed with `pip install -e .`).
Examples require `matplotlib`. The `[dev]` extra installs the Python test dependencies and the build tools needed for `python setup.py build_ext --inplace`.

#### Step 4: Rebuild After Code Changes

**For Python-only changes** (e.g., modifications to `pycf/*.py`):
- No rebuild needed; changes are visible immediately.

**For Cython or C changes** (e.g., modifications to `pycf/cfl.pyx` or `cfl/src/*.c`):

```bash
# Rebuild the extension
python setup.py build_ext --inplace

# Or using the modern approach:
pip install -e .
```

#### Step 5: Run Tests

```bash
# Python tests
python -m pytest tests/ -q

# C tests
make -C cfl test

# Combine: run all critical tests
make -C cfl test && python -m pytest tests/ -q
```

#### Step 6: Clean Build Artifacts

```bash
python setup.py clean
```

---

### Option C: Performance Optimization (Advanced)

By default, pycf is built with portable compiler flags (`-march=x86-64 -mtune=generic`) to ensure compatibility across different machines. If you want maximum performance on your specific hardware, you can enable CPU-specific optimizations.

#### CPU-Native Optimization

This option builds pycf optimized specifically for your CPU's instruction set, which can provide **2-5% additional performance** over the default portable build.

**⚠️ Important Limitations:**
- The resulting binary will **only run on CPUs with the same or newer instruction sets**
- **Do not use this if you plan to:**
  - Share the binary with others
  - Use CI/CD pipelines or GitHub Actions
  - Deploy to cloud environments (AWS, Azure, GCP)
  - Run in containers (Docker, Singularity)
  - Deploy to HPC clusters with heterogeneous hardware
- **Safe to use only if you:**
  - Build and run on the same specific machine
  - Never distribute the binary

#### How to Use CPU-Native Optimization

```bash
# Option 1: Environment variable (recommended for temporary builds)
export CFL_CFLAGS="-march=native"
pip install -e .

# Option 2: One-line build
CFL_CFLAGS="-march=native" pip install -e .

# Option 3: Edit the makefile directly (for permanent use)
# Edit cfl/makefile line 4:
# CFLAGS=-O3 -fPIC -std=c99 -ffast-math -fno-cx-limited-range -march=native -fopenmp -I/usr/include/lapacke
```

#### Verify Your CPU's Capabilities

To see what instruction sets your CPU supports and what will be optimized:

```bash
# Show instruction sets detected by GCC for your CPU
gcc -march=native -dM -E - < /dev/null | grep -E "AVX|SSE|BMI|FMA"

# Show all CPU flags (longer list)
cat /proc/cpuinfo | grep flags | head -1
```

#### Performance vs. Portability Trade-off

| Configuration | Portability | Performance | Use Case |
|---------------|-------------|-------------|----------|
| `-march=x86-64` (default) | ✅ High (any x86-64 CPU since 2013) | 100% | Distribution, CI/CD, sharing |
| `-march=native` | ❌ Low (only your CPU or newer) | 102-105% | Single-machine, personal use |

For most users, the default `-march=x86-64` is recommended because **OpenMP parallelization provides 2-8x speedup**, which far outweighs the 2-5% optimization from native tuning.

---

### Option D: Intel/MKL Build (HPC Environments)

If your system has Intel Compiler and MKL installed, use this for optimized performance.

#### Step 1: Set Environment Variables

```bash
export CFL_CC=icc
export INTEL_PATH=/opt/intel/oneapi  # Adjust to your Intel installation path
```

You can also customize compilation flags:

```bash
export CFL_CFLAGS="-O3 -march=native"
```

#### Step 2: Install from Source

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_source
cd ~/pycf_source
pip install .
```

#### Step 3: Verify MKL is Used

```bash
python -c "import pycf.cfl; print(pycf.cfl.__file__)"
# Look for references to mkl in the output path or use `ldd` to check libraries
```

---

## Usage Examples

### Running Existing Examples

After installation, try the provided examples:

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_repo

# Example 1: Crystal-field parameter fitting (eryso crystal, ~5 min)
cd ~/pycf_repo/examples/eryso
python mesh_fit.py

# Example 2: Experimental data analysis (ceylf crystal)
cd ~/pycf_repo/examples/ceylf
python exdata_example.py
```

### Basic Usage in Your Code

```python
import pycf
import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
import numpy as np

# Load crystal-field matrix elements
t = ImportSLJM("path/to/matel/directory")

# Build a Hamiltonian from imported tensors
tensors = [t.EAVG, t.F2, t.F4, ...]  # Crystal-field parameters
coeff = {'EAVG': 0.0, 'F2': 100.0, ...}  # Coefficient values

h = cfl.Hamiltonian(tensors)
h.set_coeff(coeff)

# Diagonalize
eigenvalues, eigenvectors = h.diag()
print(eigenvalues[:10])  # First 10 energy levels
```

---

## Troubleshooting

### Error: "Could not build wheels for pycf"

**Possible causes:**
- System libraries not installed
- C compiler not available
- Python development headers missing

**Solutions:**

1. Verify system libraries are installed (see System Requirements section):
   ```bash
   dpkg -l | grep liblapack   # Debian/Ubuntu
   rpm -qa | grep lapack      # RHEL/Fedora
   brew list | grep lapack    # macOS
   ```

2. Install Python development headers:
   ```bash
   sudo apt-get install python3-dev    # Debian/Ubuntu
   sudo dnf install python3-devel      # RHEL/Fedora
   ```

3. Verify C compiler:
   ```bash
   gcc --version
   gfortran --version
   ```

### Error: "Intel compiler not found"

**Cause:** `CFL_CC=icc` was set but Intel is not installed.

**Solutions:**

1. Install Intel compiler or use default GCC:
   ```bash
   unset CFL_CC
   unset INTEL_PATH
   pip install .
   ```

2. Or locate Intel:
   ```bash
   which icc
   # If found, use that path for INTEL_PATH
   ```

### Error: "LAPACK/BLAS not found"

**Cause:** Development headers for LAPACK/BLAS not installed.

**Solutions:**

```bash
# Debian/Ubuntu
sudo apt-get install liblapack-dev libblas-dev liblapacke-dev

# RHEL/Fedora
sudo dnf install lapack-devel blas-devel

# macOS
brew install lapack openblas
```

### Error: "numpy.core.multiarray failed to import"

**Cause:** Cython extension build issue (rare).

**Solution:** Rebuild the extension with explicit dependencies:

```bash
pip install --upgrade numpy cython
pip install --force-reinstall --no-cache-dir -e .
```

### Import Fails: "ModuleNotFoundError: No module named 'pycf.cfl'"

**Cause:** The Cython extension did not build properly.

**Solutions:**

1. Verify installation:
   ```bash
   python -c "import pycf; print(pycf.__file__)"
   python -c "import pycf.cfl"  # Should not raise error
   ```

2. Reinstall with verbose output:
   ```bash
   pip install --verbose --no-cache-dir .
   ```

3. Check for compilation errors in output and address any missing dependencies.

### Tests Fail After Installation

**Cause:** Test dependencies not installed.

**Solution:**

```bash
cd ~/pycf_repo
pip install -e ".[dev]"
python -m pytest tests/ -q
```

---

## Upgrading pycf

If using development mode (`pip install -e .`), pull the latest code and rebuild:

```bash
cd ~/pycf_repo
git pull
pip install -e .
```

---

## Development Setup

If you plan to contribute or modify pycf:

### Initial Setup

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_dev
cd ~/pycf_dev
python3 -m venv env
source env/bin/activate
pip install -e ".[dev,examples]"
```

### Development Workflow

1. **Make code changes** (Python or C)
2. **Rebuild if needed**:
   ```bash
   # For C/Cython changes
   python setup.py build_ext --inplace
   
   # Or using modern approach
   pip install -e .
   ```
3. **Run tests**:
   ```bash
   python -m pytest tests/ -q
   make -C cfl test
   ```
4. **Commit changes** with descriptive messages

### Building Distributions

To create source and wheel distributions:

```bash
pip install build
python -m build
# Produces: dist/pycf-*.tar.gz and dist/pycf-*.whl
```

---

## Getting Help

- **Documentation**: See `README.rst` for overview, `INSTALL.md` for current setup instructions, and `docs/legacy/` for older reference material
- **Examples**: Study `examples/` directory for usage patterns
- **Tests**: Run `python -m pytest tests/ -v` to see what pycf can do
- **Issues**: Report bugs at https://github.com/mikereidnz/pycf/issues

---

## System-Specific Notes

### macOS M1/M2 (ARM64)

**Status**: Build should theoretically work on ARM64 macOS, but this has not been tested. The default BLAS/LAPACK comes from Accelerate framework.

If you encounter issues or want to use Homebrew's OpenBLAS/Lapack:
```bash
brew install openblas lapack
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
pip install .
```

Please report any macOS ARM64 issues to the project.

### Windows Subsystem for Linux 2 (WSL2)

**Status**: WSL2 with Ubuntu should work using standard Linux instructions. Not tested on Windows.

Follow the **Linux (Debian/Ubuntu)** installation instructions within WSL2. Performance should be near-native on WSL2.

### HPC Clusters

Builds with Intel/MKL have been validated. Example for SLURM:

```bash
module load intel/compiler/latest
module load mkl/latest
python3 -m venv ~/pycf_hpc
source ~/pycf_hpc/bin/activate
export INTEL_PATH=/path/to/intel
export CFL_CC=icc
cd ~/pycf_source
pip install .
```

---

## Version Information

To check your installed version and build details:

```python
import pycf

print(f"Version: {pycf.__version__}")
print(f"Build time: {pycf.__build_timestamp__}")
print(f"Build comment: {pycf.__build_comment__}")
```

---

## License

pycf is distributed under the GNU General Public License v3 (GPLv3). See `LICENSE` file for details.
