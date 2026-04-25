# Installation Guide for pycf

This guide covers system requirements, installation methods, and troubleshooting for **pycf** — the Python crystal field theory package.

---

## System Requirements

### Required System Libraries

pycf requires several system libraries for numerical computation. These vary by operating system.

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

## Installation Methods

### Option A: Install from PyPI (Recommended for Users)

For most users, this is the simplest method.

#### Step 1: Create Virtual Environment

```bash
python3 -m venv ~/pycf_env
source ~/pycf_env/bin/activate    # On Windows: ~/pycf_env\Scripts\activate
```

#### Step 2: Upgrade pip and Install pycf

```bash
pip install --upgrade pip
pip install pycf
```

If a pre-built wheel is available for your platform, installation is fast (no compilation). Otherwise, pip downloads the source and builds locally.

#### Step 3: Verify Installation

```bash
python -c "import pycf; print(f'pycf {pycf.__version__} installed successfully')"
```

#### Step 4: Run an Example

```bash
# Clone the repository to access examples
git clone https://github.com/mikereidnz/pycf.git ~/pycf_repo
cd ~/pycf_repo/examples/ceylf

# Run a simple example
python exdata_example.py
```

---

### Option B: Install from GitHub Source (for Latest Development)

Use this method to get the latest unreleased code.

#### Step 1: Create Virtual Environment

```bash
python3 -m venv ~/pycf_dev_env
source ~/pycf_dev_env/bin/activate
```

#### Step 2: Install from GitHub

**Option B1**: Direct install from GitHub main branch:

```bash
pip install --upgrade pip
pip install git+https://github.com/mikereidnz/pycf.git@main
```

**Option B2**: Clone and install locally (better for development):

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_source
cd ~/pycf_source
pip install --upgrade pip
pip install .
```

#### Step 3: Verify

```bash
python -c "import pycf; print(f'pycf {pycf.__version__} installed')"
python -m pytest tests/ -q
```

---

### Option C: Editable Install for Development

Use this if you plan to modify pycf's source code.

#### Step 1: Clone Repository

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_repo
cd ~/pycf_repo
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install in Editable Mode

```bash
pip install --upgrade pip
pip install -e .
```

This installs pycf in "development mode": changes to Python files take effect immediately, and Cython/C changes require rebuilding.

#### Step 4: Rebuild After Code Changes

**For Python-only changes** (e.g., modifications to `pycf/*.py`):
- No rebuild needed; changes are visible immediately.

**For Cython or C changes** (e.g., modifications to `pycf/cfl.pyx` or `cfl/src/*.c`):

```bash
# Rebuild the extension
python setup.py build_ext --inplace

# Or using the modern approach:
pip install --no-build-isolation -e .
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

#### Step 2: Install

**From PyPI:**
```bash
pip install --no-build-isolation pycf
```

**From source:**
```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_source
cd ~/pycf_source
pip install --no-build-isolation .
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
   pip install pycf
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
pip install --force-reinstall --no-cache-dir pycf
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
   pip install --verbose --no-cache-dir pycf
   ```

3. Check for compilation errors in output and address any missing dependencies.

### Tests Fail After Installation

**Cause:** Test dependencies not installed.

**Solution:**

```bash
pip install pytest matplotlib scipy
cd ~/pycf_repo
python -m pytest tests/ -q
```

---

## Upgrading pycf

To upgrade to the latest version:

```bash
pip install --upgrade pycf
```

If using development mode (`pip install -e .`), pull the latest code and rebuild:

```bash
cd ~/pycf_repo
git pull
pip install --no-build-isolation -e .
```

---

## Development Setup

If you plan to contribute or modify pycf:

### Initial Setup

```bash
git clone https://github.com/mikereidnz/pycf.git ~/pycf_dev
cd ~/pycf_dev
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install pytest matplotlib scipy
```

### Development Workflow

1. **Make code changes** (Python or C)
2. **Rebuild if needed**:
   ```bash
   # For C/Cython changes
   python setup.py build_ext --inplace
   
   # Or using modern approach
   pip install --no-build-isolation -e .
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

- **Documentation**: See `README.rst` for overview and `doc/` directory for detailed guides
- **Examples**: Study `examples/` directory for usage patterns
- **Tests**: Run `python -m pytest tests/ -v` to see what pycf can do
- **Issues**: Report bugs at https://github.com/mikereidnz/pycf/issues

---

## System-Specific Notes

### macOS M1/M2 (ARM64)

The default BLAS/LAPACK comes from Accelerate framework. Builds should work automatically.

If using Homebrew's OpenBLAS/Lapack:
```bash
brew install openblas lapack
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
pip install pycf
```

### Windows Subsystem for Linux 2 (WSL2)

Follow the **Linux (Debian/Ubuntu)** instructions. Performance is near-native on WSL2.

### HPC Clusters

Consult your cluster's module system. Example for SLURM:

```bash
module load intel/compiler/latest
module load mkl/latest
python3 -m venv ~/pycf_hpc
source ~/pycf_hpc/bin/activate
export INTEL_PATH=/path/to/intel
export CFL_CC=icc
pip install pycf
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
