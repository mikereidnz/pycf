from datetime import datetime
import sys

try:
    from pycf.__version__ import __version__, __build_timestamp__, __build_comment__
except ImportError:
    __version__ = 'unknown'
    __build_timestamp__ = 'unknown'
    __build_comment__ = ''


"""
PyCF: Python Crystal Field package for rare-earth ion modeling.

A comprehensive Python package for crystal field calculations on rare-earth ions,
combining density-functional-theory (DFT) computations with experimental fitting.

**Main Components:**

1. **Core Crystal Field Module (cfl)** - Cython-based performance-critical code
   - Hamiltonian construction and diagonalization
   - Tensor storage and matrix element calculation
   - Eigenvalue/eigenvector computation
   - Experimental data handling with ExData class

2. **Intensity Calculations (inten, paramcalc)**
   - Magnetic dipole and electric dipole transitions
   - Oscillator strengths and Einstein A coefficients
   - Spectral broadening (Lorentzian lineshapes)
   - Temperature-dependent Boltzmann populations

3. **Parameter Calculations (paramcalc)**
   - Crystal field parameters (Altp) from point-charge model
   - Radial integrals and transition intensity parameters
   - Static and dynamic coupling contributions

4. **Data Import (import_sljm)**
   - SLJM/EMP program output file parsing
   - State label extraction and tensor loading
   - Sparse matrix (CSR) tensor format

5. **Utilities**
   - Spin Hamiltonian extraction (spinh)
   - EMP program wrapper (pyemp)
   - Result formatting and analysis (cfl_util)

**Typical Workflow:**

1. Load crystal field tensors from SLJM (import_sljm.ImportSLJM)
2. Create Hamiltonian and add tensor terms (cfl.Hamiltonian)
3. Diagonalize to get eigenvalues and eigenvectors
4. Transform intensity tensors to eigenbasis
5. Calculate transitions and spectrum (inten.dipole_str, inten.inten)
6. Fit parameters to experimental data (optimize, fit_residual)
7. Extract spin Hamiltonian if needed (spinh)

**Key References:**

- Crystal field theory: J. C. Gómez-Herrero & J. Sanchez-Dehesa (1988)
- Intensity calculations: Krupke (1966), Reid (1997)
- Wigner symbols: Rasch & Yu (2003)
- Spin Hamiltonian: Golding & Halley (1984)

**Performance Notes:**

- Core C library in cfl/ provides 10-100x speedup for large matrices
- BLAS/LAPACK integration via SciPy for matrix operations
- Intel MKL support available via CFL_CC and CFL_LDLIBS environment variables
- See docs/ENVIRONMENT.md for optimization options

**License:**

See individual module headers for copyright information. Most modules under
GNU GPL v3 (Sebastian Horvath) or MIT License (as noted).
"""


def _fmt_pycf_time(value=None):
    """Format pycf timestamp."""
    if value is None:
        value = datetime.now()
    if isinstance(value, str):
        return value
    return value.strftime('%Y-%m-%d %H:%M:%S')


def pycf_info(current_time=None, stream=None):
    r"""
    Print and return a short pycf metadata block for scripts and notebooks.
    """
    if stream is None:
        stream = sys.stdout

    info = (
        "----------------------------------------------------------\n"
        "pycf details\n"
        "============\n\n"
        "pycf revision: {}  built at {}\n"
        "Build comment: {}\n"
        "Current time: {}\n\n"
        "----------------------------------------------------------"
    ).format(__version__, __build_timestamp__, __build_comment__, _fmt_pycf_time(current_time))

    print(info, file=stream)
    return info


__all__ = ['__version__', '__build_timestamp__', '__build_comment__', 'pycf_info']
