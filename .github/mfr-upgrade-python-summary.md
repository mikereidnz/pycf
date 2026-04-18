# Python upgrade change summary

- **`setup.py`**
  - Switched from `distutils` imports to `setuptools` and `shutil.which`.
  - Made the `numpy.distutils.intelccompiler` import optional.
  - Fixed git revision handling so `pycf/__version__.py` gets a clean string.
  - Changed the package metadata version from a raw git hash to a valid PEP 440 form like `0+<hash>`.
  - **Reason:** Python 3.14 removed `distutils`, NumPy 2 removed `numpy.distutils`, and modern `setuptools` rejects the old raw-hash version string.

- **`pycf/cfl.pyx`**
  - Changed `cimport cfl` to `from pycf cimport cfl`.
  - Replaced old `cpdef public` attribute declarations with `cdef public`.
  - Added `noexcept` to C function-pointer declarations and casts used with the minimization APIs.
  - Replaced remaining `np.complex(...)` calls with `complex(...)`.
  - Added `Tensor.__rmul__` so scalar-left multiplication works (`scalar * tensor`).
  - Fixed the stray `6#` line near the file header on `devel`.
  - **Reason:** current Cython is stricter than the older version this code was written for, and the importer and examples relied on reverse tensor scaling for `MAGX/MAGY/MAGZ`.

- **`pycf/import_sljm.py`**
  - Cast CSR `indptr` and `indices` arrays to `np.intc` before passing them into `cfl.Tensor`.
  - Replaced `np.complex(0, -1)` with `complex(0, -1)`.
  - **Reason:** newer SciPy and NumPy can produce sparse index arrays with `long` dtype, but the Cython wrapper expects C `int`; NumPy also removed `np.complex`.

- **`pycf/matel.py`**
  - Replaced `np.complex` with `complex` and used `dtype=complex`.
  - **Reason:** NumPy 2 removed `np.complex`.

- **`pycf/pyemp.py`**
  - Replaced `np.complex(0,1)` with `complex(0, 1)`.
  - **Reason:** NumPy 2 compatibility.

- **`pycf/cfl_util.py`**
  - Replaced the remaining `np.complex(...)` and `dtype=np.complex` usage with modern equivalents.
  - **Reason:** NumPy 2 compatibility.

- **`examples/eryso/mcmc_analysis.py`**
  - Replaced `np.complex(0,1)` with `complex(0, 1)`.
  - **Reason:** the example failed at runtime under the upgraded NumPy.

- **`examples/eryso/mhfit_siman.py`**
  - Replaced `np.complex(0,1)` with `complex(0, 1)`.
  - **Reason:** same NumPy 2 compatibility fix in the example set.

## Local non-repo change

- **`/home/users/mfr24/calculations/f1/pycfinten_test/pycf/inten_mfr.py`**
  - Removed an unused `from scipy.special import sph_harm` import.
  - **Reason:** SciPy 1.17 in the new env no longer exposes that name, and the file did not actually use it.

## Not kept as source changes

- Cleaned out generated artifacts such as `pycf/cfl.c`, `pycf.egg-info/*`, build directories, and generated version-file noise so the branch only reflects the real source changes.
