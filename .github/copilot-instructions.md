# Copilot instructions for `pycf`

## Build and test commands

- Build the core C library only: `make -C cfl`
- Build the core C library with the makefile's debug flags: `make -C cfl debug`
- Clean C build artifacts and compiled test binaries: `make -C cfl clean`
- Rebuild the Python extension from the repo root: `python setup.py build_ext --inplace`
  - `setup.py` runs `make` in `cfl/` first, regenerates `pycf/__version__.py` from `git rev-parse --short HEAD`, and then builds the Cython extension.
- Install the package using the workflow documented in `README.rst` / `doc/install.rst`: `python setup.py install --prefix=/path/to/prefix`
- Run the full C test suite: `make -C cfl test`
- Run a single C test: `make -C cfl h_test && ./cfl/h_test`
  - Replace `h_test` with any target from `cfl/tests/*test.c` such as `csr_test`, `sh_test`, `opt_test`, or `zefoz_test`. The full suite just builds those `*_test` binaries and executes them via `cfl/cfl_testing.sh`.

## High-level architecture

- `cfl/` is the performance-critical core written in C99. `cfl/src/*.c` implements tensor storage, Hamiltonian assembly, fitting/minimization, spin-Hamiltonian helpers, and ZEFOZ-related routines. `cfl/include/*.h` defines the shared data structures used by both the standalone C tests and the Python wrapper.
- `pycf/cfl.pyx` is the main Cython bridge. It owns the lifetime of C objects through `PyCapsule`, converts NumPy arrays into the layouts expected by the C API, and exposes the user-facing crystal-field types used by the examples.
- `setup.py` is part of the architecture, not just packaging glue: it builds `cfl/libcfl.a`, generates `pycf/__version__.py`, and links the C archive into the `pycf.cfl` extension. If the C build changes, it also touches `pycf/cfl.pyx` so the extension is rebuilt.
- `pycf/import_sljm.py` is the main ingestion path for matrix-element data. It reads the `*.txt`, `*.mi_`, and `*.st_` outputs from SLJM/EMP tooling, parses state labels, builds SciPy CSR matrices, and turns them into `cfl.Tensor` instances.
- `pycf/cfl_util.py` holds formatting and summary helpers that are tightly coupled to the `cfl.pyx` API, especially energy summaries, label formatting, and experimental-data presentation.
- `pycf/spinh.py` is a separate pure-Python layer for building and inverting spin-Hamiltonian terms from dense matrices. It complements `cfl` rather than replacing it.
- `pycf/pyemp.py` is an independent pure-Python wrapper around Michael F. Reid's EMP executables (`cfit`, `inten`, `vtrans`, `spectrum`). Changes here often affect external file generation/parsing rather than the core `cfl` library.
- `examples/` are the best end-to-end usage references. They show how `ImportSLJM`, `cfl.Hamiltonian`, `cfl.ExData`, fitting helpers, and `spinh` are expected to work together for material-specific workflows.

## Key conventions

- Experimental level data is funneled through `cfl.ExData`, with mode strings such as `'A'`, `'D'`, `'AS'`, and `'DS'`. Example inputs use 1-based level indices; downstream summary helpers sort and convert as needed.
- State labels are not free-form strings. `ImportSLJM` parses them into integer arrays using a canonicalized `label_key` string, and half-integer quantum numbers are stored as doubled integers. If label parsing changes, update `import_sljm.py`, the label formatting logic in `cfl_util.py`, and any state-label-based `ExData` callers together.
- `Tensor` objects should usually be created from `ImportSLJM` or from tensor arithmetic, not by manually assembling low-level buffers. The Cython layer expects contiguous NumPy arrays and Hermitian CSR-style sparse data.
- Tensor names are semantically important. Arithmetic-created tensors can have only an `arith_name` until a real name is assigned; that name is used in summaries and coefficient sharing logic, so explicitly set `tensor.name` when reusing derived tensors across fits.
- The importer synthesizes convenience tensors such as `MAGX`, `MAGY`, `MAGZ`, and `HYP` from raw SLJM tensors. If you change tensor naming or available inputs, check these derived aliases too.
- Build configuration is controlled partly through environment variables, not only code changes. `CFL_CFLAGS` and `CFL_LDLIBS` inject nonstandard include/library paths, while `CFL_CC=icc` and `INTEL_PATH=...` switch the C build to the Intel/MKL path described in `doc/install.rst`.
- This repository still contains Python 2/early Python 3 transition idioms (`from __future__ import division`, `np.complex`, `iteritems` in Cython/Python glue). When modernizing code, inspect neighboring modules for compatibility assumptions instead of normalizing one file in isolation.
