# Changelog

All notable changes to PyCF are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Hamiltonian data accessors.** New Python-side helpers for
  inspecting fit residuals and parameter uncertainties without
  reaching into C internals:
  - `cfl.Hamiltonian.label`: optional human-readable label, rendered
    in `gen_summary()` headings and copied into every row of the
    `EData` table returned by `EFit.get_edata` / `MHFit.get_edata`.
  - `cfl_util.EData`: structured-array container with `.arr`,
    `.chi2()`, `.to_str()`, and `__len__`. Row order matches the
    objective vector consumed by `gsl_nls`, so row indices align
    with Jacobian rows and `last_jacobian` columns.
  - `EFit.get_edata()` / `MHFit.get_edata()`: snapshot of calculated
    and experimental energies (absolute and difference observations),
    weights, residuals, and weighted residuals at the current
    parameter point.
  - `EFit.fd_jacobian()` / `MHFit.fd_jacobian()`: central-difference
    energy Jacobian with optional swap-detection warning. Caches the
    result on `last_jacobian`.
  - `EFit.last_jacobian` / `MHFit.last_jacobian`: also populated by
    `gsl_nls` fits with the convergence Jacobian extracted from GSL
    (see `cfl_min.c::cfl_gsl_nls_setup`).
  - `EFit.covariance()` / `MHFit.covariance()`: variance-covariance
    matrix from `pinv(JᵀWJ)`, with `scale="reduced_chi2"` (default)
    or `"unscaled"` (matches GSL convention). Falls back to
    `last_jacobian` and warns on rank deficiency.
  - `cfl_util.gen_edata_summary()`: pretty-printer for `EData`.
  - **State-label observation support.** `EFit.get_edata()` /
    `MHFit.get_edata()` (and consequently `fd_jacobian` / `covariance`)
    now handle `'AS'` and `'DS'` ExData by reproducing the
    principal-component label-matching used by the C objective.  The
    `kind` field of the EData dtype is widened to `U2` so the rows
    can carry `'AS'`/`'DS'` instead of just `'A'`/`'D'`.
- **`pycf.pyfit.PyFit`**: pure-Python fitting wrapper around
  `scipy.optimize.least_squares` that drives an `EFit`/`MHFit`
  through its `EData` residual vector.  Provides `residuals(x)`,
  `chi2(x)`, `jacobian(x)` (weighted FD residual Jacobian),
  `fit(method=..., bounds=..., jac=...)` (with ``jac='pycf'`` to
  use the pycf FD helper), `covariance()` and `stderr()` for
  one-sigma parameter uncertainties at the optimum, and `fit_res(...)`
  for an `e_fit`/`mh_fit`-style summary + result payload. Parameter
  perturbations go through `_temporary_x` so the wrapped fit's
  persistent state is preserved.  Useful for irrep-aware extensions,
  bound-constrained fits, and any custom Python-side residual logic
  that doesn't yet exist in the C code.  See
  `examples/ceylf/pyfit_example.py` for a worked Ce:YLF fit.
- **Unified fit-output controls** across energy and intensity wrappers:
  `calculate_sigma`, `include_covariance`, and `include_jacobian`.
  `e_fit`/`mh_fit` now return `all_coeff`, `sigma`, and Jacobian
  diagnostics in `res`, and summaries now include an all-parameter
  table with fitted/fixed status.

### Changed (breaking)
- Seniority `label_key` letter renamed from `T` to `X` in `ImportSLJM`,
  `ExData`, and `States`, matching Nielson-Koster notation. User code
  passing a `label_key` containing `T` (e.g. `'TLJM'`) for seniority
  must be updated to use `X` (e.g. `'XLJM'`). All bundled examples and
  tests use `'SLJM'` and are unaffected.

### Added
- `pycf.import_sljm.ImportTensors`: in-memory wrapper that accepts
  user-supplied tensor matrices (NumPy arrays or SciPy sparse) and
  state labels, producing the same `cfl.Tensor` / `cfl.StateLabels`
  objects as `ImportSLJM` without any file I/O. Useful for unit
  testing and for sourcing matrix elements from non-jmcalc backends.
  `ImportSLJM` now delegates its tensor-wrapping path to
  `ImportTensors`, removing duplicated logic.
- Spin-half eigenvector regression test
  (`tests/unit/test_import_tensors.py::test_spin_half_eigenvectors_match_numpy`)
  that guards against the LAPACK transpose-vs-conjugate regression
  fixed by the conjugation in `solve_hermitian_block`
  (`cfl/src/cfl_h.c:160`).
- SpinH round-trip and `njsymbols`-vs-`sympy` unit tests; closed
  cfl_util / spinh coverage gaps identified in the prior audit.
- Comprehensive Sphinx-based API documentation framework
- GitHub Actions CI with AddressSanitizer and UndefinedBehaviorSanitizer
- Type hints for 100% of core modules (8 modules, 130+ annotations)
- CONTRIBUTING.md for developer guidelines
- CHANGELOG.md for version history
- mypy type checking in CI pipeline
- Module docstrings for all 8 core modules with workflow examples

### Changed
- Minimum supported Python is now 3.10; drop 3.8 and 3.9. `scipy>=1.15`
  is now required. CI matrix expanded with Python 3.13.
- Reorganised the test tree into `tests/unit/` and `tests/integration/`
  subdirectories; `matel/` fixture data moved with its parent
  integration tests.

### Improved
- Standardized C test tolerances to use shared `TEST_TOLERANCE` constant
- Enhanced module-level documentation with comprehensive docstrings
- Improved Cython error handling in Tensor class
- Better exception catching in PyCapsule validation
- pre-commit is now a CI gate (coverage threshold + lint enforcement
  enabled in CI).

### Fixed
- Fixed scipy.special.sph_harm deprecation (migrated to sph_harm_y)
- Corrected an effective transpose in `solve_hermitian_block`: the
  call into LAPACK's Hermitian eigensolver previously delivered the
  transpose of the input block. Eigenvectors of complex Hamiltonians
  with non-zero imaginary off-diagonals are now numerically correct.
- Fixed NLS double-weighting in `cfl.pyx`: GSL `wts` is now an
  all-ones array, since `nls_echisq` already encodes the full per-
  residual weight as `sqrt(w)*(calc-obs)`. Previously the per-
  Hamiltonian weight was effectively squared compared to the
  gradient-based methods, making NLS inconsistent with `echisq` for
  non-uniform `weights_list` values. Default error cases were also
  added to `cfl_gsl_min_setup` and `cfl_nlopt_min_setup` so an
  invalid algorithm enum returns `NULL` rather than dispatching
  through uninitialised function pointers.
- Fixed eigenvector percentage in `cfl_util` summaries: use
  `|z|^2 / sum(|z|^2)` (probability weight) instead of
  `|z| / sum(|z|)` (amplitude ratio). For a component
  `z = -0.78+0.38j` this changes the reported weight from 58% to the
  correct 75.3%. Applied at all four call sites.
- `CFLMin.fit(dry_run=True)` now evaluates the objective function at
  the initial parameters and returns the resulting `fmin`, instead
  of returning `fmin=0`. Users can now use a dry run to inspect how
  well the initial parameters fit the data without invoking the
  minimiser.
- Improved error messages and validation in crystal field functions
- Better parameter bounds checking in intensity calculations

### Documentation
- Added complete Sphinx documentation with autodoc
- Created comprehensive guides for parameter fitting and spin-Hamiltonian extraction
- Added quickstart guide and overview of crystal field theory
- Enhanced README.rst with clearer descriptions
- Created detailed installation guide with platform-specific instructions

## [0.1.0] - 2026-04-25

### Added
- Initial Python package structure with Cython wrapper
- Core crystal field calculation engine (cfl module)
- Tensor algebra and matrix element calculations
- Hamiltonian construction and diagonalization
- Intensity calculation framework (magnetic and electric dipole transitions)
- Parameter calculation helpers for crystal field terms
- Data import utilities for SLJM/EMP format
- Spin Hamiltonian extraction tools
- EMP program wrapper (cfit, inten, vtrans, spectrum)
- Comprehensive Python test suite (83 tests)
- C test suite (24 tests) with LAPACK/BLAS integration
- Type hints for all core modules
- Example workflows (CeYLF, ErYSO, SmNaCaF₂)

### Features
- Hamiltonian class for crystal field calculations
- Tensor sparse matrix support (CSR format)
- ExData class for experimental data handling
- Multiple experimental modes: absorption, emission, spin-selective
- Wigner symbol calculations (3j, 6j, 9j)
- Matrix element calculations for tensor operators
- Transition intensity calculations with Lorentzian lineshapes
- Temperature-dependent Boltzmann populations
- Fitting infrastructure with parameter bounds

### Architecture
- Performance-critical C99 library (cfl/)
- Cython wrapper for Python integration
- BLAS/LAPACK backend for matrix operations
- Intel MKL support via environment variables
- Pure Python utility modules with clear separation of concerns

### Documentation
- Comprehensive docstrings in all modules
- Example scripts showing typical workflows
- Installation guide with platform support matrix
- Crystal field theory overview
- API reference with full function signatures

## Version History

### Previous Versions
PyCF was previously developed as a standalone C library with Perl scripts.
This Python package represents the modernization and expansion of that work,
adding:
- Professional Python packaging
- Comprehensive test coverage
- Type hints and better error handling
- Modern documentation and examples
- CI/CD infrastructure

---

## How to Report Changes

When contributing changes, please:

1. **Update this file** in the appropriate section:
   - Added: New features
   - Improved: Enhancements to existing features
   - Fixed: Bug fixes
   - Removed: Removed features
   - Deprecated: Features marked for future removal
   - Security: Security vulnerability fixes

2. **Use clear, user-focused language**:
   - ✅ "Add type hints to 8 core modules"
   - ❌ "Add typing imports to utils"

3. **Reference relevant issues/PRs**:
   - ✅ "Fix memory leak in Hamiltonian (#123)"
   - ❌ "Fix memory leak"

4. **Follow the format** for consistency

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

- **Major version** (X.0.0): Breaking API changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes only

### Checklist for Release
- [ ] Update CHANGELOG.md with all changes
- [ ] Update version in pycf/__version__.py
- [ ] Ensure all tests pass (pytest, C tests, mypy)
- [ ] Build documentation and verify clean build
- [ ] Merge changes to main branch
- [ ] Create git tag: `git tag -a v0.X.X -m "Release 0.X.X"`
- [ ] Push tag: `git push origin v0.X.X`
- [ ] Build distribution: `python setup.py sdist bdist_wheel`
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create GitHub release with changelog excerpt

## Migration Guides

### Migrating from Version X to Y
(Document any breaking changes and how users should update their code)

---

## See Also

- [Development Guide](CONTRIBUTING.md)
- [Installation Instructions](docs/installation.rst)
- [API Reference](docs/api/index.rst)
- [GitHub Releases](https://github.com/mikereidnz/pycf/releases)
