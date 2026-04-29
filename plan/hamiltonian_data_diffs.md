# Hamiltonian data accessors — file-by-file diff summary

Branch `feat/hamiltonian-data-accessors` against `devel`.

## Diff stat

```
 cfl/include/cfl_min.h                |   4 +-
 cfl/src/cfl_min.c                    |  18 +-
 cfl/tests/nls_test.c                 |   2 +-
 examples/eryso/mesh_fit.py           |   6 +
 plan/hamiltonian_data_plan.md        | 576 ++++++++++++++++++++++++
 plan/pycf_long_term_plan.md          |   1 +
 pycf/cfl.pxd                         |   3 +-
 pycf/cfl.pyx                         | 492 +++++++++++++++++++-
 pycf/cfl_util.py                     | 169 +++++++-
 tests/unit/test_covariance.py        | 105 +++++
 tests/unit/test_e_summary_label.py   |  46 +
 tests/unit/test_edata_container.py   | 142 ++++++
 tests/unit/test_efit_get_edata.py    | 150 ++++++
 tests/unit/test_fd_jacobian.py       | 226 +++++++++
 tests/unit/test_gsl_jacobian.py      |  49 +
 tests/unit/test_hamiltonian_label.py |  49 +
 tests/unit/test_mhfit_get_edata.py   | 111 ++++
 17 files changed, 2141 insertions(+), 8 deletions(-)
```

## Per-file notes

### C library

- **`cfl/include/cfl_min.h`**
  - `cfl_nls_data` gains a `double *jac` field. Caller-owned, row-major
    `(n_obs, n_p_real)`.
  - `cfl_gsl_nls_setup` signature gains a `double *jac` parameter.
- **`cfl/src/cfl_min.c`**
  - `cfl_gsl_nls_setup` stores `jac` on the data struct.
  - `gsl_nls_f` copies `gsl_multifit_nlinear_jac(d->w)` into `d->jac`
    after `gsl_multifit_nlinear_covar` and before
    `gsl_multifit_nlinear_free`. Skipped when `d->jac == NULL`.
- **`cfl/tests/nls_test.c`**
  - Passes `NULL` for the new parameter — no behaviour change.

### Cython bridge

- **`pycf/cfl.pxd`** — prototype updated for `cfl_gsl_nls_setup`.
- **`pycf/cfl.pyx`**
  - Module imports `warnings` and `contextlib.contextmanager`.
  - `Hamiltonian` gains `cdef public object label` (default `None`)
    and `Hamiltonian.gen_summary` auto-fills `h_label` kwarg from it.
  - New module-level helpers:
    - `_x_to_coeff_dict(fit_obj, x)`
    - `_fit_hamiltonians(fit_obj)`
    - `_temporary_x(fit_obj, x)` — context manager for safe FD steps
    - `_fd_jacobian_impl(fit_obj, ...)`
    - `_covariance_impl(fit_obj, ...)`
    - `_build_edata_for_ex(h, ex, h_index, h_weight)`
  - `EFit` and `MHFit` each gain:
    - `cdef public object last_jacobian`
    - `get_edata(self)`
    - `fd_jacobian(self, x=None, *, delta, rel_delta, atol, check_swaps)`
    - `covariance(self, x=None, *, jacobian, scale, **fd_kwargs)`
  - `CFLMin.minimize` allocates an `(n_obs, n_p_real)` ndarray, passes
    its data pointer to `cfl_gsl_nls_setup`, and stores the array on
    `self.kwargs["jac"]`.
  - `EFit.fit` / `MHFit.fit` copy `kwargs["jac"]` into `last_jacobian`
    after the C-side minimisation completes.

### Python helpers

- **`pycf/cfl_util.py`**
  - New `_EDATA_DTYPE` (structured dtype with `h_index`, `h_label`,
    `kind`, `i_lo`, `i_hi`, `e_calc`, `e_obs`, `weight`, `residual`,
    `wresidual`).
  - New `EData` class: `arr`, `chi2()`, `to_str(precision, max_rows)`,
    `__len__`, `__getitem__`, `__repr__`, classmethod `empty(n)`.
  - New `gen_edata_summary(edata, **kwargs)` wrapper.
  - `gen_e_summary` accepts optional `h_label` kwarg.
  - `gen_fit_summary` skips `"jac"` and `"covar"` when iterating
    `kwargs` (otherwise `format(ndarray)` raises `TypeError`).

### Examples

- **`examples/eryso/mesh_fit.py`** — sets `h1.label = "Site 1, ground state"`
  and `h2.label = "Site 1, hyperfine"`; docstring updated to mention
  the new accessor workflow.

### Tests

- **`tests/unit/test_hamiltonian_label.py`** — round-trip and default
  `None` for `Hamiltonian.label`.
- **`tests/unit/test_edata_container.py`** — dtype, length, chi2,
  `to_str`, `gen_edata_summary` wrapper, type validation.
- **`tests/unit/test_efit_get_edata.py`** — `'A'`, `'D'`, mixed
  `('A','D')` ExData; row order matches GSL residual concatenation;
  weights and residuals match C-side chi2.
- **`tests/unit/test_mhfit_get_edata.py`** — multi-Hamiltonian
  aggregation; per-Hamiltonian weight bake-in matches C objective.
- **`tests/unit/test_fd_jacobian.py`** — central difference on
  diagonal observations and difference observations; swap-detection
  warning; state restoration on success and on exception.
- **`tests/unit/test_gsl_jacobian.py`** — `last_jacobian` populated
  by `gsl_nls`; magnitudes match FD result at the optimum.
- **`tests/unit/test_covariance.py`** — shape, unscaled vs
  reduced-chi2, last_jacobian fallback, rank-deficient warning,
  scale validation, state restoration with explicit `x`.
- **`tests/unit/test_e_summary_label.py`** — `gen_e_summary` heading
  shows `Hamiltonian: <label>`; auto-fill from `h.label` works.

### Plan

- **`plan/hamiltonian_data_plan.md`** — full plan (added in S0 commit).
- **`plan/hamiltonian_data_report.md`** — final implementation report.
