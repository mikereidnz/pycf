# Hamiltonian data accessors — implementation report

Branch: `feat/hamiltonian-data-accessors`. See
`plan/hamiltonian_data_plan.md` for the full plan.

## Summary

Added Python-side accessors that expose the fit residual data and
parameter Jacobian/covariance information already computable from the
C objective. The work is purely additive — no existing public API
behaviour changed. All 497 existing Python tests and 24 C tests
continue to pass; 28 new tests guard the new surface.

## Steps

| Step | Commit  | Scope |
|------|---------|-------|
| S1   | 11b0ee2 | `Hamiltonian.label` (optional human-readable name). |
| S2   | cc92227 | `cfl_util.EData` container + `gen_edata_summary`. |
| S3   | e260be3 | `EFit.get_edata()` returns per-observation `EData`. |
| S4   | 00d3959 | `MHFit.get_edata()` aggregates rows from each Hamiltonian. |
| S5   | bb207c5 | `EFit.fd_jacobian` / `MHFit.fd_jacobian` (central-difference) + `last_jacobian` cache. |
| S5a  | 90e0694 | Expose GSL convergence Jacobian via new `cfl_nls_data.jac` field; `EFit.last_jacobian` / `MHFit.last_jacobian` populated after `gsl_nls`. |
| S6   | 4446840 | `EFit.covariance` / `MHFit.covariance` (`scale ∈ {"reduced_chi2", "unscaled"}`). |
| S7   | 57a8cd1 | `gen_e_summary` accepts `h_label`; `Hamiltonian.gen_summary` auto-fills from `self.label`. |
| S8   | 8f92487 | Eryso `mesh_fit.py` example updated to label `h1`/`h2`. |
| S9   | —       | Existing `automodule` directives in `docs/api/cfl.rst` and `docs/api/cfl_util.rst` already pick up the new symbols. |
| S10  | this    | CHANGELOG entry + this report. |

## Tests

New tests added (28 total):

- `tests/unit/test_hamiltonian_label.py` — label round-trip.
- `tests/unit/test_edata_container.py` — `EData` container.
- `tests/unit/test_efit_get_edata.py` — `EFit.get_edata` row layout, weights, mixed A/D.
- `tests/unit/test_mhfit_get_edata.py` — `MHFit.get_edata` aggregation.
- `tests/unit/test_fd_jacobian.py` — FD Jacobian on diagonal and difference observations, including swap-detection warning and state-restoration guarantees.
- `tests/unit/test_gsl_jacobian.py` — `last_jacobian` populated by `gsl_nls`; magnitudes match FD at the optimum.
- `tests/unit/test_covariance.py` — shape, scaling, `last_jacobian` reuse, rank-deficient warning, scale validation, state restoration with explicit `x`.
- `tests/unit/test_e_summary_label.py` — `h_label` heading and auto-fill from `Hamiltonian.label`.

## C-side changes

- `cfl/include/cfl_min.h`:
  - `cfl_nls_data` gains a `double *jac` field (caller-owned buffer, row-major shape `(n_obs, n_p_real)`).
  - `cfl_gsl_nls_setup` signature gains a `double *jac` parameter.
- `cfl/src/cfl_min.c`:
  - `gsl_nls_f` copies the GSL Jacobian into the supplied buffer
    after `gsl_multifit_nlinear_covar` and before
    `gsl_multifit_nlinear_free` (the GSL-owned pointer must not
    outlive the workspace).
- `cfl/tests/nls_test.c`: passes `NULL` for the new argument.

All 24 C tests continue to pass.

## Cython / Python changes

- `pycf/cfl.pxd`: prototype of `cfl_gsl_nls_setup` updated.
- `pycf/cfl.pyx`:
  - `Hamiltonian.label` (Python `object`, default `None`).
  - `_x_to_coeff_dict`, `_temporary_x`, `_fd_jacobian_impl`,
    `_covariance_impl`, `_build_edata_for_ex` helpers.
  - `cdef public object last_jacobian` on both `EFit` and `MHFit`.
  - `EFit.get_edata`, `EFit.fd_jacobian`, `EFit.covariance`.
  - `MHFit.get_edata`, `MHFit.fd_jacobian`, `MHFit.covariance`.
  - `CFLMin.minimize` allocates and forwards a `(n_obs, n_p_real)`
    NumPy buffer for the GSL Jacobian; result stored in
    `self.kwargs["jac"]`.
  - `EFit.fit` / `MHFit.fit` copy `kwargs["jac"]` into
    `last_jacobian` after a `gsl_nls` minimization.
- `pycf/cfl_util.py`:
  - `EData` class + `_EDATA_DTYPE`.
  - `gen_edata_summary` wrapper.
  - `gen_e_summary` honours optional `h_label` kwarg.
  - `gen_fit_summary` skips `jac` and `covar` when iterating kwargs
    (avoids `TypeError` on ndarray values).

## Out-of-scope (deferred)

- State-label-indexed observations (`'AS'` / `'DS'`):
  `get_edata` / `fd_jacobian` / `covariance` raise
  `NotImplementedError` for `ExData` with `sl_index == 1`. The C
  residual code resolves these via `find_sort_indices()` on every
  objective evaluation; replicating that on the Python side is
  fragile because the mapping can change every parameter step.
  Tracked in the long-term plan.

## Risks and mitigations

- **FD noise near degeneracies.** Default to relative step
  `1e-5 * |x|` with `atol=1e-8`. `check_swaps=True` (default) emits
  a `UserWarning` when a Jacobian column magnitude exceeds a
  heuristic threshold; tests cover swap-free input to confirm no
  spurious warnings under normal use.
- **Eigenvalue ordering across FD steps.** Same `check_swaps`
  heuristic; user can pass per-parameter `delta` arrays.
- **Rank-deficient normal matrix.** `pinv` is used unconditionally;
  a `UserWarning` is emitted when the rank is below `n_p_real`.
- **Stale Hamiltonian state.** All Python-side parameter perturbations
  go through `_temporary_x`, which snapshots and restores both
  `fit_obj.coeff` and per-Hamiltonian `coeff_dict` so an exception
  inside the FD loop does not leak modified state.

## Verification

- `make -C cfl test` — 24/24 pass.
- `python -m pytest tests/ -q` — 497 passed, 16 skipped.
