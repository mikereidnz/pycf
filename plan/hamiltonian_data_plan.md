# Plan: Better Hamltonian labelling and listing of experimental and calculated energies. 

## 1. Purpose and Scope

1. Give better labelling of Hamiltonians in the output of functions such as gen_e_summary. 
2. Provide easily accessible listing of the calculated and experimental energies and weights. Note that some data is absolute energies and some is differences. It would be helpful to have those in the output arrays, i.e. the single level or the two levels between which the difference is taken to get the calculated value. 
3. Clearly it is easy to calculate differences and differences squared between the calculated and experimental given this out. 
4. Optional calculation of the derivatives of each experimental value (absolute or difference) with respect to each varied parameter the "param" fed into the minimization functions. This would have to be done by taking a difference between energies calculated with one of the parameters increased then decreased, i.e. (E(p + deltaP) - E(p - deltaP)) / (2 deltaP). 
5. The derivatives would allow a calculation of the variance-covariance matrix and hence an estimate of parameter uncertainties. 
6. These new functions would also allow the user to write new minimization approaches at the python level.
7. New printing functions for these new arrasy may be helpful for users. 

Issues to consider include the data format. 
  - Do we have individual arrays for each Hamiltonian or a consolidated list?
  - How do we represent the states used for absolute energies or diffences? Probably by the state number (1, 2, 3...)

Apart from adding an optional label to the Hamiltonian class, and some minor modifications to output functions, this work should not have to change current code. 

As in previous work (see other files in /plan such as import_tensor_plan.md)
1. This file should be modified to produce a more detailed plan. 
2. A file called hamiltonian_data_report.md should be constructed during the work. 
3. A file calle hamiltonian_data_diffs.md should be constructed at the end. 

## 2. Goals

G1. Allow the user to attach a human-readable label (e.g. `"Site 1"`,
    `"B || c"`) to a `Hamiltonian` so multi-Hamiltonian summary output
    can be unambiguously identified.

G2. Provide a structured, Python-side accessor that returns, per
    Hamiltonian, the calculated and experimental energy data in a form
    that:
      - includes both absolute and difference-type observations,
      - records which level index (or pair of level indices) each
        observation refers to,
      - includes the per-observation weight,
      - allows trivial computation of residuals and χ² in pure Python.

G3. Provide an optional facility to compute the finite-difference
    Jacobian J_iα = ∂E_i / ∂p_α at a given parameter point, using
    central differences (E(p+δ) − E(p−δ)) / (2 δ) over the *real-valued*
    parameter vector that the C minimizer already uses internally
    (so complex parameters are split into Re/Im, matching the existing
    ordering used by `gsl_nls`'s covariance matrix).

G4. From the Jacobian, optionally compute the variance–covariance
    matrix C = σ² (Jᵀ W J)⁻¹ (with σ² = χ²/(N − M)) so that parameter
    uncertainties become available to *any* fitting routine, not just
    `gsl_nls`.

G5. Provide pretty-printers for the new structured arrays that produce
    output consistent with the existing `gen_e_summary` /
    `gen_fit_summary` style.

G6. Make all the above work for both single-Hamiltonian (`EFit`) and
    multi-Hamiltonian (`MHFit`) fits.

## 3. Non-Goals

NG1. Replacing or rewriting `gen_e_summary` / `gen_fit_summary`. The
     new accessors are additional, not substitutional. Existing call
     sites continue to work unchanged.

NG2. Adding spin-Hamiltonian (`ESHFit`, `MESHFit`) support in this
     iteration. The same design will extend naturally, but those fit
     runners carry extra observables (g-tensors, A-tensors) whose data
     model is a follow-up.

NG3. Implementing a new minimizer in pure Python. We expose the data
     and derivatives that *enable* user-written minimizers; we do not
     write one ourselves here.

NG4. Changing the on-disk file formats consumed by `ImportSLJM` /
     `ImportTensors`.

NG5. ~~Changing the C layer.~~ A small additive change to the NLS
     driver in `cfl/src/cfl_min.c` is now in scope (see §5.4) so the
     Jacobian GSL already computes can be returned to Python instead
     of being discarded. The change is strictly additive: existing
     C callers passing `NULL` for the new `jac` argument see no
     behaviour change.

## 4. Current Code Anatomy

### 4.1 `cdef class Hamiltonian` (`pycf/cfl.pyx:416`)

- Public attributes today: `n`, `nt`, `tensors`, `coeff_dict`, `coeff`,
  `w`, `z`, `h_cap`. No name/label.
- `coeff` is a writable NumPy array used by `set_coeffs()` to feed the
  C `zh_set` routine. The Hamiltonian's `diag()` populates `w`, `z`.
- The class is iterable over its constituent tensors.

### 4.2 `cdef class ExData` (`pycf/cfl.pyx:1307`)

- Holds: `e` (absolute then difference, concatenated), `w` (per-level
  weights), `la` (level indices for absolute, 0-based), `ild`/`fld`
  (initial/final indices for difference), plus state-label variants
  `lah`/`ildh`/`fldh` for the `'AS'`/`'DS'` modes.
- `n_obs = n_a + n_d`. The level indices the user wants to see are
  already stored here in 0-based form.

### 4.3 Fit runners

- `EFit` (`cfl.pyx:1672`) — single Hamiltonian, single ExData. Owns
  `self.parameters`, `self.coeff` (initial values), `self.x0` (real
  vector handed to the minimizer), `self.param_types` (per-name
  `'r'`|`'c'`).
- `MHFit` (`cfl.pyx:1908`) — list of `(h, ex)` pairs sharing a global
  parameter list, with per-Hamiltonian scalar weighting.
- `ESHFit`, `MESHFit` — out of scope this iteration.

### 4.4 Existing summary path (`pycf/cfl_util.py`)

- `gen_e_summary(w, z, labels, label_key, ex=…)` formats one
  Hamiltonian's eigenvalues against an optional `ExData`. It already
  knows how to align by level index (`la`) but currently only handles
  absolute data; difference observations are silently dropped from the
  printed table.
- `gen_fit_summary` already prints σ_α from a covariance matrix when
  one is supplied via the `covar=` kwarg, but the *only* code path that
  produces such a matrix today is `gsl_nls`.

### 4.5 Covariance precedent (`cfl.pyx:3441–3465`)

- For `gsl_nls`, a `(n_p_real, n_p_real)` covariance matrix is
  allocated and filled by the C nonlinear-least-squares driver. The
  matrix is stored in real-valued parameter order, with complex
  parameters expanded to two consecutive entries (Re, Im). We will
  match this convention exactly for the new FD-Jacobian path so
  `gen_fit_summary` works unchanged.

## 5. Proposed Design

### 5.1 Hamiltonian label

Add a single new attribute:

```python
cdef public object label   # str | None
```

- Constructor gains `label: str | None = None`.
- `Hamiltonian.label` is read/write and defaults to `None`.
- All existing public APIs remain unchanged; behaviour when
  `label is None` is identical to today.

### 5.2 Per-observation data accessor — `EData`

A new lightweight container class lives in `pycf/cfl_util.py` and is
populated by helper methods on `EFit` / `MHFit`. Conceptually it is a
flat record array of all observations attached to a fit run, in the
same order the C code internally evaluates them:

| field          | dtype   | meaning                                                                     |
|----------------|---------|-----------------------------------------------------------------------------|
| `h_index`      | int32   | Index into the fit's Hamiltonian list (`0` for `EFit`).                     |
| `h_label`      | object  | The Hamiltonian's `label` (or `f"H[{h_index}]"` if `None`).                 |
| `kind`         | 'U1'    | `'A'` (absolute) or `'D'` (difference).                                     |
| `i_lo`         | int32   | Initial level index (1-based for output). For `'A'`, set equal to the level.|
| `i_hi`         | int32   | Final level index (1-based). For `'A'`, set to `0` (sentinel).              |
| `e_calc`       | float64 | Calculated value: `w[i_lo-1]` for `'A'`; `abs(w[i_hi-1] − w[i_lo-1])` for `'D'` (matches the `fabs` in `cfl_h_fit.c:661,1112`). |
| `e_obs`        | float64 | Experimental value (from `ExData.e`).                                       |
| `weight`       | float64 | Per-level weight (`ExData.w`) ⨯ per-Hamiltonian scalar weight.              |
| `residual`     | float64 | `e_calc − e_obs`.                                                           |
| `wresidual`    | float64 | `sqrt(weight) * residual`.                                                  |

Implementation:

- `EFit.get_edata() -> EData` runs `h.diag()` if needed, then walks
  `self.ex` and constructs the structured array. 1-based indices are
  used in `i_lo`/`i_hi` because that matches the user-facing convention
  in `ExData` constructor input and in `gen_e_summary` headings. The
  internal 0-based offsets in `ExData.la` etc. are converted on the
  way out.
- `MHFit.get_edata() -> EData` concatenates the per-Hamiltonian rows
  in fit-evaluation order so the row index aligns with the C residual
  vector.
- `EData` exposes:
    - `arr`: the underlying NumPy structured array,
    - `chi2()`: returns Σ weight·residual² (matches what the C
      objective minimises),
    - `__repr__` and a `to_str()` that renders a labelled table.
- For `'AS'`/`'DS'` (state-label-indexed) ExData, `i_lo` and `i_hi`
  carry the resolved 1-based level indices; the original state labels
  remain available on the source `ExData` for users who need them.

### 5.3 Finite-difference Jacobian — `EFit.fd_jacobian` / `MHFit.fd_jacobian`

A central-difference Jacobian computed in Python via repeated calls
to the existing C objective. This path is *universal* — works
regardless of which minimizer was used (or whether one was used at
all).

**Jacobian convention (single source of truth across the codebase).**
We define the Jacobian as the derivative of the *physical calculated
energy* (or absolute-difference) `E_i(p)` w.r.t. the real-valued
parameter vector — call it `J_E`. Specifically, for each observation
row `i`:

| `kind` | `E_i(p)`                    |
|--------|-----------------------------|
| `'A'`  | `w[i_lo - 1]`               |
| `'D'`  | `abs(w[i_hi - 1] − w[i_lo − 1])`   ← matches `cfl_h_fit.c:661,1112` exactly |

This convention:

- gives the user a Jacobian whose units are physical (cm⁻¹ per parameter unit);
- matches the C objective's `fabs(...)` exactly so FD residuals reproduce
  `echisq` to FD precision;
- decouples the Jacobian from per-observation weights, which simplifies
  covariance algebra (§5.5) and the GSL-Jacobian conversion (§5.4).

```python
def fd_jacobian(self, x=None, *, delta=None, rel_delta=1e-5, atol=1e-8,
                check_swaps=True):
    """
    Return J_E of shape (n_obs, n_p_real) at parameter vector x using
    central differences of E_i(p) (see §5.3 for the definition of E_i).
    If x is None, uses self.x0.

    delta : np.ndarray of shape (n_p_real,) | float | None
        Absolute step. If None, step is computed as
        max(rel_delta * abs(x_alpha), atol).
    check_swaps : bool
        If True, after computing each column raise a UserWarning when
        max|J[:, alpha]| / (per-row energy scale) exceeds a heuristic
        threshold (~1/delta_alpha), which is the signature of an
        eigenvalue swap or near-degeneracy across the FD step.
    """
```

- Restores parameters when finished using a `_temporary_x()` context
  manager (§5.7) that round-trips both the Python `x0`/`coeff` and
  the C-side Hamiltonian coefficients atomically.
- Returns a plain `np.ndarray` so users can post-process freely.
- Computational cost is `2 * n_p_real * cost(diag)`. Documented as
  expensive — appropriate for end-of-fit uncertainty estimation, not
  for inner-loop use.

**FD step rationale.** Central differences have truncation error
`O(δ²)` and roundoff error `O(eps_diag / δ)`. For double precision
eigensolver noise of ~1e-12 in the energies, the optimum is around
`δ ≈ (eps_diag)^{1/3} ≈ 1e-4 * |p|`. We default lower (`rel_delta=1e-5`,
`atol=1e-8`) because in practice eigenvalue ordering is more sensitive
than absolute eigenvalue noise, so a smaller step reduces the chance of
crossing a degeneracy.

### 5.4 Reusing the GSL-computed Jacobian (gsl_nls only)

When the most recent fit was performed with `method='gsl_nls'`, GSL
has *already* built a finite-difference Jacobian internally during
the trust-region iterations. Today this Jacobian is consumed by
`gsl_multifit_nlinear_covar()` at `cfl/src/cfl_min.c:744–746` and the
workspace is freed before Python regains control, so the matrix is
discarded.

**Important convention bridge.** GSL's Jacobian is `J_y = ∂y/∂p`,
where `y_i = sqrt(w_i) * (E_i(p) − e_i^obs)` is the *weighted residual*
returned by `nls_echisq` (`cfl/src/cfl_h_fit.c:1105-1112`). To preserve
the single-source-of-truth Jacobian convention from §5.3, we convert
`J_y → J_E` before storing on the Python side:

```
J_E[i, :] = J_y[i, :] / sqrt(w_i)              (rows where w_i > 0)
J_E[i, :] = 0          (rows where w_i == 0; UserWarning)
```

We expose it cheaply:

- **C side** (`cfl/include/cfl_min.h`, `cfl/src/cfl_min.c`): the
  `cfl_min_run`-style entry point that drives the NLS path gains an
  optional output parameter
  ```c
  double *jac   /* may be NULL; if non-NULL, n*p doubles, row-major,
                   in GSL's weighted-residual convention */
  ```
  populated by a `gsl_matrix_memcpy`-equivalent (loop over rows since
  `gsl_multifit_nlinear_jac` returns a `gsl_matrix *` whose stride
  may differ from a packed `(n_obs, n_p_real)` array) *before*
  `gsl_multifit_nlinear_free`.
- **Cython side** (`pycf/cfl.pyx` ~line 3441 — the `gsl_nls` branch):
  allocate a `(n_obs, n_p_real)` `np.float64` array alongside the
  existing `covar`, pass its pointer to the C call. After the call,
  divide each row by `sqrt(weight_i)` in Python to convert to `J_E`,
  and stash on the `CFLMin` instance / Fit runner.
- **Public surface**: `EFit.last_jacobian: np.ndarray | None` and
  `MHFit.last_jacobian: np.ndarray | None`. These are always in the
  `J_E` (unweighted-energy) convention — same as `fd_jacobian()`
  output — so callers never need to know which path produced them.

This is purely additive — no behaviour change for callers that
ignore the new attribute.

### 5.5 Covariance helper — `EFit.covariance` / `MHFit.covariance`

```python
def covariance(self, x=None, *, jacobian=None, scale="reduced_chi2",
               **fd_kwargs):
    """
    Returns (cov, sigma_alpha, edata) where:
      cov           : np.ndarray, shape (n_p_real, n_p_real)
      sigma_alpha   : np.ndarray, shape (n_p_real,)  -- sqrt(diag(cov))
      edata         : the EData snapshot used.

    Always works with the J_E convention (energy Jacobian).
    Builds W = diag(EData.weight) and computes:

        N = pinv(J_E.T @ W @ J_E)        # the inverse normal matrix

    Then:

        scale="unscaled"      ->  cov = N
                                  (matches gsl_multifit_nlinear_covar
                                  exactly when J_E was derived from GSL.)
        scale="reduced_chi2"  ->  cov = (chi2 / max(N_obs - M, 1)) * N
                                  (default; conventional when weights
                                  are relative rather than 1/variance.)

    By default, when `jacobian` is not supplied, this method uses
    `self.last_jacobian` (cached by the most recent gsl_nls fit) if
    available; otherwise it falls back to computing a fresh Jacobian
    via `self.fd_jacobian(...)`.
    """
```

- Uses `np.linalg.pinv` on `J_E.T @ W @ J_E` so a rank-deficient
  Jacobian (under-constrained parameter) yields the Moore–Penrose
  pseudo-inverse with a UserWarning rather than `LinAlgError`.
- The returned `cov` is in the same Re/Im-expanded parameter ordering
  used by `gsl_nls`, so `scale="unscaled"` output can be passed directly
  as `gen_fit_summary(..., covar=cov)` and reproduce today's σ values
  exactly.

### 5.6 New printers (`pycf/cfl_util.py`)

- `gen_edata_summary(edata) -> str` — single tabular section per
  Hamiltonian, headed by the Hamiltonian label. Columns:
  Lev., Kind, Theory, Experiment, Difference, Weighted residual, w.
- `gen_e_summary` is updated minimally to:
    - Print the Hamiltonian's `label` in its heading when present.
    - Continue working unchanged when `label is None`.
- No change to `gen_fit_summary`'s public API.

### 5.7 Internal helpers

- `_temporary_x(fit_obj, x)` — context manager that:
    - snapshots `fit_obj.x0` and `fit_obj.coeff`;
    - writes the new `x` into both Python state *and* into the C-side
      Hamiltonian coefficients via the same `parse_param_data` path
      the C objective uses (so a subsequent `h.diag()` sees the right
      coefficients);
    - on exit, restores both Python and C state byte-for-byte.

  This is the single chokepoint that `fd_jacobian()` and
  `get_edata(x=…)` both go through. Defining it once eliminates a
  whole class of "stale Hamiltonian state" bugs.

### 5.8 Scope: state-label-indexed observations (`'AS'`, `'DS'`)

The C residual code resolves state-label observations to current
sorted-eigenvalue indices via `find_sort_indices()` at every objective
evaluation (`cfl/src/cfl_h_fit.c:1122-1146`). Replicating that in
Python without introducing a new C entry point is fragile — the
mapping is allowed to change every parameter step.

For the first iteration, `EFit.get_edata` / `MHFit.get_edata` /
`fd_jacobian` / `covariance` all **raise `NotImplementedError`** when
the input `ExData.sl_index == 1`. Numeric-index `'A'` / `'D'` /
mixed `('A','D')` data is fully supported.

State-label support is queued as Future Work F5; it requires either
exposing `find_sort_indices()` to Cython or routing the state-label
resolution through a new objective-shaped helper that returns
`(la, ild, fld)` for current eigenvectors.

## 6. Test Plan

All new tests live under `tests/unit/`:

T1. `test_hamiltonian_label.py`
    - Default label is `None`.
    - Constructor accepts a string label.
    - `h.label = "x"` is read/write.
    - `gen_e_summary` heading includes the label when set; omits the
      "Hamiltonian:" line when `label is None` (regression guard for
      existing output).

T2. `test_edata_efit.py`
    - Build a small (≤8-state) Hamiltonian + `'A'`-only ExData.
    - `EFit.get_edata()` row count == `ex.n_obs`, all rows `kind='A'`,
      `i_lo` matches input level numbers, `e_calc == h.w[i_lo-1]`.
    - Same again for `'D'`-only and for the mixed `('A','D')` tuple
      form. Construct one `'D'` row where `w[i_hi]<w[i_lo]` to verify
      `e_calc == abs(w[i_hi]-w[i_lo])`, matching the C `fabs`.
    - `edata.chi2()` matches the value the C objective produces (call
      `min_object.minimize` with method='gsl_nmsimplex2' niter=1 and
      compare).
    - `'AS'`/`'DS'` ExData: assert `EFit.get_edata()` raises
      `NotImplementedError` with a clear message (until F5 lands).

T3. `test_edata_mhfit.py`
    - Two-Hamiltonian fit (e.g. ground + excited as in eryso, or a
      synthetic toy) with distinct labels and per-H ExData.
    - `MHFit.get_edata()` rows are concatenated in `(H0, H1)` order
      and carry the right `h_index`/`h_label`.
    - Per-Hamiltonian scalar weighting is reflected in the `weight`
      column.

T4. `test_fd_jacobian.py`
    - Construct a problem where one parameter has an analytically known
      zero derivative w.r.t. all energies (e.g. an irrelevant complex
      phase param), assert that column of J is `~0` to FD precision.
    - Compare the FD Jacobian columns for a real-valued CF parameter
      against the GSL covariance run: rebuilding `JᵀWJ` from FD and
      from `inv(cov_gsl)/sigma2_gsl` should agree to ~1e-3 relative.
    - `delta=` explicit override is honoured per parameter.

T5. `test_covariance.py`
    - Construct a small fit. Run `gsl_nls` to convergence so we have
      both `gen_fit_summary`'s historical `covar` (via
      `gsl_multifit_nlinear_covar`) and our new `last_jacobian`.
    - With `scale="unscaled"` and the GSL-derived `last_jacobian`,
      `EFit.covariance()` should reproduce the historical `covar`
      diagonals to ~1e-6 relative (same algebra, same data).
    - With `scale="reduced_chi2"` (default) and the same Jacobian,
      `cov_diag` should equal `unscaled_diag * chi2 / (N - M)`.
    - With non-unit per-Hamiltonian weights in MHFit, build a toy
      problem where the analytic JᵀWJ is known (a 1-parameter linear
      model in the energies) and confirm both the FD and GSL paths
      reproduce the analytic covariance to a few percent.
    - Underdetermined case (`n_p_real > n_obs`, with `ignore_ndof=True`)
      issues a UserWarning and returns the pseudo-inverse covariance
      without raising.

T6. `test_gen_edata_summary.py`
    - Snapshot of formatted output for a fixed small fit. Mirrors the
      style of the existing `tests/unit/test_cfl_util_summaries.py`.

T7. Integration smoke test (`tests/integration/`):
    - The eryso example, which already fits two Hamiltonians on a
      single site (`h1` ground state, `h2` excited-state hyperfine),
      run end-to-end. After the fit, call `MHFit.get_edata()` and
      `MHFit.covariance()`, asserting the printed σ values match the
      gold-file output. Per-Hamiltonian labels (`"GS"`, `"ES HFS"`)
      should appear in the summary.

## 7. Implementation Steps

S1. Add `label` to `Hamiltonian` (`cfl.pyx`); update its docstring;
    update one example (eryso) to set it. Tests: T1.

S2. Implement `EData` container in `cfl_util.py`. Pure-Python, no
    Cython changes. Tests: a unit test on the container alone.

S3. Implement `EFit.get_edata()`. Tests: T2.

S4. Implement `MHFit.get_edata()`. Tests: T3.

S5. Implement `EFit.fd_jacobian()` and `MHFit.fd_jacobian()` sharing a
    helper that drives the existing parameter→x0→C-objective path.
    Tests: T4.

S5a. Expose the GSL-computed Jacobian (§5.4):
     - Extend the C signature in `cfl/include/cfl_min.h` and the NLS
       driver in `cfl/src/cfl_min.c` with an optional `double *jac`
       output buffer.
     - Plumb the buffer through `pycf/cfl.pxd` and the `gsl_nls`
       branch in `cfl.pyx` so a Python-side `np.ndarray` is allocated
       and populated.
     - Set `EFit.last_jacobian` / `MHFit.last_jacobian` after a
       successful `gsl_nls` fit.
     - Tests: a focused test in `test_fd_jacobian.py` asserting
       `last_jacobian` matches `fd_jacobian()` to within FD precision
       at the converged point.

S6. Implement `.covariance()` on both runners, preferring
    `last_jacobian` when present and falling back to FD. Tests: T5.

S7. Add `gen_edata_summary` and update `gen_e_summary` heading for
    labels. Tests: T6.

S8. Update one example (eryso `mesh_fit.py`, which already drives a
    two-Hamiltonian fit — ground state + excited-state hyperfine — on
    a single crystal site) to demonstrate the new workflow end-to-end,
    including labels on `h1` and `h2`. Tests: T7.

S9. Update `docs/api/index.rst` with the new public symbols.

S10. Backfill `CHANGELOG.md` and write `hamiltonian_data_report.md`
     summarising the work, then `hamiltonian_data_diffs.md` listing
     the file-by-file diffs for review.

Each step ends with `make -C cfl test && python -m pytest tests/ -q`
and a logical commit on a feature branch
`feat/hamiltonian-data-accessors`.

## 8. Risks and Mitigations

R1. **FD Jacobian noise.** Energies near eigenvalue degeneracies move
    non-smoothly with parameters, so a single small `delta` can give a
    poor estimate. Mitigation: default to relative step
    `1e-4 * |p_α|` with an `atol=1e-6` floor, and document the option
    to override per parameter. Add an explicit warning in the docstring.

R2. **Eigenvalue ordering across FD steps.** A perturbation can swap
    adjacent eigenvalues; differencing then samples two different
    states and produces a spurious huge derivative. Mitigation:
    `fd_jacobian(check_swaps=True)` (default) computes a per-column
    heuristic — `max|J[:,α]| * δ_α / (max(|w|) − min(|w|))` — and emits
    a `UserWarning` for any column whose value exceeds a tunable
    threshold (default 0.5), suggesting the user reduce `delta` for
    that parameter or move away from the suspected near-degeneracy.
    Documented as a *diagnostic*, not a guarantee of correctness near
    level crossings.

R3. **Performance.** `n_obs` × diag-cost is fine; `2 n_p_real`
    diag-calls is the dominant cost. For typical fits with ~20
    parameters and small Hamiltonians this is negligible; for large
    Hamiltonians the user is paying ~40 diags. Document; do not
    parallelise in this iteration.

R4. **Backward compatibility.** Adding `label` to the Hamiltonian
    constructor as a keyword-only `label=None` argument keeps existing
    positional calls working. Adding methods to `EFit` / `MHFit` is
    additive.

R5. **Underdetermined fits.** `JᵀWJ` may be singular. Use `pinv` and
    issue a UserWarning rather than failing.

## 9. Deliverables

D1. `pycf/cfl.pyx` — `Hamiltonian.label`, `EFit.get_edata`,
    `EFit.fd_jacobian`, `EFit.last_jacobian`, `EFit.covariance`,
    `MHFit.get_edata`, `MHFit.fd_jacobian`, `MHFit.last_jacobian`,
    `MHFit.covariance`.

D1a. `cfl/include/cfl_min.h`, `cfl/src/cfl_min.c`, `pycf/cfl.pxd` —
     optional `double *jac` output buffer on the NLS driver, copied
     out of `gsl_multifit_nlinear_jac` before workspace teardown.

D2. `pycf/cfl_util.py` — `EData`, `gen_edata_summary`, label handling
    in `gen_e_summary`.

D3. New tests T1–T6 under `tests/unit/`, integration test T7.

D4. Updated `docs/api/index.rst`.

D5. Updated `examples/eryso/mesh_fit.py` (or a new sibling example)
    showing the end-to-end workflow.

D6. `plan/hamiltonian_data_report.md` (running narrative produced
    while the work is in flight).

D7. `plan/hamiltonian_data_diffs.md` (final file-by-file diff summary).

D8. `CHANGELOG.md` entry under the next-release section.

## 10. Out of Scope (Future Work)

F1. `ESHFit` / `MESHFit` extension: the same `EData` shape can be
    augmented with `kind='SH'` rows for spin-Hamiltonian observables.

F2. Pure-Python minimizers built on top of `EData` / `fd_jacobian`
    (e.g. a Levenberg–Marquardt that tolerates user-side bounds
    transforms or parameter sharing).

F3. Analytic derivatives via Hellmann–Feynman: ∂E_i/∂p_α =
    ⟨z_i| T_α |z_i⟩, which requires no extra diagonalisations and is
    immune to the eigenvalue-swap problem in R2. A natural follow-up
    once the data plumbing in this plan is in place.

F5. State-label-indexed (`'AS'`, `'DS'`) support for `get_edata`,
    `fd_jacobian`, and `covariance`. Requires exposing
    `find_sort_indices()` (or equivalent) to Cython so the per-x
    label→index resolution can be replicated in Python in lockstep
    with the C objective.
