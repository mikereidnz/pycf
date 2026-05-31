# Report: Extract ImportTensors from ImportSLJM

This report tracks progress against `import_tensor_plan.md`. Sections are
appended as each phase of the work completes.

## 1. Status

| Phase | Status |
| --- | --- |
| A. Add `ImportTensors`, leave `ImportSLJM` alone | Done (`cd30b51`) |
| B. Refactor `ImportSLJM` to delegate | Implemented locally; awaiting commit |

## 2. Phase A — Adding `ImportTensors`

**Status:** Implemented and tested locally; not yet committed.

### 2.1 Files added / modified

| File | Change |
| --- | --- |
| `pycf/import_sljm.py` | Added `ImportTensors` class above `ImportSLJM`. Imports `MappingABC`, `issparse`, `triu`. `ImportSLJM` itself is unchanged. |
| `pycf/__init__.py` | Re-export `ImportSLJM` and `ImportTensors` via the existing lazy `__getattr__`; added both to `__all__`. |
| `tests/unit/test_import_tensors.py` | New, 33 tests. |

### 2.2 Test coverage in the new test module

Construction and storage convention:
- dense full Hermitian (default `storage="full"`)
- full sparse Hermitian (CSR and COO)
- pre-upper-triangulated CSR with `storage="upper"`
- mixed dense + sparse dictionary
- `storage="full"` and `storage="upper"` produce identical eigenvalues for
  the same operator.

End-to-end eigenvalue checks:
- 2×2 σx + σy + σz Hamiltonian: eigenvalues match `np.linalg.eigvalsh`.
- 4×4 Hermitian operator: eigenvalues match `np.linalg.eigvalsh`.

Eigenvector regression check (parametrised real / imag / complex):
- Builds a 1S spin-half Hamiltonian from σx, σy, σz numpy arrays via
  `ImportTensors`, sets coefficients to select a real / imaginary / complex
  Hamiltonian, diagonalises through `cfl.Hamiltonian`, and compares both
  eigenvalues and (phase-matched) eigenvectors to `numpy.linalg.eig`.
- This test deliberately mirrors `tests/integration/spin-half/test_spin-half.py`,
  which was originally written to debug a bug where `cfl` was effectively
  diagonalising the transpose of the supplied matrix — the fix was the
  complex conjugation added at ~line 160 of
  `cfl/src/cfl_h.c::solve_hermitian_block`. The imaginary and complex
  parametrisations exercise exactly that code path. Passing these tests
  demonstrates that `cfl` is handling complex Hermitian matrices correctly
  end-to-end (no effective transpose / missing conjugation).
- Phase matching anchors on the first component within `1e-10` of the
  maximum magnitude (rather than bare `argmax`) so the choice is stable
  across pycf and numpy when several components have equal magnitudes —
  which is the case for Pauli eigenvectors.

Alias synthesis, validation, and miscellany:
- `add_aliases=False` by default; opt-in synthesises MAGX/MAGY/MAGZ from
  MAG10/MAG11 and HYP from AHYP/BHYP.
- Alias collision with a user-supplied tensor raises `ValueError`.
- Reserved attribute names (`tensors`, `states`, `label_key`,
  `print_names`, `_wrapped`) are rejected when `expose_attrs=True` and
  allowed when `expose_attrs=False`.
- Validation errors: empty `label_key`, invalid `storage`, wrong `states`
  rank or shape, non-square tensor, shape mismatch, non-Hermitian dense
  input (bypassable via `check_hermitian=False`), non-Mapping `tensors`,
  empty tensor name, empty `states`.
- Zero-tensor warning emitted by default; suppressible via
  `warn_zero=False`.
- `pycf.ImportTensors` is exposed as a public re-export.

### 2.3 Test results

```
$ python -m pytest tests/unit/test_import_tensors.py -q
33 passed
$ python -m pytest tests/ -q
431 passed, 16 skipped
```

(Was 398 passed / 16 skipped before this session began, then 428 / 16 after
the first batch of new tests, then 431 / 16 after the eigenvector
parametrised test was added — three new spin-half regression cases.)

### 2.4 Deviations from the plan

None of substance. Two clarifications worth noting:

- The reserved-name check in `ImportTensors.__init__` is gated on
  `expose_attrs=True` (not unconditional), so an advanced caller who
  explicitly opts out of attribute mirroring can use any tensor names.
- The tie-breaking rule in the eigenvector phase-matching test (first
  component within `1e-10` of the max magnitude) is necessary because
  bare `argmax` is non-deterministic when magnitudes are equal up to FP
  noise. Documented in the test file.

## 3. Phase B — Refactoring `ImportSLJM`

**Status:** Implemented and tested locally; not yet committed.

### 3.1 Change

`ImportSLJM.__init__` no longer carries its own `cfl.StateLabels` /
`cfl.Tensor` construction loop or alias-synthesis block. The terminal
~30 lines of the constructor (everything from `sl = cfl.StateLabels(...)`
onwards through the `MAGX/Y/Z` and `HYP` aliases) are replaced by a
single delegated call:

```python
self._wrapped = ImportTensors(
    label_key,
    sl,
    tensor_matrices,
    storage="upper",        # *.txt files store upper triangle only
    add_aliases=True,       # legacy ImportSLJM behaviour
    expose_attrs=False,     # legacy path tolerated reserved-name shadowing
    check_hermitian=False,  # upper-triangle inputs are not full-Hermitian
    warn_zero=True,
)
self.tensors = self._wrapped.tensors
self.__dict__.update(self._wrapped.tensors)
```

Public surface is preserved exactly:

- `self.tensors` — same dict of `cfl.Tensor` objects.
- `self.<NAME>` attribute mirroring — same `__dict__.update(...)` call.
- `__iter__` / `print_names` — unchanged.
- Zero-tensor warning text — unchanged (printed by `ImportTensors`).
- Alias signs and definitions — `_apply_aliases` mirrors the legacy
  formulas exactly (verified by visual diff and by the integration
  suite passing without any change).

### 3.2 Test results

```
$ python -m pytest tests/ -q
431 passed, 16 skipped in 40.30s
```

The integration tests (`tests/integration/ceylf/`, `tests/integration/eryso/`,
`tests/integration/inten/`) all rely on `ImportSLJM` reading real
`*.txt` / `*.mi_` / `*.st_` files, building tensors, and running through
`cfl.Hamiltonian` to compare numerical results against checked-in
references. Their continued green status is the equivalence regression
test for Phase B — any change in tensor wrapping or alias synthesis
would have shifted eigenvalues, fit residuals, or transition
intensities and broken these tests.

### 3.3 Deviations from the plan

- `expose_attrs=False` was used (rather than `True`) when delegating
  from `ImportSLJM`. Reason: the legacy `ImportSLJM` code did
  `self.__dict__.update(tensors)` without any reserved-name guard, so
  it tolerated tensor names like `"tensors"` or `"states"` shadowing
  attributes. Enabling the guard would have been a behaviour change
  for the legacy path. `ImportSLJM` performs its own `__dict__.update`
  on the wrapped dict to preserve the legacy attribute-mirroring
  behaviour exactly.
- No new dedicated equivalence test was added — the existing
  integration suite already covers this path more thoroughly than a
  bespoke fixture-comparison test would.


## 4. Coverage Delta

_Coverage figures before and after each phase will be recorded here using
the same `python -m pytest tests/ --cov=pycf --cov-report=term` invocation
as the audit report._

## 5. Open Questions / Follow-ups

_Any items that surface during implementation but are deferred go here._
