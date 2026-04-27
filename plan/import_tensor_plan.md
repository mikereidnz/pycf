# Plan: Extract ImportTensors from ImportSLJM

## 1. Purpose and Scope

Today `pycf.ImportSLJM` couples two distinct responsibilities:

1. **Parsing** the legacy jmcalc plain-text outputs (`*.txt`, `*.mi_`, `*.st_`)
   to produce a state-label list, a label-key string, and a dictionary of
   tensor matrices.
2. **Wrapping** those products as `cfl.StateLabels` and `cfl.Tensor` objects,
   adding convenience aliases (`MAGX`, `MAGY`, `MAGZ`, `HYP`), and exposing
   them as attributes.

This plan describes a small refactor that splits responsibility 2 into a new
class `ImportTensors`, leaving `ImportSLJM` to do parsing only and delegate
the wrapping. The intent is to:

- Decouple matrix-element ingestion from the legacy file format so that
  future sources (HDF5, JSON, in-memory results from a Python re-implementation
  of jmcalc, etc.) can produce tensors without going through text files.
- Make it easy to construct **small synthetic Hamiltonians inside test
  functions** to probe known eigenstructure, mirroring the philosophy of
  `tests/integration/spin-half/` but with arbitrary user-supplied matrices.
- Reduce duplication: `ImportSLJM` ends up calling `ImportTensors` rather
  than carrying its own copy of the wrapping code.

Scope is intentionally narrow:

- Behaviour-preserving for `ImportSLJM`. All existing tests must continue to
  pass without modification.
- No changes to `cfl.Tensor`, `cfl.StateLabels`, or the C layer.
- No changes to the file-parsing helpers (`get_tensor_dim`, `get_state_number`).

## 2. Goals

- Add a new class `ImportTensors` that accepts already-parsed inputs
  (states, label_key, tensors as ndarray/CSR/dict) and produces the same
  in-memory `cfl.Tensor` collection that `ImportSLJM` produces today.
- Refactor `ImportSLJM.__init__` to delegate the wrapping step (current
  lines ~173–202 of `pycf/import_sljm.py`) to `ImportTensors`.
- Add unit tests that build a small Hamiltonian from numpy arrays via
  `ImportTensors`, diagonalise it through `cfl.Hamiltonian`, and verify
  eigenvalues against `numpy.linalg.eigvalsh`.
- Document the new class in the public API surface (`pycf/__init__.py`,
  Sphinx autodoc, `CHANGELOG.md`).

## 3. Non-Goals

- Replacing `ImportSLJM`. The legacy path stays available indefinitely;
  this is purely additive plus an internal delegation.
- Changing the file format, the regex, or the term-symbol parsing.
- Designing a non-jmcalc importer in this task. We only enable it.
- Touching `pyemp.py`.

## 4. Current Code Anatomy

`pycf/import_sljm.py::ImportSLJM.__init__` (lines 75–202) does, in order:

| Lines | Responsibility |
| --- | --- |
| 76–80 | Read `name.mi_`, build `tensor_dims` list of `(name, dim)`. |
| 81–95 | Read `sl_name.st_`, find the state count `dim`. |
| 96–111 | Re-read `sl_name.st_`, regex-parse state labels; sanity-check counts. |
| 112–148 | Build `label_key` (canonical sort), build `sl` (2D list of int labels). |
| 149–172 | Read `name.txt`, slice it into `tensor_elements` per tensor, build a `csr_matrix` per tensor in `tensor_matrices`. |
| **173–202** | **Wrap as `cfl.StateLabels` + `cfl.Tensor`, add MAGX/Y/Z + HYP aliases, expose as attributes.** |

Lines 173–202 are the target of this refactor. Their inputs are exactly:

- `label_key: str` — canonical-order key string (e.g. `"LJM"`, `"SLJMI"`).
- `sl: list[list[int]]` — N×K integer label matrix (half-integer doubled).
- `tensor_matrices: dict[str, csr_matrix]` — Hermitian upper-triangle CSR.

Their outputs are:

- `self.tensors: dict[str, cfl.Tensor]`.
- Attributes on `self` for each tensor name (via `__dict__.update`).

## 5. Proposed Design

### 5.1 New class `ImportTensors`

Location: `pycf/import_sljm.py` (same module, above `ImportSLJM`).

```python
class ImportTensors:
    """
    Wrap pre-parsed states and matrix elements as cfl.Tensor objects.

    Parameters
    ----------
    label_key : str
        Canonical-order label key (one character per label column, e.g. "LJM",
        "SLJMI"). Half-integer quantum numbers are encoded as doubled integers
        in `states`.
    states : array-like, shape (N, len(label_key))
        Integer state-label matrix. Rows are states, columns correspond to
        characters of label_key. Must be exactly 2-D with the matching column
        count; no implicit reshape (a 1-D special case is allowed only when
        `len(label_key) == 1`).
    tensors : Mapping[str, numpy.ndarray | scipy.sparse.spmatrix]
        Mapping from tensor name to an N×N matrix.
    storage : {"full", "upper"}, optional
        Declares the storage convention of the input matrices. Default
        ``"full"``: matrices are full Hermitian; the class will validate
        Hermiticity (for dense input) and take the upper triangle. ``"upper"``
        promises the caller has supplied upper-triangle-only Hermitian CRS
        already (the legacy ImportSLJM path uses this). The C layer requires
        upper-triangle Hermitian compressed-row storage; this parameter
        protects new callers who would naturally pass `csr_matrix(M)` of a
        full Hermitian matrix.
    add_aliases : bool, optional
        Default ``False``. When True and the corresponding source tensors are
        present, synthesise the rare-earth-specific convenience aliases MAGX,
        MAGY, MAGZ (from MAG10/MAG11) and HYP (from AHYP/BHYP). Raises
        ``ValueError`` if an alias name collides with a user-supplied tensor.
        ImportSLJM passes ``True``.
    expose_attrs : bool, optional
        Default ``True``. Mirror tensors as attributes on ``self`` (the
        legacy ImportSLJM behaviour). Reserved names (``tensors``,
        ``states``, ``label_key``, ``print_names``, ``_wrapped``) are
        rejected unconditionally to prevent corruption of the instance.
    check_hermitian : bool, optional
        Default ``True``. Validate dense input matrices are Hermitian within
        ``np.allclose`` tolerances. Sparse input is not validated (sparse
        Hermiticity check is O(nnz) and the legacy path doesn't validate
        either; ``storage="upper"`` is the recommended path for sparse).
    warn_zero : bool, optional
        Default ``True``. Print a warning if a supplied tensor has no
        non-zero elements.
    """
```

**Internal flow:**

1. Validate `label_key` is a non-empty string. Validate `states` is 2-D with
   shape `(N, len(label_key))` (or 1-D iff `len(label_key) == 1`). Reject
   tensor names that collide with reserved attributes if
   `expose_attrs=True`.
2. Determine `dim = N`. If empty, raise `ValueError`.
3. For each `(name, mat)` in `tensors`:
   - If dense ndarray and `check_hermitian`: validate
     `np.allclose(M, M.conj().T)`, raise `ValueError` if not.
   - Convert to CSR `complex128`.
   - Apply storage convention:
     - `storage="full"`: take `scipy.sparse.triu(M, format='csr')`.
     - `storage="upper"`: pass through (caller's contract).
   - Validate `mat.shape == (dim, dim)`.

   Crystal-field / spin-Hamiltonian operators are Hermitian, so `cfl.Tensor`
   uses Hermitian compressed-row storage (upper triangle only); the lower
   triangle is implied. The legacy `*.txt` files contain upper-triangle
   elements only, which is why `ImportSLJM` historically passed CSR matrices
   through unchanged — that is the `storage="upper"` path.
4. Build `cfl.StateLabels(label_key, list_of_lists_of_int)`.
5. Build `cfl.Tensor` objects exactly as today (lines 180–186).
6. If `add_aliases=True`: check for collisions with existing tensor keys
   (raise `ValueError` on collision), then apply the MAG and HYP synthesis
   (lines 187–200).
7. `self.tensors = tensors_dict`. If `expose_attrs`,
   `self.__dict__.update(tensors_dict)` (after the reserved-name check).

`__iter__` and `print_names` are defined on `ImportTensors`. `ImportSLJM`
keeps its own copies (or delegates explicitly via `_wrapped`) — see 5.2.

### 5.2 ImportSLJM refactor

After parsing files, `ImportSLJM.__init__` constructs the new wrapper using
the legacy storage convention and aliases enabled, then explicitly mirrors
its public surface:

```python
self._wrapped = ImportTensors(
    label_key, sl, tensor_matrices,
    storage="upper",      # legacy *.txt files are upper-triangle only
    add_aliases=True,     # MAGX/Y/Z, HYP — current behaviour
    check_hermitian=False,
)
self.tensors = self._wrapped.tensors
self.__dict__.update(self._wrapped.tensors)

def __iter__(self):
    return iter(self._wrapped)

def print_names(self):
    return self._wrapped.print_names()
```

Composition (not inheritance) keeps `isinstance(x, ImportSLJM)` valid and
public attributes identical. `__iter__` and `print_names` are explicitly
delegated rather than relying on attribute inheritance.

### 5.3 Public API exposure

- Re-export `ImportTensors` from `pycf/__init__.py` alongside `ImportSLJM`.
- Add a one-paragraph entry to `CHANGELOG.md` under the next unreleased
  section.
- Add a docstring example showing a 3×3 toy Hamiltonian, suitable for
  Sphinx autodoc.

## 6. Test Plan

A new `tests/unit/test_import_tensors.py` covering:

- **Construction from dense ndarrays** (default `storage="full"`). Build a
  4×4 Hermitian "energy" tensor and verify `cfl.Tensor` round-trips the
  upper-triangle values.
- **Construction from full sparse Hermitian** with `storage="full"`; same
  result.
- **Construction from upper-triangle CSR** with `storage="upper"`;
  numerically identical to the dense-full case.
- **Storage-contract regression**: passing a full sparse Hermitian matrix
  with `storage="upper"` would double-count off-diagonal elements — we don't
  test this as "should work", but we do test that `storage="full"` and
  `storage="upper"` on the same mathematical operator (full vs `triu`)
  produce identical eigenvalues.
- **Synthetic Hamiltonian end-to-end.** Build a 2-tensor system, run
  `cfl.Hamiltonian.diag()` (note: the cfl.pyx method is `diag`, not
  `diagonalize`), compare eigenvalues against `numpy.linalg.eigvalsh`
  to ~1e-10.
- **Alias synthesis** with `add_aliases=True`: provide MAG10 + MAG11; assert
  MAGX/MAGY/MAGZ exist with the expected linear combinations. Provide
  AHYP + BHYP; assert HYP.
- **Alias default off**: `add_aliases` defaults to False; verify aliases are
  *not* synthesised when supplied MAG10/MAG11.
- **Alias collision**: `add_aliases=True` with a user-supplied `MAGX` raises
  `ValueError`.
- **Reserved-name rejection**: a tensor named `"tensors"` or `"print_names"`
  raises `ValueError` when `expose_attrs=True`; allowed when
  `expose_attrs=False`.
- **Hermiticity validation**: non-Hermitian dense input raises `ValueError`
  by default; bypassed by `check_hermitian=False`.
- **Validation errors:** non-square tensor, mismatched shape vs `len(states)`,
  bad `label_key` (empty / wrong length), wrong `states` rank, empty input
  → each raises a clear `ValueError`.
- **Zero-tensor warning.** Capture stdout for an all-zero tensor and assert
  the warning message is emitted when `warn_zero=True`, suppressed when False.
- **Public re-export**: `from pycf import ImportTensors` works.

A regression check for `ImportSLJM` (Phase B):

- After Phase B, the existing `tests/integration/{ceylf,eryso,inten,spin-half}/`
  suites must pass without modification.
- Add a focused equivalence test: load one fixture (e.g. `tests/integration/ceylf/matel/f1cf`)
  with `ImportSLJM`, then independently rebuild via the parsing helpers and
  call `ImportTensors(..., storage="upper", add_aliases=True)` directly;
  assert tensor names match, `t.get_matel()` is numerically equal for at
  least two tensors, and `state_labels.label_key` matches.

## 7. Implementation Steps

The user has indicated they want to proceed in two phases:

**Phase A — add `ImportTensors`, do not change `ImportSLJM` yet.**

1. Add `ImportTensors` to `pycf/import_sljm.py` above `ImportSLJM`.
2. Add `tests/unit/test_import_tensors.py` covering §6 cases.
3. Re-export from `pycf/__init__.py`.
4. Run `python -m pytest tests/ -q` and `make -C cfl test`; confirm green.
5. Get user sign-off.

**Phase B — refactor `ImportSLJM` to delegate.**

6. Replace lines 173–202 of `import_sljm.py` with a call to `ImportTensors`.
7. Re-run the full suite. The existing integration tests (which exercise
   `ImportSLJM` end-to-end against the legacy files in `tests/integration/`)
   are the regression gate.
8. Update `CHANGELOG.md` and `docs/`.
9. Get user sign-off and commit.

## 8. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Dense-vs-sparse upper-triangle handling diverges from the legacy path. | For dense input, explicitly `scipy.sparse.triu(M, format='csr')`. Sparse input passed through unchanged (matches today). Document the contract in the docstring. |
| Alias synthesis silently breaks for non-rare-earth callers. | Make `add_aliases=True` default, but allow opt-out. ImportSLJM keeps default. |
| Subtle dtype / contiguity mismatch when handing arrays to `cfl.Tensor`. | Re-use the exact `np.ascontiguousarray(..., dtype=np.intc)` / `np.ascontiguousarray(..., dtype=complex128)` pattern from the existing code. |
| Public-API churn surprises downstream users. | Refactor is composition; `ImportSLJM`'s shape and attributes are unchanged. |
| Tests for tiny Hamiltonians may exercise C-side assumptions (e.g. minimum dim) we don't currently know about. | Start with a 4×4 test (matches `tests/integration/spin-half/`), grow only if the small case is robust. |

## 9. Deliverables

1. Code: `ImportTensors` class + delegation from `ImportSLJM`.
2. Tests: `tests/unit/test_import_tensors.py`.
3. Docs: `pycf/__init__.py` re-export, docstring with example, CHANGELOG entry.
4. Companion files (this audit-style triplet):
   - `plan/import_tensor_plan.md` (this file).
   - `plan/import_tensor_report.md` (written as work proceeds; mirrors
     `audit_2026-04-27_171732_report.md` style with sections per phase).
   - `plan/import_tensor_diffs.md` (per-change diff log, one entry per
     commit, mirrors `audit_2026-04-27_130831_diffs.md`).

## 10. Out of Scope (Future Work)

- A second concrete loader (e.g. HDF5, JSON, direct from a Python jmcalc).
  Once `ImportTensors` exists this becomes a pure "read file → call
  ImportTensors" exercise.
- Replacing `njsymbols.py` 3j/6j/9j with `spherical`/`wigtools` calls.
- Generalising `cfl.Hamiltonian` test coverage to cover the non-spin-half
  cases that the new tests open up.
