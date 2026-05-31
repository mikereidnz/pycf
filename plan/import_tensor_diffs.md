# Diffs: ImportTensors refactor

One entry per commit, mirroring `audit_2026-04-27_130831_diffs.md`. Each
entry records the commit SHA, a short summary, the files touched, and any
notes (test results, manual verification, deviations from the plan).

---

## 27d569d — Phase A: add `ImportTensors`

**Summary**: Introduce `ImportTensors`, an in-memory wrapper that
decouples `cfl.Tensor` construction from the legacy jmcalc text-file
path. `ImportSLJM` is left untouched; Phase B will refactor it to
delegate.

**Files**:
- `pycf/import_sljm.py` — new `ImportTensors` class (+`_RESERVED_TENSOR_NAMES`).
- `pycf/__init__.py` — lazy re-export of `ImportTensors` (and `ImportSLJM`).
- `tests/unit/test_import_tensors.py` — 33 new unit tests.
- `plan/import_tensor_plan.md`, `plan/import_tensor_report.md`,
  `plan/import_tensor_diffs.md` — planning triplet.

Diffstat: 6 files changed, 1055 insertions(+), 3 deletions(-).

**Tests**: full suite **431 passed, 16 skipped** (was 398/16 pre-task;
+33 new, no regressions).

**Notes**:
- `storage='full'|'upper'` parameter makes the Hermitian CRS storage
  contract explicit (prevents the doubled-off-diagonal footgun when
  passing a full Hermitian sparse matrix straight to `csr_matrix`).
- Aliases `MAGX/MAGY/MAGZ/HYP` are opt-in (`add_aliases=False` default).
  `ImportSLJM` will pass `True` in Phase B.
- Reserved-name guard (`tensors`, `states`, `label_key`, `print_names`,
  `_wrapped`) only triggers when `expose_attrs=True` (the default).
- Parametrised real/imag/complex spin-half eigenvector test guards the
  complex-conjugation fix at `cfl/src/cfl_h.c::solve_hermitian_block`
  (~line 160), which addresses an effective-transpose bug in the
  LAPACK call. Phase matching uses tolerance-based first-index anchor
  to remain deterministic for Pauli eigenvectors with degenerate
  component magnitudes (1/√2).
## eaeda72 — Phase B: delegate `ImportSLJM` tensor wrapping

**Summary**: Replace the terminal ~30 lines of `ImportSLJM.__init__`
(the `cfl.StateLabels`/`cfl.Tensor` construction loop and the
`MAGX/MAGY/MAGZ/HYP` alias block) with a single call to
`ImportTensors(..., storage='upper', add_aliases=True,
expose_attrs=False, check_hermitian=False)`. Behaviour-preserving.

**Files**:
- `pycf/import_sljm.py` — `ImportSLJM.__init__` body shortened.
- `plan/import_tensor_report.md` — Phase B section populated.

Diffstat: 2 files changed, 87 insertions(+), 33 deletions(-).

**Tests**: full suite **431 passed, 16 skipped** (same as Phase A; no
regressions). The integration tests under `tests/integration/{ceylf,
eryso, inten}` exercise `ImportSLJM` end-to-end against real
`*.txt`/`*.mi_`/`*.st_` files and compare numerical results to
checked-in references — they collectively serve as the equivalence
regression test for this refactor.

**Notes**:
- `expose_attrs=False` deliberately bypasses the reserved-name guard
  so the legacy `__dict__.update(tensors)` shadowing behaviour is
  preserved.
- `check_hermitian=False` because the `*.txt` files store only the
  upper triangle; a Hermitian round-trip check would (incorrectly)
  fail on these inputs.
- `_apply_aliases` formulas are a verbatim port of the legacy block,
  including the standard spherical-tensor signs for `MAGX/MAGY`.
