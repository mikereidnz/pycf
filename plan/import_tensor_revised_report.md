# ImportTensors / zhcsr2zha revision — outcome report

Branch: `fix/import-tensors-non-hermitian` (4 commits, off `devel`).

## Problem

`Tensor.get_matel()` and the spin-Hamiltonian projection in
`cfl/src/cfl_sh.c` corrupted non-Hermitian tensors. The C routine
`zhcsr2zha` (CSR → dense block) had a "PASS 2" pass that
Hermitian-completed the dense buffer by mirroring the upper triangle
into the lower with a complex conjugate. That is correct for an
assembled Hamiltonian (which is Hermitian by construction) but wrong
for individual T_kq crystal-field tensors, which are not Hermitian on
their own.

The `ImportTensors` Python API also exposed two parameters
(`storage`, `check_hermitian`) whose only real consumer was the
internal `ImportSLJM` call site. With the C bug fixed, those
parameters are no longer needed.

## What changed

### C layer
- `cfl/src/cfl_csr.c::zhcsr2zha`: PASS 2 removed. The dense buffer
  is now zeroed and only the upper-triangle CSR data is written into
  it, verbatim. The docstring was expanded with the bug history and
  cross-reference to `zhcsr2zcsr` (the H-assembly path, which is
  unchanged and still does its own Hermitian fill on the summed
  CSR — that buffer is genuinely Hermitian).
- `cfl/src/cfl_sh.c::zshp_p`: gained a local Hermitian-fill loop
  immediately after the `zhcsr2zha` call. `cblas_zhemm(CblasUpper,
  CblasColMajor, ...)` reads the row-major *lower* triangle of our
  buffer, so without this completion the spin-Hamiltonian projection
  silently returned zero. The operators handled here (Zeeman,
  hyperfine, quadrupole) are all genuinely Hermitian, so the fill is
  safe.
- `solve_hermitian_block` in `cfl/src/cfl_h.c` is unchanged. Its
  conjugation at line 160 compensates for the row-major ↔ column-major
  transpose that LAPACKE applies — not for one-triangle storage. The
  H-assembly path produces both triangles via `zhcsr2zcsr`, so the
  LAPACK `'U'` flag works.

### C tests
- `cfl/tests/csr_test.c` and `cfl/tests/h_test.c` gained
  `equ_chk_upper` / `zequ_chk_upper` helpers for upper-triangle-only
  comparisons. Five `csr_test` and two `h_test` checks switched to
  these helpers; everything still passes.

### Python — ImportTensors
- `pycf/import_sljm.py::ImportTensors.__init__` no longer takes
  `storage` or `check_hermitian` parameters. Every input matrix is
  unconditionally upper-triangulated via `scipy.sparse.triu` before
  being handed to `cfl.Tensor`. The strict lower triangle is silently
  discarded (callers must arrange the data they care about into the
  upper triangle, which is the C layer's storage convention).
- `pycf/import_sljm.py::ImportSLJM` updated to call `ImportTensors`
  without the removed kwargs.
- Class docstring and parameter table updated to document the new
  upper-triangle-only contract.

### Python — ZEFOZSearch (post-audit follow-up)
- `pycf/cfl.pyx::ZEFOZSearch.__cinit__`: Hermitian-fill the dense
  MX/MY/MZ matrices before handing them to the C layer. The
  ZEFOZ inner-product routine `cfl_zefoz.c::inprod` uses
  `cblas_zgemv(CblasNoTrans, ...)` — a full matrix-vector multiply
  that reads both triangles. Without Hermitian completion, the
  off-diagonal contributions to the ZEFOZ gradient and curvature
  would be silently dropped.

### Python — inten.py
- The vtrans docstring comment dated "Mike Reid 3 April 2026" is
  updated to note that the `M - np.tril(M, k=-1)` step is now a
  no-op (because `get_matel()` no longer adds the lower triangle).
  The line is retained as a defensive measure and to keep the q==0
  Hermitian-completion logic that follows it self-evidently correct.

### Python tests
- `tests/unit/test_import_tensors.py`:
  - Removed tests that exercised the removed `storage` /
    `check_hermitian` parameters.
  - Updated round-trip tests to assert against `np.triu(input)`
    (matches the new upper-triangle-only storage contract).
  - Replaced `test_non_hermitian_dense_rejected` and
    `test_check_hermitian_false_bypasses` with a single positive
    test (`test_non_hermitian_dense_accepted_lower_dropped`) that
    documents the new contract.
  - Added `test_strictly_upper_tensor_round_trips` as a positive
    case for non-Hermitian operators with content above the
    diagonal only.
  - Added Hamiltonian reconstruction tests using the identity
    `H == z @ diag(w) @ z.conj().T` (3 parametrised spin-half
    cases + 1 4×4 case). These are order- and phase-independent,
    so they make a much simpler regression guard than the
    phase-matched eigenvector test. This identity is the
    diagnostic that originally pinpointed the conjugation bug
    in `solve_hermitian_block`.

## Verification

- `make -C cfl test` — all 24 C tests pass.
- `python -m pytest tests/ -q` — 444 passed, 16 skipped, 0 failed
  (was 433 passed, 16 skipped before this work; net +11 from the
  new tests).

## Commits

```
660cd63  Fix zhcsr2zha to preserve non-Hermitian tensors verbatim
b5cffd7  ImportTensors: drop storage and check_hermitian parameters
3639b1b  Add Hamiltonian reconstruction tests for h.diag()
fd0a8db  Hermitian-fill ZEFOZ Zeeman matrices in ZEFOZSearch
```

## Follow-ups

- **ZEFOZ example / integration test.** The Hermitian-fill in
  `ZEFOZSearch.__cinit__` was correct by inspection, but no Python
  ZEFOZ test exercises it. Mike intends to construct a small ZEFOZ
  example (a useful demonstration in its own right) which can then
  be turned into an integration test along the same lines as the
  existing `tests/integration/spin-half`, `ceylf`, `eryso`, and
  `inten` directories. That test should construct a Hamiltonian
  with non-trivial off-diagonal Zeeman matrix elements, run a
  ZEFOZ search, and compare against an analytic or numpy reference
  for the gradient and curvature at a known field. This is the
  proper regression guard for the `cfl.pyx` Hermitian-fill.

- **CHANGELOG.md.** Behavioural change to `Tensor.get_matel()`
  (now returns upper triangle only) and breaking change to
  `ImportTensors.__init__` (parameter removal). Should be noted in
  the next release entry.

- **Backups.** `cfl/src/cfl_csr.c.bak.20260428`, etc., should be
  removed before merging.
