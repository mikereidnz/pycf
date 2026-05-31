# Plan: Mofify ImportTensors to be correct. 

## 1. Purpose and Scope

I have realised that the new ImportTenosors persists with some incorrect assumptions built into the legacy code. 

The key point is that the tensor matrices are *not* necessarily Hermitian. The Hamiltonian is Hermitian. The Hamiltonian is a sum of tensor matrices multipled by parameters, and both can be complex. 

The state ordering used in the current input files are chosen so that tensors (Tkq) with q>0 only have matrix elements above the diagonal. For q<0 all matrix elements are below the diagonal. For q=0 the matrix elements can be above or below.

It is *not correct* to make the input matrices Hermitian. The code in function zhcsr2zha of clf_csr.c incorrectly does this on PASS 2 (lines 396 to 399)

The function vtrans, in inten.py, which I rewrote recently to be correct, first deletes the lower triangle (lines 76-79) to negate what zhcsr2zha does. 
It then adds the lower triangle if q==0 (lines 81-84) and constructs the matrices for q<0  (lines 94-100). 

In brief, I would like to modify ImportTensors to simply accept the matrix provided, which may or may not be Hermitian. This actually simplifies the code and removes most options, including 
storage (full/upper)
  The docstring comment is not correct. 
check_hermitian 
  Again the docstring is misleading. 

I would also like to remove PASS 2 (lines 396 to 399) from clf_csr.c

With these changes, the matrices entered could include the lower diagonal but would not be tested for that. 

I believe that we would also have to modify cfl_h.c, specifically the wrapper I installed, solve_hermitian_block, which wraps the call to LAPACKE_zheevr_work and complex conjugatse before that call (lines 158-160).

I think that the combination of options LAPACK_COL_MAJOR and 'U' (upper diagonal) actually result in an effective transposition when the FORTRAN code is called, which is why the conj() is necessary. However, if we remove the code from lines 396 to 399 of clf_csr.c, the FORTRAN code may only see a diagonal matrix, so some debugging may be necessary. 


## 2. Goals

Make the import and use of tensor matrices correct and transparent. 

## 3. Approach

1. Simplify ImportTensor as suggested above, so it simply accepts whatever matrix it is given. 

2. *Comment out* (not delete!) PASS 2 (lines 396 to 399) from clf_csr.c

3. Run the tests that you constructed that allow you to imput arbitrary states and tensors. 

4. Run my test cases, such as test_spin-half.py and the tests in the inten subdirectory. 

I expect the tests in 3 and 4 to fail, then we will need to look at the call to LAPACKE_zheevr_work, and perhaps modify the input parameters and/or the conj() operation. 

As with the previous coding, please create an import_tensor_revised_report.md and import_tensor_revised_diff.md as you go along.

## 4. Critique and recommendations (added by review pass)

The intent of the plan is sound and matches the physics: individual T_kq
matrices are not Hermitian; only the assembled H = Σ p_kq T_kq is.

**Important reframing after experimental probing (28 Apr 2026):** the
Hamiltonian assembly path is *not* corrupted by the upper-only
storage shortcut, so eigenvalues / eigenvectors come out correct on
devel today. The bug is narrower than I first thought: it only
shows up when individual tensor matrices are introspected (e.g. via
`tensor.get_matel()` or in the spin-Hamiltonian projection in
`cfl_sh.c`). This matches the user's observation that `inten.vtrans`
needed strip-and-rebuild logic to recover faithful matrices.

### 4.0 Why the Hamiltonian path is robust

Trace of a non-Hermitian tensor through Hamiltonian assembly:

1. `ImportTensors._normalise_matrix` with `storage="upper"`,
   `check_hermitian=False` stores the input CSR verbatim (no triu).
2. `cfl_h.c` builds H by:
   a. `zhcsrsama` — sums weighted tensor CSRs into one zhcsr.
      `zhcsrsama_alloc` constructs the *union* of input sparsity
      patterns, so q>0 (upper) and q<0 (lower) entries coexist in
      the summed zhcsr.
   b. `zhcsr2zcsr` (cfl_csr.c:299) — expands to a plain zcsr.
      For lower-triangle output entries it **re-derives them from
      conj(upper)** of the summed zhcsr (lines 309-315), ignoring
      the lower-triangle values that are already present in the
      hcsr storage.
   c. The dense block fill in cfl_h.c:282-290 reads the zcsr
      verbatim.
3. LAPACK reads the upper triangle of the dense block.

Because H is required to be Hermitian, conj(upper(H)) = lower(H)
*always*. So step 2(b) reconstructing lower from upper is correct
for any Hermitian H regardless of how it was assembled. The
information lost in step 2(b) (the "true" lower entries before
re-derivation) is exactly the information that would only differ
from conj(upper) if H were non-Hermitian — and a non-Hermitian H
would be a misuse anyway.

I confirmed this experimentally: built `H = T+ + T-` and
`H = (a+ib)T+ + (a-ib)T-` from strictly-upper T+ and strictly-
lower T-, ran them through `cfl.Hamiltonian.diag()`, and the
eigenvalues match `numpy.linalg.eigvalsh` to 1e-10 on devel.

### 4.1 Where the bug actually manifests

`zhcsr2zha` (cfl_csr.c:378) is used by:

- `cfl_tensor.c::zt_get_matel` (line 258) — the public Cython
  entry point used by `tensor.get_matel()`. **Bug visible here.**
- `cfl_sh.c` line 364 — spin-Hamiltonian projection. Operates on
  individual tensors; **bug visible here**.
- `csr_test.c`, `h_test.c` — C unit tests with hard-coded expected
  results that assume PASS-2 Hermitian fill. Need updating.

PASS 1 of `zhcsr2zha` iterates `j >= i`, dropping every below-
diagonal CSR entry. PASS 2 then fills the lower from conj(upper).
For a strictly-lower input, the upper is empty, so PASS 2 fills
the lower with zeros. Round-trip returns the zero matrix.

For a strictly-upper input, PASS 1 stores it correctly, but
PASS 2 fills the lower with conj(upper) — so the round-trip
returns the Hermitian completion, not the input.

Diagnostic tests (Phase 0, now committed):

- `test_strictly_lower_tensor_round_trips` — fails today.
- `test_strictly_upper_tensor_round_trips` — fails today.
- `test_mixed_lower_tensor_with_complex_values` — fails today.

### 4.2 LAPACK conjugation: leave it alone

The conj() at cfl_h.c:160 acts on the **assembled Hamiltonian
block**, which is correctly Hermitian for the reasons in §4.0.
No change required there. The spin-half regression test pins
this behaviour.

### 4.3 PASS 1 must be replaced, not just PASS 2

Just commenting out PASS 2 leaves PASS 1 dropping the lower
triangle, and the round-trip remains broken (lower becomes
zero). The fix:

```c
memset(a, 0, n*n*sizeof(complex double));
for (i = 0; i < n; i++) {
  for (k = hcsr_m->row_ptr[i]; k < hcsr_m->row_ptr[i+1]; k++) {
    a[i*n + hcsr_m->col_in[k]] = hcsr_m->val[k];
  }
}
```

PASS 2 is then redundant and can be commented out.

### 4.4 `zhcsr2zcsr` is intentionally Hermitian-fill — leave alone

`zhcsr2zcsr` (cfl_csr.c:299) runs only on the assembled-H zhcsr,
not on individual tensors. Its conj-fill is exactly what makes
the Hamiltonian path correct (§4.0). Touching it would corrupt
diagonalisation. Same reasoning for `zhcsr2zhpa`. So contrary
to my earlier critique, these stay as-is.

### 4.5 Type-name honesty

The `zhcsr` type still represents a Hermitian matrix at the
*Hamiltonian-assembly* level. At the *individual-tensor* level
it now stores arbitrary CSR. The type name is somewhat
overloaded but acceptable. Add a comment in `cfl_csr.h`
noting the dual usage.

### 4.6 `ImportTensors` simplification

The `storage="full"|"upper"` and `check_hermitian` knobs become
meaningless once `zhcsr2zha` faithfully round-trips:

- The "is upper triangle?" check is wrong as a *generic* check
  because q<0 tensors are lower-only.
- The Hermiticity check is wrong because non-q=0 tensors are not
  Hermitian.

User has confirmed (28 Apr 2026) that `ImportTensors` is only
called from `ImportSLJM` and the new unit tests, so the parameters
can be removed outright rather than deprecated. Plan: delete
`storage` and `check_hermitian` from the signature, the validation
logic, and the docstring; update `ImportSLJM` and the unit tests
to match.

### 4.7 Other call sites of `zhcsr2zha`

- `cfl/src/cfl_tensor.c:258` — `zt_get_matel`, fixes the public
  Cython API after the C change.
- `cfl/src/cfl_sh.c:364` — spin-Hamiltonian projection. Currently
  receives the Hermitian completion of each tensor. After the
  fix it will receive the raw tensor. Need to verify whether
  callers compensate (likely they do; spinh-style projections
  use traces of products that are invariant under the choice).
  Run the full spinh test suite after the change.
- `cfl/tests/csr_test.c:74,122,182,196,217` and
  `h_test.c:141,152` — C unit tests. Hard-coded expected outputs
  will need updating (or replaced with full-CSR round-trip
  expectations).

### 4.8 Revised phased execution

1. **Phase 0 — diagnostic tests (DONE):** added three failing
   `test_strictly_*_round_trips` tests in
   `tests/unit/test_import_tensors.py`.
2. **Phase 1 — C-layer fix:** rewrite PASS 1 of `zhcsr2zha` to
   copy all CSR entries verbatim; comment out PASS 2. Leave
   `zhcsr2zcsr`, `zhcsr2zhpa`, and `solve_hermitian_block`
   untouched. Update `csr_test.c` and `h_test.c` expectations.
3. **Phase 2 — Python-layer simplification:** remove `storage` and
   `check_hermitian` from `ImportTensors`; update `ImportSLJM`
   call site (line 401-414) and unit tests.
4. **Phase 3 — full test suite:** C tests, `tests/unit/`,
   `tests/integration/spin-half/`, `tests/inten/`. Spin-half
   should still pass (it tests Hermitian inputs); inten tests
   exercise the get_matel path so any regression there is real.
5. **Phase 4 — diagnose `cfl_sh.c` only if Phase 3 fails.**

### 4.9 Reports and backups

Per project convention I will create
`plan/import_tensor_revised_report.md` and
`plan/import_tensor_revised_diff.md` and update them as I go. I
will also back up any non-trivially modified file with a
`.bak.YYYYMMDD` suffix before the first edit.
