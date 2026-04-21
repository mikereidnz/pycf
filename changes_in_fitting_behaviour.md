# Changes in fitting behaviour

All of the changes described here correct pre-existing bugs.  The new
behaviour is the correct behaviour; any differences in fit results relative
to runs made before the `mfr-upgrade-python` branch are because those runs
were affected by one or more of the bugs below.

## Dangling pointer in weight array (commit `106cf51`, 21 April 2026)

**Affected routines:** all fitting routines — `EFit`, `MHFit`, `ESHFit`,
`MESHFit`.

In `exdata_alloc_helper` the temporary NumPy array holding the scaled
per-level weights (`ex.w * weight`) was immediately eligible for garbage
collection after the function returned, leaving `ex_data.w` as a dangling
pointer.  The C fitting code subsequently read garbage values from that
memory when evaluating the weighted chi-square.

The effect was most severe for multi-Hamiltonian fits (`MHFit`) with
non-uniform `weights_list`, because the corrupted weight array determined the
relative importance of each Hamiltonian.  For example, in the YbK2YF5 fit
the change reduced `fmin` from ~4933 to ~2434 — the lower value is correct.

Whether a given fit was affected depends on the **sizes of the weight arrays**
for each Hamiltonian:

- Each energy-level Hamiltonian allocates a weight array of `n_obs × 8` bytes,
  where `n_obs` is the number of observed levels.  Each g-value Hamiltonian
  typically has `n_obs = 1`, giving an 8-byte array.
- glibc's malloc uses per-size free-lists (fastbins).  If the energy-level
  array and the g-value arrays are the **same size** (i.e. the energy-level
  Hamiltonian also has one observed level), they share the same fastbin.  The
  LIFO fastbin then causes every allocation to reuse the same address, so after
  the loop over g-value Hamiltonians the energy-level `ex_data.w` pointer still
  points at that address — but the value it holds is now the last-written
  `gg_w` rather than `e_w`.
- If the energy-level array is a **different size** (e.g. 39 levels → 312
  bytes), it lives in a separate bin and the g-value loop cannot overwrite it.
  The dangling pointer still points at unfreed memory that still contains the
  original `e_w` values, so the corruption is absent.

This explains why the YbK2YF5 fit (1 energy level, 36 g-value Hamiltonians:
all 8-byte arrays) changed dramatically, while the ErK2YF5 fit (39 energy
levels → 312-byte array, 12 g-value Hamiltonians: 8-byte arrays each) was
essentially unaffected.  The effect has been verified by reading back the
weight memory after simulating the old allocation pattern in Python.

Importantly, the corrupted results were **exactly reproducible** across many
runs and even across Python versions (3.8 → 3.13).  This had two causes:
first, GSL's basin-hopping RNG is seeded deterministically (via
`gsl_rng_env_setup()`, which uses seed 0 by default), so every run takes
the identical random walk; second, Python's memory allocator reuses freed
small allocations in a predictable LIFO pattern, so the dangling `ex_data.w`
pointer consistently read the same wrong value (`gg_w`) on every run.  The
consistent wrong answer made the bug invisible until the weights were
inspected directly.

The fix keeps a reference to the weight backing array (`_ex_w_backing`) alive
for the lifetime of the fit object.

## GSL nonlinear least-squares residuals (commit `82ddced`, 21 April 2026)

**Affected routines:** `gsl_nls` fits.

The residual vector returned to GSL's trust-region NLS solver was
`w * calc - obs` instead of `sqrt(w) * (calc - obs)`.  GSL minimises the
sum of squared residuals, so only the second form makes that sum equal to the
standard weighted chi-square `Σ w_i (calc - obs)²`.  The old form was
neither a proper chi-square nor a simple scaling of one.

## NLS double-weighting of per-Hamiltonian scalars (commit `ae3eb4a`, 22 April 2026)

**Affected routines:** `gsl_nls` fits with non-uniform `weights_list`.

After fixing the residuals (above), the per-Hamiltonian weight scalar
`weights_list[j]` was also being passed as the GSL `wts` array, causing GSL
to apply it a second time internally.  The effective weight for NLS was
therefore `weights_list[j]² × ex.w[i]`, while gradient-based methods used
`weights_list[j] × ex.w[i]`.  The `wts` array is now set to all-ones so that
NLS and gradient methods minimise the same weighted chi-square.

If you were using `gsl_nls` with non-uniform `weights_list` and wish to
reproduce old results with a gradient method, you would need to square each
entry of `weights_list`.

## Spin-Hamiltonian chi-square sign (commit `82ddced`, 21 April 2026)

**Affected routines:** `ESHFit` and `MESHFit` when a `SpinHamiltonian` is
present.

`shchisq` compared `|pa[i]| - |xpa[i]|` rather than `pa[i] - xpa[i]`.
Taking absolute values before subtracting treated parameters of opposite sign
as equivalent, suppressing physically meaningful sign differences in the
spin-Hamiltonian parameters.
