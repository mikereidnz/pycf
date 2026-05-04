# Folded Magnetic Quantum Number (mu) Implementation Plan

## Problem Statement

State identification in crystal-field fitting currently relies on (S, L, J, M) quantum numbers via the `sl_diff` method in ExData. This approach fails for realistic problems where states are heavily mixed superpositions of multiple SLJ components, making automated state matching unreliable.

## Solution: Folded Magnetic Quantum Number (mu)

Introduce a robust, basis-independent state identifier based on the **folded magnetic quantum number** mu (also known as the "crystal quantum number"), which wraps the principal component's m value into a fundamental domain determined by the smallest non-zero q in the crystal-field tensor expansion.

This approach is based on classical crystallography references (Dieke, 1968; Wybourne, 1965) with a simplified implementation. It works because the Hamiltonian matrix can be block-diagonalized by magnetic quantum number. The C_kq tensor terms with q>0 can mix states whose m differ by ±q, and those connected states form the blocks. The folded mu value identifies which block (and sub-block) a state belongs to.
>>> This is usually referred to as  "crystal quantum number". See for example, G. H. Dieke, 1968, "Spectra and Energy Level of Rare Earth Ions in Crystals", B. G, Wybourne 1965, "Spectroscopic Properties of Rare Earths". Those references have a little more nuance, but the definition we are going to use is easy to implement. It works becasuse the Hamiltonian matrix can be block-diagonalized (which pycf does as part of the calculation). The Ckq with q>0 can mix states whose m differ by +-q and it is those connected states that form the blocks. 

### Algorithm

For each energy level:
1. Extract m from the principal component of the eigenvector
2. Account for doubled m values (when half-integers are avoided via integer storage)
3. Fold m into fundamental domain:
   ```python
   mu = abs(m) % min_q
   if mu > min_q // 2:
       mu = min_q - mu
   ```

### Parameters

- **minimum_q**: Smallest non-zero q value across all C_kq tensors in Hamiltonian
  - User-controllable (passed to Hamiltonian or energy summary functions)
  - Typical values: 2 (for C20, C22, ...), 4 (for C40, C44, ...), etc.
  - Note: If `half_integer_state_labels=True`, multiply minimum_q by 2 in the folding calculation

- **half_integer_state_labels**: Boolean flag indicating whether m values represent half-integers
  - If True: stored m values are actual m (e.g., 1/2, 3/2, ...)
  - If False: stored m values are doubled (e.g., 1, 3, ... for half-integer j)
  - Affects minimum_q in folding calculation: use `2 * minimum_q` if True

- **n**: Ordinal index within each mu group
  - n=1: lowest energy state with that mu value
  - n=2: next lowest energy state with that mu value
  - etc.

## Implementation Phases

### Phase 1: Output mu and n in Energy Summaries
- Calculate mu for each energy level during diagonalization
- Display in tabular format alongside eigenvalues and eigenvector composition
- Format: `Level | mu | n | Energy | Composition`
- Location: Extend `gen_e_summary()` in cfl_util.py or cfl.pyx
- User provides: minimum_q, half_integer_state_labels parameters
- **SCOPE**: Implement printout only for Phase 1 to evaluate on realistic problems 

### Phase 2: Accept (mu, n) in ExData
- Add new ExData mode: "mu_diff" (similar to "sl_diff")
- User specifies: `(mu, n, energy_diff)`
- System matches to calculated (mu, n) pairs
- Replaces fragile (S, L, J, M) matching for robust state identification
>>> Note that in the "sl_diff" example we can specify absolute energies or differences. see these lines in the exmaple: 
 >>>> # The first four elements for each energy are the state label values (SLJM),
>    # the last is the energy.
 >   ex_asl = np.array(
        [[2, 3, 5, 5, 0], [2, 3, 5, 1, 216], [2, 3, 7, 7, 2216.10], [2, 3, 7, 3, 2312.80]]
    )
    # The first eight elements are state label values for the initial and final
    # states, and the last entry is the energy level difference.
    ex_dsl = np.array([[2, 3, 7, 3, 2, 3, 7, 1, 116.0], [2, 3, 7, 1, 2, 3, 7, 5, 729.0]])

>    exdata = cfl.ExData((ex_asl, ex_dsl), ("AS", "DS"), label_key="SLJM")


## Key Design Decisions

- **User Control**: minimum_q and half_integer_state_labels are user-specified parameters
  - Cannot be reliably auto-detected from eigenvector composition alone
  - Provides flexibility for different basis conventions

- **Phase 1 First**: Output only, then integration into ExData
  - Allows users to validate mu/n assignments before using in fits
  - Clear feedback loop for debugging state identification
  - Focus on printout implementation first to evaluate on realistic problems

- **Tabular Display**: Show mu, n, and energy together
  - Users can visually verify the folding and grouping
  - Enables manual entry of (mu, n, energy) into ExData

## Files to Modify

1. **pycf/cfl_util.py** or **pycf/cfl.pyx**
   - Add `calc_mu()` helper function
   - Add `minimum_q` and `half_integer_state_labels` parameters to `gen_e_summary()`
   - Calculate mu for each eigenvalue
   - Calculate n (ordinal within each mu group) for each eigenvalue
   - Format output table with mu and n columns

2. **pycf/cfl.pyx** (ExData class, Phase 2)
   - Add "mu_diff" mode to ExData initialization
   - Implement (mu, n, energy) matching in experimental data parsing

## Backward Compatibility

- All changes are additive (new parameters with defaults)
- Existing energy summaries remain functional (without mu/n columns if parameters not specified)
- ExData "mu_diff" mode is new, doesn't affect existing "sl_diff" or "sl" modes

---

## Implementation Scope for Phase 1

**Focus**: Printout implementation only

Start with calculating and displaying mu and n in energy summary tables. This allows users to:
1. Evaluate how mu/n looks on realistic problems
2. Verify that mu folding is working correctly for their systems
3. Provide feedback before committing to ExData integration


>>> Please change the terminology as decribed above:

>>> "half_integer_state_labels" and "minimum_q"
Plese update the plan to reflect this, 
implement the printout part, and I can evaluate. 
