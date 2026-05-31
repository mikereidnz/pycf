# Folded Magnetic Quantum Number (mu) - Development Report

## Status: Planning Phase

### Current Focus
Creating planning and specification documents for Phase 1 implementation (output mu and n in energy summaries).

### Motivation
The existing (S, L, J, M) state identification method (`sl_diff` in ExData) is fragile for realistic crystal-field problems with heavily mixed eigenstates. The folded magnetic quantum number (mu) provides a robust, basis-independent alternative tied directly to eigenvector principal components.

### Algorithm Overview

For each energy level, the folded magnetic quantum number mu is calculated as:

```python
def calc_mu(m, min_q, m_is_half_integer=False):
    """
    Calculate folded magnetic quantum number.
    
    Parameters
    ----------
    m : int or float
        Magnetic quantum number from principal eigenvector component
    min_q : int
        Smallest non-zero q in C_kq tensor expansion (e.g., 2 for C20, C22)
    m_is_half_integer : bool
        If True, m is actual (may be half-integer as float)
        If False, m is doubled integer (to avoid half-integers in storage)
    
    Returns
    -------
    mu : int
        Folded magnetic quantum number in range [0, min_q//2]
    """
    if m_is_half_integer and min_q % 2 == 1:
        raise ValueError("min_q must be even when m_is_half_integer=True")
    
    # Account for doubled storage
    m_eff = m if m_is_half_integer else m
    if not m_is_half_integer:
        min_q_eff = min_q * 2  # Convert to doubled space
    else:
        min_q_eff = min_q
    
    mu = abs(m_eff) % min_q_eff
    if mu > min_q_eff // 2:
        mu = min_q_eff - mu
    return mu
```

### Key Parameters

1. **min_q**: Smallest non-zero q value across all C_kq tensors
   - User-specified based on Hamiltonian structure
   - Determines the period of the folding

2. **m_is_half_integer**: Flag for m value representation
   - True: m values are actual (floats, may be half-integer)
   - False: m values are doubled integers (to avoid half-integers)
   - Affects how min_q is used in folding calculation

3. **n**: Ordinal index of states with same mu
   - Computed by sorting levels with same mu by energy
   - n=1 is lowest energy state with that mu
   - Used for unambiguous state specification

### Output Format Example

For a d-electron system (J=2, so m ∈ {-2, -1, 0, 1, 2}), with min_q=2:

```
Level | mu | n | Energy (cm⁻¹) | Composition
------+----+---+---------------+----------------------
  1   | 0  | 1 |      0.0      | 85% |d,0⟩ + 10% other
  2   | 1  | 1 |    123.4      | 92% |d,1⟩ + 5% other
  3   | 1  | 2 |    145.2      | 88% |d,-1⟩ + 8% other
  4   | 0  | 2 |    234.5      | 75% |d,0⟩ + 20% other
  5   | 1  | 3 |    378.9      | 79% |d,1⟩ + 15% other
```

### Development Phases

#### Phase 1: Output mu and n (Current)
- [ ] Add `calc_mu()` helper function
- [ ] Add `min_q` and `m_is_half_integer` parameters to energy summary functions
- [ ] Calculate n (ordinal within mu group) for each level
- [ ] Format output table with mu and n columns
- [ ] Update docstrings and parameter documentation
- [ ] Add comprehensive tests for folding algorithm
- [ ] Test with realistic crystal-field problems

#### Phase 2: ExData Integration (Future)
- [ ] Add "mu_diff" mode to ExData class
- [ ] Implement (mu, n, energy) matching against eigenvector metadata
- [ ] Add tests for state matching accuracy
- [ ] Update examples to show (mu, n) usage

### Testing Strategy

1. **Unit Tests**: Verify folding algorithm
   - Edge cases: m=0, m=min_q//2, m=-1, etc.
   - Both m_is_half_integer=True and False
   - Various min_q values (2, 4, 6, etc.)

2. **Integration Tests**: Verify output in energy summaries
   - Compute mu for known eigenvectors
   - Verify n ordering is correct
   - Compare with manual calculations

3. **Validation Tests**: Test with realistic problems
   - Verify (mu, n) uniquely identifies states
   - Compare with sl_diff for simple cases
   - Test cases where sl_diff fails but mu succeeds

### Known Issues / Considerations

1. **Half-Integer Handling**: Need to clarify:
   - When are m values stored as half-integers vs. doubled?
   - How to detect this automatically if possible?
   - Guidance for users on how to set m_is_half_integer correctly

2. **min_q Selection**: User must specify correctly
   - Could provide helper to extract min_q from Hamiltonian tensors
   - Should validate that min_q makes sense for the system

3. **Principal Component**: Assumes principal component m is reliable
   - Should verify eigenvector concentration on one SLJ component
   - May need warning if no clear principal component

### Questions for User

1. Should min_q and m_is_half_integer be properties of Hamiltonian or passed per-call?
2. Should we try to auto-detect min_q from the tensor set?
3. How should we handle degenerate or near-degenerate states when extracting principal m?

### Next Steps

1. Review and refine mu_plan.md and mu_diff.md
2. Implement Phase 1 (output mu and n)
3. Create comprehensive test suite
4. Validate with realistic crystal-field problems
5. Plan Phase 2 integration with ExData
