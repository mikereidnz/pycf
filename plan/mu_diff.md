# Folded Magnetic Quantum Number (mu) - Implementation Differences

## Overview

This document tracks changes between current energy-level display and the new mu-based display with (mu, n) state identification.

## Current Behavior (Before mu Implementation)

### Energy Summary Output

Current format from `gen_e_summary()`:

```
Energy levels (cm⁻¹)
====================

Level |  Energy  | Eigenvalue | Composition
------+----------+------------+--------------------
  1   |    0.0   |    0.0     | 85% |7F0, 0⟩ + ...
  2   |  123.4   |  123.4     | 92% |7F1, 1⟩ + ...
  3   |  145.2   |  145.2     | 88% |7F1, -1⟩ + ...
  4   |  234.5   |  234.5     | 75% |7F0, 0⟩ + ...
  5   |  378.9   |  378.9     | 79% |7F1, 1⟩ + ...
```

### ExData State Specification

Current methods:
- **"sl"**: Specify as (S, L, J, M) - simple systems only
- **"sl_diff"**: Specify relative energy between (S, L, J, M) states - fragile for mixed states
- **Default**: Energy differences relative to reference level

## Proposed Changes (After mu Implementation)

### Phase 1: Output Format Addition

New format from extended `gen_e_summary(..., min_q=2, m_is_half_integer=False)`:

```
Energy levels (cm⁻¹)
====================

Level | mu | n |  Energy  | Composition
------+----+---+----------+--------------------
  1   | 0  | 1 |    0.0   | 85% |7F0, 0⟩ + ...
  2   | 1  | 1 |  123.4   | 92% |7F1, 1⟩ + ...
  3   | 1  | 2 |  145.2   | 88% |7F1, -1⟩ + ...
  4   | 0  | 2 |  234.5   | 75% |7F0, 0⟩ + ...
  5   | 1  | 3 |  378.9   | 79% |7F1, 1⟩ + ...
```

**Changes**:
- Add two columns: `mu` (folded m quantum number) and `n` (ordinal within mu group)
- mu values fold m into fundamental domain based on min_q
- n numbers states sequentially by increasing energy for each mu

**Parameters Added**:
- `min_q: int` - smallest non-zero q in C_kq expansion
- `m_is_half_integer: bool` - whether m values are half-integers (default: False)

**Backward Compatibility**: New columns added after Level, existing columns shift right but remain present

### Phase 2: ExData New Mode

New ExData specification mode:

```python
from pycf import ExData

# Using (mu, n, energy_diff) format
exdata_mu = ExData(
    [
        {"mu": 0, "n": 1, "energy": 0.0},
        {"mu": 1, "n": 1, "energy": 123.4},
        {"mu": 0, "n": 2, "energy": 234.5},
    ],
    mode="mu_diff",
    min_q=2,
    m_is_half_integer=False,
    reference_level=1  # E(mu=0, n=1) = 0
)
```

**Advantages over sl_diff**:
- Directly tied to eigenvector principal components (not idealized quantum numbers)
- Works for heavily mixed states
- No fragile (S, L, J, M) decomposition required
- Cleaner specification format (2 parameters instead of 4)

## Code Changes Summary

### Files Modified (Phase 1)

1. **pycf/cfl_util.py** or **pycf/cfl.pyx**
   - Add `calc_mu(m, min_q, m_is_half_integer)` helper
   - Extend `gen_e_summary()` signature:
     ```python
     def gen_e_summary(..., min_q=None, m_is_half_integer=False, ...):
     ```
   - Calculate mu and n for each eigenlevel
   - Format output table with mu and n columns

2. **pycf/cfl.pyx** (Hamiltonian.gen_summary)
   - Forward min_q and m_is_half_integer to gen_e_summary()
   - Store as optional metadata on Hamiltonian if needed

3. **tests/**
   - Add `test_mu_folding.py` for algorithm validation
   - Add integration tests showing mu/n in summaries
   - Test both m_is_half_integer modes

### Files Modified (Phase 2)

1. **pycf/cfl.pyx** (ExData class)
   - Add "mu_diff" mode to mode selection
   - Add mu, n, min_q, m_is_half_integer to ExData.__init__
   - Implement state matching via (mu, n) lookup

2. **tests/**
   - Add ExData tests for "mu_diff" mode
   - Test state matching accuracy
   - Compare results with sl_diff for validation

## Migration Path for Users

### Current Workflow → New Workflow

**Before** (fragile sl_diff):
```python
exdata = ExData(
    [(7, 6, 1, 0, 0.0),      # S=7/2, L=6, J=0, M=0
     (7, 6, 1, 1, 123.4),    # S=7/2, L=6, J=1, M=1
     (7, 6, 1,-1, 145.2)],   # S=7/2, L=6, J=1, M=-1
    mode="sl_diff"
)
# Often fails: actual states are heavily mixed!
```

**After** (robust mu):
```python
# Step 1: Check energy summary to find mu, n values
print(h.gen_summary(min_q=2))  # Shows mu and n columns

# Step 2: Use (mu, n) values in ExData
exdata = ExData(
    [
        {"mu": 0, "n": 1, "energy": 0.0},
        {"mu": 1, "n": 1, "energy": 123.4},
        {"mu": 1, "n": 2, "energy": 145.2},
    ],
    mode="mu_diff",
    min_q=2,
    m_is_half_integer=False
)
# Works robustly: tied to actual eigenvectors!
```

## Edge Cases and Considerations

1. **Doubled m values**: When m stored as integer (2*m_actual)
   - Must double min_q as well for correct folding
   - Example: if m ∈ {1, 3, 5} (representing 1/2, 3/2, 5/2), min_q=4 becomes min_q_eff=8

2. **States with identical mu**: Ordered by energy
   - Multiple states can have same mu (common in d and f electrons)
   - n uniquely identifies them

3. **Principal component extraction**: May be ambiguous
   - For nearly degenerate eigenvectors, principal component is ill-defined
   - Should warn user if no clear principal component
   - May need confidence threshold

## Testing Strategy

1. **Algorithm**: Verify folding for all m, min_q combinations
2. **Output**: Check mu/n appear correctly in summaries
3. **Matching**: Verify (mu, n) uniquely identifies states
4. **Edge cases**: Half-integer m, near-degenerate states, edge m values
