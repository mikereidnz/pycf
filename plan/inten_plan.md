# Plan: Intensity Output Formatting and Experimental Data Integration

## Status Summary

### ✅ COMPLETED
- **Output Formatting (BRIEF/VERBOSE/ULTRA)**: All three formats implemented (a81db7f, 4482f50, a7e9f8b)
- **Altp Parameter Fitting**: Working implementation with uncertainty estimation (commits 17b5c3e, 378b1f3, 2610221)
  - `fit_altp()` with scipy.optimize.minimize
  - Hessian-based parameter uncertainties
  - Two example files: magnetic field and zero-field cases
  - All 529 tests pass

---

## CURRENT PHASE: Experimental Data Integration with BRIEF Output

### Next Task: Integrate Experimental Values into BRIEF Line

**Goal**: Enhance BRIEF output to include experimental intensity values alongside calculated values

**Required output format** (for each group line):
```
Calculated_f  Experimental_f  χ² (residual metric)
```

**Details needed**:
1. How experimental values are provided (spectrum object method? external dict?)
>>> At this point I think an external list would be OK: 
  [ [i1, f1, expt1],
    [i2, f2, expt]]
2. Which χ² function to use (e.g., `((calc - exp) / (calc + exp))²` or simple relative error?)
>>> the `((calc - exp) / (calc + exp))²` function. 
3. Whether to aggregate χ² across all groups or show per-group
>>> list for each group, with a total at the bottom. 

**Examples**: 
- No magnetic field: 6 transition groups, 2 initial states
- With magnetic field: 8 transition groups, 1 initial state
>>> Yes, do both examples. 
---

## Possible Future Enhancements

>>> 1. To make this useful for testing I will need to be able to specify that the experimental data should be normalized to a particular transition. Usually a pure magnetic dipole transition. 

>>> 2. Then I want to plot the spectrum and the experimental data. 
