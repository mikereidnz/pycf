# Plan: Improve inten.py user output and polarization support

## Status Summary

### ✅ COMPLETED: Output Formatting (BRIEF/VERBOSE/ULTRA)
**Commits:** a81db7f, 4482f50, a7e9f8b

All three format levels implemented and working:
- **BRIEF**: Compact group summary (f_ED, f_MD, f_Total columns)
- **VERBOSE**: BRIEF + individual transitions with ED/MD alignment
- **ULTRA**: VERBOSE + dipole moment components (-1, 0, +1) for each transition
- Column order: ED → MD → Total (consistent alignment)
- State labels: Fixed to use principal components (pc_i/pc_f)
- Dipole moments: Cleaned at source via clean_complex(tolerance=1e-12)

**Tested:** All 529 tests pass; examples show correct output

---

## NEXT PHASE: Altp Parameter Fitting

### Problem
Currently, Altp parameters are fixed at input time. To optimize experimental parameter determination and verify the intensity calculation pipeline, we need:
- A fitting routine that adjusts Altp parameters to match target intensity data
- Ability to use computed intensities as target data (self-consistency check)
- Robust minimization (nonlinear parameter space)
- Simple example showing fitting workflow

### Approach
Implement Altp fitting similar to energy-level fitting (cfl_util.py::e_fit pattern):

1. **Fit function**: `fit_altp(h, spectrum, target_intensities, param_names, cfl_min, **kwargs) -> result`
   - Input: Hamiltonian h, target f/A values (from experimental or computed data)
   - Vary: Altp parameter values (e.g., ["A210", "A230"])
   - Minimize: χ² = Σ[(f_computed - f_target)² / σ²]
   - Return: fitted parameters, χ², convergence details

2. **Data structure**: Modify Spectrum or introduce IntensityData class
   - Store computed f/A values per transition group
   - Support both isotropic (MVP) and later polarization types
   - Enable easy access to f/A for fitting

3. **Minimization**: Use nlopt_bobyqa (or user-specified solver)
   - Robust for nonlinear parameter spaces
   - Reuse infrastructure from CFLMin in cfl.py

4. **Example**: `inten_fit_example.py`
   - Load SLJM data
   - Build Hamiltonian with known Altp values
   - Compute "experimental" f/A data using current Altp
   - Fit Altp starting from random/perturbed initial guess
   - Verify fitted parameters converge back to original values
   - Compare fit residuals and convergence details

### Tasks
- [ ] **fit_altp() function**: Design and implement in pycf/inten.py
  - Accept intensity_data (dict of f/A per group)
  - Accept param_names (which Altp params to fit)
  - Accept optional weights (uncertainties in f/A targets)
  - Return result dict with fitted values, χ², details
  
- [ ] **IntensityData or fit-friendly storage**: Store f/A alongside groups in Spectrum
  - Simplify extraction of fit targets from Spectrum
  - Enable easy serialization (CSV export)
  
- [ ] **Test case**: `test_fit_altp.py` in tests/
  - Known f/A values (from compute at fixed Altp)
  - Fit with perturbed initial guess
  - Verify convergence and parameter recovery
  - Spot-check χ² and correlation matrix
  
- [ ] **Example**: `inten_fit_example.py`
  - Similar to inten_example.py, but with fitting step
  - Use isotropic f/A only (MVP)
  - Show before/after parameter and f/A values
  - Optional: plot residuals or convergence

### Implementation Strategy (MVP)
**Phase 1: Isotropic fitting only**
- Fit to isotropic f (no polarization choice yet)
- Single spectrum (absorption or emission, not both simultaneously)
- Symmetric residual: `χ² = Σ abs((f_i^computed - f_i^target) / (f_i^computed + f_i^target))²`
  (More robust than simple relative error, symmetric in computed/target)
>>> Should be abs((computed - target)/(computed + target))^2. 

**Phase 2: Enhancements (future)**
- Support multiple spectra (simultaneous fit to absorption + emission)
- Weighted fits (user-provided uncertainties)
- Polarization-dependent fits (linear, circular)
- Correlation matrix output
- Confidence intervals on fitted parameters

### Notes
- Function naming: follow e_fit pattern (verb + noun: fit_altp, not altp_fit or fitaltp)
- Solver choice: CFLMin with 'nlopt_bobyqa' method (robust for nonlinear)
- Initial guess: start from user-provided Altp or random perturbation
- Robustness: handle edge cases (zero f values, singular matrices, non-convergence)
- Documentation: docstrings explain parameter space, solver method, return format

---

## Reference: Energy-Level Fitting (e_fit pattern)
```python
# Similar pattern in cfl_util.py:
def e_fit(param_names, h, exdata, cfl_min, **kwargs):
    """
    Minimize χ² for energy level fitting.
    param_names: list of tensor names to vary
    h: Hamiltonian
    exdata: experimental data (ExData object)
    cfl_min: CFLMin solver object
    Returns: result dict with fitted parameters, χ², details
    """
```

For Altp fitting, we follow a similar structure:
```python
def fit_altp(h, spectrum_config, target_intensities, param_names, cfl_min, **kwargs):
    """
    Minimize χ² for Altp parameter fitting.
    param_names: list of Altp parameter names to vary (e.g., ["A210", "A230"])
    target_intensities: dict of f or A values per group
    Returns: result dict with fitted parameters, χ², details
    """
```

---

## File Organization
- **Primary**: pycf/inten.py (fit_altp function + supporting code)
- **Tests**: tests/unit/test_fit_altp.py + tests/integration/test_inten_fit_c3.py
- **Examples**: examples/ceylf/inten_fit_example.py (based on inten_example.py)
- **Documentation**: Update EXAMPLES.md with fitting example

