# Plan: Intensity Output Formatting and Experimental Data Integration

## Status Summary (Updated 2026-05-02 17:50)

### ✅ COMPLETED PHASES

#### Phase 1: Output Formatting (BRIEF/VERBOSE/ULTRA)
- All three formats implemented with unified architecture ✅
- Refactored from 3 separate functions → unified `_format_inten()` with format parameter
- Eliminated 300+ lines of duplicated code (commits f4dd51c → cc85723)
- Helper sub-functions: `_format_group_line()`, `_format_transition_line()`, `_format_dipole_moments()`
- **All 522 tests passing**

#### Phase 2: Experimental Data Integration
- Enhanced BRIEF format with f_Expt and χ² columns ✅
- χ² formula: `((calc - exp) / (calc + exp))²` per-group with total aggregation
- Spectrum.expt_data field: list of [group_idx, f_expt] pairs
- Two working examples with before/after fitting demonstration
- BEFORE FIT shows perfect baseline agreement; AFTER FIT shows fit to noisy data
- **All 522 tests passing**

#### Phase 3: Intensity Plotting
- Renamed `lorentzian()` → `lorentzian_constant_height()` for clarity ✅
- Implemented `inten_plot()` with experimental data overlay
- High-resolution plots: 10,000 grid points, FWHM=0.5 cm⁻¹
- PDF output for examples
- **All 522 tests passing**

#### Phase 4: Format Output Refactoring
- Successfully consolidated three format functions into unified architecture ✅
- Eliminated wrapper stubs (`_format_inten_brief`, `_format_inten_verbose`, `_format_inten_ultra`)
- Renamed `_format_inten_unified()` → `_format_inten()`
- **Net -112 lines of code deleted**
- **All 522 tests passing**

#### Phase 5: Function Cleanup
- Deleted obsolete `inten()` function, replaced by more versatile `inten_plot()` ✅
- Removed TestIntenBounds test class (validation now covered by Spectrum class)
- Updated integration tests to remove inten() calls
- **Net -81 lines deleted**
- **All 522 tests passing**

#### Phase 6: Deep Code Review & Guard Code Audit
- Comprehensive code review by rubber-duck agent identified 8 issues ✅
- **Fixed 3 BLOCKING issues**: Array bounds violations (IndexError hazards)
  - Guard state_labels access in _format_group_line()
  - Guard state_labels access in _format_transition_line()
  - Guard state_labels access in CSV formatter
- **Fixed 2 HIGH-SEVERITY issues**: Division by zero, type coercion
  - Added g_i > 0 validation in A_and_f_calc()
  - Added type conversion with error handling for expt_data
- **Fixed 3 MEDIUM issues**: Logic consistency, parameter validation
  - Added explicit validation in is_absorption logic
  - Consistent error handling in expt_lookup parsing
- **All 522 tests passing after fixes**

#### Phase 7: State Numbering & Plot Enhancements
- Fixed state numbering to use energy level indices instead of principal components ✅
  - Transitions store both 'i','f' (actual level indices) and 'pc_i','pc_f' (for SLJM matching)
  - Now displays intuitive energy-ordered level numbers (1,2,3...) not eigenvector dominance
- Added ylim parameter to inten_plot() for intensity (y-axis) range control ✅
  - Complements existing xlim for energy (x-axis) range
  - Auto-scales if ylim=None (default behavior)
- Implemented unique figure naming for simultaneous multiple plots ✅
  - Each plot named: spectrum.name + random UUID suffix
  - Allows same spectrum plotted with different zoom levels without replacement
- Added Energy column to output formatting ✅
  - Shows energy difference (cm⁻¹) for each transition group
  - Takes absolute value for both absorption and emission
  - New header: "Group, Energy, Initial State, Final State, ..."
- Fixed energy sign handling in plots ✅
  - Energy stored as (e_f - e_i): positive for absorption, negative for emission
  - inten_plot() uses abs() for consistent positive display
- Updated examples (inten_fit_example.py, inten_fit_example_nofield.py) ✅
  - Assigned unique names to before/after fit spectra
- **All 499 tests passing** (test count reduced by removing excessive parametrization)

---

## Code Quality Metrics (Current)

- **Lines eliminated**: 193 total (formatting consolidation -112, obsolete function -81)
- **Guard code improvements**: 8 issues fixed in code audit
- **Test coverage**: 499 passing, 16 skipped (reduced from 522 by optimizing njsymbols tests)
- **Backward compatibility**: Fully maintained (lorentzian wrapper, old format names)
- **Documentation**: Updated docstrings with guard code rationale, state numbering, and plot controls

---

## Recent Enhancements (Phase 7)

### State Numbering Fix
- Changed output to display energy level indices (1-based) instead of principal component indices
- More intuitive display: "Level 1, Level 2, ..." matches energy order
- Transitions still use pc_i, pc_f internally for SLJM state matching

### Plot Range Controls
- Added `ylim` parameter to inten_plot() for intensity range control
- Each plot gets unique figure identifier (spectrum name + UUID suffix)
- Allows viewing same spectrum with different zoom levels

### Output Formatting
- Added Energy column to printout (shows |e_f - e_i| for each group)
- Consistent positive energy display for both absorption and emission

---

## Known Limitations & Future Work

### Identified but Deferred
1. Chi-square residuals visualization across transition groups
2. Before/after parameter comparison plotting
3. Using experimental energies instead of calculated (for fitting to measured spectra)
4. Variable line shapes beyond Lorentzian
5. Normalization of experimental data to reference transition (pure magnetic dipole)

### Design Decisions Made
- expt_data structure: `[group_idx, f_expt]` pairs (simple, efficient)
- Type coercion: Graceful fallback (skip malformed entries) vs strict validation
- Default FWHM: 0.5 cm⁻¹ for sharp spectral features
- Grid resolution: 10,000 points for smooth curves
- Error handling: Defensive bounds checking on all array access

---

## Architecture Notes

### _format_inten() unified formatter (1062 lines)
- Single function handles all three formats via `format` parameter
- Helper sub-functions eliminate repetition:
  - `_format_group_line()`: State labels, energies, dipole decomposition
  - `_format_transition_line()`: Individual transitions with dipole values  
  - `_format_dipole_moments()`: ED/MD component formatting
- Expt_data lookup: group_idx → f_expt dictionary with type validation
- Is_absorption determination: First group energy > 0, with empty-check guard

### inten_plot() visualization (1850 lines)
- Operates on Spectrum objects (modern integrated design)
- Parameters: fwhm (0.5), npoints (10,000), xlim (auto), figsize (12,6)
- Lorentzian convolution via lorentzian_constant_height()
- Experimental overlay: red stick lines with type-safe expt_data parsing
- Returns (fig, ax) for user customization

### A_and_f_calc() physics engine (470 lines)
- Validates g_i > 0 at entry (prevents division by zero)
- Handles energy=0 edge case (λ, ω = 0)
- Refractive index correction factor
- Returns absolute values (A, f)

---

## Commit History (Last 2 Days)

```
b000a51 feat: use energy level indices for state numbering + add plot controls
e5c4be4 Performance optimization for njsymbols tests (13x speedup)
f92b439 Fix example format names (verbose/ultra → detailed/moments)
2db0a68 Guard code test suite creation
4b06bd3 Documentation updates (plan, report, diff files)
b481343 fix: add guard code for array bounds and division by zero
16fa219 refactor: delete obsolete inten() function, replaced by inten_plot()
cc85723 cleanup: remove trivial wrapper stubs, rename unified formatter
f4dd51c refactor: consolidate intensity output formatting functions
93e2c84 feat: add inten_plot() function for intensity visualization
```

**Latest impact**: 
- State numbering corrected (more intuitive energy-ordered display)
- Plot controls added (ylim parameter, unique figure naming)
- Energy column added to output formatting
- Test suite optimized (13x faster without losing coverage)

## Usability and streamlining (Phase 8)

The intensity calculation, display, plotting, and fitting are now working, but
the current user workflow is still too complicated. Phase 8 will align the
intensity workflow more closely with the ergonomics of the energy-level fitting
workflow (e.g., `mh_fit`) while prioritizing simple user-facing calls.

### Goals

1. Recalculate a spectrum without rebuilding the object.
2. Refactor `fit_altp` to accept `Spectrum` objects directly.
3. Support fitting across multiple `Spectrum` objects.
4. Standardize data structures and output summaries for easier scripting.

### Scope and design decisions

#### 1) Recalculation and caching within `Spectrum`

- A `Spectrum` should be reusable after construction.
- Changing Altp values should trigger intensity recomputation without forcing
  users to rebuild the object.
- The expensive `vtrans` step should be skipped when Hamiltonian eigenvectors
  have not changed.
- Cache invalidation should be explicit:
  - If Hamiltonian/eigenvectors change, recompute `vtrans`.
  - If only Altp changes, reuse cached transformed tensors.

#### 2) Refactor `fit_altp` API

- `fit_altp` should operate on `Spectrum` objects (single or list), not on
  constructor ingredients that rebuild spectra internally.
- `fit_altp` should not require a separate Hamiltonian argument, since each
  `Spectrum` already owns its Hamiltonian reference.
- Return both:
  - numerical results (fitted parameters, chi², uncertainties)
  - a formatted text summary suitable for printing (like `mh_fit` style output)

#### 3) Multi-spectrum fitting

- Add a list-based fitting path analogous to `mh_fit`.
- Inputs should support:
  - list of `Spectrum` objects
  - list of experimental intensity datasets
  - shared list of fit parameter names (common Altp parameters)
- Defer weighting and non-isotropic polarization options until a later phase.

#### 4) Data model simplification

- Use dictionary-style Altp parameter storage (consistent with Hamiltonian
  coefficient dictionaries) instead of list-of-lists.
- Experimental data can remain user-provided dictionaries/lists for now, as
  long as accepted formats are clearly documented and validated.

### Proposed user workflow (target state)

1. Build and diagonalize Hamiltonian (`h.diag()`).
2. Assemble intensity tensors.
3. Create one or more `Spectrum` objects.
4. Attach/set experimental intensity data for each spectrum.
5. Print summary tables for all spectra.
6. Call `fit_altp` with spectrum object(s), experimental data, and parameter
   names.
7. Receive fitted values plus a printable summary string.

### Phase 8 implementation order

1. Single-spectrum API cleanup (`Spectrum` reuse + recalculation semantics).
2. `fit_altp(spec, ...)` refactor and summary return.
3. Multi-spectrum `fit_altp([spec1, spec2, ...], ...)`.
4. Tests/examples refresh and documentation update.
