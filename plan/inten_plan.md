# Plan: Intensity Output Formatting and Experimental Data Integration

## Status Summary

### ✅ COMPLETED
- **Output Formatting (BRIEF/VERBOSE/ULTRA)**: All three formats implemented (a81db7f, 4482f50, a7e9f8b)
- **Altp Parameter Fitting**: Working implementation with uncertainty estimation (commits 17b5c3e, 378b1f3, 2610221)
  - `fit_altp()` with scipy.optimize.minimize
  - Hessian-based parameter uncertainties
  - All 529 tests pass

- **Experimental Data Integration with BRIEF Output** ✅ (Phase completed)
  - Enhanced BRIEF format with f_Expt and χ² columns
  - χ² formula: `((calc - exp) / (calc + exp))²`
  - Per-group χ² with total aggregation
  - Spectrum.expt_data field added: list of [group_idx, f_expt] pairs
  - Two working examples with before/after fitting demonstration
    - Magnetic field: 1 initial → 8 final states (8 transition groups)
    - Zero-field: 2 initial → 6 final state groups (degenerate states)
  - BEFORE FIT shows perfect baseline agreement (target intensities)
  - AFTER FIT shows improved fit to synthetic noisy experimental data

---

## CURRENT PHASE: Planning Plotting Output

**Goal**: Add simple plotting capability to visualize:
- Chi-square residuals across transition groups
- Before/after parameter comparison
- Intensity comparison (calculated vs experimental)

**User notes for planning**:
>>> My main aim is to create the Intensity comparison. 

>>>  inten_plot()

>>> 1. Plot of the calculated f (or A) convoluted with the lorentzian() funtion given in the inten.pyt file. 
I suggest renaming this lorentzian_constant_height() to make it clear that it preserves the height not the area. 
The only place it is used is in the inten() function, which inten_plot() will supercede. 
Do this plot with a default fwhm (full width half maximum) of 1 cm-1. 

>>> 2.  Stick plot of the experimental data. I.e. vertical line at the calculated energy. 


---

## Possible Future Enhancements

>>> - Chi-square residuals across transition groups
- Before/after parameter comparison
- Using experimental rather than calculated energies. 
- Variable line widths and/or other line shapes. 