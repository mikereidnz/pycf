# Plan: Improve inten.py user output and polarization support

Problem
-------
inten.py currently emits low-level arrays and dictionaries as diagnostic output. Users need higher-level, human-friendly summaries and more flexible polarization options (isotropic, linear, circular), plus machine-readable exports (CSV/JSON/LaTeX). This will improve usability and enable downstream analysis (plots, fitting pipelines).

Goal
----
Refactor inten.py to provide:
- A stable public API for intensity generation: gen_intensity(h, ex, polarization=..., basis=..., normalize=..., output_format=...)
- A human-readable summary printer: gen_inten_summary(edata, format='text'|'json'|'csv')
- Support for named polarizations: 
    - Currently: isotropic, axial, sigma, pi. 
    - Future extensions: 
        - Circular (sigma_plus and sigma_minus).
        - Ability to specify arbitrary k vector (propagation direction) and E vector (electric field of the radiation) (for linear or circular cases).
- Unit tests and example scripts showing linear and circular polarization outputs
- Backwards compatibility is not very important, since there is very little code that uses the current format. 

Approach (implementation steps)
-------------------------------
1. **Define Spectrum dataclass** in pycf/inten.py:
   - Input fields: name (str), lrange (list of lists), intensity_tensors (dict from ImportSLJM), 
     altp (optional), group_tol (default 1e-3), nrefractive (default 1.0).
   - Computed fields: transformed_tensors (from vtrans), dipole_str_output, groups, oscillator_strengths.

2. **Implement gen_intensity(hamiltonian, spectrum) -> list of dicts**:
   - Extract eigenvectors/eigenvalues from hamiltonian.diag().
   - Call vtrans() on spectrum.intensity_tensors.
   - Call dipole_str() with spectrum.lrange and optional spectrum.altp (ED only).
   - Call group_transitions() with spectrum.group_tol.
   - Call add_oscillator_strengths_and_A_coefficients() with spectrum.nrefractive.
   - Store results in spectrum; return results list.

3. **Implement gen_inten_summary(spectrum, format='text') -> str**:
   - Iterate groups; format each as: initial_label (principal component), initial_energy, 
     final_label, final_energy, f_E, f_M, f_total (or A coefficients for emission).
   - Use cfl_util.py label formatters for state labels.
   - Text output: pretty table with aligned columns.
   - CSV output: columns matching text table.

4. **Polarization support (MVP)**:
   - Accept polarization='isotropic' parameter; later extend to axial/sigma/pi.
   - Internally, dipole_str always computes all components; select aggregate in summary.

5. **Unit tests** (test_inten_ui.py expansion):
   - Test Spectrum creation and field validation.
   - Test gen_intensity() with C1/C3 example tensors; verify against group_transitions output.
   - Test gen_inten_summary() text/CSV formatting; spot-check state labels and energies.

6. **Example script**: examples/ceylf/inten_example.py
   - Load CF and intensity tensors (ImportSLJM).
   - Define two Spectrum objects (absorption, emission).
   - Call gen_intensity() for each.
   - Print summaries and write CSV.

7. **Update docs/CHANGELOG**: Document new Spectrum class, gen_intensity(), gen_inten_summary().

Deliverables
------------
- plan/inten_plan.md (this file)
- plan/inten_report.md (progress/log)
- plan/inten_diff.md (design diffs and API notes)
- pycf/inten.py refactor, new inten/_polarization.py, tests, examples, docs.

Design decisions (answered by user)
-----------------------------------
**Spectrum abstraction**: Each Hamiltonian can have multiple named spectra (e.g., "ground state absorption", "emission from 2F₇/₂"). Gen_intensity() should accept a Hamiltonian index and spectrum label for readability and specificity.

**Oscillator strength output**: Print electric dipole **f_E** and magnetic dipole **f_M** separately for physics insight, but also compute and display total **f_total = f_E + f_M** as the observable. Same structure for A coefficients (emission).

**Weighted residuals**: Use |(expt − calc)/(expt+calc)|² as the weighted loss, with optional per-spectrum and per-transition-group weighting factors for flexibility in fitting.

**Implement spectrum abstraction now** (not deferred) since examples already mix absorption and emission.

**Polarization convention**: sigma_plus = (1, +i)/√2 (right-circular, S₃ > 0). Document explicitly; provide future option for alternative sign if needed.

**Output formats** (short-term): text (pretty-print) and CSV. JSON/LaTeX as phase 2.

**Short-term output format** (initial implementation):
- Print: initial state label (SLJM major component) + energy, final state label + energy.
- For absorption (final_energy > initial_energy): f_E, f_M, f_total oscillator strengths.
- For emission (final_energy < initial_energy): A coefficients (similarly decomposed).
- If experimental data available: e_obs, weighted residual, chi² contribution, weight.
- Optionally: per-transition dipole moments and individual transition data.
- Reuse SLJM label formatting from cfl_util.py (e.g., gen_e_summary style).

Core data structures:
- `Spectrum`: named collection of transitions. Each transition holds (initial_state, final_state, polarization). Optional experimental data per transition.
- `IntensityResult`: structured output with columns: initial_label, initial_energy, final_label, final_energy, polarization, f_E, f_M, f_total, a_coeff, e_obs, weight, residual, chi2_contribution.
- `gen_intensity(hamiltonian, spectrum_label, polarization='isotropic', normalize=True) -> IntensityResult`
- `gen_inten_summary(result, format='text'|'csv') -> str`


Next steps (implementation roadmap)
-----------------------------------
**MVP Phase 1** (isotropic polarization, no experimental data):
1. Define Spectrum dataclass with input/computed fields.
2. Implement gen_intensity(hamiltonian, spectrum) orchestrating vtrans→dipole_str→group→strengths.
3. Implement gen_inten_summary() text and CSV output with state labels and energies.
4. Write unit tests (Spectrum validation, gen_intensity correctness, label formatting).
5. Create examples/ceylf/inten_example.py (two-spectrum absorption/emission).
6. Update CHANGELOG and inline docs.

**Phase 2** (post-MVP): 
- Extend to axial/sigma/pi polarizations with per-spectrum selection.
- Add experimental data flow (analogous to ExData).
- **Multiple output formats per user feedback:**
  - Detailed multi-line (current MVP: state labels, energies, f/A, lifetime)
  - Compact one-line-per-group (for quick scanning)
  - Data-oriented: JSON (programmatic), LaTeX (publication), enhanced CSV (metadata)
- Per-transition-group weights for fitting pipelines.

See also: 
- plan/inten_research.md for polarization background.
- plan/inten_diff.md for API design notes.
- pycf/polarization.py for working example of Jones vectors and helpers.
- tests/unit/test_inten_ui.py for test structure.



Design clarifications (from C3 example review)
-----------------------------------------------

**Spectrum class design**: Spectrum contains input parameters (lrange, intensity tensors, altp) 
and computed outputs (transformed tensors, dipole moments, oscillator strengths, groups).
Initialize with file paths/params; call gen_intensity() to compute derived data.

**Intensity tensor loading**: Users provide file paths (via ImportSLJM) just like CF tensors.
Spectrum will store the imported tensor objects; gen_intensity() receives Spectrum with 
tensors already loaded.

**Altp parameters**: Like Ckq parameters in Hamiltonian. Passed to Spectrum during init
or provided per-call to gen_intensity(). Initially support isotropic only; defer multi-polarization UI.

**Output structure**: Keep list-of-dicts (from dipole_str/group_transitions). Lightweight; 
most CPU is in vtrans. Minimal wrapper overhead is acceptable.

**Experimental data**: Future enhancement. Similar pattern to ExData: user creates list
of observed intensities with weights. For now, focus on computed spectra only.

**Polarization**: Isotropic only for MVP (phase 1). State labeling uses principal component
(max abs value in eigenvector). Group tolerance is user-configurable (default 1e-3, override
for hyperfine structure). Refractive index exposed as parameter to A_and_f_calc().

**Workflow**: The procedural pattern (load tensors → vtrans → dipole_str → group → add_strengths)
is wrapped cleanly. Spectrum class encapsulates state; gen_intensity() orchestrates computation.

      update_coeff(coeff, res)


Architecture Refactoring (Phase 2 Preparation)
-----------------------------------------------

**Completed Refactoring:**
1. Renamed t_list keys for consistency: pci -> pc_i, pcf -> pc_f
2. Optimized format functions: removed O(n) energy-matching lookup
   - Now extract level indices from t_list[0]["pc_i"], t_list[0]["pc_f"]
   - Benefit: MEDIUM, BRIEF, VERBOSE formats all efficient
3. Added direction validation: each Spectrum must be purely absorption OR emission

**t_list Structure (ready for all formats):**
```
Each transition in group["t_list"][n]:
  "i": initial level (0-based)
  "f": final level (0-based)
  "pc_i": principal component index of initial level  ✨ newly added
  "pc_f": principal component index of final level    ✨ newly added
  "ei": initial energy
  "ef": final energy
  "e": transition energy
  "isotropic": isotropic dipole strength
  "md_-1", "md_0", "md_+1": magnetic dipole moments
  "ed_-1", "ed_0", "ed_+1": electric dipole moments
  "S_ED_-1", "S_ED_0", "S_ED_+1": ED dipole strengths
  "S_MD_-1", "S_MD_0", "S_MD_+1": MD dipole strengths
  "S_ED_isotropic", "S_MD_isotropic": isotropic strengths
  (also: axial, sigma, pi for each)
```


Output Formats Design (BRIEF and VERBOSE)
-----------------------------------------

**MEDIUM Format (Current - MVP)**
>>> This will become redundant with the implementation of the verbose formats. 
```
Spectrum: absorption
================================================================================
Altp (electric dipole coupling) parameters:
  A210: 1e-10
  A230: -1e-10

Transition 1: [2,3,5,-5] -> [2,3,7,7]
  Initial state: [2,3,5,-5]              E =   0.000000 cm-1 (g=2)
  Final state:   [2,3,7,7]               E = 2169.756 cm-1 (g=2)
  Transition energy: 2169.756000 cm-1
  Oscillator strength f: 4.482614e-08

Transition 2: [2,3,5,-5] -> [2,3,7,-5]
  ...

Total oscillator strength (f): 8.631216e-08
================================================================================
```
- Pros: Clear, human-readable, shows state labels and energies
- Cons: Verbose for many transitions


**BRIEF Format (Completed - Commit a85de1c)**
✅ Tabular format, one line per transition group
✅ Three columns: f_MD, f_ED, f_Total (or A_MD, A_ED, A_Total for emission)
✅ Shows Altp parameters at top
✅ State labels with level indices (1-based) and quantum numbers in "| S L J M >" format
✅ Includes initial and final state energies
✅ 132-character width


**VERBOSE and More Verbose Formats (To be designed)**

All more verbose formats will use BRIEF as the base line, then expand with additional detail on subsequent lines.

Hierarchy of verbosity:
1. BRIEF: One line per group (completed)
2. VERBOSE: BRIEF line + additional transition details below each group
3. ULTRA: VERBOSE + dipole moment components
4. DEBUG: All available data for each transition



IMPLEMENTATION STATUS

**COMPLETED: BRIEF Format (Commit a85de1c)**
```
Spectrum: Ground state absorption (Z1 -> Y1 + Y2)
====================================================================================================================================
Altp (electric dipole coupling) parameters:
  A210: 1e-10
  A230: -1e-10
  A233: (1e-10+2e-10j)

Group  Initial State                                      Final State                                        f_MD           f_ED           f_Total       
------------------------------------------------------------------------------------------------------------------------------------
1      2: | 2 3 5 5 > (E =     0.000000 cm-1)             14: | 2 3 7 -7 > (E =  2169.756474 cm-1)            1.103784e-04   7.235053e-07   4.482614e-08
2      2: | 2 3 5 5 > (E =     0.000000 cm-1)             3: | 2 3 7 5 > (E =  2313.748603 cm-1)              2.736630e-05   7.545708e-05   4.148602e-08
------------------------------------------------------------------------------------------------------------------------------------
Total                                                                                                                                       8.631216e-08
====================================================================================================================================
```

- ✅ Tabular format with one line per transition group (Option A chosen)
- ✅ Three columns of f or A values: MD, ED, Total (user preference)
- ✅ 132-character width for ample column space
- ✅ Shows Altp parameters at top
- ✅ State labels include level index (1-based) and quantum numbers in "| S L J M >" format
- ✅ Shows both initial and final state info with energies
- ✅ Added to gen_inten_summary() format dispatch
- ✅ Works for both absorption and emission spectra
- ✅ Updated examples/ceylf/inten_example.py to demonstrate
- ✅ All 50 tests pass

**Implementation Details (Commit a85de1c):**
1. Added _format_state_label_with_energy() helper to format "level: | S L J M > (E = x.xxx cm-1)"
2. Added _format_inten_brief() function that:
   - Extracts state labels from t_list[0]["pc_i"] and t_list[0]["pc_f"] (principal components)
   - Formats header with "Group", "Initial State", "Final State", "f/A_MD", "f/A_ED", "f/A_Total"
   - Handles both absorption (f) and emission (A) modes automatically
   - Pads all columns to 132-character total width
3. Updated gen_inten_summary() to dispatch 'brief' format to _format_inten_brief()

**Known Issues/Corrections needed for BRIEF:**
1. f_MD and f_ED calculations: Use same logic as total calculation for ED/MD decomposition. Total f and A should be sum of MD and ED.
2. For emission, A_MD and A_ED: Remove placeholder zeros; use ED/MD decomposition logic.
3. Remove misleading total dipole strength (only meaningful when nrefractive=1).


VERBOSE Format (To be implemented)
----------------------------------

VERBOSE expands BRIEF by adding individual transition details below each group line.

Structure:
- Print BRIEF line for each group (same as BRIEF format)
- Below each BRIEF line, print: "Individual transitions:"
- List each transition in the group with a tabular format
- Blank line before next group

Example layout:
```
Spectrum: Ground state absorption (Z1 -> Y1 + Y2)
====================================================================================================================================
Altp (electric dipole coupling) parameters:
  A210: 1e-10
  A230: -1e-10
  A233: (1e-10+2e-10j)

Group  Initial State                                      Final State                                        f_MD           f_ED           f_Total       
------------------------------------------------------------------------------------------------------------------------------------
1      2: | 2 3 5 5 > (E =     0.000000 cm-1)             14: | 2 3 7 -7 > (E =  2169.756474 cm-1)            1.103784e-04   7.235053e-07   4.482614e-08
       Individual transitions:
       i(0b)  f(0b)  Energy(cm-1)  S_ED_iso     S_MD_iso     f_ED           f_MD           f_Total
         1      13     2169.756    6.33e-03     1.75e-05     7.235e-07      1.103e-04      4.482e-08
         0      13     2169.756    1.75e-05     6.33e-03     7.235e-07      1.103e-04      4.482e-08

2      2: | 2 3 5 5 > (E =     0.000000 cm-1)             3: | 2 3 7 5 > (E =  2313.748603 cm-1)              2.736630e-05   7.545708e-05   4.148602e-08
       Individual transitions:
       i(0b)  f(0b)  Energy(cm-1)  S_ED_iso     S_MD_iso     f_ED           f_MD           f_Total
         1       2     2313.749    ...          ...          ...            ...            ...
         0       2     2313.749    ...          ...          ...            ...            ...

------------------------------------------------------------------------------------------------------------------------------------
Total                                                                                                                                       8.631216e-08
====================================================================================================================================
```