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


**BRIEF Format (Compact, one line per group)**

Design Question 1: Line format?

Option A (Tabular, aligned):
```
Group  State (i)          State (f)          Energy      f or A
  1    [2,3,5,-5]        [2,3,7,7]        2169.756    4.48e-08
  2    [2,3,5,-5]        [2,3,7,-5]       2313.749    4.15e-08
  
Total oscillator strength (f): 8.63e-08
```

Option B (Compact):
```
1: [2,3,5,-5] → [2,3,7,7]     2169.756 cm⁻¹  f=4.48e-08
2: [2,3,5,-5] → [2,3,7,-5]    2313.749 cm⁻¹  f=4.15e-08
Total f: 8.63e-08
```

Option C (CSV-like but readable):
```
Group, i_label, f_label, energy_cm1, f_or_A
1, [2,3,5,-5], [2,3,7,7], 2169.756, 4.48e-08
2, [2,3,5,-5], [2,3,7,-5], 2313.749, 4.15e-08
Total, , , , 8.63e-08
```

Design Question 2: Show Altp parameters?
- Yes? (like MEDIUM)
- No? (assume implicit, too verbose)

Design Question 3: Show state labels or level indices?
- Labels: [2,3,5,-5] (current)
- Indices: 0→6 (simpler, but less informative)
- Both? 0:[2,3,5,-5] (hybrid, verbose)


**VERBOSE Format (All transitions + dipole moments)**

Design Question 4: Structure?

Option A (Hierarchical - group as header):
```
Transition Group 1
  Energy range: 2169.756 cm-1
  Initial: [2,3,5,-5] (E=0.0), Final: [2,3,7,7] (E=2169.756)
  Group total f: 4.48e-08
  Individual transitions:
    0 → 7  e=2169.756
      Isotropic: S_ED=6.33e-03, S_MD=1.75e-05, Total=6.34e-03
      ED moments: -1: (1.25e-18+4.02e-17j), 0: (0.00308-0.00031j), +1: (3.92e-16-5.88e-17j)
      MD moments: -1: (-9.45e-17+3.56e-17j), 0: (-0.00651+0.00067j), +1: (-1.28e-17+3.23e-18j)
    
    1 → 6  e=2169.756
      Isotropic: S_ED=1.75e-05, S_MD=6.33e-03, Total=6.34e-03
      ...
```

Option B (Tabular - all transitions listed):
```
Group  i    f   e_i        e_f      e(cm-1)  S_ED_iso  S_MD_iso  Total
  1    0    7   0.0     2169.76    2169.76   6.33e-03  1.75e-05  6.34e-03
       1    6   0.0     2169.76    2169.76   1.75e-05  6.33e-03  6.34e-03
  Total group f: 4.48e-08
```

Option C (Condensed group, expandable transitions):
```
Group 1 (absorption): [2,3,5,-5] → [2,3,7,7]  E=2169.756  f=4.48e-08
  Transitions (4):
    0→7 (e=2169.76): S_iso=6.34e-03, ED: (-,0,+)=(...), MD: (-,0,+)=(...)
    1→6 (e=2169.76): S_iso=6.34e-03, ED: (-,0,+)=(...), MD: (-,0,+)=(...)
    ...
```

Design Question 5: Which dipole moments to show?
- All (-1, 0, +1 for both ED and MD)? Very verbose
- Isotropic only? Loses debugging detail
- Isotropic + decomposition (ED vs MD)? Good balance
- Component-wise (show as complex values)? Useful for debugging

Design Question 6: Should VERBOSE show state labels on every transition?
- Yes: [2,3,5,-5][i=0] → [2,3,7,7][f=7]
- No: Just indices: 0 → 7 (shorter, but need context)
- Hybrid: Show once per group, indices on transitions


**Implementation Strategy**

1. Implement BRIEF first (simpler, no significant refactoring needed)
   - Add elif format == 'brief': in gen_inten_summary()
   - Call new _format_inten_brief() function

2. Implement VERBOSE second (iterate t_list, show all dipole moments)
   - Add elif format == 'verbose': in gen_inten_summary()
   - Call new _format_inten_verbose() function
   - May need to extract _format_dipole_moments() helper

3. Update gen_inten_summary() docstring with all format options

4. Add tests: test_inten_brief.py and test_inten_verbose.py

5. Update examples/ceylf/inten_example.py to show all formats



ANSWERS TO QUESTIONS

  ⚠️ One inefficiency to fix before scaling: Both MEDIUM and VERBOSE formats need to find level indices by energy matching (loop through 
  principal_components). This should be stored directly in the group dict or we should rely on t_list[n]["i"] and t_list[n]["f"] which already have the
  level indices.

  Key questions for you to decide:

   1. Are t_list[n]["i"] and t_list[n]["f"] already the 0-based level indices? If yes, VERBOSE format can use them directly without any refactoring
      - yes, we can use them. We need to store the principal components in that same place, so we have 
        keys ei, ef, e, i, f, pc_i, pc_f
   2. Should a Spectrum mix absorption and emission groups? (The current code assumes all groups have the same direction for totals display)
      - no, it should be one or the other. 
   3. What's your priority order? BRIEF (compact scanning), VERBOSE (detailed debugging), or both equally?
      - we can do both together, but I will need to think about the verbose one in more detail. 

  Once you answer these, I can either:

   - Add a small optimization now (store level indices in group dict for efficiency)
   - Wait and handle it when implementing the new formats
   - Or confirm the current structure needs no changes

  What's your thought?
    
   - Is my answer clear enough? 