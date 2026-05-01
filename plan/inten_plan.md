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

