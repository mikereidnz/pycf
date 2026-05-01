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

Approach (refined)
------------------
1. **Design data structures** (Spectrum, IntensityResult) and API stubs.
2. **Implement gen_intensity()** to compute oscillator strengths (f_E, f_M) and A coefficients from Hamiltonian eigenvectors and dipole transitions.
3. **Implement gen_inten_summary()** to render text and CSV outputs with state labels, energies, and cross-sections.
4. **Polarization handling**: Start with isotropic, axial, sigma, pi as named variants. Circular (sigma_plus/sigma_minus) deferred to phase 2.
5. **Unit tests**: polarization sign/convention, oscillator strength conservation, normalized residual computation, label formatting.
6. **Example**: examples/ceylf/inten_example.py showing absorption and emission spectra, CSV output, fitting integration.
7. **Reuse**: cfl_util.py label formatters (gen_e_summary), existing dipole tensor infrastructure, ExData/EData patterns.

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


Next steps
----------
1. Create Spectrum and IntensityResult dataclasses in pycf/inten.py.
2. Implement gen_intensity(hamiltonian, spectrum_label, ...) → IntensityResult.
3. Implement gen_inten_summary() for text and CSV formats.
4. Add unit tests and examples/ceylf/inten_example.py.
5. Update CHANGELOG and docs.

See also: 
- plan/inten_research.md for polarization background.
- plan/inten_diff.md for API design notes.
- pycf/polarization.py for working example of Jones vectors and helpers.
- tests/unit/test_inten_ui.py for test structure.

