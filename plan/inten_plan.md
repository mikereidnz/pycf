# Plan: Improve inten.py user output and polarization support

Problem
-------
inten.py currently emits low-level arrays and dictionaries as diagnostic output. Users need higher-level, human-friendly summaries and more flexible polarization options (linear, circular, and Stokes-like summaries), plus machine-readable exports (CSV/JSON). This will improve usability and enable downstream analysis (plots, fitting pipelines).

Goal
----
Refactor inten.py to provide:
- A stable public API for intensity generation: gen_intensity(h, ex, polarization=..., basis=..., normalize=..., output_format=...)
- A human-readable summary printer: gen_inten_summary(edata, format='text'|'json'|'csv')
- Support for named polarizations: 'x','y','z','pi','sigma_plus','sigma_minus' and an optional Stokes output
- Unit tests and example scripts showing linear and circular polarization outputs
- Backwards compatibility wrapper for existing callers

Approach
--------
1. Design API and small helper module inten/_polarization.py to encode Jones vectors and basis transforms.
2. Implement gen_intensity() that returns a structured EData-like table with fields: e_calc, e_obs, weight, residual, wresidual, intensity(polarization) per observation.
3. Add gen_inten_summary() to produce pretty tables and optional JSON/CSV outputs.
4. Add unit tests covering: named polarizations, circular polarization sign/convention, summary formatting, backwards-compatibility alias.
5. Add examples/ceylf/inten_example.py showing spectra for sigma+/sigma- and linear polarizations and saving CSV.
6. Update docs and CHANGELOG.

Deliverables
------------
- plan/inten_plan.md (this file)
- plan/inten_report.md (progress/log)
- plan/inten_diff.md (design diffs and API notes)
- pycf/inten.py refactor, new inten/_polarization.py, tests, examples, docs.

Questions / Decisions needed
--------------------------
- Polarization convention: use spherical basis with sigma_plus = (1, i)/sqrt(2)? Recommend explicit names 'sigma_plus'/'sigma_minus' mapping to Jones vectors (1,+i)/sqrt(2) and (1,-i)/sqrt(2). Confirm.
- Output formats required: text (pretty), json, csv. Any others (HDF5)?

Next steps
----------
- On approval of polarization convention, implement helper module and core API.
- Write unit tests for numeric conventions.
- Add an example script and update docs.

See also: plan/inten_research.md for detailed background research on Jones vectors, Stokes parameters, recommended conventions (sigma_plus=(1,+i)/sqrt(2)), numerical testing notes, and references.

