# Inten design diffs and API notes

This file records intended diffs and rationale; keep it small so it can be reviewed in PRs.

Proposed API surface (pycf.inten)
---------------------------------
- gen_intensity(fit_or_hamiltonian, exdata, polarization='pi', basis='crystal') -> EData-like object with extra fields 'intensity'
- gen_inten_summary(edata, format='text'|'json'|'csv') -> str or write file
- helper functions:
  - polarization_vector(name) -> complex ndarray(2,) (Jones vector in chosen basis)
  - rotate_polarization_to_crystal(jones, basis) -> transformed vector

Backwards-compatibility
-----------------------
- Provide an alias inten.compute_intensity(...) that forwards to gen_intensity with default args.

Diff notes
----------
- Add a new module inten/_polarization.py with ~120 lines implementing named vectors and transforms.
- Modify inten.py (or create pycf/inten_ext.py) to add gen_intensity and gen_inten_summary that call existing intensity computation routines and augment EData with intensity fields.
- Tests in tests/unit/test_inten_ui.py (5-8 tests).

Conventions proposed
--------------------
- Use right-handed coordinate system. Represent circular polarizations as Jones vectors in [x, y] basis:
  - sigma_plus = (1/sqrt(2)) * [1, +1j]
  - sigma_minus = (1/sqrt(2)) * [1, -1j]
- Document this explicitly in docs/ and in the function docstrings.

