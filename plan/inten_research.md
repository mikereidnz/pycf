# Research: Polarization conventions and implementation notes

Summary
-------
This document records background research for inten.py improvements focusing on polarization conventions, numerical implementation, and recommended tests.

Key findings
------------
- Jones-vector convention (x,y basis) for circular polarization is commonly:
  - sigma_plus (right-circular, RCP): (1, +i) / sqrt(2)
  - sigma_minus (left-circular, LCP): (1, -i) / sqrt(2)
  - With the usual optics time dependence exp(-i omega t), sigma_plus gives Stokes S3 > 0.
- Stokes parameter S3 = 2 * Im(E_x^* E_y) is positive for sigma_plus and negative for sigma_minus.
- Different communities sometimes swap the sigma+ / sigma- naming or use the opposite time-phase convention. The code should therefore document the used convention clearly and provide an option for alternative sign conventions if consumers require it.

Recommended convention for pycf
--------------------------------
- Adopt the (1, +i)/sqrt(2) for "sigma_plus" (RCP) and (1, -i)/sqrt(2) for "sigma_minus" (LCP).
- Document explicitly in the API docstrings and examples that this uses the optical exp(-i omega t) phase convention and the right-hand rule (S3 > 0 for sigma_plus).
- Provide an alias or a boolean argument (e.g., `flip_circular_sign=False`) in higher-level API if downstream users expect the alternative quantum-optics sign.

Implementation notes
--------------------
- Encapsulate polarization definitions in a small helper module (suggested: `pycf.inten._polarization`) exposing:
  - polarization_vector(name, basis='xy') -> complex ndarray(2,)
  - stokes_from_jones(jones) -> (S0,S1,S2,S3)
  - rotate_jones_to_crystal(jones, basis_transform) -> jones_in_crystal_basis
- gen_intensity should accept either named polarizations or explicit Jones vectors. When a named polarization is given, resolve to Jones via the helper.
- Intensity for a transition is computed generically via a complex transition amplitude A (float or complex) whose squared magnitude yields intensity. Numerically, compute intensities carefully with dtype complex128 and avoid unnecessary cast-to-real until final output.

Numerical & testing recommendations
----------------------------------
- Unit test numeric sign of S3 for sigma_plus/sigma_minus using simple analytic Hamiltonian where selection rules are known (two-level system with dipole oriented along x +/- i y).
- Test invariance under global phase (intensity should be unchanged) and proper scaling with Jones vector normalization.
- Test full round-trip: named -> jones -> stokes -> named, to ensure no accidental sign flip.

References
----------
- Born, M. and Wolf, E., "Principles of Optics" (standard reference on polarization conventions).
- Hecht, E., "Optics" (student-friendly exposition with Jones/Stokes conventions).

