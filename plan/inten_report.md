# Inten work report

Status: Not started — plan drafted in inten_plan.md.

Notes from initial thinking session
----------------------------------
- The major value is improving user-facing output, not changing the physics. Therefore the implementation can be purely in Python and reuse existing tensor and Hamiltonian evaluation infrastructure in pycf.cfl and pycf.cfl_util.
- Polarization transforms should be isolated in a small helper to make unit testing straightforward.
- Use numpy complex dtype for Jones vectors; support both real (linear) and complex (circular) inputs.

Risks
-----
- Ambiguities in polarization convention (phase sign for sigma+ vs sigma-). This affects sign of circular dichroism outputs. Clarify before widespread use.
- Backwards-incompatible API changes for external callers. Provide a compatibility wrapper.

Planned test matrix
-------------------
- Verify intensity symmetry under linear polarization basis rotations.
- Confirm sigma_plus / sigma_minus conventions against a small analytic Hamiltonian with known selection rules.
- End-to-end example: Ce:YLF example producing CSV with intensities for sigma+/sigma-/pi.

