Polarization references and conventions
=====================================

This directory collects short notes and links useful when implementing
polarization features in pycf.inten.

Recommended convention (used in plan/sample_code/jones_stokes_example.py):
- Jones vectors in the (x,y) basis.
- sigma_plus ("right circular") = (1, +i) / sqrt(2)
- sigma_minus ("left circular")  = (1, -i) / sqrt(2)
- Optical time dependence: exp(-i omega t). With this choice,
  Stokes S3 > 0 for sigma_plus.

Useful references
-----------------
- Born, M. & Wolf, E., "Principles of Optics" (standard reference)
- Hecht, E., "Optics" (student-level exposition)
- Many online resources on Jones and Stokes calculus; for code-style
  conventions consult the plan/inten_research.md file in the repo.

Notes
-----
If consumers prefer the opposite circular sign (quantum-optics users
sometimes do), provide a small compatibility switch (flip_circular_sign)
or explicit alternate names (RCP/LCP) mapping to the desired vectors.
