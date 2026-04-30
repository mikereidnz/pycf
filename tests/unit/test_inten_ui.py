"""Unit tests for inten polarization helpers (initial).

These are intentionally small and focused: they exercise the
polarization_vector and stokes_from_jones helpers so the full inten
refactor can build on a tested foundation.
"""

from __future__ import annotations

import numpy as np

from pycf.polarization import polarization_vector, stokes_from_jones, quarter_wave_plate


def test_sigma_plus_has_positive_S3():
    v = polarization_vector("sigma_plus")
    S = stokes_from_jones(v)
    assert S[3] > 0


def test_sigma_minus_has_negative_S3():
    v = polarization_vector("sigma_minus")
    S = stokes_from_jones(v)
    assert S[3] < 0


def test_qwp_converts_45_linear_to_circular():
    # 45-degree linear polarisation -> QWP at 45deg should give circular
    E45 = polarization_vector("45")
    Q = quarter_wave_plate(np.pi / 4)
    E_after = Q @ E45
    S = stokes_from_jones(E_after)
    # |S1|,|S2| small, |S3| ~ S0
    assert abs(S[1]) < 1e-12
    assert abs(S[2]) < 1e-12
    assert abs(abs(S[3]) - S[0]) < 1e-9
