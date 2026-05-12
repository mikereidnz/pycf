"""Polarization helpers exposed at package root for convenience.

This mirrors the helpers originally intended for pycf.inten._polarization
so tests and callers can import them as ``from pycf.polarization import ...``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

__all__ = ["polarization_vector", "stokes_from_jones", "rotator", "quarter_wave_plate"]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.complex128)
    s0 = float((v.conj() * v).sum().real)
    if s0 == 0:
        return v
    return v / math.sqrt(s0)


def polarization_vector(name: str) -> np.ndarray:
    """Return a normalized 2-component Jones vector for a named polarization.

    Parameters
    ----------
    name : str
        Polarization name. Supported values (case-insensitive):
        ``"x"``, ``"y"``, ``"45"``, ``"sigma_plus"``, ``"sigma_minus"``.

    Returns
    -------
    numpy.ndarray
        Length-2 complex Jones vector with unit total intensity.

    Raises
    ------
    KeyError
        If ``name`` is not one of the supported polarization names.
    """
    name = name.lower()
    if name == "x":
        v = np.array([1.0, 0.0], dtype=complex)
    elif name == "y":
        v = np.array([0.0, 1.0], dtype=complex)
    elif name == "45":
        v = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
    elif name == "sigma_plus":
        v = np.array([1.0, 1j], dtype=complex) / math.sqrt(2)
    elif name == "sigma_minus":
        v = np.array([1.0, -1j], dtype=complex) / math.sqrt(2)
    else:
        raise KeyError(f"Unknown polarization name: {name}")
    return _normalize(v)


def stokes_from_jones(E: Iterable[complex]) -> np.ndarray:
    """Convert a 2-component Jones vector to its 4-component Stokes vector.

    Parameters
    ----------
    E : iterable of complex
        Length-2 Jones vector ``(Ex, Ey)``.

    Returns
    -------
    numpy.ndarray
        Length-4 real Stokes vector ``[S0, S1, S2, S3]``.
    """
    E = np.asarray(E, dtype=np.complex128)
    if E.shape != (2,):
        raise ValueError("Jones vector must be length-2 array-like")
    Ex, Ey = E[0], E[1]
    S0 = float((Ex.conj() * Ex + Ey.conj() * Ey).real)
    S1 = float((Ex.conj() * Ex - Ey.conj() * Ey).real)
    S2 = float(2.0 * (Ex.conj() * Ey).real)
    S3 = float(2.0 * (Ex.conj() * Ey).imag)
    return np.array([S0, S1, S2, S3])


def rotator(theta: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for the given angle in radians."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def quarter_wave_plate(phi: float = 0.0) -> np.ndarray:
    """Return the Jones matrix of a quarter-wave plate rotated by ``phi`` rad.

    With the sign convention used here, a QWP at ``+45deg`` converts linear
    polarization at 45deg into right-circular (``sigma_minus``).
    """
    Rm = rotator(-phi)
    R = rotator(phi)
    # Use -1j so a QWP at +45deg converts linear 45deg -> circular
    Q = R @ np.diag([1.0, -1j]) @ Rm
    return Q
