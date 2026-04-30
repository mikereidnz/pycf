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
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def quarter_wave_plate(phi: float = 0.0) -> np.ndarray:
    Rm = rotator(-phi)
    R = rotator(phi)
    Q = R @ np.diag([1.0, 1j]) @ Rm
    return Q
