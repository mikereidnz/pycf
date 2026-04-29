"""Jones & Stokes example

Run as: python plan/sample_code/jones_stokes_example.py

Demonstrates named Jones vectors, conversion to Stokes, and simple
Jones optical elements (polarizer, rotator, quarter-wave plate).

Convention: optical exp(-i omega t). "sigma_plus" == (1, +i)/sqrt(2).
"""

from __future__ import annotations

import csv
import math
import numpy as np
from pathlib import Path


def norm(v: np.ndarray) -> np.ndarray:
    """Normalize a Jones vector to unit total intensity (S0==1)."""
    v = np.asarray(v, dtype=np.complex128)
    s0 = float((v.conj() * v).sum().real)
    if s0 == 0:
        return v
    return v / math.sqrt(s0)


def jones(name: str) -> np.ndarray:
    """Return a named Jones vector (x,y basis)."""
    if name == "x":
        return np.array([1.0, 0.0], dtype=complex)
    if name == "y":
        return np.array([0.0, 1.0], dtype=complex)
    if name == "45":
        return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
    if name == "sigma_plus":
        return np.array([1.0, 1j], dtype=complex) / math.sqrt(2)
    if name == "sigma_minus":
        return np.array([1.0, -1j], dtype=complex) / math.sqrt(2)
    raise KeyError(f"Unknown polarization name: {name}")


def stokes_from_jones(E: np.ndarray) -> np.ndarray:
    """Return Stokes vector (S0, S1, S2, S3) from Jones vector E.

    Uses S0 = |Ex|^2 + |Ey|^2, S1 = |Ex|^2 - |Ey|^2,
    S2 = 2 Re(Ex* Ey.conj()), S3 = 2 Im(Ex* Ey.conj()).
    """
    Ex, Ey = E[0], E[1]
    S0 = (Ex.conj() * Ex + Ey.conj() * Ey).real
    S1 = (Ex.conj() * Ex - Ey.conj() * Ey).real
    S2 = 2.0 * (Ex.conj() * Ey).real
    S3 = 2.0 * (Ex.conj() * Ey).imag
    return np.array([float(S0), float(S1), float(S2), float(S3)])


def rotator(theta: float) -> np.ndarray:
    """Rotation matrix of the linear basis by theta radians."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def quarter_wave_plate(phi: float = 0.0) -> np.ndarray:
    """Quarter-wave plate with fast axis at angle phi (radians).

    Fast axis along x for phi=0; retardance +pi/2 between axes.
    """
    Rm = rotator(-phi)
    R = rotator(phi)
    Q = R @ np.diag([1.0, 1j]) @ Rm
    return Q


def polarizer(theta: float) -> np.ndarray:
    """Linear polarizer transmitting polarization at angle theta."""
    c = math.cos(theta)
    s = math.sin(theta)
    P = np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)
    return P


def apply(J: np.ndarray, E: np.ndarray) -> np.ndarray:
    return J @ E


def pretty_print(name: str, E: np.ndarray) -> None:
    S = stokes_from_jones(norm(E))
    print(f"{name:12s} Jones={np.round(E,3)}  Stokes={np.round(S,6)}")


def main():
    print("Named Jones vectors and corresponding Stokes vectors:\n")
    for name in ("x", "y", "45", "sigma_plus", "sigma_minus"):
        E = norm(jones(name))
        pretty_print(name, E)

    print("\nEffect of a quarter-wave plate (QWP) at 45deg on 45deg linear:")
    E45 = norm(jones("45"))
    Q45 = quarter_wave_plate(math.pi / 4)
    E_after = apply(Q45, E45)
    pretty_print("after QWP(45deg)", E_after)

    print("\nWrite Jones+Stokes table to plan/sample_code/jones_stokes.csv")
    rows = []
    for name in ("x", "y", "45", "sigma_plus", "sigma_minus", "after_QWP_45"):
        if name == "after_QWP_45":
            E = E_after
        else:
            E = norm(jones(name))
        S = stokes_from_jones(E)
        rows.append({
            "name": name,
            "Ex_real": float(E[0].real),
            "Ex_imag": float(E[0].imag),
            "Ey_real": float(E[1].real),
            "Ey_imag": float(E[1].imag),
            "S0": S[0],
            "S1": S[1],
            "S2": S[2],
            "S3": S[3],
        })

    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "jones_stokes.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
