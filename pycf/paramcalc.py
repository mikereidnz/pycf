#!/usr/bin/env python
# Filename = paramcalc.py
"""
Crystal field parameter calculations and atomic structure utilities.
This module computes intensities, transition parameters, and lattice coupling
terms for rare-earth ion crystal fields. Main categories:
**Intensity Parameters (Xi):**
- Xi(t, λ) parameters for magnetic-dipole and electric-quadrupole transitions
- From Krupke (1966) systematic tables for Pr, Nd, Eu, Tb, Er, Tm, Yb
**Radial Integrals (RInt4f):**
- <4f|r^λ|4f> radial integrals for 4f electrons
- From Freeman & Watson tables for Ce through Yb
**Spherical Harmonics (Ckq):**
- Solid spherical harmonics in crystal field normalization
- Used for building static and dynamic coupling terms
**Static Coupling (A_SC):**
- Point-charge model (Krupke) and dynamic polarizability effects
- Includes charge and polarizability contributions
**Dynamic Coupling (A_DC):**
- Isotropic ligand polarizability effects
- Rank-dependent contributions to A parameters
**Ligand Data:**
- Ligand class for coordinate and property storage
- AltpData class for managing multi-ligand systems
Key workflow for complete CF model:
  1. Load ligand coordinates and charges from experiment
  2. Calculate Altp parameters using this module
  3. Build Hamiltonian with these parameters (cfl module)
  4. Fit to experimental data and refine
"""

from typing import List, Tuple

import numpy as np
from scipy.special import sph_harm_y  # type: ignore[import-untyped]

from pycf.cfl_util import uline_char
from pycf.constants import BOHR_RADIUS
from pycf.njsymbols import wigner_3j


def Xi_val(t: int, lam: int, Ln: str) -> float:
    """
    Xi(t, lam) parameters in Angstrom^(t+1) erg^-1 from Krupke Phys Rev 145, 1
    (1966).
    Available ions are Pr, Nd, Eu, Tb, Er, Tm, Yb.
    Yb values are linearly interpolated from values for Er and Tm.

    Parameters
    ----------
    t : int
        Degree of the parameter. Must be in {1, 3, 5, 7}.
    lam: int
        Transition intensity lambda parameter, with values of {2, 4, 6}.
    Ln : string
        The chemical symbol of the Lanthanide dopant.

    Returns
    -------
    xi : float
        value

    Raises
    ------
    ValueError
        If t, lam, or Ln are not in the valid set.
    """
    xi_tl = {
        "12": [-1.78, -1.58, -1.08, -0.83, -0.57, -0.40, -0.23],
        "32": [1.54, 1.35, 0.88, 0.64, 0.36, 0.30, 0.24],
        "34": [1.75, 1.50, 0.90, 0.64, 0.37, 0.29, 0.21],
        "54": [-2.26, -1.98, -1.27, -0.89, -0.44, -0.30, -0.16],
        "56": [-5.45, -4.62, -2.70, -1.84, -0.92, -0.66, -0.40],
        "76": [4.54, 3.96, 2.58, 1.78, 0.71, 0.43, 0.15],
    }
    Ln_list = ["Pr", "Nd", "Eu", "Tb", "Er", "Tm", "Yb"]
    # Validate inputs
    if t not in {1, 3, 5, 7}:
        raise ValueError(f"t must be in {{1, 3, 5, 7}} (got {t})")
    if lam not in {2, 4, 6}:
        raise ValueError(f"lam must be in {{2, 4, 6}} (got {lam})")
    try:
        i = Ln_list.index(Ln)
    except ValueError:
        raise ValueError(f"Invalid lanthanide: Ln={Ln} (valid: {', '.join(Ln_list)})")
    try:
        xi_values = xi_tl["%i%i" % (t, lam)]
    except (ValueError, KeyError):
        raise ValueError("Invalid parameters: t=%i, lam=%i" % (t, lam))
    if Ln == "Yb":
        v = 0.5 * (xi_values[Ln_list.index("Er")] + xi_values[Ln_list.index("Tm")])
    else:
        v = xi_values[i]
    v *= 1e10  # Scale units to Angstrom^(t+1) erg^-1
    return v


def RInt4f(lam: int, Ln: str) -> float:
    r"""
    Radial integrals of the form <4f|r^\lambda|4f> for the RE3+ ions, from
    Freeman and Watson, 10.1103/PhysRev.127.2058.

    Parameters
    ----------
    lam : int
        The power lambda, with available values of {2, 4, 6}.
    Ln : string
        The chemical symbol of the Lanthanide dopant.

    Returns
    -------
    rint : float
        The radial integral, in units of Angstrom^2, Angstrom^4, and
        Angstrom^6 depending on lambda.

    Raises
    ------
    ValueError
        If lam or Ln are not in the valid set.
    """
    # Bohr radius in Angstrom
    # (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
    a0 = BOHR_RADIUS
    # Units in Freeman and Watson are specified as a0^{-lambda}, but I can't
    # make sense of inverse length for the radial integrals?  Treating as
    # a0^{lambda} gives consistent units throughout and reasonable values; going
    # with that for now, but this is disconcerting.
    rint = [
        [1.200, 3.455, 21.226],
        [1.086, 2.822, 15.726],
        [1.001, 2.401, 12.396],
        [0.883, 1.897, 8.775],
        [0.938, 2.273, 11.670],
        [0.726, 1.322, 5.102],
        [0.666, 1.126, 3.978],
        [0.613, 0.960, 3.104],
    ]
    Ln_list = ["Ce", "Pr", "Nd", "Sm", "Eu", "Dy", "Er", "Yb"]
    lam_list = [2, 4, 6]
    if lam not in lam_list:
        raise ValueError(f"lam must be in {lam_list} (got {lam})")
    try:
        i = Ln_list.index(Ln)
    except ValueError:
        raise ValueError(f"Invalid lanthanide: Ln={Ln} (valid: {', '.join(Ln_list)})")
    li = lam_list.index(lam)
    val = rint[i][li] * a0**lam
    return val


class Ligand(object):
    """
    Class for holding data of a specific ligand type.

    Parameters
    ----------
    coords : np.array
        Array of ligand coordinates, of the form [R, theta, phi], that is,
        the ligand radius (Angstrom), polar angle (radians), and azimuthal
        angle (radians), respectively.
    q : float
        The charge of the ligand ion, in units of proton charge.
    alpha_bar : float
        Mean polarizability of ligand species in Angstrom^3.
    """

    def __init__(self, coords: np.ndarray, q: float, alpha_bar: float) -> None:
        self.coords = coords
        self.q = q
        self.alpha_bar = alpha_bar


def Ckq(k: int, q: int, theta: float, phi: float) -> np.complexfloating:
    """
    Solid spherical harmonic functions in normalization conventionally used for CF calcs.

    Parameters
    ----------
    k : int
        Degree of the harmonic. Must be >= 0.
    q : int
        Order of the harmonic. Must satisfy -k <= q <= k.
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.

    Returns
    -------
    Ckq : float
        Value of spherical harmonic.

    Raises
    ------
    ValueError
        If k < 0 or ``|q|`` > k.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0 (got {k})")
    if abs(q) > k:
        raise ValueError(f"q must satisfy |q| <= k (got q={q}, k={k})")
    C = np.sqrt((4 * np.pi) / (2 * k + 1)) * sph_harm_y(k, q, theta, phi)
    return C


def A_SC(
    lam: int, t: int, p: int, Ln: str, q_Ln: float, ligands: List[Ligand]
) -> Tuple[float, float]:
    r"""
    Calculate the A^lambda_tp parameters for static coupling using a
    point-charge model, following Reid and Richardson, J. Chem. Phys. 79(12)
    1983, pg 5739.

    Parameters
    ----------
    lam: int
        Transition intensity lambda parameter (2, 4, 6).
    t : int
        Degree of the parameter.
    p : int
        Order of the parameter.
    Ln : string
        The chemical symbol of the Lanthanide dopant.
    q_Ln : float
        The charge of the Lanthanide dopant.
    ligands : list
        List of Ligand objects.

    Returns
    -------
    A : float
        The transition intensity parameter A^{\lambda}_{tp} in cm^(-1).
    """
    Xi = Xi_val(t, lam, Ln)
    # To avoid overflowing our 64bit double, we'll rescale Xi by a factor
    # 10^(-10) and e2 by 10^(10).  These variables are always multiplied later,
    # so this avoids tiny numbers.
    Xi = Xi * 10 ** (-10)
    e2 = (4.80320425**2) * 10 ** (-10)  # proton charge squared in esu
    prefac = -((-1) ** p) * e2 * Xi * (2 * lam + 1) / (np.sqrt(2 * t + 1))
    A_chg = 0
    A_pol = 0
    for L in ligands:
        c = L.coords
        C = Ckq(t, -p, c[1], c[2])
        A_chg += C * c[0] ** (-(t + 1)) * L.q
        A_pol += C * c[0] ** (-(t + 4)) * L.alpha_bar
    A_chg = prefac * (-1) * A_chg
    A_pol = prefac * 2 * q_Ln * (t + 1) * A_pol
    A_chg = np.real(A_chg)
    A_pol = np.real(A_pol)
    return (A_chg, A_pol)


def A_DC(lam: int, t: int, p: int, Ln: str, ligands: List[Ligand]) -> float:
    r"""
    Calculate the A^lambda_tp parameters for dynamic coupling assuming isotropic
    ligands, following Reid and Richardson, J. Chem. Phys. 79(12) 1983, pg 5739.

    Parameters
    ----------
    lam: int
        Transition intensity lambda parameter (2, 4, 6).
    t : int
        Degree of the parameter.
    p : int
        Order of the parameter.
    Ln : string
        The chemical symbol of the Lanthanide dopant.
    ligands : list
        List of Ligand objects.

    Returns
    -------
    A : float
        The transition intensity parameter A^{\lambda}_{tp} in cm^(-1).
    """
    A = 0
    if t == lam + 1:
        rint = RInt4f(lam, Ln)
        prefac = (
            7
            * wigner_3j(3, lam, 3, 0, 0, 0)
            * np.sqrt((lam + 1) * (2 * lam + 1))
            * rint
            * (-1) ** p
        )
        for L in ligands:
            c = L.coords
            A += Ckq(lam + 1, -p, c[1], c[2]) * c[0] ** (-(lam + 2)) * L.alpha_bar
        # Convert to cm from A
        A *= prefac * 10 ** (-8)
        A = np.real(A)
    return A


class AltpData(object):
    """
    Class for holding data required for calculating Altp parameters.

    Parameters
    ----------
    Ln : string
        The lanthanide chemical symbol, with available options Pr, Nd, Eu, Er,
        and Yb. Tb and Tm are currently missing radial integrals, whereas Ce,
        Sm, and Dy are missing values for Xi(t, lambda), and Ho is missing both.
        To implement them, update the appropriate functions and add them to the
        available list.
    q_Ln : int
        The charge of the lanthanide dopant ion, in units of proton charge.
    ligands : list
        List of Ligand objects.
    """

    def __init__(self, Ln: str, q_Ln: int, ligands: List[Ligand]) -> None:
        self.Ln = Ln
        self.q_Ln = q_Ln
        self.ligands = ligands
        self.nL = len(ligands)

    def eval_params(self):
        """
        Evaluate the Altp parameters and return them.  After running this
        function, the AltpData object also has the attributes A_statchg,
        A_statpol, A_dyniso, and A_total, corresponding to static charge
        contribution, static polarziation contribution, and dynamic contribution
        assuming isotropic ligands.

        Returns
        -------
        A_list : list
            Elements are lists of length two, with the first element in the sublist
            containing a string that specifies the parameter designation.  The
            second element in the sublist is a list with four elements, the static
            coupling charge and polarization parameters, the dynamic coupling
            parameter for isotropic ligands, and the total A parameter value.
        """
        A_list = []
        l_list = [2, 4, 6]
        for lam in l_list:
            for t in [lam - 1, lam + 1]:
                for p in range(0, t + 1):
                    # Static portion
                    A_statchg, A_statpol = A_SC(lam, t, p, self.Ln, self.q_Ln, self.ligands)
                    # Dynamic portion
                    A_dyniso = A_DC(lam, t, p, self.Ln, self.ligands)
                    A_total = A_statchg + A_statpol + A_dyniso
                    if (np.abs(A_total)) > 1e-15:
                        A_list += [
                            [
                                "A%i%i%i" % (lam, t, p),
                                [
                                    A_statchg,
                                    A_statpol,
                                    A_dyniso,
                                    A_statchg + A_dyniso,
                                    A_total,
                                ],
                            ]
                        ]
        self.A_list = list(A_list)
        return A_list

    def gen_summary(self):
        s = ""
        heading = "Param   A_statchg    A_statpol     A_dyniso  A_statchg+A_dyniso      A_total\n"
        s += uline_char(heading)
        for A in self.A_list:
            s += "%s  % .4e  % .4e  % .4e         % .4e  % .4e\n" % (A[0], *A[1])
        s += "\n"
        return s
