#!/usr/bin/env python
# Filename = inten.py
"""
A rewrite of the intensity calculation to follow the old Pascal code more closely,
"""

from dataclasses import dataclass, field
from datetime import datetime
from operator import itemgetter
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
from scipy.optimize import (
    minimize,
    basinhopping,
    differential_evolution,
    dual_annealing,
    shgo,
    direct,
    least_squares,
)

from pycf.cfl_util import (
    L2term,
    gen_completed_str,
    gen_pycf_summary,
    jacobian_diagnostics,
)
from pycf.constants import (
    BOLTZMANN_CM_INVERSE,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    EPSILON_0,
    HBAR,
    SPEED_OF_LIGHT,
)
from pycf.njsymbols import wigner_3j


def clean_complex(value: Union[complex, float], tolerance: float = 1e-12) -> Union[complex, float]:
    """
    Clean up rounding errors in complex numbers by zeroing small real/imaginary parts.

    Parameters
    ----------
    value : complex or float
        The complex or real number to clean
    tolerance : float, optional
        Threshold for zeroing components (default 1e-12)

    Returns
    -------
    complex or float
        If input is complex: returns complex with small parts zeroed
        If input is real: returns float unchanged
    """
    if isinstance(value, complex):
        real = value.real if abs(value.real) > tolerance else 0.0
        imag = value.imag if abs(value.imag) > tolerance else 0.0
        if imag == 0.0:
            return real
        return complex(real, imag)
    return value


def _normalize_altp(
    altp: Optional[Union[Mapping[str, Any], Sequence[Sequence[Any]]]]
) -> Optional[Dict[str, Any]]:
    """Normalize Altp inputs to an internal dict representation."""
    if altp is None:
        return None
    if isinstance(altp, MappingABC):
        return dict(altp)
    normalized: Dict[str, Any] = {}
    for item in altp:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Altp entries must be (name, value) pairs.")
        name, value = item
        normalized[str(name)] = value
    return normalized


def _altp_dict_to_list(altp: Optional[Mapping[str, Any]]) -> Optional[List[List[Any]]]:
    """Convert dict-style Altp to dipole_str's pair-list format."""
    if altp is None:
        return None
    return [[name, value] for name, value in altp.items()]


def vtrans(tensors: List[Any], z: np.ndarray) -> Dict[str, Any]:
    """
    Transform tensor matrix elements into eigenbasis previously calculated by
    diagonalizing a Hamiltonian.

    This does the transformation part of the Pascal vtrans program, but not the
    construction of the electric-dipole operators, which is now done in the
    dipole_str function.

    Mike Reid 3 April 2026:
        Delete lower-diagonal elements that *were* mistakenly added by
        t.get_matel(). Since the 2026-04 fix to ``zhcsr2zha`` (see
        ``cfl/src/cfl_csr.c``), ``get_matel()`` returns only the upper
        triangle, so the ``M - np.tril(M, k=-1)`` step below is now a
        no-op. It is retained as a defensive measure and to keep the q=0
        Hermitian-completion logic that follows it self-evidently correct.

    Parameters
    ----------
    tensors : list
        Elements are tensors of type cfl.Tensor.
    z : np.ndarray
        The eigenvectors, columnwise, used for the transformation.  This is
        generally the second output variable from h.diag() where H is a
        cfl.Hamiltonian.

    Returns
    -------
    tensor_dict : list
        Transformed tensors
    """
    tensor_dict = {}
    vtrans_ten = [
        "U20",
        "U21",
        "U22",
        "U23",
        "U40",
        "U41",
        "U42",
        "U43",
        "U44",
        "U60",
        "U61",
        "U62",
        "U63",
        "U64",
        "U65",
        "U66",
        "M10",
        "M11",
    ]
    if len(tensors) == 0:
        raise ValueError("vtrans requires at least one tensor.")
    for t in tensors:
        if t.name not in vtrans_ten:
            raise ValueError("Unsupported tensor passed to vtrans: %s" % t.name)
        M = t.get_matel()  # get a numpy matrix M from the compressed form
        # Delete lower diagonal, in case it was added when reading in
        # Include the q==0 case, even though we put it back below
        # just to be sure.
        M = M - np.tril(M, k=-1)  # subtract the lower triangle
        q = int(t.name[2])
        if q == 0:
            # in this case we need a Hermitian matrix
            # so we add the conjugate, omitting diagonal
            M = M + np.tril(M.conj().T, k=-1)
        matel = z.conj().T @ M @ z  # eigenvector transformation of M
        # discard small real or imaginary parts of the transformed matrix
        # matel.imag[np.abs(matel.imag) < tolerance] = 0
        # matel.real[np.abs(matel.real) < tolerance] = 0
        if q == 0:
            tensor_dict[t.name] = matel
        else:  # q != 0, specifically q > 0
            # +q case
            tensor_dict[t.name] = matel
            # -q case
            # Use V^dagger M^\dagger V  = (V^dagger M  V)^\dagger
            matel = matel.conj().T
            # Then add (-1)^q phase.
            # This will work for any q!=0.
            if q % 2 == 1:
                matel *= -1
            tensor_dict["%s-%i" % (t.name[:2], q)] = matel
    return tensor_dict


def dipole_str(
    i_range: List[int],
    f_range: List[int],
    tensor_dict: Dict[str, Any],
    h: Any,
    E: np.ndarray,
    V: np.ndarray,
    md: bool = True,
    ed: bool = False,
    Altp: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate dipole strengths and transition properties in eigenbasis.

    Parameters
    ----------
    i_range : list of int
        1-based initial state level indices (user-friendly convention).
        Example: [1, 2] for ground state doublet.
    f_range : list of int
        1-based final state level indices (user-friendly convention).
        Example: [7, 8, 9, 10] for excited multiplet.
    tensor_dict : dict
        Dictionary of transformed dipole tensors with keys (M10, M11, M1-1, etc.)
        pointing to matrix elements in the eigenbasis.
    h : cfl.Hamiltonian
        Hamiltonian object containing state information and labels.
    E : np.ndarray
        Array of eigenvalues from Hamiltonian diagonalization.
    V : np.ndarray
        Array of eigenvectors (columnwise) from Hamiltonian diagonalization.
    md : bool, optional
        Include magnetic dipole transitions (default True).
    ed : bool, optional
        Include electric dipole transitions via Altp parameters (default False).
    Altp : list, optional
        Altp coupling parameters for electric dipole calculation, required if ed=True.

    Returns
    -------
    trs : list
        List of dictionaries containing transition data:
        - 'e': transition energy (cm^-1)
        - 'ei': initial state energy (cm^-1)
        - 'ef': final state energy (cm^-1)
        - dipole strengths by polarization
    """
    e = 1e10  # means that diople moments are in e-10 cm
    # e = 4.803246e-10       # esu
    clight = 2.997925e10  # cm/sec
    hbar = 1.0545903e-27  # erg-sec
    me = 9.109553e-28  # gm
    md_prefac = -(e * hbar) / (2 * me * clight)

    # Convert 1-based user input to 0-based internal indices
    i_range = [i - 1 for i in i_range]
    f_range = [i - 1 for i in f_range]

    w = E
    z = V
    # Validate eigenvector dimensions
    if not isinstance(z, np.ndarray) or z.ndim != 2:
        shape = getattr(z, "shape", None)
        raise ValueError(
            "Eigenvector V must be 2-dimensional (nstates x nstates), got shape %s" % (shape,)
        )
    # find principal components
    pc = np.argmax(np.abs(z), axis=0)
    if ed:
        D_factor = {}
        if Altp is None:
            raise ValueError("ed is True but no Altp parameters were provided")
        for A in Altp:
            lam = int(A[0][1])
            t = int(A[0][2])
            pp = int(A[0][3])
            # Evaluate the Clebsch-Gordon coefficient of Eq. (9), Reid and Richardson J.
            # Chem. Phys. 79, 5735 (1983). Note: sign factor includes additional (-1)^q
            for q in [-1, 0, 1]:
                for p in np.unique([-pp, pp]):
                    CG_coeff = np.sqrt(2 * t + 1) * wigner_3j(lam, 1, t, p + q, -q, -p)
                    if (lam - 1 + p + q) % 2 != 0:
                        CG_coeff *= -1
                    D_factor["%i%i%i%i" % (lam, t, p, q)] = CG_coeff
    trs = []
    for i in i_range:
        for f in f_range:
            md_mom = [0, 0, 0]
            ed_mom = [0, 0, 0]
            if md:
                keys = ["M1-1", "M10", "M11"]
                if not all(k in tensor_dict for k in keys):
                    raise ValueError(
                        "Missing all or some of the magnetic dipole "
                        "operator matrix elements. Required tensors are 'M1-1', "
                        "'M10', 'M11'"
                    )
                # md_mom = [np.real(md_prefac*tensor_dict[k][i, f]) for k in keys]
                # moments can be complex for complex eigenvectors.
                md_mom = [(md_prefac * tensor_dict[k][i, f]) for k in keys]
            if ed:
                if Altp is None:
                    raise ValueError("Altp must be provided when ed=True")
                for A in Altp:
                    lam = int(A[0][1])
                    t = int(A[0][2])
                    pp = int(A[0][3])
                    for q in [-1, 0, 1]:
                        for p in np.unique([-pp, pp]):
                            A_val = A[1]
                            if p < 0:  # symmetry of Altp parameter
                                A_val = A_val.conjugate()
                                if (1 + t + p) % 2 != 0:  # if 1+t+p is odd
                                    A_val = -A_val
                            if -lam <= (p + q) <= lam:
                                k = "U%i%i" % (lam, p + q)
                                if k not in tensor_dict:
                                    msg = (
                                        "Missing electric dipole tensor '{}' "
                                        "required by Altp.".format(k)
                                    )
                                    raise ValueError(msg)
                                D = (
                                    -e
                                    * A_val
                                    * D_factor["%i%i%i%i" % (lam, t, p, q)]
                                    * tensor_dict[k][i, f]
                                )
                                ed_mom[q + 1] += D  # order is -1, 0, 1
            # Keep all transitions, otherwise our degeneracy calculations will be wrong.
            # Clean dipole moments to remove rounding errors before calculating strengths
            ed_mom = [clean_complex(m) for m in ed_mom]  # type: ignore[misc]
            md_mom = [clean_complex(m) for m in md_mom]  # type: ignore[misc]

            # Electric and magnetic dipole strengths for -1, 0, +1 components
            S_ED_m = np.abs(ed_mom[0]) ** 2
            S_ED_0 = np.abs(ed_mom[1]) ** 2
            S_ED_p = np.abs(ed_mom[2]) ** 2
            S_MD_m = np.abs(md_mom[0]) ** 2
            S_MD_0 = np.abs(md_mom[1]) ** 2
            S_MD_p = np.abs(md_mom[2]) ** 2
            # In future we should consider more flexible polarization choices,
            # but for now will stick with the same definitions as in the old Pascal code.
            # For electric dipole, the isotropic component is the average of the three q components.
            # For axial symmetries:
            # the axial component is the average of the q=±1 components,
            # the sigma component is the q=0 component,
            # the pi component is the average of the q=±1 components.
            S_ED_isotropic = (S_ED_m + S_ED_0 + S_ED_p) / 3
            S_ED_axial = (S_ED_m + S_ED_p) / 2
            S_ED_sigma = S_ED_axial
            S_ED_pi = S_ED_0
            # Note that E and B are perpedicular for linear polarization.
            # For magnetic dipole, the sigma component is the one with q=0,
            # and the pi component is the average of the q=±1 components.
            S_MD_isotropic = (S_MD_m + S_MD_0 + S_MD_p) / 3
            S_MD_axial = (S_MD_m + S_MD_p) / 2
            S_MD_sigma = S_MD_0
            S_MD_pi = S_MD_axial
            # Keep the totals that the Pascal code calculates,
            # which are the sum of the electric and magnetic dipole contributions.
            # However, these are problematical as ED and MD have different
            # refractive index prefactors, so we should be careful about how we
            # use these totals.
            isotropic = S_ED_isotropic + S_MD_isotropic
            axial = S_ED_axial + S_MD_axial
            sigma = S_ED_sigma + S_MD_sigma
            pi = S_ED_pi + S_MD_pi
            # transition moments for a single intial and final state,
            # which we will later group by energy to get the total transition moment
            # for a given transition.
            trs += [
                {
                    # states and principal components
                    "i": i,
                    "f": f,
                    "pc_i": pc[i],
                    "pc_f": pc[f],
                    # energies
                    "ei": w[i],
                    "ef": w[f],
                    "e": w[f] - w[i],
                    # dipole moments
                    "md_-1": md_mom[0],
                    "md_0": md_mom[1],
                    "md_+1": md_mom[2],
                    "ed_-1": ed_mom[0],
                    "ed_0": ed_mom[1],
                    "ed_+1": ed_mom[2],
                    # dipole strengths
                    "S_ED_-1": S_ED_m,
                    "S_ED_0": S_ED_0,
                    "S_ED_+1": S_ED_p,
                    "S_MD_-1": S_MD_m,
                    "S_MD_0": S_MD_0,
                    "S_MD_+1": S_MD_p,
                    "S_ED_isotropic": S_ED_isotropic,
                    "S_MD_isotropic": S_MD_isotropic,
                    "S_ED_axial": S_ED_axial,
                    "S_MD_axial": S_MD_axial,
                    "S_ED_sigma": S_ED_sigma,
                    "S_MD_sigma": S_MD_sigma,
                    "S_ED_pi": S_ED_pi,
                    "S_MD_pi": S_MD_pi,
                    "isotropic": isotropic,
                    "axial": axial,
                    "sigma": sigma,
                    "pi": pi,
                }
            ]
    trs.sort(key=itemgetter("e"))
    return trs


def group_transitions(items: List[Dict[str, Any]], tol: float = 1e-4) -> List[Dict[str, Any]]:
    """
    Group transition dictionaries by (ei, ef) level-pair and annotate each group
    with initial/final degeneracies.

    Parameters
    ----------
    items : list of dict
        Transition dictionaries as returned by :func:`dipole_str`, containing
        keys ``ei``, ``ef``, ``e``, ``i`` and ``f``.
    tol : float
        Tolerance for comparing energies when determining degeneracies and
        when grouping transitions by level pair.

    Returns
    -------
    list of dict
        A list with entries of the form:
        ``{"Energy", "e_i", "e_f", "g_i", "g_f", "t_list"}``.
    """
    if not items:
        return []

    def _level_degeneracies(entries, energy_key, state_key):
        """Build a list of (anchor_energy, degeneracy_count)."""
        pairs = sorted(set((d[energy_key], d[state_key]) for d in entries))
        clusters = []
        for energy, state in pairs:
            if clusters and abs(energy - clusters[-1][0]) <= tol:
                clusters[-1][1].add(state)
            else:
                clusters.append([energy, set([state])])
        return [(anchor, len(states)) for anchor, states in clusters]

    def _lookup_degeneracy(anchors, energy):
        for anchor, count in anchors:
            if abs(energy - anchor) <= tol:
                return count
        return 1

    ei_deg = _level_degeneracies(items, "ei", "i")
    ef_deg = _level_degeneracies(items, "ef", "f")
    # Sort by transition energy first, then by level pair so equivalent pairs are
    # contiguous before grouping.
    ordered = sorted(items, key=lambda d: (d["e"], d["ei"], d["ef"]))
    groups = []
    cur_ei = ordered[0]["ei"]
    cur_ef = ordered[0]["ef"]
    cur_list = [ordered[0]]
    for tr in ordered[1:]:
        same_ei = abs(tr["ei"] - cur_ei) <= tol
        same_ef = abs(tr["ef"] - cur_ef) <= tol
        if same_ei and same_ef:
            cur_list.append(tr)
        else:
            groups.append(
                {
                    "Energy": cur_ef - cur_ei,
                    "e_i": cur_ei,
                    "e_f": cur_ef,
                    "g_i": _lookup_degeneracy(ei_deg, cur_ei),
                    "g_f": _lookup_degeneracy(ef_deg, cur_ef),
                    "t_list": cur_list,
                }
            )
            cur_ei = tr["ei"]
            cur_ef = tr["ef"]
            cur_list = [tr]
    groups.append(
        {
            "Energy": cur_ef - cur_ei,
            "e_i": cur_ei,
            "e_f": cur_ef,
            "g_i": _lookup_degeneracy(ei_deg, cur_ei),
            "g_f": _lookup_degeneracy(ef_deg, cur_ef),
            "t_list": cur_list,
        }
    )
    return groups


def A_and_f_calc(
    S_ED: float, S_MD: float, energy: float, g_i: float, nrefractive: float = 1.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate the Einstein A coefficient and oscillator strength for a transition
    with given electric and magnetic dipole strengths and transition energy.

    Parameters
    ----------
    S_ED : float
        Electric dipole strength (in 10^-20 cm^2 units).
    S_MD : float
        Magnetic dipole strength (in 10^-20 cm^2 units).
    energy : float
        Transition energy (in cm^-1).
    g_i : float
        Degeneracy of the initial state.
    nrefractive : float, optional
        Refractive index of the medium (default is 1.0 for vacuum).

    Returns
    -------
    f_ED : float
        Electric-dipole contribution to oscillator strength.
    f_MD : float
        Magnetic-dipole contribution to oscillator strength.
    f : float
        Total oscillator strength (f_ED + f_MD).
    A_ED : float
        Electric-dipole contribution to Einstein A coefficient (s^-1).
    A_MD : float
        Magnetic-dipole contribution to Einstein A coefficient (s^-1).
    A : float
        Total Einstein A coefficient (A_ED + A_MD).

    Raises
    ------
    ValueError
        If g_i <= 0 (invalid degeneracy).
    """
    # Validate degeneracy
    if g_i <= 0:
        raise ValueError(f"Initial state degeneracy g_i must be positive (got {g_i})")

    # Constants, in SI units
    melectron = ELECTRON_MASS
    echarge = ELEMENTARY_CHARGE
    epsilon0 = EPSILON_0
    hbar = HBAR
    clight = SPEED_OF_LIGHT
    rpi = np.pi
    if energy == 0:
        lambda_ = 0.0
        omega = 0.0
    else:
        lambda_ = 1e-2 / energy  # {cm-1}; {m}
        if lambda_ != 0:
            omega = 2 * rpi * clight / lambda_  # {hz}
        else:
            omega = 0.0
    chilocal = ((nrefractive**2 + 2) / 3) ** 2
    # our dipole strengths are in units of 10-20 cm2,
    # so we need to convert to SI units of C2m2 for the A and f calculations.
    sed = echarge * echarge * S_ED * 1e-20 * 1e-4  # {C2m2}
    smd = echarge * echarge * S_MD * 1e-20 * 1e-4  # {C2m2}
    oscfactor = 2 * melectron / hbar / echarge / echarge
    afactor = 1 / (4 * rpi * epsilon0) * 4 / (hbar * clight * clight * clight)
    f_ED = omega * oscfactor * (sed * 1 / nrefractive * chilocal) / g_i
    f_MD = omega * oscfactor * (smd * nrefractive) / g_i
    f = f_ED + f_MD
    A_ED = (
        omega
        * omega
        * omega
        * afactor
        * (sed * nrefractive * chilocal)
        / g_i
    )
    A_MD = (
        omega
        * omega
        * omega
        * afactor
        * (smd * nrefractive * nrefractive * nrefractive)
        / g_i
    )
    A = A_ED + A_MD
    return abs(f_ED), abs(f_MD), abs(f), abs(A_ED), abs(A_MD), abs(A)


def add_oscillator_strengths_and_A_coefficients(
    groups: List[Dict[str, Any]], refractive_index: float = 1.0
) -> None:
    """
    Add oscillator strengths and Einstein A coefficients to transition groups.

    Parameters
    ----------
    groups : list of dict
        List of transition groups from group_transitions(), modified in place.
    refractive_index : float, optional
        Refractive index of the medium (default 1.0 for vacuum).

    Returns
    -------
    None
        Modifies input groups list in place, adding 'A' and 'f' fields.
    """
    for group in groups:
        group["S_ED_isotropic"] = sum(tr["S_ED_isotropic"] for tr in group["t_list"])
        group["S_MD_isotropic"] = sum(tr["S_MD_isotropic"] for tr in group["t_list"])
        f_ED, f_MD, f, A_ED, A_MD, A = A_and_f_calc(
            group["S_ED_isotropic"],
            group["S_MD_isotropic"],
            group["Energy"],
            group["g_i"],
            nrefractive=refractive_index,
        )
        group["f_ED"] = f_ED
        group["f_MD"] = f_MD
        group["A"] = A
        group["A_ED"] = A_ED
        group["A_MD"] = A_MD
        group["f"] = f
    # no return value since we are modifying the input list in place.


def boltzmann_factor(e: float, t: float) -> float:
    """
    Calculate the Boltzmann factor for a given energy and temperature.

    Parameters
    ----------
    e : float
        Energy difference (cm^-1).
    t : float
        Temperature (K). Must be >= 0.

    Returns
    -------
    float
        Boltzmann factor (dimensionless).

    Raises
    ------
    ValueError
        If temperature is negative.
    """
    if t < 0:
        raise ValueError(f"Temperature must be non-negative (got {t} K)")
    elif t == 0:
        ans = 1
    else:
        ans = np.exp(-e / (t * BOLTZMANN_CM_INVERSE))
    return ans


def lorentzian_constant_height(
    x: Union[float, np.ndarray], x0: float, fwhm: float
) -> Union[float, np.ndarray]:
    """
    Calculate Lorentzian line shape (constant height).

    This Lorentzian preserves peak height (not area) regardless of fwhm.

    Parameters
    ----------
    x : float or array
        Energy points (cm^-1).
    x0 : float
        Center of the Lorentzian (cm^-1).
    fwhm : float
        Full width at half maximum (cm^-1).

    Returns
    -------
    float or array
        Lorentzian function values.

    Raises
    ------
    ValueError
        If fwhm <= 0.
    """
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive (got {fwhm})")
    gamma_sq = (fwhm / 2) ** 2
    return gamma_sq / ((x - x0) ** 2 + gamma_sq)


def lorentzian(x: Union[float, np.ndarray], x0: float, fwhm: float) -> Union[float, np.ndarray]:
    """Deprecated: use lorentzian_constant_height() instead."""
    return lorentzian_constant_height(x, x0, fwhm)


@dataclass
class Spectrum:
    """
    Named collection of transitions with intensity calculation parameters and results.

    A Spectrum encapsulates the parameters and state needed to compute dipole strengths
    and oscillator strengths for a specific set of transitions (e.g., absorption from
    ground state, emission from excited state). Similar to Hamiltonian.diag(), call
    calculate_intensities() to compute derived data after creating or updating the spectrum.

    Parameters
    ----------
    hamiltonian : cfl.Hamiltonian
        Reference to the Hamiltonian object (must be diagonalized before computing intensities).
    name : str
        Name of the spectrum (e.g., 'ground absorption', 'excited emission').
    i_range : list of int
        Initial state level indices (1-based, matching energy level printouts Z1, Z2, ...).
        Example: [1, 2] for ground state Z1 doublet.
    f_range : list of int
        Final state level indices (1-based). Example: [7, 8, 9, 10] for excited Y multiplet.
    intensity_tensors : list
        List of intensity tensor objects (e.g., from ImportSLJM).
    group_tol : float, optional
        Tolerance for grouping transitions by (ei, ef) level pair (default 1e-3).
        Smaller values (1e-5 or less) may be needed for hyperfine structure.
    nrefractive : float, optional
        Refractive index of the medium (default 1.0 for vacuum).
    md : bool, optional
        Include magnetic dipole transitions (default True).
    ed : bool, optional
        Include electric dipole transitions via altp (default False).
    expt_data : list of [group_idx, f_calc, f_expt], optional
        Experimental intensity data for comparison. When provided, BRIEF output includes
        experimental values and χ² residuals: χ² = ((f_calc - f_expt) / (f_calc + f_expt))².

    Attributes
    ----------
    altp : list, optional
        Altp coupling parameters for electric dipole calculation.
        Each element is [name_str, value], e.g., ['A210', 1e-10].
    expt_data : list, optional
        Experimental data (format: [[group_idx, f_calc, f_expt], ...]).
    transformed_tensors : dict
        Cached transformed tensors (computed by calculate_intensities).
    groups : list of dict
        Computed transition groups (from calculate_intensities).
    total_f : float
        Total oscillator strength across all groups (absorption only).
    total_A : float
        Total Einstein A coefficient across all groups (emission only).

    Methods
    -------
    set_altp(altp)
        Update Altp parameters without recreating spectrum.
    calculate_intensities(polarization='isotropic')
        Compute transformed tensors, dipole strengths, and oscillator strengths.
    """

    hamiltonian: Any
    name: str
    i_range: List[int]
    f_range: List[int]
    intensity_tensors: List[Any]
    group_tol: float = 1e-3
    nrefractive: float = 1.0
    md: bool = True
    ed: bool = False

    # Mutable Altp parameters (can be changed without rebuilding Spectrum)
    altp: Optional[Dict[str, Any]] = field(default=None)

    # Experimental data (optional): list of [group_idx, f_calc, f_expt]
    expt_data: Optional[List[List[float]]] = field(default=None)
    altp_uncertainties: Dict[str, Any] = field(default_factory=dict, init=False)

    # Computed fields
    transformed_tensors: Dict[str, Any] = field(default_factory=dict, init=False)
    dipole_strengths: List[Dict[str, Any]] = field(default_factory=list, init=False)
    groups: List[Dict[str, Any]] = field(default_factory=list, init=False)
    total_f: float = field(default=0.0, init=False)
    total_A: float = field(default=0.0, init=False)
    eigenvalues: Optional[np.ndarray] = field(default=None, init=False)
    principal_components: Optional[np.ndarray] = field(default=None, init=False)
    _cached_eigenvectors: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs."""
        if not self.name:
            raise ValueError("Spectrum name must be non-empty")
        if not self.i_range or not isinstance(self.i_range, list):
            raise ValueError("i_range must be non-empty list of initial state indices (1-based)")
        if not self.f_range or not isinstance(self.f_range, list):
            raise ValueError("f_range must be non-empty list of final state indices (1-based)")
        if not self.intensity_tensors:
            raise ValueError("intensity_tensors must be non-empty")
        if self.group_tol <= 0:
            raise ValueError(f"group_tol must be positive (got {self.group_tol})")
        if self.nrefractive <= 0:
            raise ValueError(f"nrefractive must be positive (got {self.nrefractive})")
        if not hasattr(self.hamiltonian, "diag"):
            raise ValueError("hamiltonian must be a cfl.Hamiltonian object")
        self.altp = _normalize_altp(self.altp)

    def set_altp(self, altp: Optional[Union[Mapping[str, Any], Sequence[Sequence[Any]]]]) -> None:
        """Replace Altp parameters for this spectrum."""
        self.altp = _normalize_altp(altp)
        self.altp_uncertainties = {}

    def update_altp(self, **updates: Any) -> None:
        """Update one or more Altp entries in-place."""
        if self.altp is None:
            self.altp = {}
        self.altp.update(updates)
        for name in updates:
            self.altp_uncertainties.pop(name, None)

    def set_expt_data(self, expt_data: Optional[List[List[float]]]) -> None:
        """Set or replace experimental intensity data."""
        self.expt_data = expt_data

    def _eigenvectors_match(self, z: np.ndarray) -> bool:
        return (
            self._cached_eigenvectors is not None
            and self._cached_eigenvectors.shape == z.shape
            and np.array_equal(self._cached_eigenvectors, z)
        )

    def calculate_intensities(
        self, polarization: str = "isotropic", *, reuse_transformed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute intensity data for this spectrum.

        Orchestrates: vtrans (basis transformation) → dipole_str (compute dipole strengths) →
        group_transitions (group by level pair) → add_oscillator_strengths_and_A_coefficients.

        Parameters
        ----------
        polarization : str, optional
            Polarization type ('isotropic', 'axial', 'sigma', 'pi'). Default 'isotropic'.
            (Only 'isotropic' fully supported in MVP; others deferred to phase 2.)

        Returns
        -------
        list of dict
            List of transition group dictionaries with keys:
            - 'Energy': transition energy (cm^-1)
            - 'e_i': initial state energy
            - 'e_f': final state energy
            - 'g_i': initial state degeneracy
            - 'g_f': final state degeneracy
            - 't_list': list of individual transitions in the group
            - 'S_ED_isotropic', 'S_MD_isotropic': electric/magnetic dipole strengths
            - 'A': Einstein A coefficient
            - 'f': oscillator strength

        Raises
        ------
        ValueError
            If hamiltonian has not been diagonalized, or spectrum parameters are invalid.
        """
        if polarization != "isotropic":
            raise ValueError(f"Polarization '{polarization}' not yet supported (MVP phase 1)")

        # Extract eigenvectors and eigenvalues
        w, z = self.hamiltonian.diag()
        self.eigenvalues = w
        self.principal_components = np.argmax(np.abs(z), axis=0)

        # Validate eigenvectors
        if not isinstance(z, np.ndarray) or z.ndim != 2:
            raise ValueError(
                "Hamiltonian eigenvectors must be 2D (got shape "
                f"{z.shape if hasattr(z, 'shape') else 'unknown'})"
            )

        # Transform intensity tensors to eigenbasis (cache-aware).
        if not (reuse_transformed and self.transformed_tensors and self._eigenvectors_match(z)):
            self.transformed_tensors = vtrans(self.intensity_tensors, z)
            self._cached_eigenvectors = np.array(z, copy=True)

        # Compute dipole strengths for all transitions
        # dipole_str() expects 1-based indices and converts internally
        self.dipole_strengths = dipole_str(
            self.i_range,
            self.f_range,
            self.transformed_tensors,
            self.hamiltonian,
            w,
            z,
            md=self.md,
            ed=self.ed,
            Altp=_altp_dict_to_list(self.altp),
        )

        # Group transitions by level pair
        self.groups = group_transitions(self.dipole_strengths, tol=self.group_tol)

        # Add oscillator strengths and A coefficients
        add_oscillator_strengths_and_A_coefficients(self.groups, refractive_index=self.nrefractive)

        # Compute totals
        self.total_f = sum(group.get("f", 0.0) for group in self.groups)
        self.total_A = sum(group.get("A", 0.0) for group in self.groups)

        # Validate that all groups have the same direction (all absorption or all emission)
        if self.groups:
            first_direction = self.groups[0]["Energy"] > 0
            for group in self.groups[1:]:
                if (group["Energy"] > 0) != first_direction:
                    raise ValueError(
                        "Spectrum contains mixed absorption and emission groups. "
                        "Each Spectrum must be purely absorption (Energy > 0) or "
                        "purely emission (Energy < 0)."
                    )

        return self.groups

    def recalculate(self, polarization: str = "isotropic", *, force_vtrans: bool = False) -> List[Dict[str, Any]]:
        """Recalculate intensities after parameter updates."""
        return self.calculate_intensities(
            polarization=polarization, reuse_transformed=not force_vtrans
        )


def gen_inten_summary(
    spectrum: Spectrum,
    format: str = "text",
    state_labels: Optional[List[Any]] = None,
) -> str:
    """
    Generate a human-readable summary of intensity data for a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object with computed groups (from calculate_intensities()).
    format : str, optional
        Output format: 'text' (default, pretty table) or 'csv' (comma-separated).
    state_labels : list of Any, optional
        Human-readable state labels (from spectrum.hamiltonian.tensors[0].states.labels).
        If not provided, uses principal component indices from cached spectrum data.

    Returns
    -------
    str
        Formatted string suitable for printing or writing to file.
    """
    if not spectrum.groups:
        return "No transitions computed."

    hamiltonian = spectrum.hamiltonian

    # Get state labels if not provided
    if state_labels is None:
        if hasattr(hamiltonian, "tensors") and hamiltonian.tensors:
            state_labels = hamiltonian.tensors[0].states.labels
        else:
            # Fallback: use level indices
            w = spectrum.eigenvalues if spectrum.eigenvalues is not None else hamiltonian.diag()[0]
            state_labels = [f"Level {i}" for i in range(len(w))]

    # Use cached eigenvalues and principal components (computed during calculate_intensities)
    w = spectrum.eigenvalues if spectrum.eigenvalues is not None else hamiltonian.diag()[0]
    pc = (
        spectrum.principal_components
        if spectrum.principal_components is not None
        else np.argmax(np.abs(hamiltonian.diag()[1]), axis=0)
    )

    if format == "text":
        return _format_inten_text(spectrum, w, pc, state_labels)
    elif format in ("brief", "detailed", "moments"):
        return _format_inten(spectrum, w, pc, state_labels, format=format)
    elif format == "csv":
        return _format_inten_csv(spectrum, w, pc, state_labels)
    else:
        raise ValueError(
            f"Unknown format: {format}. Use 'text', 'brief', 'detailed', 'moments', or 'csv'."
        )


def _format_altp_param_line(spectrum: Spectrum, name: str, value: Any) -> str:
    """Format one Altp parameter line, appending uncertainty when available."""
    def _fmt(v: Any) -> str:
        if isinstance(v, complex):
            return f"{float(v.real):.6e}{float(v.imag):+.6e}j"
        if isinstance(v, (int, float, np.floating)):
            return f"{float(v):.6e}"
        return str(v)

    unc = spectrum.altp_uncertainties.get(name)
    value_s = _fmt(value)
    if unc is None:
        return f"  {name:<6}: {value_s:>14}"
    unc_s = _fmt(unc)
    return f"  {name:<6}: {value_s:>14} +/- {unc_s:>14}"


def _format_inten_text(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as a human-readable text table."""
    title = f"Spectrum: {spectrum.name}"
    lines = ["=" * len(title), title, "=" * len(title), ""]

    # Print Altp parameters if present
    if spectrum.altp:
        lines.append("Altp (electric dipole coupling) parameters:")
        for name, value in spectrum.altp.items():
            lines.append(_format_altp_param_line(spectrum, name, value))
        lines.append("")

    for group_idx, group in enumerate(spectrum.groups, start=1):
        e_i = group["e_i"]
        e_f = group["e_f"]
        energy = group["Energy"]
        g_i = group["g_i"]
        g_f = group["g_f"]
        f = group["f"]
        A = group["A"]
        t_list = group["t_list"]

        # Get principal component state labels from first transition in group
        if t_list:
            initial_level = t_list[0]["pc_i"]
            final_level = t_list[0]["pc_f"]
        else:
            initial_level = None
            final_level = None

        # Format state labels (with bounds checking)
        if initial_level is not None and 0 <= initial_level < len(state_labels):
            initial_label = state_labels[initial_level]
        else:
            initial_label = f"State {initial_level}" if initial_level is not None else "Unknown"

        if final_level is not None and 0 <= final_level < len(state_labels):
            final_label = state_labels[final_level]
        else:
            final_label = f"State {final_level}" if final_level is not None else "Unknown"

        if isinstance(initial_label, (list, tuple)):
            initial_label_str = " ".join(str(x) for x in initial_label)
        else:
            initial_label_str = str(initial_label)

        if isinstance(final_label, (list, tuple)):
            final_label_str = " ".join(str(x) for x in final_label)
        else:
            final_label_str = str(final_label)

        direction = "->" if energy > 0 else "<-"

        lines.append("")
        lines.append(f"Transition {group_idx}: {initial_label_str} {direction} {final_label_str}")
        lines.append(
            f"  Initial state: {initial_label_str:30s} E = {e_i:12.6f} cm-1 (g={int(g_i)})"
        )
        lines.append(f"  Final state:   {final_label_str:30s} E = {e_f:12.6f} cm-1 (g={int(g_f)})")
        lines.append(f"  Transition energy: {energy:12.6f} cm-1")

        if energy > 0:
            # Absorption
            lines.append(f"  Oscillator strength f:      {f:.6e}")
        else:
            # Emission
            lines.append(f"  Einstein A coefficient:     {A:.6e} s-1")
            if A > 0:
                lifetime_s = 1.0 / A
                lifetime_ms = lifetime_s * 1e3
                lines.append(
                    f"  Lifetime:                   {lifetime_s:.6e} s ({lifetime_ms:.6e} ms)"
                )

    # Add totals summary (f for absorption, A for emission)
    lines.append("")
    if spectrum.groups:
        # Determine if absorption (Energy > 0) or emission (Energy < 0)
        is_absorption = spectrum.groups[0]["Energy"] > 0

        if is_absorption:
            if spectrum.total_f > 0:
                lines.append(f"Total oscillator strength (f): {spectrum.total_f:.6e}")
        else:  # emission
            if spectrum.total_A > 0:
                lines.append(f"Total A coefficient:          {spectrum.total_A:.6e} s-1")
                lifetime_s = 1.0 / spectrum.total_A
                lifetime_ms = lifetime_s * 1e3
                lines.append(
                    f"Lifetime (from total A):      {lifetime_s:.6e} s ({lifetime_ms:.6e} ms)"
                )

    lines.append("=" * 80)
    return "\n".join(lines)


def _format_state_label_content(label: Any, label_key: Optional[str] = None) -> str:
    """
    Format state label content (without pipes or brackets).

    Parameters
    ----------
    label : tuple or list
        State label (tuple of quantum numbers)
    label_key : str, optional
        Label key specifying quantum number types (S, L, J, M, etc.).
        If provided, formats with proper L->term letter conversion.
        If None, returns space-separated quantum numbers.

    Returns
    -------
    str
        Formatted label content without decorative pipes or brackets.
    """
    if label_key:
        try:
            label_str = ""
            for i, l in enumerate(label):
                if i < len(label_key):
                    if label_key[i] == "S":
                        label_str += "{:d}".format(l)
                    elif label_key[i] == "L":
                        label_str += L2term(l)
                    elif label_key[i] == "J":
                        label_str += "{: >2d},".format(l)
                    elif label_key[i] == "M":
                        # M is last element
                        label_str += "{: >3d}".format(l)
                    elif label_key[i] == "X":
                        label_str += "{:d},".format(l)
                    elif label_key[i] == "F":
                        if l:
                            label_str += "(2F)"
                        else:
                            label_str += "    "
                    elif label_key[i] == "T":
                        label_str += "{:d},".format(l)
                    else:
                        # Other types - use default formatting
                        if i < len(label_key) - 1:
                            label_str += "{: >3d},".format(l)
                        else:
                            label_str += "{: >3d}".format(l)
                else:
                    # Beyond label_key length, format normally
                    label_str += "{: >3d}".format(l)
            return label_str
        except Exception:
            # Fallback to simple format if something goes wrong
            if isinstance(label, (list, tuple)):
                return " ".join(str(x) for x in label)
            else:
                return str(label)
    else:
        # No label_key, use simple formatting
        if isinstance(label, (list, tuple)):
            return " ".join(str(x) for x in label)
        else:
            return str(label)


def _format_state_label_short(label: Any, label_key: Optional[str] = None) -> str:
    """
    Format a state label in short form using label_key convention if provided.

    Returns formatted label with pipes and brackets: |2F 5, -5>
    """
    content = _format_state_label_content(label, label_key)
    return f"|{content}>"


def _format_state_label_with_energy(
    label: Any, level: int, energy: float, label_key: Optional[str] = None
) -> str:
    """Format a state label with 1-based level index and energy."""
    content = _format_state_label_content(label, label_key)
    return f"{level:4}: |{content}> ({energy:8.2f})"


def _format_complex_dipole(value: Union[complex, float]) -> str:
    """Format a complex dipole moment value for display."""
    if isinstance(value, complex):
        if value.imag == 0:
            return f"{value.real:>13.6e}"
        elif value.real == 0:
            return f"{value.imag:>13.6e}j"
        else:
            return f"{value.real:>10.3e}{value.imag:+.3e}j"
    else:
        return f"{value:>13.6e}"


def _format_group_line(
    group: Dict[str, Any],
    group_idx: int,
    state_labels: List[Any],
    spectrum: Spectrum,
    is_absorption: bool,
    label_w: int = 42,
) -> str:
    """Format one group line (Group, Initial State, Final State, f_ED, f_MD, f_Total/A_Total)."""
    t_list = group["t_list"]
    e_i = group["e_i"]
    e_f = group["e_f"]

    # Get actual level indices and state labels from first transition in group
    if t_list:
        i_level_idx = t_list[0]["i"]  # Actual level index (0-based)
        f_level_idx = t_list[0]["f"]  # Actual level index (0-based)
        pc_i = t_list[0]["pc_i"]  # Principal component index for SLJM state
        pc_f = t_list[0]["pc_f"]  # Principal component index for SLJM state
        initial_level = i_level_idx + 1  # Convert to 1-based for display
        final_level = f_level_idx + 1
    else:
        initial_level = None
        final_level = None
        pc_i = None
        pc_f = None

    # Format state labels with energies (with bounds checking)
    if (
        t_list
        and pc_i is not None
        and pc_f is not None
        and 0 <= pc_i < len(state_labels)
        and 0 <= pc_f < len(state_labels)
    ):
        label_key = None
        if (
            spectrum.hamiltonian
            and spectrum.hamiltonian.tensors
            and spectrum.hamiltonian.tensors[0]
        ):
            try:
                label_key = spectrum.hamiltonian.tensors[0].states.label_key
            except (AttributeError, IndexError):
                pass
        initial_label = _format_state_label_with_energy(
            state_labels[pc_i], initial_level, e_i, label_key
        )
        final_label = _format_state_label_with_energy(
            state_labels[pc_f], final_level, e_f, label_key
        )
    else:
        initial_label = f"State {initial_level}" if initial_level is not None else "Unknown"
        final_label = f"State {final_level}" if final_level is not None else "Unknown"

    energy = group["Energy"]
    g_i = group.get("g_i", 1)

    # Format energy as absolute value (for both absorption and emission)
    abs_energy = abs(energy)
    energy_str = f"{abs_energy:>9.2f}"

    # Sum dipole strengths over all transitions in group
    total_S_ED = sum(t.get("S_ED_isotropic", 0.0) for t in t_list)
    total_S_MD = sum(t.get("S_MD_isotropic", 0.0) for t in t_list)

    # Calculate f_total and A_total using the group's values
    f_total = group.get("f", 0.0)
    A_total = group.get("A", 0.0)

    # Decompose into ED and MD components
    f_ED = group.get("f_ED")
    f_MD = group.get("f_MD")
    A_ED = group.get("A_ED")
    A_MD = group.get("A_MD")
    if any(v is None for v in (f_ED, f_MD, A_ED, A_MD)):
        f_ED, f_MD, _, A_ED, A_MD, _ = A_and_f_calc(
            total_S_ED, total_S_MD, energy, g_i, nrefractive=spectrum.nrefractive
        )

    # Format group line based on absorption or emission
    if is_absorption:
        line = (
            f"{group_idx:<4} {energy_str} {initial_label:<{label_w}} {final_label:<{label_w}} "
            f"{f_ED:.6e} {f_MD:.6e} {f_total:.6e}"
        )
    else:
        line = (
            f"{group_idx:<4} {energy_str} {initial_label:<{label_w}} {final_label:<{label_w}} "
            f"{A_ED:.6e} {A_MD:.6e} {A_total:.6e}"
        )

    return line


def _format_transition_line(
    trans: Dict[str, Any],
    g_i: float,
    state_labels: List[Any],
    spectrum: Spectrum,
    is_absorption: bool,
) -> str:
    """Format one transition line (indent + state labels + dipole strengths + f values)."""
    i_level_idx = trans["i"]  # Actual level index (0-based)
    f_level_idx = trans["f"]  # Actual level index (0-based)
    i_pc_idx = trans["pc_i"]  # Principal component index for SLJM state
    f_pc_idx = trans["pc_f"]  # Principal component index for SLJM state
    i_1b = i_level_idx + 1  # Convert to 1-based for display
    f_1b = f_level_idx + 1
    e_trans = trans["e"]
    s_ed = trans.get("S_ED_isotropic", 0.0)
    s_md = trans.get("S_MD_isotropic", 0.0)

    # Get state labels using principal component indices (with bounds checking)
    label_key = None
    if spectrum.hamiltonian and spectrum.hamiltonian.tensors and spectrum.hamiltonian.tensors[0]:
        try:
            label_key = spectrum.hamiltonian.tensors[0].states.label_key
        except (AttributeError, IndexError):
            pass

    if 0 <= i_pc_idx < len(state_labels):
        i_label = _format_state_label_short(state_labels[i_pc_idx], label_key)
    else:
        i_label = f"State {i_1b}"

    if 0 <= f_pc_idx < len(state_labels):
        f_label = _format_state_label_short(state_labels[f_pc_idx], label_key)
    else:
        f_label = f"State {f_1b}"

    # Get transition-level ED/MD and total values
    f_ED_t, f_MD_t, f_t, A_ED_t, A_MD_t, A_t = A_and_f_calc(
        s_ed, s_md, e_trans, g_i, nrefractive=spectrum.nrefractive
    )

    if is_absorption:
        trans_line = (
            f"        {i_1b:<4} | {i_label} >                    \t"
            f"{f_1b:<4} | {f_label} >                  {s_ed:>10.6e}  "
            f"{f_ED_t:>13.6e}  {s_md:>10.6e}  {f_MD_t:>13.6e}  {f_t:>13.6e}"
        )
    else:
        trans_line = (
            f"        {i_1b:<4} | {i_label} >                    \t"
            f"{f_1b:<4} | {f_label} >                  {s_ed:>10.6e}  "
            f"{A_ED_t:>13.6e}  {s_md:>10.6e}  {A_MD_t:>13.6e}  {A_t:>13.6e}"
        )

    return trans_line


def _format_dipole_moments(trans: Dict[str, Any]) -> List[str]:
    """Return list of formatted strings for ED and MD dipole moment components."""
    ed_m1 = clean_complex(trans.get("ed_-1", 0.0))
    ed_0 = clean_complex(trans.get("ed_0", 0.0))
    ed_p1 = clean_complex(trans.get("ed_+1", 0.0))
    md_m1 = clean_complex(trans.get("md_-1", 0.0))
    md_0 = clean_complex(trans.get("md_0", 0.0))
    md_p1 = clean_complex(trans.get("md_+1", 0.0))

    return [
        (
            f"             D_ED :      -1: {_format_complex_dipole(ed_m1):>13}    "
            f"0: {_format_complex_dipole(ed_0):>13}   +1: {_format_complex_dipole(ed_p1):>13}"
        ),
        (
            f"             D_MD :      -1: {_format_complex_dipole(md_m1):>13}    "
            f"0: {_format_complex_dipole(md_0):>13}   +1: {_format_complex_dipole(md_p1):>13}"
        ),
    ]


def _format_inten(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
    format: str = "brief",
) -> str:
    """
    Unified formatter for intensity output supporting brief, detailed, and moments formats.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object with computed groups.
    eigenvalues : np.ndarray
        Eigenvalues (not currently used but kept for API consistency).
    principal_components : np.ndarray
        Principal components (not currently used but kept for API consistency).
    state_labels : List[Any]
        State labels for formatting.
    format : str, optional
        Output format: 'brief' (default, one line per group),
        'detailed' (brief + individual transitions),
        'moments' (detailed + dipole moment components).

    Returns
    -------
    str
        Formatted intensity output.
    """
    if format not in ("brief", "detailed", "moments"):
        raise ValueError(f"Unknown format: {format}. Use 'brief', 'detailed', or 'moments'.")

    title = f"Spectrum: {spectrum.name}"
    lines = [
        "=" * len(title),
        title,
        "=" * len(title),
        f"Refractive index: {spectrum.nrefractive}",
        "",
    ]

    # Print Altp parameters if present
    if spectrum.altp:
        lines.append("Altp (electric dipole coupling) parameters:")
        for name, value in spectrum.altp.items():
            lines.append(_format_altp_param_line(spectrum, name, value))
        lines.append("")

    # Determine if absorption or emission
    if not spectrum.groups:
        raise ValueError("Cannot format spectrum with no groups")
    is_absorption = spectrum.groups[0]["Energy"] > 0

    # Build expt_data lookup (group_idx -> f_expt) - only used in brief format
    # Parse with error handling for type consistency
    expt_lookup = {}
    if spectrum.expt_data and format == "brief":
        for expt_entry in spectrum.expt_data:
            # Check if entry is iterable (list, tuple) and has at least 2 elements
            try:
                if isinstance(expt_entry, (list, tuple)) and len(expt_entry) >= 2:
                    group_idx = int(expt_entry[0])  # Convert to int with validation
                    f_expt = float(expt_entry[1])  # Convert to float with validation
                    expt_lookup[group_idx] = f_expt
            except (ValueError, TypeError, IndexError):
                continue  # Skip malformed entries silently

    # Compute label column width from actual rendered labels
    min_label_w = max(len("Initial State"), len("Final State"))  # header text minimum
    label_w = min_label_w
    # Resolve label_key once (same logic as _format_group_line)
    label_key: Optional[str] = None
    if (
        spectrum.hamiltonian
        and spectrum.hamiltonian.tensors
        and spectrum.hamiltonian.tensors[0]
    ):
        try:
            label_key = spectrum.hamiltonian.tensors[0].states.label_key
        except (AttributeError, IndexError):
            pass
    for group in spectrum.groups:
        t_list = group.get("t_list", [])
        if t_list:
            i_idx = t_list[0].get("i", 0)
            f_idx = t_list[0].get("f", 0)
            pc_i = t_list[0].get("pc_i", i_idx)
            pc_f = t_list[0].get("pc_f", f_idx)
            i_label = _format_state_label_with_energy(
                state_labels[pc_i] if 0 <= pc_i < len(state_labels) else None,
                i_idx + 1,
                group["e_i"],
                label_key,
            )
            f_label = _format_state_label_with_energy(
                state_labels[pc_f] if 0 <= pc_f < len(state_labels) else None,
                f_idx + 1,
                group["e_f"],
                label_key,
            )
            label_w = max(label_w, len(i_label), len(f_label))

    label_w += 2  # always leave a small gap after the longest label

    # Header
    if is_absorption:
        header = (
            f"{'Grp':<4} {'Energy':>9} {'':6}{'Initial State':<{label_w - 6}} "
            f"{'':6}{'Final State':<{label_w - 6}} {'f_ED':<12} {'f_MD':<12} {'f_Total':<12}"
        )
    else:
        header = (
            f"{'Grp':<4} {'Energy':>9} {'':6}{'Initial State':<{label_w - 6}} "
            f"{'':6}{'Final State':<{label_w - 6}} {'A_ED':<12} {'A_MD':<12} {'A_Total':<12}"
        )

    # Append expt columns for brief format only
    if spectrum.expt_data and format == "brief":
        expt_label = "f_Expt" if is_absorption else "A_Expt"
        header += f" {expt_label:<12} {'chisqr':<12}"

    sep_width = len(header)
    lines.append("=" * sep_width)
    lines.append(header)
    lines.append("-" * sep_width)

    # Print each group
    total_chi2 = 0.0
    for group_idx, group in enumerate(spectrum.groups, start=1):
        # Print group line
        group_line = _format_group_line(group, group_idx, state_labels, spectrum, is_absorption, label_w)

        # Append experimental data if present (brief format only)
        if spectrum.expt_data and format == "brief":
            f_expt = expt_lookup.get(group_idx)
            f_calc = group.get("f", 0.0) if is_absorption else group.get("A", 0.0)
            if f_expt is None:
                group_line += f" {'---':<12} {'---':<12}"
            else:
                # Calculate chisqr = ((calc - exp) / (calc + exp))^2
                if f_calc + f_expt != 0:
                    chi2 = ((f_calc - f_expt) / (f_calc + f_expt)) ** 2
                else:
                    chi2 = 0.0
                total_chi2 += chi2
                group_line += f" {f_expt:.6e} {chi2:.6e}"

        lines.append(group_line)

        # Print individual transitions if in detailed or moments format
        if format in ("detailed", "moments"):
            lines.append("        Individual transitions:")
            if is_absorption:
                trans_header = (
                    "        i     Initial State                 f      "
                    "Final State                  S_ED_iso      f_ED           "
                    "S_MD_iso      f_MD           f_Total"
                )
            else:
                trans_header = (
                    "        i     Initial State                 f      "
                    "Final State                  S_ED_iso      A_ED           "
                    "S_MD_iso      A_MD           A_Total"
                )
            lines.append(trans_header)

            g_i = group.get("g_i", 1)
            t_list = group["t_list"]

            # List each transition
            for trans in t_list:
                trans_line = _format_transition_line(
                    trans, g_i, state_labels, spectrum, is_absorption
                )
                lines.append(trans_line)

                # Add dipole moment components if in moments format
                if format == "moments":
                    dipole_lines = _format_dipole_moments(trans)
                    lines.extend(dipole_lines)

            lines.append("")  # Blank line between groups

    lines.append("-" * sep_width)

    # Add totals
    if spectrum.groups:
        if is_absorption:
            total_line = (
                f"Total{'':<9} {'':<{label_w}} {'':<{label_w}} {'':<12} {'':<12} {spectrum.total_f:.6e}"
            )
        else:
            total_line = (
                f"Total{'':<9} {'':<{label_w}} {'':<{label_w}} {'':<12} {'':<12} {spectrum.total_A:.6e}"
            )

        if spectrum.expt_data and format == "brief":
            total_line += f" {'':<12} {total_chi2:.6e}"

        lines.append(total_line)

    lines.append("=" * sep_width)
    return "\n".join(lines)


def _format_inten_csv(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as CSV (comma-separated values)."""
    lines = [
        "# Spectrum: " + spectrum.name,
        "initial_level,initial_label,initial_energy_cm-1,final_level,"
        "final_label,final_energy_cm-1,transition_energy_cm-1,"
        "g_i,g_f,f_or_A,A_type",
    ]

    for group in spectrum.groups:
        e_i = group["e_i"]
        e_f = group["e_f"]
        energy = group["Energy"]
        g_i = group["g_i"]
        g_f = group["g_f"]
        f = group["f"]
        A = group["A"]
        t_list = group["t_list"]

        # Get principal component level indices from first transition in group
        if t_list:
            initial_level = t_list[0]["pc_i"]
            final_level = t_list[0]["pc_f"]
        else:
            initial_level = None
            final_level = None

        # Format state labels (with bounds checking) for CSV export
        if initial_level is not None and 0 <= initial_level < len(state_labels):
            initial_label = state_labels[initial_level]
        else:
            initial_label = f"State {initial_level}" if initial_level is not None else "Unknown"

        if final_level is not None and 0 <= final_level < len(state_labels):
            final_label = state_labels[final_level]
        else:
            final_label = f"State {final_level}" if final_level is not None else "Unknown"
        if isinstance(initial_label, (list, tuple)):
            initial_label_str = " ".join(str(x) for x in initial_label)
        else:
            initial_label_str = str(initial_label)

        if isinstance(final_label, (list, tuple)):
            final_label_str = " ".join(str(x) for x in final_label)
        else:
            final_label_str = str(final_label)

        a_type = "f" if energy > 0 else "A"
        a_value = f if energy > 0 else A

        lines.append(
            f"{initial_level},{initial_label_str},{e_i:.6f},"
            f"{final_level},{final_label_str},{e_f:.6f},"
            f"{energy:.6f},{int(g_i)},{int(g_f)},{a_value:.6e},{a_type}"
        )

    return "\n".join(lines)


# ============================================================================
# ALTP PARAMETER FITTING
# ============================================================================


class AltpFit:
    """Fit Altp parameters to match target intensity data."""

    def __init__(
        self,
        param_names: List[str],
        spectra: Union[Spectrum, Sequence[Spectrum]],
        target_intensities: Optional[Union[Dict[int, float], Sequence[Dict[int, float]]]] = None,
        weights: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
        **kwargs: Any,
    ):
        self.param_names = param_names
        if isinstance(spectra, Spectrum) or not isinstance(spectra, SequenceABC):
            self.spectra = [spectra]
        else:
            self.spectra = list(spectra)
        if not self.spectra:
            raise ValueError("At least one Spectrum is required.")

        self.target_intensities = self._normalize_target_intensities(target_intensities)
        self.weights = weights
        self.n_obs = sum(len(t) for t in self.target_intensities)
        self.base_altp_by_spectrum = [dict(spec.altp or {}) for spec in self.spectra]

        # Build parameter info: track which are complex, indices in flat vector
        self.param_info = self._build_param_info()
        self.n_p = self._count_flat_params()
        self.initial_x = self._extract_initial_params()

    def _normalize_target_intensities(
        self,
        target_intensities: Optional[Union[Dict[int, float], Sequence[Dict[int, float]]]],
    ) -> List[Dict[int, float]]:
        if target_intensities is None:
            out = []
            for spec in self.spectra:
                if not spec.expt_data:
                    # No data for this spectrum — contributes 0 to chi-square
                    out.append({})
                    continue
                mapped: Dict[int, float] = {}
                for entry in spec.expt_data:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        mapped[int(entry[0])] = float(entry[1])
                out.append(mapped)
            return out

        if isinstance(target_intensities, dict):
            if len(self.spectra) != 1:
                raise ValueError(
                    "For multiple spectra, target_intensities must be a list of dicts."
                )
            return [target_intensities]

        out = list(target_intensities)
        if len(out) != len(self.spectra):
            raise ValueError(
                "Number of target_intensity datasets must match number of spectra."
            )
        return out

    def _weights_for(self, spec_index: int, n_local: int, offset: int) -> Optional[np.ndarray]:
        if self.weights is None:
            return None
        if isinstance(self.weights, np.ndarray):
            return np.asarray(self.weights[offset : offset + n_local], dtype=float)
        local = np.asarray(self.weights[spec_index], dtype=float)
        if len(local) != n_local:
            raise ValueError(
                f"Weight length mismatch for spectrum index {spec_index}: expected {n_local}, got {len(local)}."
            )
        return local

    def _build_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Build parameter tracking dict for complex/real parameters."""
        info = {}
        flat_idx = 0

        # Use first spectrum's Altp as source of truth.
        altp_dict = self.spectra[0].altp or {}

        for pname in self.param_names:
            if pname not in altp_dict:
                raise ValueError(f"Parameter {pname} not found in Spectrum.altp")

            value = altp_dict[pname]
            is_complex = isinstance(value, complex)

            if is_complex:
                info[pname] = {
                    "type": "complex",
                    "real_index": flat_idx,
                    "imag_index": flat_idx + 1,
                    "initial_value": value,
                }
                flat_idx += 2
            else:
                info[pname] = {
                    "type": "real",
                    "index": flat_idx,
                    "initial_value": value,
                }
                flat_idx += 1

        return info

    def _count_flat_params(self) -> int:
        """Count total flat parameters (real + 2*complex)."""
        count = 0
        for info in self.param_info.values():
            if info["type"] == "complex":
                count += 2
            else:
                count += 1
        return count

    def _extract_initial_params(self) -> np.ndarray:
        """Extract initial parameter vector from current Altp."""
        x = np.zeros(self.n_p)
        for pname, info in self.param_info.items():
            if info["type"] == "complex":
                val = info["initial_value"]
                x[info["real_index"]] = val.real
                x[info["imag_index"]] = val.imag
            else:
                x[info["index"]] = info["initial_value"]
        return x

    def _x_to_altp(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert flat parameter vector to dict-style Altp."""
        altp: Dict[str, Any] = {}
        for pname in self.param_names:
            info = self.param_info[pname]
            if info["type"] == "complex":
                value = complex(x[info["real_index"]], x[info["imag_index"]])
            else:
                value = float(x[info["index"]])
            altp[pname] = value
        return altp

    def _compute_grouped_intensities(self, x: np.ndarray) -> List[np.ndarray]:
        """Compute grouped intensities for all spectra for parameter vector x."""
        fitted_altp_subset = self._x_to_altp(x)
        computed_all: List[np.ndarray] = []
        for i, (spec, target_map) in enumerate(zip(self.spectra, self.target_intensities)):
            # Keep non-fitted Altp parameters fixed for each spectrum.
            full_altp = dict(self.base_altp_by_spectrum[i])
            full_altp.update(fitted_altp_subset)
            spec.set_altp(full_altp)
            spec.recalculate(polarization="isotropic")
            if not spec.groups:
                computed_all.append(np.full(len(target_map), np.nan))
                continue
            is_absorption = spec.groups[0]["Energy"] > 0
            key = "f" if is_absorption else "A"
            computed = {group_idx: group.get(key, 0.0) for group_idx, group in enumerate(spec.groups, start=1)}
            indices = sorted(target_map.keys())
            values = np.array([computed.get(i, np.nan) for i in indices], dtype=float)
            computed_all.append(values)
        return computed_all

    def per_spectrum_chi2(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """Return per-spectrum chi² contributions and observation counts."""
        computed_all = self._compute_grouped_intensities(x)
        rows: List[Dict[str, Any]] = []
        offset = 0
        for spec_index, (spec, computed, target_map) in enumerate(
            zip(self.spectra, computed_all, self.target_intensities)
        ):
            if np.any(np.isnan(computed)):
                raise ValueError(f"Computation failed for spectrum '{spec.name}'.")
            target_vals = np.array([target_map[i] for i in sorted(target_map.keys())], dtype=float)
            epsilon = 1e-20
            denominator = np.abs(computed) + np.abs(target_vals) + epsilon
            residuals = (computed - target_vals) / denominator
            local_weights = self._weights_for(spec_index, len(target_vals), offset)
            if local_weights is not None:
                residuals = residuals * np.sqrt(local_weights)
            chi2_val = float(np.sum(residuals**2))
            rows.append({"name": spec.name, "chi2": chi2_val, "n_obs": len(target_vals)})
            offset += len(target_vals)
        return rows

    def compute_residual(self, x: np.ndarray) -> float:
        """
        Compute symmetric residual for objective function.

        χ² = Σ w_i * abs((computed_i - target_i) / (computed_i + target_i))²

        This is symmetric and robust to scaling.
        """
        try:
            rows = self.per_spectrum_chi2(x)
        except Exception:
            return 1e10
        return float(sum(row["chi2"] for row in rows))

    def objective_fn(self, x: np.ndarray) -> float:
        """Objective function for minimizer."""
        return self.compute_residual(x)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """
        Return residuals vector for least_squares optimization.

        Each entry is (computed_i - target_i) / (|computed_i| + |target_i| + epsilon),
        optionally scaled by sqrt(weight_i).  When squared and summed, this yields
        the same chi² as compute_residual.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            One-dimensional residuals vector.
        """
        try:
            computed_all = self._compute_grouped_intensities(x)
        except Exception:
            return np.full(self.n_obs, 1e5)

        residuals_list = []
        offset = 0
        for spec_index, (spec, computed, target_map) in enumerate(
            zip(self.spectra, computed_all, self.target_intensities)
        ):
            if np.any(np.isnan(computed)):
                # Return large residuals on NaN
                target_vals = np.array([target_map[i] for i in sorted(target_map.keys())], dtype=float)
                residuals_list.append(np.full_like(target_vals, 1e5))
                continue

            target_vals = np.array([target_map[i] for i in sorted(target_map.keys())], dtype=float)
            epsilon = 1e-20
            denominator = np.abs(computed) + np.abs(target_vals) + epsilon
            r = (computed - target_vals) / denominator
            
            local_weights = self._weights_for(spec_index, len(target_vals), offset)
            if local_weights is not None:
                r = r * np.sqrt(local_weights)
            
            residuals_list.append(r)
            offset += len(target_vals)

        return np.concatenate(residuals_list)



def fit_altp(
    param_names: List[str],
    spectra: Union[Spectrum, Sequence[Spectrum]],
    target_intensities: Optional[Union[Dict[int, float], Sequence[Dict[int, float]]]] = None,
    cfl_min: Optional[Any] = None,
    weights: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Fit Altp parameters to match target intensity data using Spectrum object(s).

    Parameters
    ----------
    param_names : list of str
        Altp parameter names to fit (e.g., ["A210", "A230", "A233"])
    spectra : Spectrum or sequence of Spectrum
        Spectrum object(s) to fit. Each spectrum carries its Hamiltonian and setup.
    target_intensities : dict or list of dict, optional
        Per-spectrum target intensity mappings (group_idx -> f/A value).
        If omitted, each spectrum must already have expt_data set.
    cfl_min : cfl.CFLMin, optional
        Minimization object specifying solver and options (not used, kept for API consistency)
    weights : np.ndarray, optional
        Per-group weights (default: equal weights)
    **kwargs
        Additional options passed to the optimizer. Key kwargs:

        ``minimizer`` : str, optional
            Which scipy optimizer to use. Choices:

            * ``"minimize"`` (default) — ``scipy.optimize.minimize``.
              Pass ``method`` (e.g. ``"Nelder-Mead"``, ``"Powell"``,
              ``"L-BFGS-B"``) and ``options`` dict.
            * ``"least_squares"`` — ``scipy.optimize.least_squares``.
              Pass ``method`` (``"lm"``, ``"trf"``, or ``"dogbox"``,
              default ``"lm"``). Efficient for well-behaved least-squares
              problems; ``"lm"`` is Levenberg-Marquardt. ``"trf"`` and
              ``"dogbox"`` support ``bounds``.
            * ``"basinhopping"`` — global search via random perturbations +
              local minimization. Pass ``minimizer_kwargs`` (dict with
              ``method``, ``options``) and ``niter`` (default 100).
            * ``"differential_evolution"`` — population-based global search.
              Requires ``bounds`` (list of ``(min, max)`` per parameter).
            * ``"dual_annealing"`` — simulated annealing global search.
              Requires ``bounds``; optionally pass ``x0``.
            * ``"shgo"`` — simplicial homology global optimization.
              Requires ``bounds``.
            * ``"direct"`` — DIRECT algorithm global search.
              Requires ``bounds``.

        All other kwargs (except ``dry_run``) are forwarded directly to the
        chosen optimizer.

        ``dry_run`` : bool, optional
            If True, evaluate chi² at the current parameters without fitting.

        ``calculate_sigma`` : bool, optional
            If True (default), estimate parameter uncertainties from the Hessian.
        ``include_covariance`` : bool, optional
            If True, include the estimated covariance matrix in the result dict.
        ``include_jacobian`` : bool, optional
            If True, include optimizer Jacobian (when available) in the result dict.

    Returns
    -------
    dict
        Result dictionary with keys:
        - 'fitted_params': dict of fitted parameter names to values
        - 'uncertainties': dict of parameter uncertainties (1-sigma estimates)
        - 'chi2': final chi-squared value
        - 'n_obs': number of observables (intensity groups)
        - 'n_params': number of fitted parameters
        - 'initial_altp': initial Altp values
        - 'summary': human-readable summary string
    """
    calculate_sigma = bool(kwargs.pop("calculate_sigma", True))
    include_covariance = bool(kwargs.pop("include_covariance", False))
    include_jacobian = bool(kwargs.pop("include_jacobian", False))

    started_at = datetime.now()

    # Create fitter
    fitter = AltpFit(
        param_names,
        spectra,
        target_intensities,
        weights=weights,
        **kwargs,
    )
    initial_x = np.array(fitter.initial_x, copy=True)
    initial_chi2 = fitter.objective_fn(initial_x)

    dry_run = bool(kwargs.get("dry_run", False))
    reverted_to_initial = False
    optimizer_result = None

    if dry_run:
        x_opt = initial_x
        fmin = initial_chi2
    else:
        minimizer_name = str(kwargs.get("minimizer", "minimize")).lower()
        optimizer_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("dry_run", "minimizer", "calculate_sigma", "include_covariance", "include_jacobian")
        }

        _BOUNDS_REQUIRED = {"differential_evolution", "dual_annealing", "shgo", "direct"}
        if minimizer_name in _BOUNDS_REQUIRED and "bounds" not in optimizer_kwargs:
            raise ValueError(
                f"minimizer='{minimizer_name}' requires a 'bounds' kwarg: "
                f"a list of (min, max) tuples, one per fitted parameter."
            )

        if minimizer_name == "minimize":
            method = optimizer_kwargs.pop("method", "Nelder-Mead")
            options = optimizer_kwargs.pop("options", {})
            result = minimize(
                fitter.objective_fn, fitter.initial_x,
                method=method, options=options, **optimizer_kwargs,
            )
        elif minimizer_name == "least_squares":
            method = optimizer_kwargs.pop("method", "lm")
            result = least_squares(
                fitter.residuals, fitter.initial_x,
                method=method, **optimizer_kwargs,
            )
        elif minimizer_name == "basinhopping":
            niter = optimizer_kwargs.pop("niter", 100)
            minimizer_kwargs = dict(
                optimizer_kwargs.pop("minimizer_kwargs", {"method": "Nelder-Mead"})
            )
            # Allow users to pass bounds at fit_altp() top-level for consistency
            # with other optimizers; route them into the local minimizer kwargs.
            if "bounds" in optimizer_kwargs:
                bh_bounds = optimizer_kwargs.pop("bounds")
                minimizer_kwargs.setdefault("bounds", bh_bounds)
            # SciPy minimize() passes bounds separately and then expands options.
            # If options also contains "bounds", Nelder-Mead receives duplicate
            # bounds and raises TypeError; normalize by promoting/removing it.
            options = minimizer_kwargs.get("options")
            if isinstance(options, dict) and "bounds" in options:
                options = dict(options)
                option_bounds = options.pop("bounds")
                minimizer_kwargs["options"] = options
                minimizer_kwargs.setdefault("bounds", option_bounds)
            result = basinhopping(
                fitter.objective_fn, fitter.initial_x,
                niter=niter, minimizer_kwargs=minimizer_kwargs,
                **optimizer_kwargs,
            )
        elif minimizer_name == "differential_evolution":
            result = differential_evolution(
                fitter.objective_fn, **optimizer_kwargs,
            )
        elif minimizer_name == "dual_annealing":
            x0 = optimizer_kwargs.pop("x0", fitter.initial_x)
            result = dual_annealing(
                fitter.objective_fn, x0=x0, **optimizer_kwargs,
            )
        elif minimizer_name == "shgo":
            result = shgo(fitter.objective_fn, **optimizer_kwargs)
        elif minimizer_name == "direct":
            result = direct(fitter.objective_fn, **optimizer_kwargs)
        else:
            raise ValueError(
                f"Unknown minimizer '{minimizer_name}'. "
                f"Choose from: minimize, least_squares, basinhopping, "
                f"differential_evolution, dual_annealing, shgo, direct."
            )

        x_opt = result.x
        # least_squares returns result.cost (0.5 * sum(residuals^2)); others return result.fun (scalar)
        if minimizer_name == "least_squares":
            fmin = 2.0 * result.cost  # chi2 = sum(residuals^2) = 2 * cost
        else:
            fmin = result.fun
        optimizer_result = result
        # Guard against regressions from non-smooth objectives/group re-ordering:
        # never accept a solution worse than the starting point.
        if fmin > initial_chi2:
            x_opt = initial_x
            fmin = initial_chi2
            reverted_to_initial = True

    # Reconstruct fitted Altp
    fitted_dict = fitter._x_to_altp(x_opt)

    for spec in fitter.spectra:
        if hasattr(spec, "update_altp"):
            spec.update_altp(**fitted_dict)
        else:
            merged_altp = dict(getattr(spec, "altp", {}) or {})
            merged_altp.update(fitted_dict)
            spec.set_altp(merged_altp)
        spec.recalculate(polarization="isotropic")

    # Estimate parameter uncertainties from Hessian
    covariance = None
    uncertainty_diagnostics: Dict[str, Any] = {}
    if dry_run or reverted_to_initial or not calculate_sigma:
        uncertainties = {}
    else:
        unc_result = _estimate_parameter_uncertainties(
            fitter, x_opt, fmin, param_names, return_details=True
        )
        if isinstance(unc_result, tuple) and len(unc_result) == 3:
            uncertainties, covariance, uncertainty_diagnostics = unc_result
        else:
            uncertainties = unc_result

    jacobian = None
    jacobian_info: Dict[str, Any] = {}
    if optimizer_result is not None and hasattr(optimizer_result, "jac"):
        jac = getattr(optimizer_result, "jac")
        if jac is not None:
            # least_squares may return a sparse Jacobian (e.g. with jac_sparsity).
            # Convert safely for diagnostics without breaking the fit result path.
            try:
                if hasattr(jac, "toarray"):
                    jacobian = np.asarray(jac.toarray(), dtype=float)
                else:
                    jacobian = np.asarray(jac, dtype=float)
                jacobian_info = jacobian_diagnostics(jacobian, fitter.n_p)
            except Exception:
                jacobian = None
                jacobian_info = {}
    chi2_rows = fitter.per_spectrum_chi2(x_opt)
    for spec in fitter.spectra:
        if hasattr(spec, "altp_uncertainties"):
            spec.altp_uncertainties = dict(uncertainties)
    completed_at = datetime.now()

    # Build summary (parameters first; diagnostics at the end).
    summary_main = "\n====================\n"
    summary_main += "fit_altp summary\n"
    summary_main += "====================\n"
    summary_main += gen_pycf_summary(started_at, suppress_input=True)
    summary_main += gen_completed_str(completed_at)
    summary_main += "\n"

    summary_diag = "\nFit diagnostics\n"
    summary_diag += "===============\n"
    summary_diag += f"Fitted parameters: {', '.join(param_names)}\n"
    summary_diag += f"Number of spectra: {len(fitter.spectra)}\n"
    summary_diag += f"Number of observations: {fitter.n_obs}\n"
    summary_diag += f"Number of parameters: {fitter.n_p}\n"
    summary_diag += f"Initial chisqr: {initial_chi2:.6e}\n"
    if dry_run:
        summary_diag += "Mode: dry_run (no optimization performed)\n"
    elif reverted_to_initial:
        summary_diag += "Mode: optimization reverted to initial parameters (no improvement)\n"
    summary_diag += f"Final chisqr: {fmin:.6e}\n\n"
    summary_diag += "Per-spectrum chisqr contributions:\n"
    summary_diag += f"{'Spectrum':<40} {'n_obs':>8} {'chi2':>16}\n"
    summary_diag += f"{'-'*40} {'-'*8:>8} {'-'*16:>16}\n"
    for row in chi2_rows:
        summary_diag += f"{row['name']:<40} {row['n_obs']:>8d} {row['chi2']:>16.6e}\n"
    summary_diag += "\nFitted parameter details:\n"
    summary_diag += f"{'Parameter':<12} {'Initial':>22} {'Fitted':>22} {'Difference':>22} {'Uncertainty':>22}\n"
    summary_diag += "-" * 108 + "\n"
    for pname, value in fitted_dict.items():
        initial_val = fitter.param_info[pname]["initial_value"]
        diff = value - initial_val
        unc = uncertainties.get(pname, None)
        unc_str = str(unc) if unc is not None else "n/a"
        summary_diag += (
            f"{pname:<12} {str(initial_val):>22} {str(value):>22} "
            f"{str(diff):>22} {unc_str:>22}\n"
        )
    if uncertainty_diagnostics:
        summary_diag += "\nUncertainty diagnostics:\n"
        summary_diag += f"  rank(H): {uncertainty_diagnostics.get('rank', 'n/a')} / {uncertainty_diagnostics.get('n_params', 'n/a')}\n"
        summary_diag += f"  cond(H): {uncertainty_diagnostics.get('condition_hessian', 'n/a')}\n"
    if jacobian_info:
        summary_diag += "\nJacobian diagnostics:\n"
        summary_diag += f"  rank(J): {jacobian_info.get('rank', 'n/a')} / {jacobian_info.get('n_params', 'n/a')}\n"
        summary_diag += f"  cond(J^T J): {jacobian_info.get('condition_jtj', 'n/a')}\n"
    summary = summary_main + "\n" + summary_diag

    return {
        "fitted_params": fitted_dict,
        "uncertainties": uncertainties,
        "sigma": uncertainties,
        "chi2": fmin,
        "n_obs": fitter.n_obs,
        "n_params": fitter.n_p,
        "n_spectra": len(fitter.spectra),
        "chi2_by_spectrum": chi2_rows,
        "initial_altp": {pname: fitter.param_info[pname]["initial_value"] for pname in param_names},
        "initial_chi2": initial_chi2,
        "improved": fmin < initial_chi2,
        "reverted_to_initial": reverted_to_initial,
        "dry_run": dry_run,
        "summary_main": summary_main,
        "summary_diagnostics": summary_diag,
        "summary": summary,
        "optimizer_result": optimizer_result,
        "covariance": covariance if include_covariance else None,
        "jacobian": jacobian if include_jacobian else None,
        "jacobian_diagnostics": jacobian_info,
        "uncertainty_diagnostics": uncertainty_diagnostics,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers for multi-spectrum operations
# ---------------------------------------------------------------------------


def inten_calculate(
    spectra: Sequence[Spectrum], polarization: str = "isotropic"
) -> None:
    """Call calculate_intensities() on each spectrum in the list."""
    for spec in spectra:
        spec.calculate_intensities(polarization=polarization)


def inten_print(
    spectra: Union[Spectrum, Sequence[Spectrum]],
    format: str = "brief",
    **kwargs: Any,
) -> None:
    """Print gen_inten_summary() for each spectrum in the list."""
    if isinstance(spectra, Spectrum):
        spectra = [spectra]
    for spec in spectra:
        print(gen_inten_summary(spec, format=format, **kwargs))

    # Print grand total chi-square across all spectra with expt data
    grand_total_chi2 = 0.0
    has_any_expt = False
    for spec in spectra:
        if spec.expt_data and spec.groups:
            has_any_expt = True
            expt_lookup: Dict[int, float] = {}
            for expt_entry in spec.expt_data:
                try:
                    if isinstance(expt_entry, (list, tuple)) and len(expt_entry) >= 2:
                        expt_lookup[int(expt_entry[0])] = float(expt_entry[1])
                except (ValueError, TypeError, IndexError):
                    continue
            is_absorption = spec.groups[0]["Energy"] > 0
            for group_idx, group in enumerate(spec.groups, start=1):
                f_expt = expt_lookup.get(group_idx)
                if f_expt is not None:
                    f_calc = group.get("f", 0.0) if is_absorption else group.get("A", 0.0)
                    if f_calc + f_expt != 0:
                        grand_total_chi2 += ((f_calc - f_expt) / (f_calc + f_expt)) ** 2
    if has_any_expt:
        print(f"Total chisqr: {grand_total_chi2:.6e}")


def inten_set_expt_data(
    spectra: Sequence[Spectrum],
    data_list: Sequence[Optional[List[List[float]]]],
) -> None:
    """Call set_expt_data() on each spectrum with its corresponding data."""
    if len(spectra) != len(data_list):
        raise ValueError(
            f"spectra and data_list must have the same length "
            f"(got {len(spectra)} and {len(data_list)})"
        )
    for spec, data in zip(spectra, data_list):
        spec.set_expt_data(data)


def inten_set_altp(
    spectra: Sequence[Spectrum],
    altp: Optional[Union[Mapping[str, Any], Sequence[Sequence[Any]]]],
) -> None:
    """Call set_altp() on every spectrum with the same Altp dict."""
    for spec in spectra:
        spec.set_altp(altp)


def inten_recalculate(
    spectra: Sequence[Spectrum], polarization: str = "isotropic"
) -> None:
    """Call recalculate() on each spectrum in the list."""
    for spec in spectra:
        spec.recalculate(polarization=polarization)


def ms_fit_altp(
    param_names: List[str],
    spectra: Sequence[Spectrum],
    target_intensities: Optional[Sequence[Dict[int, float]]] = None,
    cfl_min: Optional[Any] = None,
    weights: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Multi-spectrum Altp parameter fitting.

    Requires a list/sequence of Spectrum objects. After fitting, prints the fit
    summary followed by an updated brief intensity table for each spectrum.

    Parameters are parallel to fit_altp; see that function for details.
    Extra kwarg ``print_summary`` (default True) can be set to False to suppress
    automatic printing.
    """
    if isinstance(spectra, Spectrum) or not isinstance(spectra, SequenceABC):
        raise TypeError("ms_fit_altp requires a sequence of Spectrum objects.")
    if len(spectra) == 0:
        raise ValueError("ms_fit_altp requires at least one Spectrum.")
    if not all(isinstance(spec, Spectrum) for spec in spectra):
        raise TypeError("ms_fit_altp requires a sequence containing only Spectrum objects.")
    print_summary = kwargs.pop("print_summary", True)
    result = fit_altp(
        param_names,
        list(spectra),
        target_intensities=target_intensities,
        cfl_min=cfl_min,
        weights=weights,
        **kwargs,
    )
    if print_summary:
        print(result.get("summary_main", result["summary"]))
        inten_print(list(spectra), format="brief")
        if "summary_diagnostics" in result:
            print(result["summary_diagnostics"])
    return result


def _estimate_parameter_uncertainties(
    fitter: "AltpFit",
    x_opt: np.ndarray,
    fmin: float,
    param_names: List[str],
    return_details: bool = False,
) -> Any:
    """
    Estimate parameter uncertainties from the Hessian matrix.

    Uses numerical differentiation to compute the Hessian (second derivatives)
    at the optimum. The covariance matrix is estimated as the inverse of the
    Hessian scaled by the reduced χ² (χ²/dof).

    Parameters
    ----------
    fitter : AltpFit
        The fitter object containing parameter structure
    x_opt : np.ndarray
        Optimal parameter vector
    fmin : float
        Final objective function value (chi-squared)
    param_names : list of str
        Parameter names

    Returns
    -------
    dict
        Parameter uncertainties keyed by parameter name
    """
    from scipy.optimize import approx_fprime

    n_obs = fitter.n_obs
    n_params = len(x_opt)

    # Reduced chi-squared (normalizes for dof)
    dof = max(1, n_obs - n_params)
    chi2_red = fmin / dof if dof > 0 else 1.0

    # Compute Hessian numerically
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(x_opt))
    hessian = np.zeros((n_params, n_params))

    for i in range(n_params):
        x_plus = x_opt.copy()
        x_plus[i] += eps[i]
        grad_plus = approx_fprime(x_plus, fitter.objective_fn, eps)

        x_minus = x_opt.copy()
        x_minus[i] -= eps[i]
        grad_minus = approx_fprime(x_minus, fitter.objective_fn, eps)

        hessian[i, :] = (grad_plus - grad_minus) / (2 * eps[i])

    # Make Hessian symmetric
    hessian = 0.5 * (hessian + hessian.T)

    # Estimate covariance from inverse Hessian
    try:
        cov = np.linalg.inv(hessian) * chi2_red
        uncertainties = {}

        # Map uncertainties back to parameter names
        idx = 0
        for pname in param_names:
            param_info = fitter.param_info[pname]
            if param_info["type"] == "complex":
                real_unc = np.sqrt(max(0, cov[idx, idx]))
                imag_unc = np.sqrt(max(0, cov[idx + 1, idx + 1]))
                uncertainties[pname] = (real_unc, imag_unc)
                idx += 2
            else:
                unc = np.sqrt(max(0, cov[idx, idx]))
                uncertainties[pname] = unc
                idx += 1

        diagnostics = {
            "rank": int(np.linalg.matrix_rank(hessian)),
            "n_params": int(n_params),
            "condition_hessian": float(np.linalg.cond(hessian)),
            "well_conditioned": bool(np.isfinite(np.linalg.cond(hessian)) and np.linalg.cond(hessian) < 1e12),
        }
        if return_details:
            return uncertainties, cov, diagnostics
        return uncertainties
    except np.linalg.LinAlgError:
        # Hessian is singular or near-singular
        uncertainties = {pname: None for pname in param_names}
        diagnostics = {
            "rank": 0,
            "n_params": int(n_params),
            "condition_hessian": np.inf,
            "well_conditioned": False,
        }
        if return_details:
            return uncertainties, None, diagnostics
        return uncertainties


def inten_plot(  # pragma: no cover
    spectrum: "Spectrum",
    fwhm: float = 0.5,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    npoints: int = 10000,
    figsize: Tuple[float, float] = (12, 6),
    comment: Optional[str] = None,
) -> "Tuple[Any, Any]":
    """
    Plot calculated and experimental intensities.

    Creates a plot with:
    - Calculated intensities convoluted with a Lorentzian line shape
    - Experimental data as vertical stick lines

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object containing transition groups with energies and intensities.
    fwhm : float, optional
        Full width at half maximum of the Lorentzian (cm^-1). Default: 0.5 cm^-1.
    xlim : list of float, optional
        Energy range [E_min, E_max] for plot. If None, determined from transition energies.
    ylim : list of float, optional
        Intensity range [I_min, I_max] for plot. If None, auto-scaled to data.
    npoints : int, optional
        Number of points for energy grid (controls resolution). Default: 10000.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (12, 6).
    comment : str, optional
        If provided, prepended to the plot title as ``"comment": spectrum_name``.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.

    Raises
    ------
    ValueError
        If spectrum has no groups or matplotlib cannot be imported.
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for inten_plot(). Install with: pip install matplotlib"
        )

    if not spectrum.groups:
        raise ValueError("Spectrum must have at least one transition group")

    # Determine if absorption or emission based on first group's energy sign
    is_absorption = spectrum.groups[0]["Energy"] > 0 if spectrum.groups else True

    # Extract transition energies and intensities
    energies: list[float] = []
    intensities: list[float] = []
    for group in spectrum.groups:
        energies.append(abs(group.get("Energy", 0.0)))
        if is_absorption:
            intensities.append(group.get("f", 0.0))
        else:
            intensities.append(group.get("A", 0.0))

    energies_arr = np.array(energies)
    intensities_arr = np.array(intensities)

    # Determine energy range for plotting
    if xlim is None:
        e_min, e_max = float(energies_arr.min()), float(energies_arr.max())
        margin = (e_max - e_min) * 0.1 if e_max > e_min else 10.0
        xlim = (e_min - margin, e_max + margin)

    # Generate energy grid for convolution (fine step size for smooth curve)
    energy_grid = np.linspace(xlim[0], xlim[1], npoints)

    # Convolute with Lorentzian
    convoluted = np.zeros(npoints)
    for e, inten in zip(energies_arr, intensities_arr):
        convoluted += inten * lorentzian_constant_height(energy_grid, e, fwhm)

    # Create plot using spectrum name as figure identifier, with random suffix
    # to allow multiple plots of same spectrum with different zoom levels
    figure_name = f"{spectrum.name} #{str(uuid4())[:8]}"
    fig, ax = plt.subplots(figsize=figsize, num=figure_name)

    # Plot convoluted spectrum
    ax.plot(energy_grid, convoluted, "b-", linewidth=2, label="Calculated")

    # Plot stick spectrum for calculated
    ax.vlines(
        energies_arr, 0, intensities_arr, colors="blue", alpha=0.5, linewidth=1, linestyles="solid"
    )

    # Mark each transition position on the x-axis
    ax.plot(energies_arr, np.zeros_like(energies_arr), "o", color="blue", markersize=4, alpha=0.7)

    # If experimental data available, plot as stick lines
    if spectrum.expt_data:
        expt_energies = []
        expt_intensities = []
        for expt_entry in spectrum.expt_data:
            # Check if entry is iterable (list, tuple) and has at least 2 elements
            try:
                if isinstance(expt_entry, (list, tuple)) and len(expt_entry) >= 2:
                    group_idx = int(expt_entry[0])  # Convert to int with validation
                    f_expt = float(expt_entry[1])  # Convert to float with validation
                else:
                    continue
            except (ValueError, TypeError, IndexError):
                continue  # Skip malformed entries

            if 1 <= group_idx <= len(spectrum.groups):
                e = abs(spectrum.groups[group_idx - 1].get("Energy", 0.0))
                expt_energies.append(e)
                expt_intensities.append(f_expt)

        if expt_energies:
            expt_energies_arr = np.array(expt_energies)
            expt_intensities_arr = np.array(expt_intensities)
            ax.vlines(
                expt_energies_arr,
                0,
                expt_intensities_arr,
                colors="red",
                alpha=0.8,
                linewidth=2,
                linestyles="solid",
                label="Experimental",
            )

    # Labels and formatting
    ax.set_xlabel("Energy (cm$^{-1}$)", fontsize=14)
    if is_absorption:
        ylabel = "Oscillator Strength (dimensionless)"
    else:
        ylabel = "A Coefficient (s$^{-1}$)"
    ax.set_ylabel(ylabel, fontsize=14)
    title = f"{spectrum.name} (FWHM = {fwhm} cm$^{{-1}}$)"
    if comment is not None:
        title = f"{comment}: {title}"
    ax.set_title(title, fontsize=13, pad=20)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Increase font size for axis tick labels
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)

    # Add secondary x-axis for wavelength in nanometers
    # Conversion: λ (nm) = 10^7 / wavenumber (cm^-1)
    ax2 = ax.twiny()
    # Get the energy limits and convert to wavelength
    assert xlim is not None, "xlim should be set by this point"
    e_min, e_max = xlim[0], xlim[1]
    # Avoid division by zero at very low energies
    if e_min > 0:
        lambda_max = 1e7 / e_min  # Lower energy = longer wavelength
        lambda_min = 1e7 / e_max  # Higher energy = shorter wavelength
    else:
        lambda_min = 1e7 / e_max if e_max > 0 else 10000
        lambda_max = 1e7 / 0.1 if e_min <= 0 else 1e7 / e_min

    ax2.set_xlim(lambda_max, lambda_min)  # Reversed to match energy axis direction
    ax2.set_xlabel("Wavelength (nm)", fontsize=14)
    ax2.tick_params(axis="x", labelsize=13)

    return fig, ax
