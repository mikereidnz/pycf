#!/usr/bin/env python
# Filename = inten.py
"""
A rewrite of the intensity calculation to follow the old Pascal code more closely,
"""

from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pycf.constants import (
    BOLTZMANN_CM_INVERSE,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    EPSILON_0,
    HBAR,
    SPEED_OF_LIGHT,
)
from pycf.njsymbols import wigner_3j


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
    lrange: List[List[int]],
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
    lrange : list
        A list of two lists: [initial_levels, final_levels], where each
        sub-list contains 0-based level indices.
        Example: [[0, 1], [6, 7, 8, 9]] selects levels 0–1 as initial
        and levels 6–9 as final.
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
    for i in lrange[0]:
        for f in lrange[1]:
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
                    "pci": pc[i],
                    "pcf": pc[f],
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
) -> Tuple[float, float]:
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
    A : float
        Einstein A coefficient for the transition (s^-1).
    f : float
        Oscillator strength (dimensionless).
    """
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
    f = omega * oscfactor * (sed * 1 / nrefractive * chilocal + smd * nrefractive) / g_i
    A = (
        omega
        * omega
        * omega
        * afactor
        * (sed * nrefractive * chilocal + smd * nrefractive * nrefractive * nrefractive)
        / g_i
    )
    return abs(A), abs(f)


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
        A, f = A_and_f_calc(
            group["S_ED_isotropic"],
            group["S_MD_isotropic"],
            group["Energy"],
            group["g_i"],
            nrefractive=refractive_index,
        )
        group["A"] = A
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


def lorentzian(x: Union[float, np.ndarray], x0: float, fwhm: float) -> Union[float, np.ndarray]:
    """
    Calculate Lorentzian line shape.

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


def inten(
    trs: List[Dict[str, Any]],
    polarization: str,
    linewidth: float,
    T: float,
    xlim: Optional[List[float]] = None,
    npoints: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate intensity spectrum from transitions and produce spectral data.

    Parameters
    ----------
    trs : list of dict
        List of transition dictionaries, each containing 'e' (transition energy),
        'ei' (initial state energy), and transition strengths by polarization.
    polarization : str
        Polarization type for intensity calculation (e.g., 'pi', 'sigma').
    linewidth : float
        Full width at half maximum (FWHM) of the Lorentzian line shape (cm^-1).
    T : float
        Temperature in Kelvin for Boltzmann factor calculation.
    xlim : list of float, optional
        Energy range [xmin, xmax] for plotting. If None, computed from transitions.
    npoints : int, optional
        Number of points in the intensity curve (default 1000).

    Returns
    -------
    tuple of numpy.ndarray
        ``(line_energies, line_inten, curve_energies, curve_inten)``:

        - ``line_energies``: array of line positions
        - ``line_inten``: array of line intensities
        - ``curve_energies``: array of energy values
        - ``curve_inten``: array of cumulative-curve intensities

    Raises
    ------
    ValueError
        If linewidth <= 0, T < 0, npoints < 1, or trs is empty.
    """
    if len(trs) == 0:
        raise ValueError("inten requires at least one transition.")
    if linewidth <= 0:
        raise ValueError(f"linewidth must be positive (got {linewidth})")
    if T < 0:
        raise ValueError(f"Temperature T must be non-negative (got {T})")
    if npoints < 1:
        raise ValueError(f"npoints must be >= 1 (got {npoints})")
    # Determine the smallest initial energy level, which we assume to be the
    # ground state (used for scaling other energies for boltzmann factor... this
    # behavior is not really obvious.
    min_energy = min(trs, key=lambda tr: tr["ei"])["ei"]
    if xlim is None:
        xmin = min(trs, key=lambda tr: tr["e"])["e"]
        xmax = max(trs, key=lambda tr: tr["e"])["e"]
        # add some padding to plotting range
        xmin -= 50 * linewidth
        xmax += 50 * linewidth
        xlim = [xmin, xmax]
    ntrs = len(trs)
    curve_energies = np.linspace(xlim[0], xlim[1], npoints)
    curve_inten = np.zeros(npoints)
    line_energies = np.zeros(ntrs)
    line_inten = np.zeros(ntrs)
    for i, tr in enumerate(trs):
        line_energies[i] = tr["e"]
        # Calculate the individual line intensities.
        line_inten[i] = boltzmann_factor(tr["ei"] - min_energy, T) * tr[polarization]
        # Calculate the cumulative curve intensity.
        curve_inten += line_inten[i] * lorentzian(curve_energies, line_energies[i], linewidth)
    return (line_energies, line_inten, curve_energies, curve_inten)


@dataclass
class Spectrum:
    """
    Named collection of transitions with intensity calculation parameters and results.

    A Spectrum encapsulates the parameters needed to compute dipole strengths and
    oscillator strengths for a specific set of transitions (e.g., absorption from
    ground state, emission from excited state).

    Parameters
    ----------
    name : str
        Name of the spectrum (e.g., 'ground absorption', 'excited emission').
    lrange : list of list of int
        Level ranges [initial_levels, final_levels], where each sub-list contains
        0-based level indices. Example: [[0, 1], [6, 7, 8, 9]].
    intensity_tensors : list
        List of intensity tensor objects (e.g., from ImportSLJM).
    altp : list, optional
        Altp coupling parameters for electric dipole calculation.
        Each element is [name_str, value], e.g., ['A210', 1e-10].
    group_tol : float, optional
        Tolerance for grouping transitions by (ei, ef) level pair (default 1e-3).
        Smaller values (1e-5 or less) may be needed for hyperfine structure.
    nrefractive : float, optional
        Refractive index of the medium (default 1.0 for vacuum).
    md : bool, optional
        Include magnetic dipole transitions (default True).
    ed : bool, optional
        Include electric dipole transitions via Altp (default False).
    """

    name: str
    lrange: List[List[int]]
    intensity_tensors: List[Any]
    altp: Optional[List[Any]] = None
    group_tol: float = 1e-3
    nrefractive: float = 1.0
    md: bool = True
    ed: bool = False
    # Computed fields
    transformed_tensors: Dict[str, Any] = field(default_factory=dict, init=False)
    dipole_strengths: List[Dict[str, Any]] = field(default_factory=list, init=False)
    groups: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate inputs."""
        if not self.name:
            raise ValueError("Spectrum name must be non-empty")
        if len(self.lrange) != 2:
            raise ValueError("lrange must be [initial_levels, final_levels]")
        if not self.intensity_tensors:
            raise ValueError("intensity_tensors must be non-empty")
        if self.group_tol <= 0:
            raise ValueError(f"group_tol must be positive (got {self.group_tol})")
        if self.nrefractive <= 0:
            raise ValueError(f"nrefractive must be positive (got {self.nrefractive})")


def gen_intensity(
    hamiltonian: Any,
    spectrum: Spectrum,
    polarization: str = "isotropic",
) -> List[Dict[str, Any]]:
    """
    Generate intensity data for a spectrum (absorption or emission).

    Orchestrates the intensity calculation pipeline: vtrans (basis transformation) →
    dipole_str (compute dipole strengths) → group_transitions (group by level pair) →
    add_oscillator_strengths_and_A_coefficients (compute f and A).

    Parameters
    ----------
    hamiltonian : cfl.Hamiltonian
        Hamiltonian object with eigenvectors/eigenvalues from diag().
    spectrum : Spectrum
        Spectrum object with lrange, intensity_tensors, and optional parameters.
    polarization : str, optional
        Polarization type ('isotropic', 'axial', 'sigma', 'pi'). Default 'isotropic'.
        (Only 'isotropic' fully supported in MVP; others deferred to phase 2.)

    Returns
    -------
    list of dict
        List of transition group dictionaries, each with keys:
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
    if not hasattr(hamiltonian, "diag"):
        raise ValueError("hamiltonian must be a cfl.Hamiltonian object")

    # Extract eigenvectors and eigenvalues
    w, z = hamiltonian.diag()

    # Validate eigenvectors
    if not isinstance(z, np.ndarray) or z.ndim != 2:
        raise ValueError(
            f"Hamiltonian eigenvectors must be 2D (got shape {z.shape if hasattr(z, 'shape') else 'unknown'})"
        )

    # Transform intensity tensors to eigenbasis
    spectrum.transformed_tensors = vtrans(spectrum.intensity_tensors, z)

    # Compute dipole strengths for all transitions
    spectrum.dipole_strengths = dipole_str(
        spectrum.lrange,
        spectrum.transformed_tensors,
        hamiltonian,
        w,
        z,
        md=spectrum.md,
        ed=spectrum.ed,
        Altp=spectrum.altp,
    )

    # Group transitions by level pair
    spectrum.groups = group_transitions(spectrum.dipole_strengths, tol=spectrum.group_tol)

    # Add oscillator strengths and A coefficients
    add_oscillator_strengths_and_A_coefficients(spectrum.groups, refractive_index=spectrum.nrefractive)

    return spectrum.groups


def gen_inten_summary(
    spectrum: Spectrum,
    hamiltonian: Any,
    format: str = "text",
    state_labels: Optional[List[Any]] = None,
) -> str:
    """
    Generate a human-readable summary of intensity data for a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum object with computed groups (from gen_intensity()).
    hamiltonian : cfl.Hamiltonian
        Hamiltonian object (needed for state labels and energies).
    format : str, optional
        Output format: 'text' (default, pretty table) or 'csv' (comma-separated).
    state_labels : list of Any, optional
        Human-readable state labels (from hamiltonian.tensors[0].states.labels).
        If not provided, uses principal component indices from diag().

    Returns
    -------
    str
        Formatted string suitable for printing or writing to file.
    """
    if not spectrum.groups:
        return "No transitions computed."

    # Get state labels if not provided
    if state_labels is None:
        if hasattr(hamiltonian, "tensors") and hamiltonian.tensors:
            state_labels = hamiltonian.tensors[0].states.labels
        else:
            # Fallback: use level indices
            w, z = hamiltonian.diag()
            state_labels = [f"Level {i}" for i in range(len(w))]

    # Get eigenvalues and eigenvectors for label determination
    w, z = hamiltonian.diag()
    pc = np.argmax(np.abs(z), axis=0)

    if format == "text":
        return _format_inten_text(spectrum, w, pc, state_labels)
    elif format == "csv":
        return _format_inten_csv(spectrum, w, pc, state_labels)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'text' or 'csv'.")


def _format_inten_text(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as a human-readable text table."""
    lines = [f"Spectrum: {spectrum.name}", "=" * 80]

    for group_idx, group in enumerate(spectrum.groups):
        e_i = group["e_i"]
        e_f = group["e_f"]
        energy = group["Energy"]
        g_i = group["g_i"]
        g_f = group["g_f"]
        f = group["f"]
        A = group["A"]

        # Get principal component state labels
        initial_level = None
        final_level = None
        for i, pc_idx in enumerate(principal_components):
            if abs(eigenvalues[i] - e_i) < 1e-6:
                if initial_level is None:
                    initial_level = i
            if abs(eigenvalues[i] - e_f) < 1e-6:
                if final_level is None:
                    final_level = i

        # Format state labels (handle both list and string formats)
        initial_label = state_labels[initial_level] if initial_level is not None else f"State {initial_level}"
        final_label = state_labels[final_level] if final_level is not None else f"State {final_level}"
        
        if isinstance(initial_label, (list, tuple)):
            initial_label_str = " ".join(str(x) for x in initial_label)
        else:
            initial_label_str = str(initial_label)
            
        if isinstance(final_label, (list, tuple)):
            final_label_str = " ".join(str(x) for x in final_label)
        else:
            final_label_str = str(final_label)

        direction = "→" if energy > 0 else "←"

        lines.append("")
        lines.append(
            f"Transition {group_idx}: {initial_label_str} {direction} {final_label_str}"
        )
        lines.append(f"  Initial state: {initial_label_str:30s} E = {e_i:12.6f} cm⁻¹ (g={int(g_i)})")
        lines.append(f"  Final state:   {final_label_str:30s} E = {e_f:12.6f} cm⁻¹ (g={int(g_f)})")
        lines.append(f"  Transition energy: {energy:12.6f} cm⁻¹")

        if energy > 0:
            # Absorption
            lines.append(f"  Oscillator strength f:      {f:.6e}")
        else:
            # Emission
            lines.append(f"  Einstein A coefficient:     {A:.6e} s⁻¹")
            if A > 0:
                lifetime_s = 1.0 / A
                lifetime_ms = lifetime_s * 1e3
                lines.append(f"  Lifetime:                   {lifetime_s:.6e} s ({lifetime_ms:.6e} ms)")

    lines.append("")
    lines.append("=" * 80)
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

        # Find principal component level indices
        initial_level = None
        final_level = None
        for i, pc_idx in enumerate(principal_components):
            if abs(eigenvalues[i] - e_i) < 1e-6:
                if initial_level is None:
                    initial_level = i
            if abs(eigenvalues[i] - e_f) < 1e-6:
                if final_level is None:
                    final_level = i

        initial_label = state_labels[initial_level] if initial_level is not None else f"State {initial_level}"
        final_label = state_labels[final_level] if final_level is not None else f"State {final_level}"

        # Format labels (handle both list and string formats)
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
