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
            ed_mom = [clean_complex(m) for m in ed_mom]
            md_mom = [clean_complex(m) for m in md_mom]
            
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

    Attributes
    ----------
    altp : list, optional
        Altp coupling parameters for electric dipole calculation.
        Each element is [name_str, value], e.g., ['A210', 1e-10].
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
    
    # Mutable Altp parameters (can be changed via set_altp)
    altp: Optional[List[Any]] = field(default=None)
    
    # Computed fields
    transformed_tensors: Dict[str, Any] = field(default_factory=dict, init=False)
    dipole_strengths: List[Dict[str, Any]] = field(default_factory=list, init=False)
    groups: List[Dict[str, Any]] = field(default_factory=list, init=False)
    total_f: float = field(default=0.0, init=False)
    total_A: float = field(default=0.0, init=False)
    eigenvalues: Optional[np.ndarray] = field(default=None, init=False)
    principal_components: Optional[np.ndarray] = field(default=None, init=False)

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

    def set_altp(self, altp: Optional[List[Any]]) -> None:
        """Update Altp parameters. Call calculate_intensities() to recompute."""
        self.altp = altp

    def calculate_intensities(self, polarization: str = "isotropic") -> List[Dict[str, Any]]:
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
                f"Hamiltonian eigenvectors must be 2D (got shape {z.shape if hasattr(z, 'shape') else 'unknown'})"
            )

        # Transform intensity tensors to eigenbasis
        self.transformed_tensors = vtrans(self.intensity_tensors, z)

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
            Altp=self.altp,
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
                        "Each Spectrum must be purely absorption (Energy > 0) or purely emission (Energy < 0)."
                    )

        return self.groups


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
        Spectrum object with computed groups (from calculate_intensities()).
    hamiltonian : cfl.Hamiltonian
        Hamiltonian object (needed for state labels and energies if cached values not available).
    format : str, optional
        Output format: 'text' (default, pretty table) or 'csv' (comma-separated).
    state_labels : list of Any, optional
        Human-readable state labels (from hamiltonian.tensors[0].states.labels).
        If not provided, uses principal component indices from cached spectrum data.

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
            w = spectrum.eigenvalues if spectrum.eigenvalues is not None else hamiltonian.diag()[0]
            state_labels = [f"Level {i}" for i in range(len(w))]

    # Use cached eigenvalues and principal components (computed during calculate_intensities)
    w = spectrum.eigenvalues if spectrum.eigenvalues is not None else hamiltonian.diag()[0]
    pc = spectrum.principal_components if spectrum.principal_components is not None else np.argmax(np.abs(hamiltonian.diag()[1]), axis=0)

    if format == "text":
        return _format_inten_text(spectrum, w, pc, state_labels)
    elif format == "brief":
        return _format_inten_brief(spectrum, w, pc, state_labels)
    elif format == "verbose":
        return _format_inten_verbose(spectrum, w, pc, state_labels)
    elif format == "csv":
        return _format_inten_csv(spectrum, w, pc, state_labels)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'text', 'brief', 'verbose', or 'csv'.")



def _format_inten_text(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as a human-readable text table."""
    lines = [f"Spectrum: {spectrum.name}", "=" * 80]

    # Print Altp parameters if present
    if spectrum.altp:
        lines.append("Altp (electric dipole coupling) parameters:")
        for altp_item in spectrum.altp:
            if isinstance(altp_item, (list, tuple)) and len(altp_item) == 2:
                name, value = altp_item
                lines.append(f"  {name}: {value}")
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

        direction = "->" if energy > 0 else "<-"

        lines.append("")
        lines.append(
            f"Transition {group_idx}: {initial_label_str} {direction} {final_label_str}"
        )
        lines.append(f"  Initial state: {initial_label_str:30s} E = {e_i:12.6f} cm-1 (g={int(g_i)})")
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
                lines.append(f"  Lifetime:                   {lifetime_s:.6e} s ({lifetime_ms:.6e} ms)")

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
                lines.append(f"Lifetime (from total A):      {lifetime_s:.6e} s ({lifetime_ms:.6e} ms)")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def _format_state_label_with_energy(label: Any, level: int, energy: float) -> str:
    """Format a state label with 1-based level index and energy."""
    if isinstance(label, (list, tuple)):
        label_str = " ".join(str(x) for x in label)
    else:
        label_str = str(label)
    return f"{level}: | {label_str} > (E = {energy:12.6f} cm-1)"


def _format_state_label_short(label: Any) -> str:
    """Format a state label in short form (just the quantum numbers, no brackets)."""
    if isinstance(label, (list, tuple)):
        label_str = " ".join(str(x) for x in label)
    else:
        label_str = str(label)
    return label_str


def _format_inten_brief(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as a brief tabular summary (one line per group)."""
    lines = [f"Spectrum: {spectrum.name}"]
    lines.append("=" * 132)

    # Print Altp parameters if present
    if spectrum.altp:
        lines.append("Altp (electric dipole coupling) parameters:")
        for altp_item in spectrum.altp:
            if isinstance(altp_item, (list, tuple)) and len(altp_item) == 2:
                name, value = altp_item
                lines.append(f"  {name}: {value}")
        lines.append("")

    # Determine if absorption or emission
    is_absorption = spectrum.groups[0]["Energy"] > 0 if spectrum.groups else True

    # Header
    if is_absorption:
        header = f"{'Group':<6} {'Initial State':<50} {'Final State':<50} {'f_MD':<14} {'f_ED':<14} {'f_Total':<14}"
    else:
        header = f"{'Group':<6} {'Initial State':<50} {'Final State':<50} {'A_MD':<14} {'A_ED':<14} {'A_Total':<14}"

    lines.append(header)
    lines.append("-" * 132)

    # Print each group as one line
    for group_idx, group in enumerate(spectrum.groups, start=1):
        t_list = group["t_list"]
        e_i = group["e_i"]
        e_f = group["e_f"]
        
        # Get state labels from first transition in group
        if t_list:
            initial_level = t_list[0]["pc_i"] + 1  # Convert to 1-based
            final_level = t_list[0]["pc_f"] + 1    # Convert to 1-based
        else:
            initial_level = None
            final_level = None

        # Format state labels with energies
        initial_label = _format_state_label_with_energy(state_labels[initial_level - 1], initial_level, e_i) if initial_level is not None else "Unknown"
        final_label = _format_state_label_with_energy(state_labels[final_level - 1], final_level, e_f) if final_level is not None else "Unknown"
        
        energy = group["Energy"]
        g_i = group.get("g_i", 1)
        
        # Sum dipole strengths over all transitions in group
        total_S_ED = sum(t.get("S_ED_isotropic", 0.0) for t in t_list)
        total_S_MD = sum(t.get("S_MD_isotropic", 0.0) for t in t_list)
        
        # Calculate f_total and A_total using the group's values
        f_total = group.get("f", 0.0)
        A_total = group.get("A", 0.0)
        
        # Decompose into ED and MD components using A_and_f_calc
        # Call with ED only (S_ED, 0), MD only (0, S_MD), get separate values
        A_ED, f_ED = A_and_f_calc(total_S_ED, 0.0, energy, g_i, nrefractive=spectrum.nrefractive)
        A_MD, f_MD = A_and_f_calc(0.0, total_S_MD, energy, g_i, nrefractive=spectrum.nrefractive)
        
        # Format line based on absorption or emission
        if is_absorption:
            line = f"{group_idx:<6} {initial_label:<50} {final_label:<50} {f_MD:>13.6e}  {f_ED:>13.6e}  {f_total:>13.6e}"
        else:
            line = f"{group_idx:<6} {initial_label:<50} {final_label:<50} {A_MD:>13.6e}  {A_ED:>13.6e}  {A_total:>13.6e}"

        lines.append(line)

    lines.append("-" * 132)
    
    # Add totals
    if spectrum.groups:
        if is_absorption:
            lines.append(f"{'Total':<6} {'':<50} {'':<50} {'':<14} {'':<14} {spectrum.total_f:>13.6e}")
        else:
            lines.append(f"{'Total':<6} {'':<50} {'':<50} {'':<14} {'':<14} {spectrum.total_A:>13.6e}")
    
    lines.append("=" * 132)
    return "\n".join(lines)


def _format_inten_verbose(
    spectrum: Spectrum,
    eigenvalues: np.ndarray,
    principal_components: np.ndarray,
    state_labels: List[Any],
) -> str:
    """Format spectrum as verbose output (BRIEF + individual transitions for each group)."""
    lines = [f"Spectrum: {spectrum.name}"]
    lines.append("=" * 132)

    # Print Altp parameters if present
    if spectrum.altp:
        lines.append("Altp (electric dipole coupling) parameters:")
        for altp_item in spectrum.altp:
            if isinstance(altp_item, (list, tuple)) and len(altp_item) == 2:
                name, value = altp_item
                lines.append(f"  {name}: {value}")
        lines.append("")

    # Determine if absorption or emission
    is_absorption = spectrum.groups[0]["Energy"] > 0 if spectrum.groups else True

    # Header
    if is_absorption:
        header = f"{'Group':<6} {'Initial State':<50} {'Final State':<50} {'f_MD':<14} {'f_ED':<14} {'f_Total':<14}"
    else:
        header = f"{'Group':<6} {'Initial State':<50} {'Final State':<50} {'A_MD':<14} {'A_ED':<14} {'A_Total':<14}"

    lines.append(header)
    lines.append("-" * 132)

    # Print each group with its individual transitions
    for group_idx, group in enumerate(spectrum.groups, start=1):
        t_list = group["t_list"]
        e_i = group["e_i"]
        e_f = group["e_f"]
        
        # Get state labels from first transition in group (use transition i/f indices, not pc indices)
        if t_list:
            initial_idx = t_list[0]["i"] + 1  # Convert to 1-based (transition index)
            final_idx = t_list[0]["f"] + 1    # Convert to 1-based (transition index)
        else:
            initial_idx = None
            final_idx = None

        # Format state labels with energies using first transition's indices
        initial_label = _format_state_label_with_energy(state_labels[t_list[0]["i"]], initial_idx, e_i) if t_list else "Unknown"
        final_label = _format_state_label_with_energy(state_labels[t_list[0]["f"]], final_idx, e_f) if t_list else "Unknown"
        
        energy = group["Energy"]
        g_i = group.get("g_i", 1)
        
        # Sum dipole strengths over all transitions in group
        total_S_ED = sum(t.get("S_ED_isotropic", 0.0) for t in t_list)
        total_S_MD = sum(t.get("S_MD_isotropic", 0.0) for t in t_list)
        
        # Calculate f_total and A_total using the group's values
        f_total = group.get("f", 0.0)
        A_total = group.get("A", 0.0)
        
        # Decompose into ED and MD components using A_and_f_calc
        A_ED, f_ED = A_and_f_calc(total_S_ED, 0.0, energy, g_i, nrefractive=spectrum.nrefractive)
        A_MD, f_MD = A_and_f_calc(0.0, total_S_MD, energy, g_i, nrefractive=spectrum.nrefractive)
        
        # Format group line based on absorption or emission
        if is_absorption:
            line = f"{group_idx:<6} {initial_label:<50} {final_label:<50} {f_MD:>13.6e}  {f_ED:>13.6e}  {f_total:>13.6e}"
        else:
            line = f"{group_idx:<6} {initial_label:<50} {final_label:<50} {A_MD:>13.6e}  {A_ED:>13.6e}  {A_total:>13.6e}"

        lines.append(line)
        
        # Print individual transitions for this group
        lines.append("        Individual transitions:")
        if is_absorption:
            trans_header = "        i     Initial State                 f      Final State                  S_MD_iso      f_MD           S_ED_iso      f_ED           f_Total"
        else:
            trans_header = "        i     Initial State                 f      Final State                  S_MD_iso      A_MD           S_ED_iso      A_ED           A_Total"
        lines.append(trans_header)
        
        # List each transition
        for trans in t_list:
            i_1b = trans["i"] + 1  # Convert to 1-based
            f_1b = trans["f"] + 1  # Convert to 1-based
            e_trans = trans["e"]
            s_ed = trans.get("S_ED_isotropic", 0.0)
            s_md = trans.get("S_MD_isotropic", 0.0)
            
            # Get state labels
            i_label = _format_state_label_short(state_labels[trans["i"]])
            f_label = _format_state_label_short(state_labels[trans["f"]])
            
            # Calculate f_ED, f_MD for this individual transition
            A_ED_t, f_ED_t = A_and_f_calc(s_ed, 0.0, e_trans, g_i, nrefractive=spectrum.nrefractive)
            A_MD_t, f_MD_t = A_and_f_calc(0.0, s_md, e_trans, g_i, nrefractive=spectrum.nrefractive)
            
            # Get transition-level total (not group total)
            A_t, f_t = A_and_f_calc(s_ed, s_md, e_trans, g_i, nrefractive=spectrum.nrefractive)
            
            if is_absorption:
                trans_line = f"        {i_1b:<4} | {i_label} >                    \t{f_1b:<4} | {f_label} >                  {s_md:>10.6e}  {f_MD_t:>13.6e}  {s_ed:>10.6e}  {f_ED_t:>13.6e}  {f_t:>13.6e}"
            else:
                trans_line = f"        {i_1b:<4} | {i_label} >                    \t{f_1b:<4} | {f_label} >                  {s_md:>10.6e}  {A_MD_t:>13.6e}  {s_ed:>10.6e}  {A_ED_t:>13.6e}  {A_t:>13.6e}"
            
            lines.append(trans_line)
        
        lines.append("")  # Blank line between groups

    lines.append("-" * 132)
    
    # Add totals
    if spectrum.groups:
        if is_absorption:
            lines.append(f"{'Total':<6} {'':<50} {'':<50} {'':<14} {'':<14} {spectrum.total_f:>13.6e}")
        else:
            lines.append(f"{'Total':<6} {'':<50} {'':<50} {'':<14} {'':<14} {spectrum.total_A:>13.6e}")
    
    lines.append("=" * 132)
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
