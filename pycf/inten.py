#!/usr/bin/env python
# Filename = inten.py
"""
A rewrite of the intensity calculation to follow the old Pascal code more closely,
"""
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pycf.constants import (BOLTZMANN_CM_INVERSE, ELECTRON_MASS,
                            ELEMENTARY_CHARGE, EPSILON_0, HBAR, SPEED_OF_LIGHT)
from pycf.njsymbols import wigner_3j
def vtrans(tensors: List[Any], z: np.ndarray) -> Dict[str, Any]:
    """
    Transform tensor matrix elements into eigenbasis previously calculated by
    diagonalizing a Hamiltonian.
    This does the transformation part of the Pascal vtrans program, but not the
    construction of the electric-dipole operators, which is now done in the
    dipole_str function.
    Mike Reid 3 April 2026:
        Delete lower-diagonal elements that are mistakenly added by t.get_matel()
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
        # print(t.name)
        # print('before deleting lower diagonal')
        # print(M.real)
        M = M - np.tril(M, k=-1)  # subtract the lower triangle
        q = int(t.name[2])
        if q == 0:
            # in this case we need a Hermitian matrix
            # so we add the conjugate, omitting diagonal
            M = M + np.tril(M.conj().T, k=-1)
        # print('after deleting lower diagonal, but hermetizing if q==0')
        # print(M.real)
        matel = z.conj().T @ M @ z  # eigenvector transformation of M
        # discard small real or imaginary parts of the transformed matrix
        # matel.imag[np.abs(matel.imag) < tolerance] = 0
        # matel.real[np.abs(matel.real) < tolerance] = 0
        # print('transformed matrix')
        # print(matel.real)
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
    Altp: Optional[List[float]] = None,
) -> Dict[str, Any]:
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
            "Eigenvector V must be 2-dimensional (nstates x nstates), got shape %s"
            % (shape,)
        )
    # find principal components
    pc = np.argmax(np.abs(z), axis=0)
    # print('\n##############')
    # print('w', w)
    # print('z', z)
    # print('states')
    # for i,s in enumerate(labels):
    #      print(i, s)
    # print('principal components')
    # for i, p in enumerate(pc):
    #    print(i, '\t', w[i], '\t', p, '\t', labels[p], '\t', np.abs(z[p,i]), '\t', z[p,i])
    # print('##############\n')
    if ed:
        D_factor = {}
        if Altp is None:
            raise ValueError("ed is True but no Altp parameters were provided")
        for A in Altp:
            l = int(A[0][1])
            t = int(A[0][2])
            pp = int(A[0][3])
            # Evaluate the Clebsch-Gordon coefficient of Eq. (9), Reid and Richardson J.
            # Chem. Phys. 79, 5735 (1983). Note: sign factor includes additional (-1)^q
            for q in [-1, 0, 1]:
                for p in np.unique([-pp, pp]):
                    CG_coeff = np.sqrt(2 * t + 1) * wigner_3j(l, 1, t, p + q, -q, -p)
                    if (l - 1 + p + q) % 2 != 0:
                        CG_coeff *= -1
                    # print('% i % i % i % i % i  %f' % (l, t, p+ q, p, q, factor))
                    D_factor["%i%i%i%i" % (l, t, p, q)] = CG_coeff
        # print('D_factor')
        # for Df in D_factor:
        #    print(Df, D_factor[Df])
    trs = []
    for i in lrange[0]:
        for f in lrange[1]:
            # print('\nTransition: i', i, w[i], labels[pc[i]], 'f', f, w[f], labels[pc[f]])
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
                # print('md_mom', md_mom)
            if ed:
                for A in Altp:
                    # print('\n###', A)
                    l = int(A[0][1])
                    t = int(A[0][2])
                    pp = int(A[0][3])
                    for q in [-1, 0, 1]:
                        # print('>>> q',q)
                        for p in np.unique([-pp, pp]):
                            # print('## p',p)
                            A_val = A[1]
                            if p < 0:  # symmetry of Altp parameter
                                A_val = A_val.conjugate()
                                if (1 + t + p) % 2 != 0:  # if 1+t+p is odd
                                    A_val = -A_val
                            # print(A_val)
                            # print('Altp', l, t, p, q, A_val)
                            if -l <= (p + q) <= l:
                                k = "U%i%i" % (l, p + q)
                                if k not in tensor_dict:
                                    msg = (
                                        "Missing electric dipole tensor '{}' "
                                        "required by Altp.".format(k)
                                    )
                                    raise ValueError(msg)
                                # print('Altp', l, t, p, A_val, 'q', q, 'k', k)
                                # print('D_factor', '%i%i%i%i' % (l, t, p, q),
                                #       D_factor['%i%i%i%i' % (l, t, p, q)])
                                # print('tensor_dict', k, i, f,
                                #       tensor_dict[k][i, f])
                                D = (
                                    -e
                                    * A_val
                                    * D_factor["%i%i%i%i" % (l, t, p, q)]
                                    * tensor_dict[k][i, f]
                                )
                                # print('D', D)
                                ed_mom[q + 1] += D  # order is -1, 0, 1
                                # print('ed_mom', ed_mom)
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
            # print('trs')
            # print(trs)
    trs.sort(key=itemgetter("e"))
    return trs
def group_transitions(
    items: List[Dict[str, Any]], tol: float = 1e-4
) -> List[Dict[str, Any]]:
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
        lambda_ = 0
        omega = 0
    else:
        lambda_ = 1e-2 / energy  # {cm-1}; {m}
        if lambda_ != 0:
            omega = 2 * rpi * clight / lambda_  # {hz}
        else:
            omega = 0
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
def lorentzian(x: float | np.ndarray, x0: float, fwhm: float) -> float | np.ndarray:
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
    xlim: Optional[Tuple[float, float]] = None,
    npoints: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
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
    dict
        Dictionary containing:
        - 'curve_energies': numpy array of energy values
        - 'curve_inten': numpy array of intensities
        - 'line_energies': numpy array of line positions
        - 'line_inten': numpy array of line intensities
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
        curve_inten += line_inten[i] * lorentzian(
            curve_energies, line_energies[i], linewidth
        )
    return (line_energies, line_inten, curve_energies, curve_inten)
