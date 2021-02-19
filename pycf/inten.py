#!/usr/bin/env python
# Filename = inten.py


from __future__ import division
import numpy as np
from scipy.special import sph_harm
from pycf.njsymbols import wigner_3j
from pycf.cfl_util import *

def Xi_val(t, l, Ln):
    """
    Xi(t, l) parameters in Angstrom^(t+1) erg^-1 from Krupke Phys Rev 145, 1
    (1966).
   
    Available ions are Pr, Nd, Eu, Tb, Er, Tm, Yb. 

    Yb values are linearly interpolated from values for Er and Tm.
 
    Parameters
    ----------
    t : int
        Degree of the parameter.
    l: int
        Transition intensity lambda parameter, with values of (2, 4, 6).
    Ln : string
        The chemical symbol of the Lanthanide dopant.

    Returns
    -------
    xi : float
        value
    """
    
    xi_tl = {
    '12' : [-1.78, -1.58, -1.08, -0.83, -0.57, -0.40, -0.23],
    '32' : [ 1.54,  1.35,  0.88,  0.64,  0.36,  0.30,  0.24],
    '34' : [ 1.75,  1.50,  0.90,  0.64,  0.37,  0.29,  0.21],
    '54' : [-2.26, -1.98, -1.27, -0.89, -0.44, -0.30, -0.16],
    '56' : [-5.45, -4.62, -2.70, -1.84, -0.92, -0.66, -0.40],
    '76' : [ 4.54,  3.96,  2.58,  1.78,  0.71,  0.43,  0.15],
    }
   
    Ln_list =  ['Pr', 'Nd', 'Eu', 'Tb', 'Er', 'Tm', 'Yb']
    try:
        i = Ln_list.index(Ln)
    except ValueError:
        raise ValueError("Invalid parameter: Ln=%s" % Ln)
    
    try:
        v = xi_tl['%i%i' % (t, l)][i]
    except ValueError:
        raise ValueError("Invalid parameters: t=%i, l=%i" % (t, l))
    
    v *= 1e10 # Scale units to Angstrom^(t+1) erg^-1 

    return v


def RInt4f(l, Ln):
    """
    Radial integrals of the form <4f|r^\lambda|4f> for the RE3+ ions, from
    Freeman and Watson, 10.1103/PhysRev.127.2058. 

    Parameters
    ----------
    l : int
        The power lambda, with available values of (2, 4, 6). 
    Ln : string
        The chemical symbol of the Lanthanide dopant.

    Returns
    -------
    rint : float
        The radial integral, in units of Angstrom^2, Angstrom^4, and
        Angstrom^6 depending on lambda. 

    """
    # Bohr radius in Angstrom
    # (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
    a0 = 0.529177210903
    
    # Units in Freeman and Watson are specified as a0^{-lambda}, but I can't
    # make sense of inverse length for the radial integrals?  Treating as
    # a0^{lambda} gives consistent units throughout and reasonable values; going
    # with that for now, but this is disconcerting. 
    rint = [[1.200, 3.455, 21.226],
            [1.086, 2.822, 15.726], 
            [1.001, 2.401, 12.396],
            [0.883, 1.897, 8.775 ],
            [0.938, 2.273, 11.670],
            [0.726, 1.322, 5.102 ],
            [0.666, 1.126, 3.978 ],
            [0.613, 0.960, 3.104 ]]
    
    Ln_list =  ['Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Dy', 'Er', 'Yb']
    l_list = [2, 4, 6]
    
    try:
        i = Ln_list.index(Ln)
    except ValueError:
        raise ValueError("Invalid parameter: Ln=%s" % Ln)
    try:
        li = l_list.index(l)
    except ValueError:
        raise ValueError("Invalid parameter: l=%s" % l)
    
    val = rint[i][li]*a0**l

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
    def __init__(self, coords, q, alpha_bar):
        self.coords = coords
        self.q = q
        self.alpha_bar = alpha_bar


def Ckq(k, q, theta, phi):
    """
    Solid spherical harmonic functions in normalization conventionally used for CF calcs.

    Parameters
    ----------
    k : int
        Degree of the harmonic.
    q : int
        Order of the harmonic.
    theta : float
        Polar angle in radians.
    phi : float
        Azimuthal angle in radians.

    Returns
    -------
    Ckq : float
        Value of spherical harmonic.
    """
    C = np.sqrt((4*np.pi)/(2*k+1)) * sph_harm(q, k, phi, theta)
    
    return C 


def A_SC(l, t, p, Ln, q_Ln, ligands):
    """
    Calculate the A^lambda_tp parameters for static coupling using a
    point-charge model, following Reid and Richardson, J. Chem. Phys. 79(12)
    1983, pg 5739. 

    Parameters
    ----------
    l: int
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
    Xi = Xi_val(t, l, Ln)
    # To avoid overflowing our 64bit double, we'll rescale Xi by a factor
    # 10^(-10) and e2 by 10^(10).  These variables are always multiplied later,
    # so this avoids tiny numbers. 
    Xi = Xi*10**(-10)
    e2 = (4.80320425**2)*10**(-10)   # proton charge squared in esu
    prefac = -(-1)**p * e2 * Xi *(2*l+1)/(np.sqrt(2*t+1))
    A_chg = 0
    A_pol = 0
    for L in ligands:
        c = L.coords
        C = Ckq(t, -p, c[1], c[2])
        A_chg += C * c[0]**(-(t+1)) * L.q
        A_pol += C * c[0]**(-(t+4)) * L.alpha_bar
    
    A_chg = prefac * (-1) * A_chg
    A_pol = prefac * 2 * q_Ln * (t+1) * A_pol
    
    A_chg = np.real(A_chg)
    A_pol = np.real(A_pol)

    return (A_chg, A_pol)


def A_DC(l, t, p, Ln, ligands):
    """
    Calculate the A^lambda_tp parameters for dynamic coupling assuming isotropic
    ligands, following Reid and Richardson, J. Chem. Phys. 79(12) 1983, pg 5739. 

    Parameters
    ----------
    l: int
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
    if t == l+1:
        rint = RInt4f(l, Ln)
        prefac = 7*wigner_3j(3,l,3,0,0,0)*np.sqrt((l+1)*(2*l+1)) * rint * (-1)**p
        
        for L in ligands:
            c = L.coords
            A += Ckq(l+1, -p, c[1], c[2]) * c[0]**(-(l+2)) * L.alpha_bar

        # Convert to cm from A
        A *= prefac*10**(-8)
        
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
    def __init__(self, Ln, q_Ln, ligands):
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

        for l in l_list:
            for t in [l-1, l+1]:
                for p in range(0, t+1):
                    # Static portion
                    (A_statchg, A_statpol) = A_SC(l, t, p, self.Ln, self.q_Ln, self.ligands)

                    # Dynamic portion
                    A_dyniso = A_DC(l, t, p, self.Ln, self.ligands)

                    A_total = A_statchg+A_statpol+A_dyniso
                    if (np.abs(A_total)) > 1e-15:
                        A_list += [['A%i%i%i' % (l, t, p), [A_statchg, 
                            A_statpol, A_dyniso, A_statchg+A_dyniso, A_total]]]
                        
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


def vtrans(tensors, z):
    """
    Transform tensor matrix elements into eigenbasis previously calculated by
    diagonalizing a Hamiltonian. 

    Parameters
    ----------
    tensors : list
        Elements are tensors of type cfl.Tensor.
    z : np.ndarray 
        The eigenvectors, columnwise, used for the transformation.  This is
        generally the second output variable from h.diag() where H is a
        cfl.Hamiltonian. 
    """
    tensor_dict = {}
    vtrans_ten = ['U20', 'U21', 'U22', 'U23', 'U40', 'U41', 'U42', 'U43', 'U44',
            'U60', 'U61', 'U62', 'U63', 'U64', 'U65', 'U66', 'M10', 'M11']
    
    for t in tensors:
        if t.name not in vtrans_ten:
            raise ValueError("Unsupported tensor passed to vtrans: %s" % t.name)
        
        matel = z.conj().T@t.get_matel()@z

        q = int(t.name[2])
        if q != 0:
            tensor_dict[t.name] = 1/np.sqrt(2)*matel
            tensor_dict['%s-%i' % (t.name[:2], q)] = -1/np.sqrt(2)*matel.conj().T
#            tensor_dict[t.name] = matel
#            tensor_dict['%s-%i' % (t.name[:2], q)] = -matel.conj().T
        else:
            tensor_dict[t.name] = matel

    return tensor_dict


def dipole_str(lrange, tensor_dict, w, md=True, ed=False, Altp=None):
    """
    Parameters
    ----------
    lrange : list
        The level range, with entries: [first initial level, last initial level,
        first final level, last final level].  The level index convention calls
        the ground state level 1.
    tensor_dict : dict
        Expects specific keys (M10, M11, M1-1) pointing to matrix elements of
        the dipole operator in the eigenbasis of the Hamiltonian. 
    """
    e =  1e10# 4.803246e-10       # esu 
    clight = 2.997925e10    # cm/sec 
    hbar = 1.0545903e-27    # erg-sec 
    me = 9.109553e-28        # gm 
    md_prefac = -(e*hbar) / (2 * me * clight)
    dipole_cutoff = 1e-15 # Throw out dipole moments less than this
   

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
                    CG_coeff = np.sqrt(2*t+1)*wigner_3j(l,1,t,p+q,-q,-p)
                    if (l-1+p+q) % 2 != 0:
                        CG_coeff *= -1
                    #print('% i % i % i % i % i  %f' % (l, t, p+ q, p, q, factor))
                    D_factor['%i%i%i%i' % (l, t, p, q)] = CG_coeff
    trs = []
    for i in lrange[0]:
        for f in lrange[1]:
            md_str = [0, 0, 0]
            ed_str = [0, 0, 0]
            if md:
                keys = ['M1-1', 'M10', 'M11']
                if not any(k in tensor_dict for k in keys):
                    raise ValueError("Missing all or some of the magnetic dipole "\
                            "operator matrix elements. Required tensors are 'M1-1', "\
                            "'M10', 'M11'")
                md_str = [np.real(md_prefac*tensor_dict[k][i, f])**2 for k in keys]
            if ed:
                for A in Altp:
                    l = int(A[0][1])
                    t = int(A[0][2])
                    pp = int(A[0][3])
                    for q in [-1, 0, 1]:
                        for p in np.unique([-pp, pp]):
                            if -l <= (p+q) <= l:
                                k = 'U%i%i' % (l, p+q)
                                D = -e * A[1] * D_factor['%i%i%i%i' % (l, t, p, q)] * tensor_dict[k][i, f]
                                ed_str[q] += np.real(D)**2

            if any(d > dipole_cutoff for d in md_str) or any(d > dipole_cutoff for d in ed_str):
                isotropic = sum(md_str)/3 + sum(ed_str)/3
                pi = (md_str[0]+md_str[2])/2 + ed_str[1]
                sigma = md_str[1] + ed_str[0]+ed_str[2]

                trs += [{'md_-1': md_str[0], 'md_0': md_str[1], 'md_+1': md_str[2], 
                    'ed_-1': ed_str[0], 'ed_0': ed_str[1], 'ed_+1': ed_str[2],
                    'isotropic': isotropic, 'pi': pi, 'sigma': sigma, 
                    'ei': w[i], 'ef': w[f], 'e': w[f]-w[i],'i': i, 'f': f}]

    return trs


def boltzmann_factor(e, t):
    """
    The Boltzmann factor for a given energy e and temperature t in units
    of cm^-1 K^-1.
    """
    if t < 0:
        ans = 0
    elif t == 0:
        ans = 1
    else:
        ans = np.exp(-e / (t * 0.6952))

    return(ans)
      

def lorentzian(x, x0, fwhm):
    """
    Lorentzian function.
    """
    gamma_sq = (fwhm / 2)**2

    return(gamma_sq / ((x - x0)**2 + gamma_sq))


def inten(trs, polarization, linewidth, T, xlim=None, npoints=1000):
    """
    
    Parameters
    ----------
    lrange : list
        The level range, with entries: [first initial level, last initial level,
        first final level, last final level].  The level index convention calls
        the ground state level 1.
    """
    
    # Determine the smallest initial energy level, which we assume to be the
    # ground state (used for scaling other energies for boltzmann factor... this
    # behavior is not really obvious.
    min_energy = min(trs, key = lambda tr: tr['ei'])['ei']

    if xlim is None:
        xmin = min(trs, key = lambda tr: tr['e'])['e']
        xmax = max(trs, key = lambda tr: tr['e'])['e']
        # add some padding to plotting range
        xmin -= 50*linewidth
        xmax += 50*linewidth
        xlim = [xmin, xmax]
    
    ntrs = len(trs)
    initial_energies = np.zeros(ntrs)
    curve_energies = np.linspace(xlim[0], xlim[1], npoints)
    curve_inten = np.zeros(npoints)
    line_energies = np.zeros(ntrs)
    line_inten = np.zeros(ntrs)
    
    
    for i,tr in enumerate(trs):
        line_energies[i] = tr['e']

        # Calculate the individual line intensities.
        line_inten[i] = boltzmann_factor(tr['ei'] - min_energy, T) * tr[polarization]
        # Calculate the cumulative curve intensity.  
        curve_inten += line_inten[i] * lorentzian(curve_energies, line_energies[i], linewidth)
    

    return (line_energies, line_inten, curve_energies, curve_inten)

