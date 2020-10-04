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


