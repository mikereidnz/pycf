#!/usr/bin/env python
# Filename = inten.py


from __future__ import division
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt


def Xi_val(t, l, re):
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
    re : string
        Rare-earth ion.

    Returns
    -------
    xi : float
        value
    """
    
    
    xi_tl = {
    '12' : [-1.78, -1.58, -1.08, -0.83, -0.57, -0.4, -0.23],
    '32' : [1.54, 1.35, 0.88, 0.64, 0.36, 0.30, 0.24],
    '34' : [1.75, 1.50, 0.90, 0.64, 0.37, 0.29, 0.21],
    '54' : [-2.26, -1.98, -1.27, -0.89, -0.44, -0.30, -0.16],
    '56' : [-5.45, -4.62, -2.70, -1.84, -0.92, -0.66, -0.4],
    '76' : [4.54, 3.96, 2.58, 1.78, 0.71, 0.43, 0.15],
    }
    
    re_list =  ['Pr', 'Nd', 'Eu', 'Tb', 'Er', 'Tm', 'Yb']
    try:
        i = re_list.index(re)
    except ValueError:
        raise ValueError("Invalid parameter: re=%s" % re)
    
    try:
        v = xi_tl['%i%i' % (t, l)][i]
    except ValueError:
        raise ValueError("Invalid parameters: t=%i, l=%i" % (t, l))

    return v*1e10


class LData(object):
    """
    Class for holding data about ligands that comprise the crystal lattice.

    Parameters
    ----------
    q_Ln : int
        The charge of the lanthanide ion, in units of proton charge. 
    q_L : int
        The charge of the ligand ion, in units of proton charge. 
    alpha_L_bar : float
        Mean polarizability of ligand species in Angstrom^3. 
    L_a : np.array
        Array of ligand coordinates, n by 3, where n is the number of
        ligands. Each row consists of coordinates [R, theta, phi], that is,
        the ligand radius (Angstrom), polar angle (radians), and azimuthal
        angle (radians), respectively.
    """
    def __init__(self, q_Ln, q_L, alpha_L_bar, L_a):

        self.q_L = q_L
        self.q_Ln = q_Ln
        self.alpha_L_bar = alpha_L_bar 
        self.L_a = L_a
        self.nL = len(L_a)

    def __iter__(self):
        """
        L is a vector of ligand coordinates [R, theta, phi].
        """
        for L in self.L_a:
            yield L



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
    C = np.sqrt((2*k+1)/(4*np.pi)) * sph_harm(q, k, phi, theta)
    
    return C 


def A_SC(l, t, p, lat, Xi):
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
    lat : Lattice
        Ligand data for next nearest neighbors.
    Xi : float
        Xi(t, l) parameter, in Angstrom^(t+1) erg^-1. 


    Returns
    -------
    A : float
        The transition intensity parameter A^{\lambda}_{tp} in cm^(-1).

    """
    
    # To avoid overflowing our 64bit double, we'll rescale Xi to somewhere near
    # unity (of magnitude ~10^10 in units of Angstrom^(t+1) erg^-1) and use an
    # exponent of -10 rather than -20 for e^2. All instances of proton charge
    # e^2 are later multiplied by Xi.
    Xi = Xi*10**(-10)
    e2 = (4.80320425**2)*10**(-10)   # proton charge squared in esu
    prefac_chg = -(-1)**p * e2 * lat.q_L
    prefac_pol = 2*(-1)**p * e2 * lat.q_Ln * (t+1)
    A_chg = 0
    A_pol = 0
    for L in lat:
        C = Ckq(t, -p, L[1], L[2])
        A_chg += C * L[0]**(-(t+1))
        A_pol += C * L[0]**(-(t+4))
    
    A = -(prefac_chg*A_chg + prefac_pol*A_pol)*Xi*(2*l+1)/(np.sqrt(2*t+1))

    return A

