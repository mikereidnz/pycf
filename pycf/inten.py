#!/usr/bin/env python
# Filename = inten.py

"""
A rewrite of the intensity calculation to follow the old Pascal code more closely, 
"""

from __future__ import division
import numpy as np
from pycf.njsymbols import wigner_3j
from pycf.cfl_util import *
from operator import itemgetter

def vtrans(tensors, z):
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
    vtrans_ten = ['U20', 'U21', 'U22', 'U23', 'U40', 'U41', 'U42', 'U43', 'U44',
            'U60', 'U61', 'U62', 'U63', 'U64', 'U65', 'U66', 'M10', 'M11']
    if len(tensors) == 0:
        raise ValueError("vtrans requires at least one tensor.")
    labels = tensors[0].states.labels
    tolerance = 1e-10 # for deleting small values 

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
        M = M - np.tril(M, k=-1) # subtract the lower triangle 
        
        q = int(t.name[2])
        if q == 0: 
            # in this case we need a Hermitian matrix
            # so we add the conjugate, omitting diagonal
            M = M + np.tril(M.conj().T, k=-1)
        
        # print('after deleting lower diagonal, but hermetizing if q==0')
        # print(M.real)
        
        matel = z.conj().T @ M @ z # eigenvector transformation of M 
        # discard small real or imaginary parts of the transformed matrix
        #matel.imag[np.abs(matel.imag) < tolerance] = 0
        #matel.real[np.abs(matel.real) < tolerance] = 0
        
        # print('transformed matrix')
        # print(matel.real)

        if q == 0:
            tensor_dict[t.name] = matel
        else: # q != 0, specifically q > 0   
            # +q case 
            tensor_dict[t.name] = matel
            # -q case
            # Use V^dagger M^\dagger V  = (V^dagger M  V)^\dagger
            matel = matel.conj().T 
            # Then add (-1)^q phase. 
            # This will work for any q!=0. 
            if q % 2 == 1: 
                matel *= -1
            tensor_dict['%s-%i' % (t.name[:2], q)] = matel

    return tensor_dict


def dipole_str(lrange, tensor_dict, h, E, V, md=True, ed=False, Altp=None):
    """
    Parameters
    ----------
    lrange : list
        A list of two lists: [initial_levels, final_levels], where each
        sub-list contains 0-based level indices.
        Example: [[0, 1], [6, 7, 8, 9]] selects levels 0–1 as initial
        and levels 6–9 as final.
    tensor_dict : dict
        Expects specific keys (M10, M11, M1-1) pointing to matrix elements of
        the dipole operator in the eigenbasis of the Hamiltonian. 
    Returns:
    --------
    trs : list
        A list of dictionaries for each transition component that contains
        energies, dipole moments, and dipole strengths. 
    """
    e =  1e10                # means that diople moments are in e-10 cm
    # e = 4.803246e-10       # esu 
    clight = 2.997925e10    # cm/sec 
    hbar = 1.0545903e-27    # erg-sec 
    me = 9.109553e-28        # gm 
    md_prefac = -(e*hbar) / (2 * me * clight)
    dipole_cutoff = 1e-10 # Throw out dipole moments of magnitude less than this

    w = E
    z = V
    # Validate eigenvector dimensions
    if not isinstance(z, np.ndarray) or z.ndim != 2:
        raise ValueError("Eigenvector V must be 2-dimensional (nstates x nstates), got shape %s" % (z.shape,))
    labels = h.tensors[0].states.labels
    # find principle components
    pc = np.argmax(np.abs(z), axis=0)
    
    #print('\n##############')
    #print('w', w)
    #print('z', z)
    #print('states')
    #for i,s in enumerate(labels):
    #      print(i, s)
    #print('principal components')
    #for i, p in enumerate(pc):
    #    print(i, '\t', w[i], '\t', p, '\t', labels[p], '\t', np.abs(z[p,i]), '\t', z[p,i])
    #print('##############\n')

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
        #print('D_factor')
        #for Df in D_factor:
        #    print(Df, D_factor[Df])
    trs = []
    for i in lrange[0]:
        for f in lrange[1]:
            #print('\nTransition: i', i, w[i], labels[pc[i]], 'f', f, w[f], labels[pc[f]])
            md_mom = [0, 0, 0]
            ed_mom = [0, 0, 0]
            if md:
                keys = ['M1-1', 'M10', 'M11']
                if not all(k in tensor_dict for k in keys):
                    raise ValueError("Missing all or some of the magnetic dipole "\
                            "operator matrix elements. Required tensors are 'M1-1', "\
                            "'M10', 'M11'")
                #md_mom = [np.real(md_prefac*tensor_dict[k][i, f]) for k in keys]
                # moments can be complex for complex eigenvectors. 
                md_mom = [(md_prefac*tensor_dict[k][i, f]) for k in keys]
                #print('md_mom', md_mom)
            if ed:
                for A in Altp:
                    #print('\n###', A)
                    l = int(A[0][1])
                    t = int(A[0][2])
                    pp = int(A[0][3])
                    for q in [-1, 0, 1]:
                        #print('>>> q',q)
                        for p in np.unique([-pp, pp]):
                            #print('## p',p)
                            A_val  = A[1]
                            if p < 0: # symmetry of Altp parameter
                                A_val = A_val.conjugate()
                                if (1+t+p) % 2 != 0: # if 1+t+p is odd
                                    A_val = - A_val
                            #print(A_val) 
                            #print('Altp', l, t, p, q, A_val)
                            if -l <= (p+q) <= l:
                                k = 'U%i%i' % (l, p+q)
                                if k not in tensor_dict:
                                    raise ValueError("Missing electric dipole tensor '{}' required by Altp.".format(k))
                                #print('Altp', l, t, p, A_val, 'q', q, 'k', k)
                                #print('D_factor', '%i%i%i%i' % (l, t, p, q), D_factor['%i%i%i%i' % (l, t, p, q)])
                                #print('tensor_dict', k, i, f, tensor_dict[k][i, f])
                                D = -e * A_val * D_factor['%i%i%i%i' % (l, t, p, q)] * tensor_dict[k][i, f]
                                #print('D', D)
                                ed_mom[q+1] += (D) # order is -1, 0, 1
                                #print('ed_mom', ed_mom)
                                

            if any(abs(d) > dipole_cutoff for d in md_mom) or any(abs(d) > dipole_cutoff for d in ed_mom):
                # temporarily stick with the totals that were in Sebastian's version
                isotropic = sum(np.abs(md_mom)**2)/3 + sum(np.abs(ed_mom)**2)/3
                axial = (np.abs(md_mom[0])**2 + np.abs(md_mom[2])**2)/2 + (np.abs(ed_mom[0])**2 + np.abs(ed_mom[2])**2)/2
                sigma = np.abs(md_mom[1])**2 + (np.abs(ed_mom[0])**2+np.abs(ed_mom[2])**2)/2
                pi = (np.abs(md_mom[0])**2+np.abs(md_mom[2])**2)/2 + np.abs(ed_mom[1])**2
                

                trs += [{'md_-1': md_mom[0], 'md_0': md_mom[1], 'md_+1': md_mom[2], 
                    'ed_-1': ed_mom[0], 'ed_0': ed_mom[1], 'ed_+1': ed_mom[2],
                    'isotropic': isotropic, 'axial': axial,'sigma': sigma, 'pi': pi, 
                    'ei': w[i], 'ef': w[f], 'e': w[f]-w[i],'i': i, 'f': f, 'pci': pc[i], 'pcf': pc[f] }]
            #print('trs')
            #print(trs)
            
    trs.sort(key=itemgetter('e'))
    return trs

def group_transitions(items, tol=1e-4):
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

    ei_deg = _level_degeneracies(items, 'ei', 'i')
    ef_deg = _level_degeneracies(items, 'ef', 'f')

    # Sort by transition energy first, then by level pair so equivalent pairs are
    # contiguous before grouping.
    ordered = sorted(items, key=lambda d: (d['e'], d['ei'], d['ef']))

    groups = []
    cur_ei = ordered[0]['ei']
    cur_ef = ordered[0]['ef']
    cur_list = [ordered[0]]

    for tr in ordered[1:]:
        same_ei = abs(tr['ei'] - cur_ei) <= tol
        same_ef = abs(tr['ef'] - cur_ef) <= tol
        if same_ei and same_ef:
            cur_list.append(tr)
        else:
            groups.append({
                'Energy': cur_ef - cur_ei,
                'e_i': cur_ei,
                'e_f': cur_ef,
                'g_i': _lookup_degeneracy(ei_deg, cur_ei),
                'g_f': _lookup_degeneracy(ef_deg, cur_ef),
                't_list': cur_list,
            })
            cur_ei = tr['ei']
            cur_ef = tr['ef']
            cur_list = [tr]

    groups.append({
        'Energy': cur_ef - cur_ei,
        'e_i': cur_ei,
        'e_f': cur_ef,
        'g_i': _lookup_degeneracy(ei_deg, cur_ei),
        'g_f': _lookup_degeneracy(ef_deg, cur_ef),
        't_list': cur_list,
    })

    return groups

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
    """
    if len(trs) == 0:
        raise ValueError("inten requires at least one transition.")
    
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
