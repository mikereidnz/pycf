#!/usr/bin/env python
# Filename = spinh.py

# Copyright (C) 2013-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
import warnings
import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import block_diag
from scipy.optimize import minimize
from matel import matel


def bgs(v, m, t):
    r"""
    Generate the `BgS` term, an array of size `(2 \times j + 1)` by `(2 \times j
    + 1)`, with `j` the angular momentum of the rank one tensor `S`.

    Parameters 
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    m : numpy.ndarray
        The `3` by `3` Zeeman parameter matrix `g`.
    t : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : array
    """
    tl = len(t[0])
    l = len(t)
    result = np.zeros([tl, tl], dtype = np.complex)
    # All states of t are iterated through by the outer two loops.  The
    # contribution due to each term in v cdot m cdot t is computed by
    # the inner two loops.  This consists of a matrix multiplication of
    # the form transpose(v) * m * t_element, where t_element is the
    # matrix element corresponding to the state enumerated by tc and tr.
    for tr in range(tl):
        for tc in range(tl):
            elem = 0
            for i in range(l):
                for j in range(l):
                    elem += v[i] * m[i, j] * t[j][tr, tc]
            result[tr , tc] = elem
    
    return(result)

def ias(t1, m, t2):
    r""" 
    Generate the `IAS` term, an array of size `(2 \times j_1+1) \times (2 \times
    j_2+1)` by `(2 \times j_1+1) \times (2 \times j_2+1)`, with `j_1` and `j_2`
    the angular momentum of the rank one tensors `A` and `I`, respectively.

    Parameters
    ----------
    t1 : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    m : numpy.ndarray
        The `3` by `3` dipole parameter matrix `A`.
    t2 : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.
   
    Returns
    -------
    result : array
    """
    
    t1l = len(t1[0])
    t2l = len(t2[0])
    l = len(t1)
    result = np.zeros([t1l * t2l, t1l * t2l], dtype =  np.complex)

    # All states of t1 and t2 are iterated through by the outer four loops.
    # The contribution of each term in the t1 cdot m cdot t2 construct is
    # computed by the inner two loops.  This is effectively a matrix
    # multiplication, but for each term the multiplicative factors due to t1
    # and t2 are looked up from the matrix element determined by the indices
    # t1c, t1r, t2c and t2r. 
    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    elem = 0
                    for i in range(l):
                        for j in range(l):
                            elem += t1[i][t1r, t1c] * m[i, j] * t2[j][t2r, t2c]
                    result[t1r+t1l*t2r, t1c+t1l*t2c] = elem

    return(result)


def iqi(t, m):
    r"""
    Generate the `IQI` term, an array of size `(2 \times j + 1)` by `(2 \times j
    + 1)`, with `j` the angular momentum of the rank one tensor `I`.

    Parameters 
    ----------
    t : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    m : numpy.ndarray
        The `3` by `3` quadrupole parameter matrix `Q`.

    Returns
    -------
    result : array
        
    """
    
    tl = len(t[0])
    l = len(t)
    result = np.zeros([tl, tl], dtype = np.complex)

    # All states of t are iterated through by the outer two loops.  The
    # contribution due to each term in t cdot m cdot t is computed by
    # the inner three loops.  The i and j loops iterate over x, y and z
    # to yield a 3 by 3 matrix, with the sqi loop evaluating the matrix
    # elements for the square t * t operator.
    for tr in range(tl):
        for tc in range(tl):
            elem = 0
            for i in range(l):
                for j in range(l):
                    components = 0
                    for ci in range(tl):
                        components +=t[i][tr, ci] * t[j][ci, tc]
                    elem += m[i, j] * components
            result[tr, tc] = elem
    return(result)


def bgs_coeff_array(v, t):
    r"""
    Generate the `BgS` coefficient array.  This consists of a `2j+1 \times 2j+1`
    by `3 \times 3` array containing the matrix elements of the terms `B_a S_b`,
    with `a,b \in \{x, y, z\}` and `j` the angular momentum of the rank one
    tensor `S`.  Here the rows enumerate the `2j+1 \times 2j+1` different state
    combinations while the columns enumerate all combinations of `a` and `b`.
    This array is independent of `S` and is intended to be computed once, then
    employed with numpy's :func:`lstsq` function to calculate `S` given a `BgS`
    matrix.

    Parameters
    ----------
    v : numpy.ndarray
        A `3` by `1` vector of magnetic field strengths `B_x`, `B_y` and `B_z`.
    t : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j+1 \times 2j+1` by `3 \times 3` array.
    """

    tl = len(t[0])
    l = len(t)
    bgs_a = np.zeros([tl, tl, l, l], dtype = np.complex)

    for tr in range(tl):
        for tc in range(tl):
            for i in range(l):
                for j in range(l):
                    bgs_a[tr, tc, i, j] = v[i] * t[j][tr, tc]

    return(np.reshape(bgs_a, (tl*tl, l*l)))


def ias_coeff_array(t1, t2):
    r"""
    Generate the `IAS` coefficient array.  This consists of a `2j_1+1 \times
    2j_2+1` by `3 \times 3` array containing the matrix elements of the
    operators `I_a S_b`, with `a,b \in \{x, y, z\}` and `j_1` and `j_2` the
    angular momentum of the rank one tensors `I` and `S`, respectively.  Here
    the rows enumerate the `2j_1+1 \times 2j_2+1` different state combinations
    while the columns enumerate all combinations of `a` and `b`.  This array is
    independent of `A` and is intended to be computed once, then employed with
    numpy's :func:`lstsq` function to calculate `A` given an `IAS` matrix.

    Parameters
    ----------
    t1 : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.
    t2 : list
        Elements consist of the matrix elements of `S_x`, `S_y` and `S_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j_1+1 \times 2j_2+1` by `3 \times 3` array.
    """

    t1l = len(t1[0])
    t2l = len(t2[0])
    l = len(t1)
    ias_a = np.zeros([t1l * t2l, t1l * t2l, l, l], dtype = np.complex)

    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(l):
                        for j in range(l):
                            ias_a[t1r+t1l*t2r , t1c+t1l*t2c, i, j] = \
                                t1[i][t1r, t1c] * t2[j][t2r, t2c]

    return(np.reshape(ias_a, (t1l*t2l*t1l*t2l, l*l)))


def iqi_coeff_array(t):
    r""" 
    Generate the `IQI` coefficient array.  This consists of a `2j+1 \times 2j+1`
    by `3 \times 3` array containing the matrix elements of the operators `I_a
    I_b`, with `a,b \in \{x, y, z\}` and `j` the angular momentum of the rank
    one tensor `I`.  Here the rows enumerate the `2j+1 \times 2j+1` different
    state combinations while the columns enumerate all combinations of `a` and
    `b`.  This array is independent of `Q` and is intended to be computed once,
    then employed with numpy's :func:`lstsq` function to calculate `Q` given an
    `IQI` matrix.

    Parameters
    ----------
    t : list
        Elements consist of the matrix elements of `I_x`, `I_y` and `I_z`.

    Returns
    -------
    result : numpy.ndarray
        A `2j+1 \times 2j+1` by `3 \times 3` array.
    """
    
    tl = len(t[0])
    l = len(t)
    iqi_a = np.zeros([tl, tl, l, l], dtype = np.complex)

    for tr in range(tl):
        for tc in range(tl):
            for i in range(l):
                for j in range(l):
                    components = 0
                    for ci in range(tl):
                        components +=t[i][tr, ci] * t[j][ci, tc]
                    iqi_a[tr, tc, i, j] = components

    return(np.reshape(iqi_a, (tl*tl, l*l)))


def invert_term(coeff_a, b):
    r"""
    Invert a spin Hamiltonian term.

    Parameters
    ----------
    term : np.ndarray
        The matrix elements of the term.
    coeff_a : np.ndarray
        The appropriate coefficient array, generated with either
        :func:`bgs_coeff_array`, :func:`ias_coeff_array` or
        :func:`iqi_coeff_array`.
    b : numpy.ndarray
        For a 'BgS' term, b must be a `3 \times 2 \times 2` array, corresponding
        to individual 'BgS' matrix elements for a field along three linearly
        independent directions. For an 'IAS' term, b must be a `2 (I + 1) \times
        2` np.ndarray corresponding to the `IAS` matrix elements. 

    Returns
    -------
    result : numpy.ndarray
        A `9` by `1` vector, consisting of stacked rows of the 3 by 3 term
        parameter matrix. 
    """
    # Reshape the matel array to a vector.
    b = b.flatten()

    # Solve the equation coeff_a * x = b for x.
    return(np.real(lstsq(coeff_a, b)[0]))


def su2_rz(p, m):
    """
    Apply an SU(2) rotation about the z-axis of the spin-half matrix elements of a
    spin Hamiltonian term, specifically, a Zeeman interaction term or a magnetic
    dipole hyperfine interaction term.  

    Parameters
    ----------
    p : float 
        The phase `\phi` of an SU(2) rotation `\mathcal{D}_z(\phi)`.
    m : ndarray
        A `2 \times 2` Zeeman interaction term, or a `2 \times (I+1) \times 2`
        by `2 \times (I+1) \times 2` magnetic dipole hyperfine interaction term.
    
    Returns
    -------
    m : ndarray
        The transformed matrix. 
    """
    t = np.exp(np.complex(0,1)*p)
    mp = np.array(m, dtype=np.complex)

    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements of the 2 by 2 spin-half blocks.
    mp[0::2, 1::2] = m[0::2, 1::2] * t
    mp[1::2, 0::2] = m[1::2, 0::2] * np.conj(t)

    return(mp)


def su2_rz_lsq_f(p, coeff_a, b):
    """
    Helper function for least squares fitting of the SU(2) rotation required to
    symmetrize spin Hamiltonian terms containing spin half matrix elements.

    Parameters
    ----------
    p : float
        The phase to be varied. 
    coeff_a : np.ndarray
        The appropriate coefficient array, generated with either
        :func:`bgs_coeff_array` or :func:`ias_coeff_array`.
    b : numpy.ndarray
        For a 'BgS' term, b must be a `3 \times 2 \times 2` array, corresponding
        to individual 'BgS' matrix elements for a field along three linearly
        independent directions. For an 'IAS' term, b must be a `2 (I + 1) \times
        2` np.ndarray corresponding to the `IAS` matrix elements. 

    Returns
    -------
    r : float
        The residue; calculated from the differences between the off diagonal
        elements of the spin Hamiltonian tensor.
    """
    # Check whether 'BgS', or 'IAS', since we need to loop through each field
    # direction for the former.
    if b.shape == (3, 2, 2):
        b_p = np.zeros((3, 2, 2), dtype=np.complex)
        for i in range(3):
            b_p[i,:,:] = su2_rz(p, b[i,:,:])
    else:
        b_p = su2_rz(p, b)

    tensor = invert_term(coeff_a, b_p)
    
    r = 0
    for i in [(1, 3), (2, 6), (5, 7)]:
        r += np.abs(tensor[i[0]] - tensor[i[1]])

    return r


class SpinH(object):
    r""" 
    Container for holding data about the spin Hamiltonian.  Can either be used
    to calculate the full spin Hamiltonian from individual terms, or to invert
    the spin Hamiltonian and recover the spin Hamiltonian parameters.  If used
    for the former, the object is instantiated and then spin Hamiltonian terms
    are added with the :func:`add_term` method.  The full Hamiltonian can then
    be returned using the :func:`get_H` method.  If used for the latter, terms
    are added as arrays with dimensions of the full Hamiltonian using the
    :func:`add_H_term` method.  The spin Hamiltonian parameters can then be
    calculated using the :func:`inv_term` method. 

    Parameters
    ----------
    terms : list
        Elements are strings with possible values 'bgs', 'ias' and 'iqi'.  The
        choice of elements affects what other keyword arguments are required;
        see below. 
    B : numpy.ndarray or list with numpy.ndarray elements
        A `3` by `1` vector containing values for the magnetic field strengths
        `B_x`, `B_y` and `B_z`; if ``terms`` contains 'bgs' this keyword
        argument must be specified.  Furthermore, if ``'inv' = True``, this must
        be a list of magnetic field strength vectors, with a minimum of three
        linearly independent vectors required for a fully determined solution.
    S : float
        The spin projection `S_z`; if ``terms`` contains 'bgs' or 'ias' this
        keyword argument must be specified.
    I : float
        The nuclear spin projection `I_z`; if ``terms`` contains 'ias' or 'iqi'
        this keyword argument must be specified.
    inv : boolean, optional
        If True, the coefficient arrays for term inversion are pre-computed.

    Returns
    -------
    object : SpinH

    """
    def __init__(self, terms, **kwargs):
        for t in terms:
            if not any(t in term for term in ['bgs', 'ias', 'iqi']):
                raise ValueError("Invalid element in terms list: {}. Allowed"
                        "values are 'bgs', 'ias' and 'iqi'.".format(terms))
            else:
                self.t_list = terms
        self.terms = {}

        # Calculate matrix elements for the specified terms.
        j_l = ['jx', 'jy', 'jz']
        if 'bgs' in terms:
            try:
                B = kwargs['B']
                self.B = B
            except KeyError:
                raise ValueError("Missing keyword argument B.")
        else:
            B = None
        if 'bgs' in terms or 'ias' in terms:
            try: 
                S = kwargs['S']
            except KeyError:
                raise ValueError("Missing keyword argument S.")
            # Calculate the matrix elements for spin.
            S_m = [None, None, None]
            for i in range(3):
                S_m[i] = matel(j_l[i], S)
            self.S = S
            self.S_m = S_m
        else:
            S_m = None

        if 'ias' in terms or 'iqi' in terms:
            try:
                I = kwargs['I']
            except KeyError:
                raise ValueError("Missing keyword argument I.")
            # Calculate the matrix elements for nuclear spin.
            I_m = [None, None, None]
            for i in range(3):
                I_m[i] = matel(j_l[i], I)

            self.I = I
            self.I_m = I_m
        else:
            I_m = None

        # Determine Hamiltonian dimension.
        if B != None:
            if I_m == None:
                # Only the bgs term.
                H_dim = 2*S + 1
            else:
                # Both the bgs and iqi terms.
                H_dim = (2*S + 1) * (2*I + 1)
        elif S_m == None:
            # Only the iqi term.
            H_dim = 2*I + 1 
        else:
            # Contains ias term.
            H_dim = (2*S + 1) * (2*I + 1)
        
        self.H_dim = H_dim

        # Calculate the coefficient arrays.
        if 'inv' in kwargs:
            if kwargs['inv'] == True:
                self.H_terms = {}
                self.coeff_a = {}
                if 'bgs' in terms:
                    if not isinstance(B, list):
                        raise TypeError("When passing inv = True, B must be a"
                                "list of numpy.ndarrays.")
                    S_dimsq = (2*S + 1)**2
                    B_a = np.zeros([len(B), S_dimsq, 9], dtype = np.complex)
                    for i,e in enumerate(B):
                        B_a[i, :, :] = bgs_coeff_array(e, S_m)
                    self.coeff_a['bgs'] = np.reshape(B_a, (len(B) * S_dimsq, 9))

                if 'ias' in terms:
                    self.coeff_a['ias'] = ias_coeff_array(I_m, S_m)
                if 'iqi' in terms:
                    self.coeff_a['iqi'] = iqi_coeff_array(I_m)
            elif kwargs['inv'] != False:
                raise ValueError("Invalid value for keyword argument 'inv'; "
                        "valid values are either True or False")
            self.inv = kwargs['inv']
    
    def add_term(self, term, m):
        r"""
        Add the specified term to the spin Hamiltonian.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            provided when the SpinH object was instantiated.
        m : numpy.ndarray
            A `3` by `3` matrix providing the parameters for the specified spin
            Hamiltonian term.
        """
        if term not in self.t_list:
            raise ValueError("This SpinH object was not instantiated "
                    "with support for the specified term: {}".format(term))

        def __add_diag(m, n):
            """
            Generate n copies of a matrix and arrange in a block diagonal form.

            Parameters
            ----------
            m : numpy.ndarray
                Matrix to for blocks.
            n : int
                Number of copies of m.

            Returns
            -------
            result : numpy.ndarray
                A len(m) * n by len(m) * n block diagonal matrix.
            """
            l = [m * i for i in np.ones(n)]
            return(block_diag(*l))

        if term == 'bgs':
            # Create list of H_dim/(2*S + 1) length and block diagonalize.
            n = self.H_dim/(2 * self.S + 1)
            self.terms['bgs'] = __add_diag(bgs(self.B, m, self.S_m), n)
        elif term == 'ias':
            # ias term is of correct dimension.
            self.terms['ias'] = ias(self.I_m, m, self.S_m)
        elif term == 'iqi':
            # Create list of H_dim/(2*I + 1) length and block diagonalize.
            n = self.H_dim/(2 * self.I + 1)
            self.terms['iqi'] = __add_diag(iqi(self.I_m, m), n)
            
    def add_H_term(self, term, val):
        r"""
        Extract the specified term from the full Hamiltonian and update the
        appropriate term value of the SpinH object.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinH object was instantiated. 
        val : numpy.ndarray or list
            The value of the specified term.  For 'bgs' this must be a list of
            numpy.ndarrays, with elements in the list in the same order as the
            ``B`` list used to instantiate the SpinH object.  
        """
        if not self.inv:
            raise TypeError("This spectrum object does not support add_H_term "
                    "method calls; to enable this pass the inv = True argument "
                    "to the constructor.")
        if term not in self.t_list:
            raise ValueError("This SpinH object was not instantiated "
                    "with support for the specified term: {}".format(term))

        if term == 'bgs':
            S_dim = 2 * self.S + 1
            self.H_terms['bgs'] = np.array([v[:S_dim, :S_dim] for v in val])
        elif term == 'ias':
            self.H_terms['ias'] = val
        elif term == 'iqi':
            I_dim = 2 * self.I + 1
            self.H_terms['iqi'] = val[:I_dim, :I_dim]
    
    def inv_term(self, term, sym=False):
        r"""
        Invert the specified term of this spin Hamiltonian.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinH object was instantiated. 
        sym : bool
            Set to True to enable spin Hamiltonian parameter symmeterization.

        Returns
        -------
        term_parameters : numpy.ndarray
            A `9` by `1` vector consisting of stacked rows of the corresponding
            `3` by `3` term parameter matrix. 
        """
        if not self.inv:
            raise TypeError("This spectrum object does not support inv_term "
                    "method calls; to enable this pass the inv = True argument "
                    "to the constructor.")
        elif term not in self.t_list:
            raise ValueError("This SpinH object was not instantiated "
                    "with support for the specified term: {}".format(term))

        if sym:
            r = minimize(lambda p: su2_rz_lsq_f(p, self.coeff_a[term],
                self.H_terms[term]), 0, method='Powell')
            if not r['success']:
                warnings.warn("The tensor symmetrization fit did not succeed.",
                    RuntimeWarning)
            # Check whether 'BgS', or 'IAS', since we need to loop through each
            # field direction for the former. 
            if self.H_terms[term].shape == (3, 2, 2):
                for i in range(3):
                    self.H_terms[term][i,:,:] = su2_rz(r['x'], self.H_terms[term][i,:,:])
            else:
                self.H_terms[term] = su2_rz(r['x'], self.H_terms[term])

            self.sym_phase = r['x']
        else:
            self.sym_phase = 0
            
        return(invert_term(self.coeff_a[term], self.H_terms[term]))

    def get_H(self):
        r"""
        Calculate the full Hamiltonian and return the result.
        """

        # Bohr magneton in MHz/T
        # http://physics.nist.gov/cgi-bin/cuu/Value?mubshhz|search_for=bohr+magneton
        mu_b = 13.996245042e3

        H = np.complex(0, 0)
        for t in self.t_list:
            try:
                if t == 'bgs':
                    H += self.terms[t]*mu_b
                else:
                    H += self.terms[t]
            except KeyError:
                raise ValueError("This object does not have data for the {} "
                    "term.  Have you run the 'add_term' method?".format(t))

        return(H)
    
