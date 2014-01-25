#!/usr/bin/env python
# Filename = spinh.py

# Copyright (C) 2013 Sebastian Horvath (sebastian.horvath@gmail.com)
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
from numpy import linalg as LA
from scipy.linalg import block_diag
from scipy.optimize import minimize, basinhopping
from matel import matel
from pyemp import Spectrum

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

    def __ci(t1i, t2i):
        r"""
        Calculate the row/column index for the ``ias_a`` array.

        Parameters
        ----------
        t1i : list 
            The index for the tensor t1;
        t2i : list 
            The index for the tensor t2.
    
        Returns
        -------
        index : integer
            The ``ias_a`` index in dimension 0, or 1.
        """
        # The t1 upper bound is the t1 length.
        return(t1i + t1l * t2i)

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
                    result[__ci(t1r, t2r), __ci(t1c, t2c)] = elem

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

    def __ci(t1i, t2i):
        r"""
        Calculate the row/column index for the ``ias_a`` array.

        Parameters
        ----------
        t1i : list 
            The index for the tensor t1;
        t2i : list 
            The index for the tensor t2.
    
        Returns
        -------
        index : integer
            The ``ias_a`` index in dimension 0, or 1.
        """
        # The t1 upper bound is the t1 length.
        return(t1i + t1l * t2i)

    ias_a = np.zeros([t1l * t2l, t1l * t2l, l, l], dtype = np.complex)

    for t1r in range(t1l):
        for t2r in range(t2l):
            for t1c in range(t1l):
                for t2c in range(t2l):
                    for i in range(l):
                        for j in range(l):
                            ias_a[__ci(t1r, t2r), __ci(t1c, t2c), i, j] = \
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


def invert_term(term, coeff_a):
    r"""
    Invert a spin Hamiltonian term.

    Parameters
    ----------
    term : list
        The matrix elements of the term.
    coeff_a : numpy array
        The appropriate coefficient array, generated with either
        :func:`bgs_coeff_array`, :func:`ias_coeff_array` or
        :func:`iqi_coeff_array`.

    Returns
    -------
    result : numpy.ndarray
        A `9` by `1` vector, consisting of stacked rows of the 3 by 3 term
        parameter matrix. 
    """
    # Reshape the term array to a vector.
    b = term.flatten()
    # Use numpy's lstsq function, which wraps LAPACK's QR factorization, to
    # solve the equation coeff_a * x = b for x.
    x = np.real(LA.lstsq(coeff_a, b)[0])

    return(x)

def su2_rz(m, p):
    """
    Apply a rotation about the z-axis in the SU(2) matrix representation.
    
    Parameters
    ----------
    m : ndarray
        The `2 \times 2` SU(2) matrix. 
    p : float
        The phase `\phi` of an SU(2) rotation `\mathcal{D}_z(\phi)`.
    
    Returns
    -------
    mp : ndarry
        The transformed `2 \times 2` matrix.
    """
    t = np.exp(np.complex(0,1)*p)
    mp = np.array(m, dtype=np.complex)
    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements.
    mp[0, 1] = m[0, 1] * t
    mp[1, 0] = m[1, 0] * np.conj(t)
    return(mp)


def su2_rz_ias(m, p):
    """
    Apply a rotation about the z-axis of the spin-half matrix elements of a
    magnetic dipole spin Hamiltonian term.

    Parameters
    ----------
    m : ndarray
        The `IAS` term of dimension `2 \times (I+1) \times 2` by `2 \times (I+1)
        \times 2`.
    p : float 
        The phase `\phi` of an SU(2) rotation `\mathcal{D}_z(\phi)`.
    
    Returns
    -------
    mp : ndarray
        The transformed `IAS` term. 
    """
    t = np.exp(np.complex(0,1)*p)
    mp = np.array(m, dtype=np.complex)
    # The rotation consists of a multiplication by t and t^* of the off-diagonal
    # elements of the 2 by 2 spin-half blocks.
    mp[0::2, 1::2] = m[0::2, 1::2] * t
    mp[1::2, 0::2] = m[1::2, 0::2] * np.conj(t)
    return(mp)


class SpinHamiltonian(object):
    r""" 
    Container for holding data about the spin Hamiltonian.  Can either be used
    to calculate the full spin Hamiltonian from individual terms, or to invert
    the spin Hamiltonian and recover the spin Hamiltonian parameters.  If used
    for the former, the object is instantiated and then spin Hamiltonian terms
    are added with the :func:`add_term` method.  The full Hamiltonian can then
    be returned using the :func:`get_H` method.  If used for the latter, terms
    are added as arrays with dimensions of the full Hamiltonian using the
    :func:`add_H_term` method.  The spin Hamiltonian parameters can then be
    calculated using the :func:`inv_term` method; this operation is quite
    efficient for repeated evaluations using the same spin Hamiltonian object,
    since the inversion coefficient matrices are precomputed when the object is
    created.

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
    object : SpinHamiltonian
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
            provided when the SpinHamiltonian object was instantiated.
        m : numpy.ndarray
            A `3` by `3` matrix providing the parameters for the specified spin
            Hamiltonian term.
        """
        if term not in self.t_list:
            raise ValueError("This SpinHamiltonian object was not instantiated "
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
            
    def add_H_term(self, term, val, phase=None):
        r"""
        Extract the specified term from the full Hamiltonian and update the
        appropriate term value of the SpinHamiltonian object.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinHamiltonian object was instantiated. 
        val : numpy.ndarray or list
            The value of the specified term.  For 'bgs' this must be a list of
            numpy.ndarrays, with elements in the list in the same order as the
            ``B`` list used to instantiate the SpinHamiltonian object.
        phase : float, optional
            If specified, an SU(2) rotation `\mathcal{D}_z(\phi)` is applied to
            terms containing spin-half matrix elements. 
        """
        if not self.inv:
            raise TypeError("This spectrum object does not support add_H_term "
                    "method calls; to enable this pass the inv = True argument "
                    "to the constructor.")
        if term not in self.t_list:
            raise ValueError("This SpinHamiltonian object was not instantiated "
                    "with support for the specified term: {}".format(term))

        if term == 'bgs':
            S_dim = 2 * self.S + 1
            if phase != None:
                if self.S != 1/2:
                    raise ValueError("The phase argument can only be specified"
                            "for terms containing spin-half matrix elements")
                self.H_terms['bgs'] = np.array([su2_rz(v[:S_dim, :S_dim], phase)\
                    for v in val])
            else:
                self.H_terms['bgs'] = np.array([v[:S_dim, :S_dim] for v in val])
        elif term == 'ias':
            if phase != None:
                if self.S != 1/2:
                    raise ValueError("The phase argument can only be specified"
                            "for terms containing spin-half matrix elements")
                val = su2_rz_ias(val, phase)
            self.H_terms['ias'] = val
        elif term == 'iqi':
            I_dim = 2 * self.I + 1
            self.H_terms['iqi'] = val[:I_dim, :I_dim]
    
    def inv_term(self, term):
        r"""
        Invert the specified term of this spin Hamiltonian.

        Parameters
        ----------
        term : string
            Specifies the term; must be one of the values of the ``term`` list
            specified when the SpinHamiltonian object was instantiated. 

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
            raise ValueError("This SpinHamiltonian object was not instantiated "
                    "with support for the specified term: {}".format(term))

        coeff_a = self.coeff_a[term]
        
        # Reshape the term array to a vector; for the 'bgs' case this stacks the
        # different spin Hamiltonian terms in addition to stacking different
        # states.
        try:
            b = self.H_terms[term].flatten()
        except:
            raise ValueError("This object does not have Hamiltonian data for {}"
                    ".  Have you run the 'add_H_term' method?".format(term))
        

        # Use numpy's lstsq function, which wraps LAPACK's QR factorization, to
        # solve the equation coeff_a * x = b for x.
        return(np.real(LA.lstsq(coeff_a, b)[0]))

    def get_H(self):
        r"""
        Calculate the full Hamiltonian and return the result.
        """
        H = np.complex(0, 0)
        for t in self.t_list:
            try:
                H += self.terms[t]
            except KeyError:
                raise ValueError("This object does not have data for the {} "
                    "term.  Have you run the 'add_term' method?".format(t))

        return(H)


def su2_rz_lsq_f(p, sh, mode, H):
    """
    Helper function for least squares fitting of the SU(2) rotation required to
    symmetrize spin Hamiltonian terms containing spin half matrix elements.

    Parameters
    ----------
    p : float
        The phase to be varied. 
    sh : SpinHamiltonian
        Must have been instantiated with the 'inv' = True kwarg and contain a
        term with spin half matrix element, i.e., either `BgS` or `IAS`. 
    mode : string
        Specify whether to symmetrize using the `BgS` or the `IAS` term using
        values 'bgs' and 'ias', respectively.  
    H : list or numpy.ndarray
        If mode='bgs' H must be a list of `2 \times 2` ndarrays corresponding
        to the `BgS` spin Hamiltonian term; the order of the elements must match
        the order of the B argument used to instantiate sh.  If mode = 'ias' H
        must be a `2 (I + 1) \times 2` ndarray corresponding to the `IAS` spin
        Hamiltonian term.
       
    Returns
    -------
    r : float
        The residue; calculated from the differences between the off diagonal
        elements of the spin Hamiltonian tensor. 
    """
    if sh.S != 1/2:
            raise ValueError("su2_rz_lsq_f only supports SU(2) rotations; you "
                    "provided a spin Hamiltonian that is not spin half.")
    
    if mode == 'bgs':
        if isinstance(H, list):
            H_current = []
            for e in H:
                H_current += [su2_rz(e, p[0])]
        else:
            raise ValueError("If mode = 'bgs' H must be a list.")
    
        sh.add_H_term('bgs', H_current)
        tensor = sh.inv_term('bgs')
    elif mode == 'ias':
        if isinstance(H, numpy.ndarray):
            H_current = su2_rz_ias(H, p[0])
        else:
            raise ValueError("If mode = 'ias' H must be a numpy.ndarray.")

        sh.add_H_term('ias', H_current)
        tensor = sh.inv_term('ias')
    else:
        raise ValueError("Invalid mode; allowed values are 'bgs' and 'ias'.")

    sym_index = [(1, 3), (2, 6), (5, 7)]
    r = 0
    for i in sym_index:
        r += np.abs(tensor[i[0]] - tensor[i[1]])
    
    return r


def su2_rz_lsq(sh, spec, phi_p=0, term=None):
    r"""
    Calculate the SU(2) `\mathcal(R)_z(\phi)` parameter `\phi` that symmeterizes
    the Zeeman and/or magentic dipole tensor of the provided
    :class:`SpinHamiltonian` object.

    Parameters
    ----------
    sh : SpinHamiltonian
        Must have been instantiated with the inv='true' argument.
    spec : Spectrum
        Must have been instantiated with spin hamiltonian support.
    phi_p : complex, optional
        Parameter `\phi`, typically from a previous evaluation, used to check
        whether re-fitting is required; defaults to 0.
    term : string, optional
        Either 'bgs', 'ias' or None; specifies which term, if any, to
        symmeterize spin-half matrix elements with.
        
    Returns
    -------
    phi : complex 
        The parameter `\phi`.
    """
    f_min = lambda p: su2_rz_lsq_f(p, sh, term, spec.sh_terms[term])
    if term == None:
        phi = 0
    elif np.abs(f_min([phi_p])) <= 10**(-10):
        phi = phi_p
    else:
        r = minimize(f_min, 0, method='Powell')
        if not r['success']:
            warnings.warn("The tensor symmetrization fit did not succeed.",
                    RuntimeWarning)
        phi = r['x']
    
    return(phi)


class BHStep(object):
    r"""
    Custom class for modifying basinhopping step size.
    """
    def __init__(self, stepsize=0.5):
        self.stepsize = np.array(stepsize)

    def __call__(self, x):
        s = self.stepsize
        s_l = len(s)
        x[:s_l] += np.random.uniform(-s, s)
        x[s_l:] += np.random.uniform(-0.5, 0.5, x[s_l:].shape)
        return x
   

class BHBounds(object):
    r"""
    Custom class for modifying bounds on parameters to be varied using
    basinhopping. 
    """
    def __init__(self, xmax, xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class SHFit(object):
    r"""
    Fit crystal field parameters using spin Hamiltonian and energy level data. 

    Parameters
    ----------
    sh : SpinHamiltonian
        The terms that this object determines the to be fit spin Hamiltonian
        terms.  Additionally, it must have been instantiated with they inv=True
        kwarg.
    spec_f : function
        A function that returns a dictionary containing all the keys required
        for instantiating a Spectrum object and takes as an argument a list of
        parameters to be fit.
    p0 : list
        Initial values for the parameters to be fit; the order of elements is
        determined by spec_f. 
    data_exp : dictionary
        Valid keys are 'bgs', 'ias', 'iqi' and 'e'.  Spin Hamiltonian tensors
        are `3 \times 3` ndarrays and each term specified when the to be fit
        SpinHamiltonian object was instantiated must have a corresponding
        experimental tensor specified here.  'e' is optional and corresponds to
        an `3 \times n` ndarray of experimental block, level and energy data.
    weights : dictionary, optional
        Allowed keys are 'bgs', 'ias', 'iqi' and 'e'; these values correspond,
        respectively, to the weighting used in the least squares fit for the
        experimental values of `g`, `A`, `Q` and the energy levels.
    niter : integer, optional
        The number of basin hopping iterations.
    step : list, optional
        Elements specify the step size of the to be fit parameters, with the
        order determined by the argument of spec_f.
    bounds : tuple, optional
        Elements are lists of the same length as p0 with elements specifying
        the upper and lower bounds of the to be fit parameters, respectively.
        
    Returns
    -------
    object : SHFit

    """
    def __init__(self, sh, spec_f, p0, data_exp, weights=None, niter=50,
            step=None, bounds=None):
        self.sh = sh
        self.spec_f = spec_f
        self.p0 = p0
        self.data_exp = data_exp
        spec = Spectrum(name = 'sh_lsq', **spec_f(p0))
        spec.cfit()
        
        # Reshape since least squares implementation requires column data.
        self.sh_exp = [data_exp[t].reshape(1, 9) for t in sh.t_list]
        if 'e' in data_exp:
            self.ble_exp = data_exp['e']
        else:
            self.ble_exp = np.zeros((0, 3))
        
        # We calculate the weighting scaled by the experimental magnitude.
        self.scaled_w = {}
        if weights != None:
            for key in data_exp:
                try:
                    self.scaled_w[key] = weights[key]/np.sum(data_exp[key])
                except KeyError:
                    raise ValueError("If the weights dictionary is specified a "
                            "value must be provided for each term in data_exp.")
        else:
            for key in data_exp:
                self.scaled_w[key] = 1
        
        # Determine which term should be used to symmeterize spin-half matrix
        # elements. 
        if 'bgs' in sh.t_list:
            self.sym_term = 'bgs'
        elif 'ias' in sh.t_list:
            self.sym_term = 'ias'
        else:
            self.sym_term = None
        self.phi = 0

        # niter, step and bounds defaults if not specified.
        self.niter = niter

        if step != None:
            if len(step) != len(p0):
                raise ValueError("The provided step list is not the same "
                        "length as the p0 list.")
            self.step = step
        else:
            self.step = [np.array(10)] * len(p0)

        if bounds != None:
            if len(bounds[0]) != len(p0):
                raise ValueError("The provided bounds tuple contains lists "
                        "that are not the same length as the p0 list.")
            self.bounds = bounds
        else:
            self.bounds = ([np.array(10**5)]*len(p0),[np.array(-10**5)]*len(p0))

    def lsq_f(self, cf_params):
        r""" Spin Hamiltonian fitting function; calculates weighted differences
        between experimental and theoretical values of spin Hamiltonian terms
        and energy levels for a given set of crystal field parameters.

        Parameters
        ----------
        cf_params : list
            Crystal field parameters to be varied. 
        full_out : boolean, optional
            If True, the function returns a list of squares of differences for
            individual terms; this is intended for calibrating the weights
            parameters.

        Returns
        -------
        res : float
            The residue.
        """
        # Create new spectrum object and run cfit to calculate the spin
        # Hamiltonian terms for the specified crystal field parameters.
        spec = Spectrum(name = 'sh_lsq', **self.spec_f(cf_params))
        spec.cfit()
        
        # Symmeterize if necessary.
        self.phi = su2_rz_lsq(self.sh, spec, phi_p=self.phi, term=self.sym_term)

        # The calculated spin Hamiltonian matrices are only solutions up to a
        # constant prefactor the value of which we determine w.r.t. the
        # experimental term data.  
        sh_sqdiff = np.zeros((len(self.sh.t_list), 9))
        for i,e in enumerate(self.sh.t_list):
            self.sh.add_H_term(e, spec.sh_terms[e], phase=self.phi)
            sh_term = self.sh.inv_term(e)
            max_index = sh_term.argmax()
            prefactor = np.abs(sh_term.sum()/self.sh_exp[i].sum())
            sh_sqdiff[i, :] = np.sum(((self.sh_exp[i] - prefactor *
                sh_term)*self.scaled_w[e])**2)
        
        sh_sq = np.sum(sh_sqdiff)
        # Get levels for which we have experimental data. 
        e_exp = self.ble_exp[:, 2]
        b_l_exp = self.ble_exp[:, 0:2]
        reduced_e = np.zeros(len(e_exp))
        for i,n in enumerate(b_l_exp):
            reduced_e[i] = spec.sh_energies[np.logical_and(
                spec.sh_b_l[:, 0] == n[0], spec.sh_b_l[:, 1] == n[1])]
        
        # Calculate the square of the difference between the experimental and
        # theoretical energy levels.
        if reduced_e != []:
            e_sq = np.sum(((e_exp - reduced_e)*self.scaled_w['e'])**2)
        else:
            e_sq = 0
            
        return(sh_sq + e_sq)

    def __print_f(self, x, f, accepted):
        print("At minima %.4f accepted %d." % (f, int(accepted)))

    def gen_fit(self):
        r"""
        Run the fitting procedure. 
        """
        step = BHStep(self.step)
        bounds = BHBounds(self.bounds[0], self.bounds[1])

        
        self.fit = basinhopping(self.lsq_f, self.p0, niter = self.niter,
                take_step = step, accept_test = bounds, callback =
                self.__print_f, minimizer_kwargs = {'method': 'Powell'})
        
        return(self.fit)
    
    def gen_summary(self):
        r"""
        Generate a summary of the fit, along with the fitted parameters and cfit
        log output.
        """
        spec = Spectrum(name = 'sh_lsq', **self.spec_f(self.fit['x']))
        spec.cfit()
        phi = su2_rz_lsq(self.sh, spec, term=self.sym_term)

        cfit_log = spec.print_log(mode='full')
    
        fit_log = "Fitting summary\n"
        fit_log += "===============\n\n"
        fit_log += str(self.fit) + "\n\n"

        sh_log = "Spin Hamiltonian log\n"
        sh_log += "====================\n\n"

        for i,e in enumerate(self.sh.t_list):
            self.sh.add_H_term(e, spec.sh_terms[e], phase=phi)
            sh_log += "{} term:\n".format(e)
            sh_theory = self.sh.inv_term(e).reshape((3,3))
            sh_log += str(sh_theory) + "\n\n"
            sh_log += "{} experimental value:\n".format(e)
            sh_log += str(self.data_exp[e]) + "\n\n"
            sh_log += "theory - experiment:\n"
            sh_log += str(sh_theory - self.data_exp[e]) + "\n\n"

        return(fit_log + sh_log + cfit_log) 

