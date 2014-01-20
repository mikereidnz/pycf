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


class SpinHamiltonian(dict):
    r""" 
    Container for holding data about the spin Hamiltonian.  Can either be used
    to calculate the full spin Hamiltonian from individual terms, or to invert
    the spin Hamiltonian and recover the spin Hamiltonian parameters.  If used
    for the former, the object is instantiated and then spin Hamiltonian terms
    are added with the :func:`add_term` method.  The full Hamiltonian can then
    be returned using the ``H`` key of the spin Hamiltonian object.  If used for
    the latter, terms are added as arrays with dimensions of the full
    Hamiltonian using the :func:`add_H_term` method.  The spin Hamiltonian
    parameters can then be calculated using the :func:`inv_term` method; this
    operation is quite efficient for repeated evaluations using the same spin
    Hamiltonian object, since the inversion coefficient matricies are
    precomputed when the object is created.

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
                raise ValueError("Invalid element in terms list: {}. Allowed \
values are 'bgs', 'ias' and 'iqi'.".format(terms))
            else:
                self['terms'] = terms
        # Calculate matrix elements for the specified terms.
        j_l = ['jx', 'jy', 'jz']
        if 'bgs' in terms:
            try:
                B = kwargs['B']
                self['B'] = B
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
            self['S'] = S
            self['S_m'] = S_m
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

            self['I'] = I
            self['I_m'] = I_m
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
        
        self['H_dim'] = H_dim

        # Calculate the coefficient arrays.
        if 'inv' in kwargs:
            if kwargs['inv'] == True:
                if 'bgs' in terms:
                    if not isinstance(B, list):
                        raise TypeError("When passing inv = True, B must be a \
list of numpy.ndarrays.")
                    S_dimsq = (2*S + 1)**2
                    B_a = np.zeros([len(B), S_dimsq, 9], dtype = np.complex)
                    for i,e in enumerate(B):
                        B_a[i, :, :] = bgs_coeff_array(e, S_m)
                    self['bgs_coeff_a'] = np.reshape(B_a, (len(B) * S_dimsq, 9))

                if 'ias' in terms:
                    self['ias_coeff_a'] = ias_coeff_array(I_m, S_m)
                if 'iqi' in terms:
                    self['iqi_coeff_a'] = iqi_coeff_array(I_m)
            elif kwargs['inv'] != False:
                raise ValueError("Invalid value for keyword argument 'inv'; \
valid values are either True or False")
            self['inv'] = kwargs['inv']

    
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
        if term not in self['terms']:
            raise ValueError("This SpinHamiltonian object was not instantiated \
with support for the specified term: {}".format(term))

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
            n = self['H_dim']/(2 * self['S'] + 1)
            self['bgs_H'] = __add_diag(bgs(self['B'], m, self['S_m']), n)
        elif term == 'ias':
            # ias term is of correct dimension.
            self['ias_H'] = ias(self['I_m'], m, self['S_m'])
        elif term == 'iqi':
            # Create list of H_dim/(2*I + 1) length and block diagonalize.
            n = self['H_dim']/(2 * self['I'] + 1)
            self['iqi_H'] = __add_diag(iqi(self['I_m'], m), n)
            
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
        if not self['inv']:
            raise TypeError("This spectrum object does not support add_H_term \
method calls; to enable this pass the inv = True argument to the constructor.")
        if term not in self['terms']:
            raise ValueError("This SpinHamiltonian object was not instantiated \
with support for the specified term: {}".format(term))

        if term == 'bgs':
            S_dim = 2 * self['S'] + 1
            if phase != None:
                if self['S'] != 1/2:
                    raise ValueError("The phase argument can only be specified"
                            "for terms containing spin-half matrix elements")
                self['bgs'] = np.array([su2_rz(v[:S_dim, :S_dim], phase) for v in val])
            else:
                self['bgs'] = np.array([v[:S_dim, :S_dim] for v in val])
        elif term == 'ias':
            if phase != None:
                if self['S'] != 1/2:
                    raise ValueError("The phase argument can only be specified"
                            "for terms containing spin-half matrix elements")
                val = su2_rz_ias(val, phase)
            self['ias'] = val
        elif term == 'iqi':
            I_dim = 2 * self['I'] + 1
            self['iqi'] = val[:I_dim, :I_dim]
    
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
        if not self['inv']:
            raise TypeError("This spectrum object does not support inv_term \
method calls; to enable this pass the inv = True argument to the constructor.")
        elif term not in self['terms']:
            raise ValueError("This SpinHamiltonian object was not instantiated \
with support for the specified term: {}".format(term))

        coeff_a = self['{}_coeff_a'.format(term)]
        
        # Reshape the term array to a vector; for the 'bgs' case this stacks the
        # different spin Hamiltonian terms in addition to stacking different
        # states.
        try:
            b = self[term].flatten()
        except:
            raise ValueError("This object does not have the {} attribute.  \
Have you run the 'add_H_term' method?".format(term))
        

        # Use numpy's lstsq function, which wraps LAPACK's QR factorization, to
        # solve the equation coeff_a * x = b for x.
        return(np.real(LA.lstsq(coeff_a, b)[0]))

    def __getitem__(self, key):
        r"""
        Redefine the __getitem__ method to return the full Hamiltonian for the
        key ``H``.
        """
        if key == 'H':
            H = np.complex(0, 0)
            for t in self['terms']:
                try:
                    H += self['{}_H'.format(t)]
                except KeyError:
                    raise ValueError("This object does not have the {}_H \
attribute.  Have you run the 'add_term' method?".format(t))
            return(H)
        else:
            return(dict.__getitem__(self, key))


def sh_lsq_f(cf_params, sh, spec_f, sh_exp, exp_en, weights, full_out=False):
    r""" 
    Spin Hamiltonian fitting function; calculates weighted differences between
    experimental and theoretical values of spin Hamiltonian terms and energy
    levels for a given set of crystal field paramaters.
    
    Parameters
    ----------
    cf_params : list
        Crystal field parameters to be varied. 
    spec_f : func
        Spectrum function; returns a dictionary that can be unpacked when
        creating a spectrum instance.  This function must, as an only argument,
        accept values for the crystal field parameters to be fit.
    sh : SpinHamiltonian
        A spin Hamiltonian object. 
    sh_exp : list
        `n` vectors, each of length `9`, corresponding to the `3` by `3`
        matrices for the spin Hamiltonian parameters `g`, `A` and `Q` with
        stacked rows.  Here, `n` is the number of terms in the spin Hamiltonian.
        The order of the vectors must be the same as for the ``terms`` argument
        that was used for instantiating the SpinHamiltonian object.
    exp_en : numpy.ndarray
        Experimental data for the energy levels.
    weights : dictionary 
        Allowed keys are 'bgs', 'ias', 'iqi' and 'E'; these values correspond,
        respectively, to the weighting used in the least squares fit for the
        experimental values of `g`, `A`, `Q` and the energy levels.
    full_out : boolean
        If True, the function returns squares of differences for individual
        terms; this is intended for callibrating the weights parameters.
    """
    
    # Create new spectrum object and run cfit to calculate the spin Hamiltonian
    # terms for the specified crystal field parameters.
    spec = Spectrum(name = 'sh_lsq', **spec_f(cf_params))
    spec.cfit()
    term_dict = spec['sh_terms']
    
    # The calculated spin Hamiltonian matrices are only solutions up to a
    # constant prefactor; we calculate this w.r.t. the experimental term data. 
    terms = sh['terms']
    sh_sqdiff = np.zeros([len(terms), 9])
    for i,e in enumerate(terms):
        sh.add_H_term(e, term_dict[e])
        sh_term = sh.inv_term(e)
        max_index = sh_term.argmax()
        prefactor = np.abs(sh_term.sum()/sh_exp[i].sum())
        sh_sqdiff[i, :] = (sh_exp[i] - prefactor * sh_term)**2 * weights[e]
    
    # Get levels for which we have experimental data. 
    exp_e = exp_en[:, 2]
    exp_b_l = exp_en[:, 0:2]
    reduced_e = np.zeros(len(exp_e))
    for i,n in enumerate(exp_b_l):
        reduced_e[i] = spec['sh_energies'][np.logical_and(spec['sh_b_l'][:, 0] == n[0], spec['sh_b_l'][:, 1] == n[1])]

    # Calculate the square of the difference between the experimental and
    # theoretical energy levels, times the energy level weighting.
    print("Exp e: %s" % exp_e)
    print("Reduced calc e: %s" % reduced_e)
    e_sq = np.sum((exp_e - reduced_e)**2 * weights['E'])

    # Return the requested data.
    if full_out:
        return([np.sum(sh_sqdiff[i,:]) for i in range(len(terms))] + [e_sq])
    else:
        return(np.sum(sh_sqdiff) + e_sq) 

