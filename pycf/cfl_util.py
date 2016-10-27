#!/usr/bin/env python
# Filename = cfl_util.py

#   Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np
import re

from datetime import datetime
import os, inspect
import __version__

def uline_char(s):
    """Underline all non-whitespace characters in a string, except for single
    spaces between non-whitespace characters."""
    ul = ""
    for i in range(len(s)-1):
        if not s[i-1].isspace() and not s[i+1].isspace():
            ul += "-"
        elif s[i].isspace():
            ul += " "
        else:
            ul += "-"
    if s[-1::] == "\n":
        return s + ul + "\n"
    else:
        return s + ul

def term2L(c):
    r"""
    Convert an L quantum number term character to its numerical value.

    Parameters
    ----------
    c : string
        The L quantum number term character to be converted. 
    """
    if c == 'S': 
        return 0
    elif c == 'P':
        return 1
    elif c == 'D':
        return 2
    elif c == 'F':
        return 3
    elif c == 'G':
        return 4
    elif c == 'H':
        return 5
    elif c == 'I':
        return 6
    elif c == 'K':
        return 7
    elif c == 'L':
        return 8
    elif c == 'M':
        return 9
    elif c == 'N':
        return 10
    elif c == 'O':
        return 11
    elif c == 'P':
        return 12
    elif c == 'Q':
        return 13
    else:
        raise NotImplementedError("L quantum number term symbols beyond Q are not supported; found: {}.".format(c))

def L2term(i):
    r"""
    Convert an L quantum number numerical value to its term character.

    Parameters
    ----------
    i : integer
        The L quantum number numerical value to be converted. 
    """
    if i == 0: 
        return 'S'
    elif i == 1:
        return 'P'
    elif i == 2:
        return 'D'
    elif i == 3:
        return 'F'
    elif i == 4:
        return 'G'
    elif i == 5:
        return 'H'
    elif i == 6:
        return 'I'
    elif i == 7:
        return 'K'
    elif i == 8:
        return 'L'
    elif i == 9:
        return 'M'
    elif i == 10:
        return 'N'
    elif i == 11:
        return 'O'
    elif i == 12:
        return 'P'
    elif i == 13:
        return 'Q'
    else:
        raise NotImplementedError("L quantum number values greater than 12 are not supported; found: {}.".format(i))


def gen_pycf_summary():
    r"""
    Print the pycf version and date/time.

    """
    s = "pycf revision: {}\n".format(__version__.__version__)
    s += "File: {}\n".format(os.path.abspath(inspect.stack()[1][1]))
    s += "Calculation completed on: {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    s += "Input file\n"
    s += "==========\n\n"
    with open(str(os.path.abspath(inspect.stack()[1][1])), 'r') as f:
        s += f.read()
    s += "\n\n"

    return s


def ex_parse_abs(ex, z, labels):
    r"""
    Helper function for extracting and formatting experimental energy level data
    from an ExData object for absolute energy level data.

    Parameters
    ----------
    ex : ExData
        The object to be parsed.
    z : np.ndarray
        Eigenvector array the principal components of which are used to sort
        state labels.
    labels : list 
        A list of state labels.

    Returns
    -------
    parsed_ex : np.ndarray
        Two column array containing level indices starting at 1 in the zeroeth
        column and corresponding experimental energy levels in the first column.
        If the ExData object contains no absolute energy levels None is
        returned.
    """
    if ex.n_a == 0:
        parsed_ex = None
    elif ex.sl_index:
        parsed_ex = np.zeros((ex.n_a, 2))
        parsed_ex[:, 1] = ex.e[:ex.n_a]
        # Determine the index of the principal component of each
        # eigenvector. 
        pc = np.argmax(np.abs(z), axis=0)
        for i,r in enumerate(ex.a_states):
            # Find the index of the principal component of each state label.
            parsed_ex[i, 0] = np.where((np.array(labels)[pc] == r).all(axis=1))[0][0]
    
    else:
        parsed_ex = np.zeros((ex.n_a, 2))
        parsed_ex[:, 1] = ex.e[:ex.n_a]
        parsed_ex[:, 0] = ex.la 
        # Sort ex according to index column.
        parsed_ex = parsed_ex[np.argsort(parsed_ex[:, 0]), :]

    return parsed_ex

def ex_parse_diff(ex, z, labels):
    r"""
    Helper function for extracting and formatting experimental energy level data
    from an ExData object for energy level differences.

    Parameters
    ----------
    ex : ExData
        The object to be parsed.
    z : np.ndarray
        Eigenvector array the principal components of which are used to sort
        state labels.
    labels : list 
        A list of state labels.
    
    Returns
    -------
    parsed_ex : np.ndarray
        Three coloumn array containing initial level indices starting at 1 in
        the zeroeth column, final level indices starting at 1 in the first
        column, and corresponding experimental energy levels differences in the
        second column.  If the ExData object contains no absolute energy levels
        None is returned.
    """
    if ex.n_a == 0:
        parsed_ex = None
    elif ex.sl_index:
        parsed_ex = np.zeros((ex.n_d, 3))
        parsed_ex[:, 2] = ex.e[ex.n_a:]
        # Determine the index of the principal component of each
        # eigenvector. 
        pc = np.argmax(np.abs(z), axis=0)
        # Find the index of the principal component of each state label.
        for i,s in enumerate(ex.id_states):
            parsed_ex[i, 0] = np.where((np.array(labels)[pc] == s).all(axis=1))[0][0]
        for i,s in enumerate(ex.fd_states):
            parsed_ex[i, 1] = np.where((np.array(labels)[pc] == s).all(axis=1))[0][0]
    else:
        parsed_ex = np.zeros((ex.n_d, 3))
        parsed_ex[:, 2] = ex.e[ex.n_a:]
        parsed_ex[:, 0] = ex.ild
        parsed_ex[:, 1] = ex.fld
        # Sort ex according to index column.
        parsed_ex = parsed_ex[np.argsort(parsed_ex[:, 0]), :]

    return parsed_ex

def gen_e_summary(w, z, labels, label_key, ex=None, nstates=2, chi2=None, ndof=None, weighting=None, e_shift=False):
    r"""
    Generate energy level summary given eigenvalues and eigenvectors. 

    Parameters
    ----------
    w : np.ndarray
        The eigenvalue vector, of length n.
    z : np.ndarray
        The eigenvectors in an n by n matrix.
    labels : list
        A list of state labels.
    label_key : str
       String identifying the type of label.  Valid characters are S, L, J, M,
       I, T, and F and their position in label_key specifies the location in
       each label.  
    ex : np.ndarray or ExData, optional
        Either a 2 by n dimensional array or an ExData object. The two
        column case is used to specify only absolute energy levels.  In this
        instance, the first column contains energy level indices starting at 1,
        and the second column contains the absolute experimental energy of the
        corresponding level.  Other types of energy level data must be passed as
        an ExData object.  
    nstates : int, optional
        The number of constituent states to display for mixed states.
    chi2 : float, optional 
        The final chi2 value of the fit. 
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : float, optional
        The weighting applied during the chi2 fit.  This should be set if ndof is set.
    e_shift : bool, optional
        Shift entire eigenvalue spectrum s.t. the first eigenvalue is zero. 
    """
    
    def fmt_label(li, labels):
        label = "|"
        for i,l in enumerate(labels[li]):
            if label_key[i] == 'T':
                label += "{:d},".format(l)
            elif label_key[i] == 'F':
                if l:
                    label += "(2F)".format(l)
                else:
                    label += "    "
            elif label_key[i] == 'S':
                    label += "{:d}".format(l)
            elif label_key[i] == 'L':
                label += L2term(l)
            elif label_key[i] == 'J':
                label += "{: >2d},".format(l)
            elif i < len(label_key)-1:
                label += "{: >3d},".format(l)
            else:
                label += "{: >3d}>".format(l)

        return label
    
    if ex != None:
        if isinstance(ex, np.ndarray):
            # Sort ex according to index column.
            ex = ex[np.argsort(ex[:, 0]), :]
            # Change to zero based indexing
            ex[:, 0] = ex[:, 0]-1
        else:
            ex = ex_parse_abs(ex, z, labels)

        if len(ex[:, 0]) != len(set(ex[:, 0])):
            raise ValueError("e_summary: ex input data contains duplicate entries in the index column.")

    if e_shift:
        e_shift = -np.min(w)
        w = w + e_shift

    s = "Energy level summary\n"
    s+= "====================\n\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[:,i]))[::-1]]
    heading = "Lev.  " + ("Percentage                 " + "State" + \
            " " * (len(fmt_label(0, labels))-4)) * nstates + "       Theory"
    if ex != None:
        heading += "     Experiment    Difference\n"
    else:
        heading += " \n"
    
    s += uline_char(heading)
    ii=0
    for i in range(len(z)):
        line = "{0:<6}".format(i+1)
        N = np.sum(np.abs(z[:, i]))
        for j in range(nstates):
            si = sort_list[i][j]
            
            line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[si,i], np.abs(z[si,i])/N, 
                    si+1, fmt_label(si, labels))
        s += line + " {: >12.4f}".format(w[i])

        if ex != None:
            if ex[ii,0] == i:
                s += "   {: >12.4f}  {: >12.4f}".format(ex[ii,1], ex[ii,1]-w[i]) + "\n"
                if ii != len(ex)-1:
                    ii += 1
            else:
                s += "         --            --\n"
        else:
            s += "\n"

    s += "Label key: {}\n".format(label_key)
    if chi2 != None:
        s += "weighted chi2 = {: .4f}\n".format(chi2)
        if ndof != None:
            if weighting == None:
                raise ValueError("The weight argument needs to be provided if you provide ndof.")
            s += "sigma = {: .4f}\n".format(np.sqrt(chi2/weighting)/ndof)
            if weighting != 1:
                s += "weighting factor= {: .4f}\n".format(weighting)

    if e_shift:
        s += "Energy level shift: {: .4f}\n".format(e_shift)
    
    return s


def gen_e_summary_trunc(w, z, labels, label_key, ex, name, nstates=2, chi2=None, ndof=None, weighting=None):
    r"""
    Generate a truncated energy level summary displaying only levels for which
    experimental energy level data is provided.

    Parameters
    ----------
    w : np.ndarray
        The eigenvalue vector, of length n.
    z : np.ndarray
        The eigenvectors in an n by n matrix.
    labels : list
        A list of state labels.
    label_key : str
       String identifying the type of label.  Valid characters are S, L, J, M,
       I, T, and F and their position in label_key specifies the location in
       each label.  
    ex : ExData
        The ExData object for which to generate the truncated energy level summary. 
    name : str
        Name used in heading for this truncated summary.
    nstates : int, optional
        The number of constituent states to display for mixed states.
    chi2 : float, optional 
        The final chi2 value of the fit. 
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : float, optional
        The weighting applied during the chi2 fit.  This should be set if ndof is set.
    """
    def fmt_label(li, labels):
        label = "|"
        for i,l in enumerate(labels[li]):
            if label_key[i] == 'T':
                label += "{:d},".format(l)
            elif label_key[i] == 'F':
                if l:
                    label += "(2F)".format(l)
                else:
                    label += "    "
            elif label_key[i] == 'S':
                    label += "{:d}".format(l)
            elif label_key[i] == 'L':
                label += L2term(l)
            elif label_key[i] == 'J':
                label += "{: >2d},".format(l)
            elif i < len(label_key)-1:
                label += "{: >3d},".format(l)
            else:
                label += "{: >3d}>".format(l)

        return label
    
    if ex.n_a + ex.n_d == 0:
        return ""

    s = "{} summary\n".format(name)
    s+= "="*len(name) + "========\n\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[:,i]))[::-1]]
    
    # Absolute energy level summary.
    if ex.n_a != 0:
        if ex.n_d != 0:
            s += uline_char("Absolute energy levels:\n")
        exa = ex_parse_abs(ex, z, labels)
        heading = "Lev.  " + ("Percentage                 " + "State" + \
                " " * (len(fmt_label(0, labels))-4)) * nstates + "       Theory"
        heading += "     Experiment    Difference\n"
        
        s += uline_char(heading)
        for ii in range(ex.n_a):
            i = int(exa[ii, 0])
            line = "{0:<6}".format(i+1)
            N = np.sum(np.abs(z[:, i]))
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[si,i], np.abs(z[si,i])/N, 
                        si+1, fmt_label(si, labels))

            s += line + " {: >12.4f}".format(w[i])
            s += "   {: >12.4f}  {: >12.4f}".format(exa[ii,1], exa[ii,1]-w[i]) + "\n"
        
        s += "\n"
    
    # Difference energy level summary. 
    if ex.n_d != 0:
        if ex.n_a != 0:
            s += uline_char("Energy level differences:\n")
        exd = ex_parse_diff(ex, z, labels)
        heading = "Lev.  " + ("Percentage                 " + "State" + \
                " " * (len(fmt_label(0, labels))-4)) * nstates + "    Th. diff."
        heading += "     Exp. diff.    Diff. diff.\n"
        
        s += uline_char(heading)
        for ii in range(ex.n_d):
            i = int(exd[ii, 0])
            line = "{0:<6}".format(i+1)
            N = np.sum(np.abs(z[:, i]))
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[si,i], np.abs(z[si,i])/N, 
                        si+1, fmt_label(si, labels))
            s += line + "\n"
            tmp_w = w[i]
            i = int(exd[ii, 1])
            line = "{0:<6}".format(i+1)
            N = np.sum(np.abs(z[:, i]))
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[si,i], np.abs(z[si,i])/N, 
                        si+1, fmt_label(si, labels))
            tmp_w = w[i]-tmp_w
            s += line + " {: >12.4f}".format(tmp_w)
            s += "   {: >12.4f}  {: >12.4f}".format(exd[ii,2], exd[ii,2] - tmp_w) + "\n"
        s += "\n"

    s += "Label key: {}\n".format(label_key)
    if chi2 != None:
        s += "weighted chi2 = {: .4f}\n".format(chi2)
        if ndof != None:
            if weighting == None:
                raise ValueError("The weight argument needs to be provided if you provide ndof.")
            s += "sigma = {: .4f}\n".format(np.sqrt(chi2/weighting)/ndof)
            if weighting != 1:
                s += "weighting factor= {: .4f}\n".format(weighting)
    s += "\n"
    
    return s


def gen_sh_summary(param, sh, shx=None, name=None, chi2=None, ndof=None, weighting=None):
    r"""
    Generate a spin Hamiltonian summary displaying calculated and experimental
    spin Hamiltonian data. 

    Parameters
    ----------
    param : list
        Elements must be `3 \times 3` np.ndarrays corresponding to the spin
        Hamiltonian parameters.  Output from
        :func:`cfl.SpinHamiltonian.calc_param` is appropriately formated to be
        passed as param.
    sh : SpinHamiltonian
        Generally the spin Hamiltonian object used to generate the param list. 
    shx : dict, optional
        Specifies the experimental spin Hamiltonian data for comparison.  Valid
        keys are 'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3
        \times 3` np.ndarrays corresponding to the experimental spin Hamiltonian
        tensor.
    name : str, optional
        If provided, the summary heading uses the provided string instead of
        "Spin Hamiltonian".
    chi2 : np.ndarray, optional 
        The final chi2 value of the fit for each spin Hamiltonian term. 
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : dict, optional
        The weighting applied during the chi2 fit; one entry for each spin
        Hamiltonian term.  This should be set if ndof is set.
    """
    np.set_printoptions(formatter={'float': lambda x: '{:8.5f}'.format(x)})
    
    if name != None:
        s = "{} summary\n".format(name)
        s+= "="*len(name) + "========\n\n"
    else:
        s = "Spin Hamiltonian summary\n"
        s+= "========================\n\n"
    
    tmp_sigma = 0
    for i,inter in enumerate(sh.interactions):
        s += uline_char("%s interaction\n" % inter)
        if shx != None:
            s += uline_char("Theory                        Experiment                    Difference\n")
        else:
            s += uline_char("Theory\n")
        for j in range(3):
            s += str(np.real(np.abs(param[i])).reshape(3,3)[j,:])
            if shx != None:
                s += "  " + str(np.abs(shx[inter].reshape(3,3)[j,:])) + "  " + str((np.abs(shx[inter]) 
                    - np.abs(np.real(param[i]))).reshape(3,3)[j,:]) + "\n"
            else:
                s += "\n"
        if chi2 != None:
            s += "weighted chi2 = {: .4f}\n".format(chi2[i])
            if weighting != None:
                s += "weighting factor= {: .4f}\n".format(weighting[inter])
                tmp_sigma += chi2[i]/weighting[inter]
        s += "\n"
    
    if chi2 != None and ndof != None:
        if weighting == None:
            raise ValueError("The weight argument needs to be provided if you provide ndof.")
        s += "sigma = {: .4f}\n".format(np.sqrt(tmp_sigma)/ndof)

    return s

def gen_fit_summary(coeff, fit_obj, method, fmin, **kwargs):
    r"""
    Create a string summarizing a crystal-field Hamiltonian fitting run.

    Parameters
    ----------
    coeff : dict
        Contains the fitted interaction coefficients.
    fit_obj : EFitRunner, MHFitRunner, ESHFitRunner, or MESHFitRunner
        Must have __iter__ method that iterates over names of tensors.
    method : str
        The optimization algorithm used for the fit.
    kwargs: dict
        Additional, optimization algorithm specific, settings to print.

    """
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})
    cov = None

    s = "Fitting summary\n"
    s+= "===============\n\n"
    
    heading = "Tensor name            Fitted coeff        Initial coeff         Difference"
    if kwargs['cov']:
        ndof = max(fit_obj.n_p_real - fit_obj.n_obs, 1)
        sigma = np.sqrt(np.sum(fit_obj.chi2))/ndof
        cov = np.linalg.inv(kwargs['cov_inv'])
        heading += "    Uncertainty"
    if 'bounds' in kwargs:
        heading += "      Lower bounds       Upper bounds"
    if 'stepsize' in kwargs:
        heading += "          Stepsize"
    heading += "\n"

    s += uline_char(heading)
    for i,p in enumerate(fit_obj):
        co = fit_obj.coeff[p]
        if co.imag == 0:
            co = co.real
        s += "'{0:<12}: {1: >20.4f} {2: >20.4f} {3: >18.4f}".format(p+"'", coeff[p], co, coeff[p]-co)
        if kwargs['cov']:
            s += "{0: >15.0f}".format(np.sqrt(np.abs(cov[i,i]))*sigma)
        if 'bounds' in kwargs:
            s += "{0: >18.0f} {1: >18.0f}".format(kwargs['bounds'][p][0], kwargs['bounds'][p][1])
        if 'stepsize' in kwargs:
            s += "{0: >18.0f}".format(kwargs['stepsize'][p])
        s += "\n"

    if 'bounds' in kwargs:
        del kwargs['bounds']
    if 'stepsize' in kwargs:
        del kwargs['stepsize']

    np.set_printoptions(formatter={'float': lambda x: '{:15.4f}'.format(x)}, linewidth=200)
    if kwargs['cov']:
        s += "\n" + uline_char("Covariance matrix:\n")
        try:
            cov = np.linalg.inv(kwargs['cov_inv'])
            s += str(cov) + "\n"
        except:
            s += "Singular covariance matrix; cannot invert.\n"

        del kwargs['cov_inv']
    
    del kwargs['cov']

    s += "\nNumber of observables: {}\n".format(kwargs['n_obs'])
    s += "Number of real-valued parameters: {}\n".format(kwargs['n_param'])
    del kwargs['n_obs']
    del kwargs['n_param']

    if method == 'basinhopping':
        kwargs['naccept'] = kwargs['retval']
        del kwargs['retval']

    s += "\n" + uline_char("Optimization routine details:\n")
    s += "{0:<20} {1: <}\n".format("fmin:", fmin)
    s += "{0:<20} {1: <}\n".format("method:", method)
    for k in kwargs:
        s += "{0:<20} {1: <}\n".format(k+":", kwargs[k])

    return s


def print_as_fortran_array(a):
    r"""
    Print a two dimensional numpy array in a form that makes it easy to include
    in a c program, using column major ordering.
    """
    s = "{"
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (np.real(a[i,j]) == 0):
                a_real = 0
            else: 
                a_real = np.real(a[i,j])
            if (np.imag(a[i,j]) > 0):
                s += "{0}+{1}*I".format(a_real, np.imag(a[i,j]))
            elif (np.imag(a[i,j]) < 0):
                s += "{0}{1}*I".format(a_real, np.imag(a[i,j]))
            else:
                s += str(a_real)
            if ((i*a.shape[1]+j)<(a.shape[0]*a.shape[1])-1):
                s += ", "
    s += "};"
    print(s)

def print_as_c_array(a):
    r"""
    Print a two dimensional numpy array in a form that makes it easy to include
    in a c program, using row major ordering.  
    """
    s = "{"
    for i in range(a.shape[0]):
        s += "{"
        for j in range(a.shape[1]):
            if (np.real(a[i,j]) == 0):
                a_real = 0
            else: 
                a_real = np.real(a[i,j])
            if (np.imag(a[i,j]) > 0):
                s += "{0}+{1}*I".format(a_real, np.imag(a[i,j]))
            elif (np.imag(a[i,j]) < 0):
                s += "{0}{1}*I".format(a_real, np.imag(a[i,j]))
            else:
                s += str(a_real)
            if j != a.shape[1]-1:
                s += ","
        s += "}"
        if i != a.shape[0]-1:
            s += ", "
    s += "};"
    print(s)

def MHz2cm1(val):
    r"Convert MHz to cm$^{-1}$."
    return (1.0/29979.2458)*val

def cm12MHz(val):
    r"Convert cm$^{-1}$ to MHz."
    return 29979.2458*val

def bal_bounds(coeff, bounds):
    r"""
    Helper function for creating balanced bounds dictionary.  That is, the
    bounds are are some constant, symmetric, $\pm$ offset from the starting
    coefficient values.

    Parameters
    ----------
    coeff : dict
        Coefficient initial value dictionary. 
    bounds : dict
        Dictionary of single bounds values for each parameter to be fit, which
        will be added/subtracted from the initial coeff value.
    Returns
    -------
    bal_bounds : dict
        The balanced bounds dictionary. 
    """
    bal_b = {}
    for c in bounds:
        bal_b[c] = (coeff[c]-bounds[c], coeff[c]+bounds[c])

    return bal_b
