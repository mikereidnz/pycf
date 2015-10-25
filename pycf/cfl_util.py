#!/usr/bin/env python
# Filename = cfl_util.py

#   Copyright (C) 2014 Sebastian Horvath (sebastian.horvath@gmail.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

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
    else:
        raise NotImplementedError("L quantum number term symbols beyond N are not supported.")

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
    else:
        raise NotImplementedError("L quantum number values greater than 10 are not supported.")

def gen_pycf_summary():
    r"""
    Print the pycf version and date/time.

    """
    s = "pycf revision: {}\n".format(__version__.__version__)
    s += "File: {}\n".format(os.path.abspath(inspect.stack()[1][1]))
    s += "Calculation completed on: {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return s

def gen_e_summary(w, z, labels, ex=None, nstates=2, sigma=None):
    r"""
    Generate energy level summary given eigenvalues and eigenvectors. 

    Parameters
    ----------
    w : np.ndarray
        The eigenvalue vector, of length n.
    z : np.ndarray
        The eigenvectors in an n by n matrix.
    labels : list
        A list of labels of state labels.
    ex : np.ndarray, optional
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    nstates : int, optional
        The number of constituent states to display for mixed states.
    sigma : float, optional
        The standard deviation for the energy level chi^2.
    """

    def fmt_label(li, labels):
        for i,l in enumerate(labels[li]):
            if i == 0:
                label = "|{:d}".format(l)
            elif i == 1:
                label += L2term(l)
            elif i == 2:
                label += "{: >2d},".format(l)
            elif i != len(labels[li])-1:
                label += "{: >3d},".format(l)
            else:
                label += "{: >3d}>".format(l)

        return label

    #z = np.transpose(z) 
    s = "Energy level summary\n"
    s+= "====================\n\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[:,i]))[::-1]]
    heading = "Lev.  " + ("Percentage                 " + "State" + " "*(len(fmt_label(0, labels))-4))*nstates + "       Theory"
    if ex != None:
        heading += "     Experiment    Difference\n"
    else:
        heading += " \n"
    
    s += uline_char(heading)
    ex_i=0
    for i in range(len(z)):
        line = "{0:<6}".format(i+1)
        N = np.sum(np.abs(z[:, i]))
        for j in range(nstates):
            si = sort_list[i][j]
            
            line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[si,i], np.abs(z[si,i])/N, si+1, fmt_label(si, labels))
        s += line + " {: >12.4f}".format(w[i])
        if ex != None:
            if ex[ex_i,0] == i+1:
                s += "   {: >12.4f}  {: >12.4f}".format(ex[ex_i,1], ex[ex_i,1]-w[i]) + "\n"
                if ex_i != len(ex)-1:
                    ex_i += 1
            else:
                s += "         --            --\n"
        else:
            s += "\n"

    if sigma != None:
        s += "sigma = {: .4f}\n".format(sigma)

    return s

def gen_sh_summary(param, sh, shx=None, sigma=None):
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
    sigma : float, optional
        The standard deviation for the spin Hamiltonian chi^2.
    """
    np.set_printoptions(formatter={'float': lambda x: '{:8.5f}'.format(x)})

    s = "Spin Hamiltonian summary\n"
    s+= "========================\n\n"
    for i,inter in enumerate(sh.interactions):
        s += uline_char("%s interaction\n" % inter)
        if shx != None:
            s += uline_char("Theory                        Experiment                    Difference\n")
        else:
            s += uline("Theory\n")
        for j in range(3):
            s += str(np.real(np.abs(param[i])).reshape(3,3)[j,:])
            if shx != None:
                s += "  " + str(np.abs(shx[inter].reshape(3,3)[j,:])) + "  " + str((np.abs(shx[inter]) - np.abs(np.real(param[i]))).reshape(3,3)[j,:]) + "\n"
            else:
                s += "\n"
    
    if sigma != None:
        s += "sigma = {: .4f}\n".format(sigma)

    return s

def gen_fit_summary(coeff, fit_obj, method, fmin, sigma=None, **kwargs):
    r"""
    Create a string summarizing a crystal-field Hamiltonian fitting run.

    Parameters
    ----------
    coeff : np.ndarray
        Contains the fitted interaction coefficients.
    fit_obj : EFitRunner, MevFitRunner, or ESHFitRunner
        Must have __iter__ method that iterates over tensors corresponding to
        parameters.
    method : str
        The optimization algorithm used for the fit.
    sigma : float, optional
        The total uncertainty for both the energy level and spin Hamiltonian
        fits; must be specified if 'cov'=True in kwargs.
    kwargs: dict
        Additional, optimization algorithm specific, settings to print.

    """
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})
    cov = None

    s = "Fitting summary\n"
    s+= "===============\n\n"
    
    heading = "Tensor name          Fitted coeff        Initial coeff           Difference"
    if kwargs['cov']:
        if sigma == None:
            raise ValueError("'cov' kwarg is specified as True, but no sigma is not provided.")
        cov = np.linalg.inv(kwargs['cov_inv'])
        heading += "    Uncertainty"
    if 'bounds' in kwargs:
        heading += "      Lower bounds       Upper bounds"
    if 'stepsize' in kwargs:
        heading += "          Stepsize"
    heading += "\n"

    s += uline_char(heading)
    for i,p in enumerate(fit_obj):
        co = coeff[fit_obj.h.index(p)]
        if co.imag == 0:
            co = co.real
        s += "{0:<12} {1: >20.4f} {2: >20.4f} {3: >20.4f}".format(p.name+":",
                co, fit_obj.param_list[i], co-fit_obj.param_list[i])
        if kwargs['cov']:
            s += "{0: >15.0f}".format(np.sqrt(np.abs(cov[i,i]))*sigma)
        if 'bounds' in kwargs:
            s += "{0: >18.0f} {1: >18.0f}".format(kwargs['bounds'][p.name][0],
                    kwargs['bounds'][p.name][1])
        if 'stepsize' in kwargs:
            s += "{0: >18.0f}".format(kwargs['stepsize'][p.name])
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

    if method == 'basinhopping':
        kwargs['naccept'] = kwargs['retval']
        del kwargs['retval']

    s += "\n" + uline_char("Optimization routine details:\n")
    s += "{0:<20} {1: <}\n".format("fmin:", fmin)
    s += "{0:<20} {1: <}\n".format("method:", method)
    for k in kwargs:
        s += "{0:<20} {1: <}\n".format(k+":", kwargs[k])

    return s

def e_fit_sigma(e, ex, ndof):
    r"""
    Calculate the standard deviation of an energy level fit assuming a model
    fit.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.

    Parameters
    ----------
    e : np.ndarray
        The energies of fitted levels.
    ex : np.ndarray
        2 by n dimensional array, with n the number of available experimental
        energy levels. The first column contains energy level indices starting
        at 1, and the second column contains corresponding experimental energy
        level values. 
    ndof : int
        The number of degrees of freedom of the chi-squared distribution, that
        is, the number of experimental data points minus the number of
        parameters.
    """
    # Experimental level index.
    ex_li = np.array(ex[:,0], dtype=int)-1
    try:
        sigma = np.sqrt(np.sum((e[ex_li] - ex[:,1])**2))/ndof
    except:
        raise IndexError("Level index in experimental energies file is out of range.")
    
    return sigma

def sh_fit_sigma(param, sh, shx, ndof):
    r"""
    Calculate the standard deviation of a spin Hamiltonian fit assuming a model
    fit.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.

    Parameters
    ----------
    param : list
        Elements must be `3 \times 3` np.ndarrays corresponding to the spin
        Hamiltonian parameters.  Output from
        :func:`cfl.SpinHamiltonian.calc_param` is appropriately formated to be
        passed as param.
    sh : SpinHamiltonian
        Generally the spin Hamiltonian object used to generate the param list. 
    shx : dict
        Specifies the experimental spin Hamiltonian data.  Valid keys are
        'zeeman', 'hyperfine', and 'quadrupole'.  Values should be `3 \times 3`
        np.ndarrays corresponding to the experimental spin Hamiltonian tensor.
    ndof : int
        The number of degrees of freedom of the chi-squared distribution, that
        is, the number of experimental data points minus the number of
        parameters.
    """

    chi2 = 0
    for i,inter in enumerate(sh.interactions):
        chi2 += np.sum((np.abs(shx[inter]) - np.abs(np.real(param[i])))**2)

    sigma = np.sqrt(chi2/ndof)

    return sigma

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
        if type(coeff[c]) != type(bounds[c]):
            raise ValueError("Coefficient %s is of type %s while corresponding bounds element is of type %s" % (c, type(coeff[c]), type(bounds[c])))
        bal_b[c] = (coeff[c]-bounds[c], coeff[c]+bounds[c])

    return bal_b
