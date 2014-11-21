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


def gen_e_summary(w, z, labels, ex=None, nstates=2, ndof=None):
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
        A 2 by m array, specifying the experimental energy levels, with m the
        number of available experimental levels.  The first column specifies the
        index of the corresponding entry in the complete eigenvalue vector, and
        the second column contains the energy level values.
    nstates : int, optional
        The number of constituent states to display for mixed states.
    ndof : int, optional
        The number of degrees of freedom of the chi-squared distribution, that
        is, the number of experimental data points minus the number of
        parameters.  If specified, in addition to ex, then the standard
        deviation will be added to the summary.
    """
    
    s = "Energy level summary\n"
    s+= "====================\n\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[i,:]))[::-1]]
    heading = "Lev.  " + ("Percentage                 " + "State" + " "*(len(labels[0])-4))*nstates + "       Theory"
    if ex != None:
        heading += "     Experiment     Difference \n"
    else:
        heading += " \n"
    
    s += uline_char(heading)
    ex_i=0
    for i in range(len(z)):
        line = "{0:<6}".format(i+1)
        N = np.sum(np.abs(z[i, :]))
        for j in range(nstates):
            si = sort_list[i][j]
            line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(z[i,si], np.abs(z[i,si])/N, si+1, labels[si])
        s += line + " {: >12.4f}".format(w[i])
        if ex != None:
            if ex[ex_i,0] == i:
                s += "   {: >12.4f}   {: >12.4f}".format(ex[ex_i,1], ex[ex_i,1]-w[i]) + "\n"
                if ex_i != len(ex)-1:
                    ex_i += 1
            else:
                s += "         --             --\n"
        else:
            s += "\n"

    if ex != None and ndof != None:
        s += "sigma = {: .4f}\n".format(e_fit_sigma(w, ex, ndof))

    return s

def gen_sh_summary(param, sh, shx=None, ndof=None):
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
    ndof : int, optional
        The number of degrees of freedom of the chi-squared distribution, that
        is, the number of experimental data points minus the number of
        parameters.  If specified, in addition to shx, then the standard
        deviation will be added to the summary.
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
            s += str(np.real(param[i]).reshape(3,3)[j,:])
            if shx != None:
                s += "  " + str(shx[inter].reshape(3,3)[j,:]) + "  " + str((shx[inter] - np.real(param[i])).reshape(3,3)[j,:]) + "\n"
            else:
                s += "\n"
    
        if shx != None and ndof != None:
            s += "sigma = {: .4f}\n".format(sh_fit_sigma(param, sh, shx, ndof))

    return s

def gen_fit_summary(coeff, param_indices, param_initial, method, fmin, **kwargs):
    r"""
    Create a string summarizing a crystal-field Hamiltonian fitting run.

    Parameters
    ----------
    coeff : np.ndarray
        Contains the fitted interaction coefficients.
    param_indices : dict
        Initial values of coefficients for tensors to be fit.
    param_initial : tuple
        The first element corresponds to the initial coefficient value and the
        second element corresponds to the tensor name.
    method : str
        The optimization algorithm used for the fit.
    kwargs: dict
        Additional, optimization algorithm specific, settings to print.

    """
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    s = "Fitting summary\n"
    s+= "===============\n\n"

    heading = "Tensor name          Fitted coeff        Initial coeff           Difference"
    if 'bounds' in kwargs:
        heading += "        Lower bounds         Upper bounds"
    if 'stepsize' in kwargs:
        heading += "  Specified stepsize"
    heading += "\n"

    s += uline_char(heading)
    for i in range(len(param_initial)):
        co = coeff[param_indices[i]]
        if co.imag == 0:
            co = co.real
        s += "{0:<12} {1: >20.4f} {2: >20.4f} {3: >20.4f}".format(param_initial[i][1]+":", co, 
                param_initial[i][0], co-param_initial[i][0])
        if 'bounds' in kwargs:
            s += "{0: >20.0f} {1: >20.0f}".format(kwargs['bounds'][param_initial[i][1]][0],
                    kwargs['bounds'][param_initial[i][1]][1])
        if 'stepsize' in kwargs:
            s += "{0: >20.0f}".format(kwargs['stepsize'][param_initial[i][1]])
        s += "\n"

    if 'bounds' in kwargs:
        del kwargs['bounds']
    if 'stepsize' in kwargs:
        del kwargs['stepsize']
  
    if kwargs['cov']:
        s += "\n" + uline_char("Covariance matrix:\n")
        try:
            cov = np.linalg.inv(kwargs['cov_inv'])
            s += str(cov) + "\n"
        except:
            s += "Singular covariance matrix; cannot invert\n"

        del kwargs['cov_inv']
    
    del kwargs['cov']

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
        A 2 by m array, specifying the experimental energy levels, with m the
        number of available experimental levels.  The first column specifies the
        index of the corresponding entry in the complete eigenvalue vector, and
        the second column contains the energy level values.
    ndof : int
        The number of degrees of freedom of the chi-squared distribution, that
        is, the number of experimental data points minus the number of
        parameters.
    """
    # Experimental level index.
    ex_li = np.array(ex[:,0], dtype=int)
    sigma = np.sqrt(np.sum((e[ex_li] - ex[:,1])**2))/ndof
    
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
        chi2 += np.sum((shx[inter] - np.real(param[i]))**2)

    sigma = np.sqrt(chi2/ndof)

    return sigma





