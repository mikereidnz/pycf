#!/usr/bin/env python
# Filename = cfl_util.py

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


def gen_e_summary(w, z, labels, ex=None, nstates=2):
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
    ex : np.ndarray
        A 2 by m array, specifing the experimental energy levels, with m the
        number of available experimental levels.  The first column specifies the
        corresponding index of the complete eigenvalue vector, and the second
        column contains the actual energy level values.
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
                ex_i += 1
            else:
                s += "         --             --\n"
        else:
            s += "\n"
    return s

def gen_sh_summary(param, inter, shx=None):
    np.set_printoptions(formatter={'float': lambda x: '{:8.5f}'.format(x)})
    s = "Spin Hamiltonian summary\n"
    s+= "========================\n\n"
    for i in inter:
        s += uline_char("%s interaction\n" % i)
        if shx != None:
            s += uline_char("Theory                        Experiment                    Difference\n")
        else:
            s += uline("Theory\n")
        for j in range(3):
            s += str(np.real(param[0]).reshape(3,3)[j,:])
            if shx != None:
                s += "  " + str(shx[i].reshape(3,3)[j,:]) + "  " + str((shx[i] - np.real(param[0])).reshape(3,3)[j,:]) + "\n"
            else:
                s += "\n"

    return s

def gen_fit_summary(coeff, param_indices, param_initial, method, fmin, bounds, **kwargs):
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
    s = "Fitting summary\n"
    s+= "===============\n\n"

    heading = "Tensor name             Fitted coeff         Initial coeff            Difference"
    if bounds != None:
        heading += "   Lower bounds   Upper bounds\n"
    else:
        heading += "\n"

    s += uline_char(heading)
    for i in range(len(param_initial)):
        s += "{0:<15} {1: >20.4f} {2: >21.4f} {3: >21.4f}".format(param_initial[i][1]+":", coeff[param_indices[i]], param_initial[i][0], coeff[param_indices[i]]-param_initial[i][0])
        if bounds != None:
            s += "{0: >15.0f} {1: >14.0f}\n".format(bounds[param_initial[i][1]][0], bounds[param_initial[i][1]][1])
        else:
            s += "\n"

    
    s += "\n" + uline_char("Optimization routine details:\n")
    s += "{0:<20} {1: <}\n".format("fmin:", fmin)
    s += "{0:<20} {1: <}\n".format("method:", method)
    for k in kwargs:
        s += "{0:<20} {1: <}\n".format(k+":", kwargs[k])

    return s
