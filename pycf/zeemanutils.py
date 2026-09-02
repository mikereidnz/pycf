#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: pycf/zeemanutils.py
"""
Legacy Utility functions for Zeeman fits.
Energy levels and g values entered in files.

Now with log_ and slog_ output files generated automatically
and summaries of energies and g values.

The logfile functions are deprecated but are there for backward compatibility.

2019-10-10 Stop subtracting lowest energy
2021-12-12 Fix a minor problem with new pycf that caused error in printparam
2021-12-16 Note that you should create a new h when doing printing,
  so you don't mess with the fitting h set.
"""

import numpy as np
import copy as copy

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.cfl_util import *

import datetime


def setup_logfiles(filestr):
    # Input: filestr e.g. "ybzeeman.txt"
    # Writes timestamp to files and returns:
    #  log_datetime_filestr.txt   Full log
    #  slog_datetime_filestr.txt  Short Summary
    datetimestr = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[0:19]
    logfile = "log_" + datetimestr + "_" + filestr
    slogfile = "slog_" + datetimestr + "_" + filestr
    s = datetimestr + "\n\n"
    print(s)
    with open(logfile, "w") as f:
        f.write(s)
    with open(slogfile, "w") as f:
        f.write(s)

    return logfile, slogfile


def logprint(fname, out):
    # to save typing when appending logfile and slogfile
    with open(fname, "a") as f:
        f.write(out)


# Function for reading the g values
def load_g_values(B0, Bhat, fname, weight, exdata_list, h_list, weights_list, t_list, coeff, mu_b):
    # read data for particular B orientation Bhat and update:
    # exdata_list h_list weights_list
    # could reduce the printing...
    exg = np.loadtxt(fname, skiprows=1)
    print("B0    ", B0)
    print("Bhat  ", Bhat)
    print("fname ", fname)
    # array must be 2D, even if there is only one data point
    if exg.ndim == 1:
        exg = np.array([exg])
    # print("exg g values")
    # print(exg)
    # need to add a column to take difference
    exg2 = np.array([exg[..., 0], exg[..., 0] + 1, exg[..., 1]]).transpose()
    # print(exg2)
    exdelta = exg2
    exdelta[:, 2] *= mu_b * B0
    # print("exdelta magnetic splittings for B0")
    # print(exdelta)
    exdata_list += [cfl.ExData(exg2, "D")]
    h = cfl.Hamiltonian(t_list)
    coeff["MX"] = Bhat[0] * B0
    coeff["MY"] = Bhat[1] * B0
    coeff["MZ"] = Bhat[2] * B0
    h.set_coeff(coeff)
    h_list += [h]
    weights_list += [weight]
    coeff["MX"] = 0
    coeff["MY"] = 0
    coeff["MZ"] = 0
    return exg, exdata_list, h_list, weights_list


# Functions to print Summaries of Parameters, Energies and g values.
def print_params(coeff):
    # Print Parameters
    out = "\nParameters\n\n"
    # for p in sorted(coeff.iterkeys()): # fails on new pycf
    for p in coeff:
        if coeff[p] != 0:
            out += "'{0:<12}: ".format(p + "'")
            v = coeff[p]
            # if np.iscomplex(v):
            # if isinstance(v, complex):
            if isinstance(v, (complex, np.complexfloating)):
                out += "{: >24}".format(
                    "({:.4f}+{:.4f}j)".format(v.real, v.imag)
                    if v.imag >= 0
                    else "({:.4f}{:.4f}j)".format(v.real, v.imag)
                )
            else:
                out += "{: >24.4f}".format(float(v))
            out += "\n"

    out += "\n\n"
    return out


# end print_params


def print_energies_g(maxlev, h, coeff, B0, mu_b, ex, exgx, exgy, exgz):
    # calculate zero-field energies and g values
    # print them, along with exerimental data
    # set maxlev to 0 if you just want to print all states

    # calculate zero-field energies
    h.set_coeff(coeff)
    w, z = h.diag()
    # w = w -np.min(w)
    E0 = w
    # print("E0")
    # print(E0)

    # Calculate g values for field along x, y, z
    coeff["MX"] = B0
    h.set_coeff(coeff)
    w, z = h.diag()
    w = (w - np.min(w)) / B0 / mu_b
    coeff["MX"] = 0
    Ex = w

    coeff["MY"] = B0
    h.set_coeff(coeff)
    w, z = h.diag()
    w = (w - np.min(w)) / B0 / mu_b
    coeff["MY"] = 0
    Ey = w

    coeff["MZ"] = B0
    h.set_coeff(coeff)
    w, z = h.diag()
    w = (w - np.min(w)) / B0 / mu_b
    coeff["MZ"] = 0
    Ez = w

    # just changing the coefficient doesn't reset h!
    h.set_coeff(coeff)

    def find_e(n, ex):
        # ex is 2D: index, experimental value
        # if n is one of the indices, return True, value
        # if ex is [[]] then ex.size==0 so it is empty
        found = False
        ee = -1
        if ex.size > 0:
            for i in range(ex.shape[0]):
                if ex[i, 0] == n:
                    found = True
                    ee = ex[i, 1]
                    break
        return found, ee

    # Formatted energies and g values
    if maxlev < 1:
        maxlev = E0.size + 1
    # Note: the loop below accesses Ex[i] where i reaches maxlev-1 = E0.size.
    # This is safe only when E0.size is even (Kramers ions always satisfy this).
    # Do not use with odd-sized state spaces (non-Kramers ions).
    # may want to set this smaller to save paper...
    out = "\n"
    out += "  Lev."
    out += "     E         Eex       dE   "
    out += "     gx        gx_ex     dgx  "
    out += "     gy        gy_ex     dgy  "
    out += "     gz        gz_ex     dgz  "
    out += "\n\n"
    for i in range(1, maxlev, 2):
        out += "{: >5}".format(int(i))
        e = E0[i - 1]
        out += "{: >10.2f}".format(e)
        found, ee = find_e(i, ex)
        if found:
            out += "{: >10.2f}".format(ee)
            out += "{: >10.2f}".format(ee - e)
        else:
            out += "        --"
            out += "        --"

        e = Ex[i] - Ex[i - 1]
        out += "{: >10.2f}".format(e)
        found, ge = find_e(i, exgx)
        if found:
            out += "{: >10.2f}".format(ge)
            out += "{: >10.2f}".format(e / ge)
        else:
            out += "        --"
            out += "        --"

        e = Ey[i] - Ey[i - 1]
        out += "{: >10.2f}".format(e)
        found, ge = find_e(i, exgy)
        if found:
            out += "{: >10.2f}".format(ge)
            out += "{: >10.2f}".format(e / ge)
        else:
            out += "        --"
            out += "        --"

        e = Ez[i] - Ez[i - 1]
        out += "{: >10.2f}".format(e)
        found, ge = find_e(i, exgz)
        if found:
            out += "{: >10.2f}".format(ge)
            out += "{: >10.2f}".format(e / ge)
        else:
            out += "        --"
            out += "        --"

        out += "\n"

    out += "\n"
    return out


# end print_energies_g
