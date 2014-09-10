#!/usr/bin/env python

from __future__ import division
import numpy as np
import re

import pycf.cfl as cfl

def get_tensor_dim(source):
    "Generator for extracting tensor dimensions from '*.mi_' files."
    parse = False
    for line in source:
        if line.startswith("CREATED"):
            parse = True
            yield ''
        elif parse:
            yield re.findall(r'(\w+)\s+(\d+)', line)
        else: 
            yield ''

def get_state_number(source):
    "Generator for extracting the number of states from a '.*st' file."
    parse = False
    done = False
    for line in source:
        if line.startswith("CREATED"):
            parse = True
            yield [0]
        elif done:
            raise StopIteration
        elif parse:
            done = True
            yield re.findall(r'(\d+)\s+STATES', line)
        else:
            yield [0]

class ImportSLJM:
    r"""
    Import the matrix elements and state labels from an SLJM calc plain text file. 

    Parameters
    ----------
    name : string
        The path/name of the SLJM calc output files, specifically, the files
        'name.txt' containing the matrix elements in plain text, and 'name.mi_'
        containing the tensor dimensions.
    dim : int
        The dimension of the tensors.
    """
    def __init__(self, name):
        # Create list of tuples of the form ('tensor_name', 'tensor_dim')
        tensor_dims = []
        with open("%s.mi_" % name, 'r' ) as f:
            for td in get_tensor_dim(f):
                tensor_dims += td

        # Get the number of states and state labels from *.st file.
        with open("%s.st_" % name, 'r') as f:
            for d in get_state_number(f):
                dim = int(d[0])

        with open("%s.st_" % name, 'r') as f:
            state_labels = re.findall(r'[^[]+(\[[\w\s,-]+[)>])', f.read())
        if dim != len(state_labels):
            raise RuntimeError("Parsing state labels file %s.st_ failed.  This "
                    "is indicative of either a limitation of the parsing regex,"
                    " or a corrupt *.st_ file." % name)

        data = np.loadtxt('%s.txt' % name, skiprows = 2)
        # Generate a dictionary of lists, with list elements [row, col, matel].
        i = 0
        tensor_elements = {}
        tensor_matrices = {}
        for td in tensor_dims:
            tensor_elements[td[0]] = data[i:i+int(td[1]), :]
            i += int(td[1])
            # Create zero matrix for each tensor. 
            tensor_matrices[td[0]] = np.zeros((dim, dim), dtype=np.complex128)
        
        # Populate tensor matrices with non-zero matrix elements.
        for t in tensor_matrices:
            for e in tensor_elements[t]:
                tensor_matrices[t][np.real(e[0])-1, np.real(e[1])-1] = e[2]
        
        if 'MAG11' in tensor_matrices and 'MAG10' in tensor_matrices:
            tensor_matrices['MAGX'] = tensor_matrices['MAG11'] * 1/np.sqrt(2)
            tensor_matrices['MAGY'] = tensor_matrices['MAGX'] * np.complex(0, -1)
            tensor_matrices['MAGZ'] = tensor_matrices['MAG10']
       
        # Create tensors; since tensors use hermitian matrix compressed row
        # storage we do not require the lower triangular half.
        sl = cfl.StateLabels(state_labels)
        tensors = {}
        for t in tensor_matrices:
            tensors[t] = cfl.Tensor(t, tensor_matrices[t], sl)

        self.tensors = tensors

    def print_available_tensors(self):
        for t in self.tensors:
            print(t)

    def get_tensors(self):
        return self.tensors

def uline_char(s):
    "Underline all non-whitespace characters in a string."
    ul = ""
    for c in s:
        if c.isspace():
            ul += " "
        else:
            ul += "-"
    if s[-1::] == "\n":
        return s + ul + "\n"
    else:
        return s + ul

def uline_str(s):
    "Underline all characters of a string."
    if s[-1::] == "\n":
        ul = "-"*(len(s)-1)
        return s + ul +"\n"
    else:
        ul = "-"*len(s)
        s += "\n"
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
    heading = "Lev.  " + ("Percentage                  " + "State" + " "*(len(labels[0])-4))*nstates + "     Theory"
    if ex != None:
        heading += "   Experiment   Exp-Theory \n"
    else:
        heading += " \n"
    
    s += uline_char(heading)
    ex_i=0
    for i in range(len(z)):
        line = "{0:<6}".format(i+1)
        N = np.sum(np.abs(z[i, :]))
        for j in range(nstates):
            si = sort_list[i][j]
            line += "({0: .3f}) {1:5.1%} {2:>5} {3} ".format(z[i,si], np.abs(z[i,si])/N, si+1, labels[si])
        s += line + " {: >10.4f}".format(w[i])
        if ex != None:
            if ex[ex_i,0] == i:
                s += "   {: >10.4f}   {: >10.4f}".format(ex[ex_i,1], ex[ex_i,1]-w[i]) + "\n"
                ex_i += 1
            else:
                s += "       --           --\n"
        else:
            s += "\n"
    return s

def gen_sh_summary(param, inter, shx=None):
    np.set_printoptions(precision=4)
    s = "Spin Hamiltonian summary\n"
    s+= "========================\n\n"
    for i in inter:
        s += uline_str("%s interaction\n" % i)
        if shx != None:
            s += uline_char("Theory                     Experiment              Exp-Theory\n")
        else:
            s += uline("Theory\n")
        for j in range(3):
            s += str(np.real(param[0]).reshape(3,3)[j,:])
            if shx != None:
                s += "  " + str(shx[i].reshape(3,3)[j,:]) + "  " + str((shx[i] - np.real(param[0])).reshape(3,3)[j,:]) + "\n"
            else:
                s += "\n"

    return s


