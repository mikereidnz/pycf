#!/usr/bin/env python
# Filename = import_sljm.py

#   Copyright (C) 2014-2015 Sebastian Horvath (sebastian.horvath@gmail.com)
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
from scipy.sparse import csr_matrix
import re
import cfl
from cfl_util import *

def get_tensor_dim(source):
    "Generator for extracting tensor dimensions from ``*.mi_`` files."
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
    "Generator for extracting the number of states from a ``*.st_`` file."
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

class ImportSLJM(object):
    r"""
    Import the matrix elements and state labels from an SLJM calc plain text file. 

    Parameters
    ----------
    name : string
        The path/name of the SLJM calc output files, specifically, the files
        ``name.txt`` containing the matrix elements in plain text, ``name.mi_``
        containing the tensor dimensions, and the states file ``name.st_``.
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
            state_labels = re.findall(r'[^[]*\[(\(?2?F?\s?\)?)(\d+)(\w)(\d?)\s*(\d+)\s*([\d-]*),?\s*([\d-]*)[)>]', f.read())
        if dim != len(state_labels):
            raise RuntimeError("Parsing state labels file %s.st_ failed.  This "
                    "is indicative of either a limitation of the parsing regex,"
                    " or a corrupt *.st_ file." % name)
       
        # Index of regex group for each label type.
        gi = {'S':1, 'L':2, 'J':4, 'M':5, 'I':6, 'T':3, 'F': 0}
        label_key = ['L', 'J', 'M']
        for l in state_labels:
            label_key += [k for k in gi if (l[gi[k]] and k not in label_key)]

        # Rearrange label key to cannonical order.
        sort_key = ['F', 'S', 'L', 'J', 'M', 'I', 'T']
        label_key.sort(key=lambda l: sort_key.index(l))
        
        sl = []
        for l in state_labels:
            lk = []
            for k in label_key:
                # Convert total orbital angular momentum label to numerical
                # label. 
                if k == 'L':
                    label = term2L(l[gi[k]])
                elif k == 'T':
                    # Set T labels to zero for states that don't specify it. 
                    if not l[gi[k]]:
                        label = 0
                # Only 2 F states seem to be labeled for F->D, so we set those F
                # labels to 1, all others to 0. 
                elif k == 'F':
                    if l[gi[k]]:
                        label = 1
                    else:
                        label = 0
                else:
                    label = int(l[gi[k]])
                lk += [label]
            sl += [lk]

        label_key = "".join(label_key) 

        # Generate a dictionary with keys for each tensor and lists of the form
        # [row, col, matel].  These are then used to create Scipy sparse CSR
        # matrices. 
        data = np.loadtxt('%s.txt' % name, skiprows = 2)
        i = 0
        tensor_elements = {}
        tensor_matrices = {}
        for td in tensor_dims:
            tensor_elements[td[0]] = data[i:i+int(td[1]), :]
            i += int(td[1])
            tensor_matrices[td[0]] = csr_matrix((tensor_elements[td[0]][:,2], (tensor_elements[td[0]][:,0]-1, 
                tensor_elements[td[0]][:,1]-1)), shape=(dim, dim), dtype=np.complex128)
        
        # Create tensors; since tensors use hermitian matrix compressed row
        # storage we do not require the lower triangular half.
        sl = cfl.StateLabels(label_key, sl)
        tensors = {}
        for t in tensor_matrices:
            if tensor_matrices[t].nnz == 0:
                print("Warning: all matrix elements of %s are zero." % t) 
            tensors[t] = cfl.Tensor(t, np.ascontiguousarray(tensor_matrices[t].indptr), 
                    np.ascontiguousarray(tensor_matrices[t].indices), np.ascontiguousarray(tensor_matrices[t].data), sl)
       
        try:
            tensors['MAGX'] = 1/np.sqrt(2) * tensors['MAG11']
            tensors['MAGX'].name = 'MAGX'
            tensors['MAGY'] = np.complex(0, -1) * tensors['MAGX']
            tensors['MAGY'].name = 'MAGY'
            tensors['MAGZ'] = tensors['MAG10']
            tensors['MAGZ'].name = 'MAGZ'
        except:
            pass
        try:
            tensors['HYP'] = tensors['AHYP'] - np.sqrt(10) * tensors['BHYP']
            tensors['HYP'].name = 'HYP'
        except: 
            pass

        self.tensors = tensors
        self.__dict__.update(tensors)

    def print_names(self):
        r"""
        Print the names of all the tensors that have been loaded.

        """
        for t in self.tensors:
            print(t)
