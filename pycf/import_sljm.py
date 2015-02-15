#!/usr/bin/env python
# Filename = import_sljm.py

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

import cfl

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
        
        # AHYP is the L.I part and BHYP is (sC2).I part
        if 'AHYP' in tensor_matrices and 'BHYP' in tensor_matrices:
            tensor_matrices['HYP'] = tensor_matrices['AHYP'] - np.sqrt(10) * tensor_matrices['BHYP']
       
        # Create tensors; since tensors use hermitian matrix compressed row
        # storage we do not require the lower triangular half.
        sl = cfl.StateLabels(state_labels)
        tensors = {}
        for t in tensor_matrices:
            tensors[t] = cfl.Tensor(t, tensor_matrices[t], sl)

        self.tensors = tensors
        self.__dict__.update(tensors)

    def print_names(self):
        r"""
        Print the names of all available the available tensors that have been loaded.

        """
        for t in self.tensors:
            print(t)
