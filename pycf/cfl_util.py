#!/usr/bin/env python

from __future__ import division
import numpy as np
import re

import pycf.cfl as cfl

def get_tensor_dim(source):
    "Generator for extracting tensor dimensions from '*.mi_' files."
    match = False
    for line in source:
        if line.startswith("CREATED"):
            yield ''
            match = True
        elif match:
            yield re.findall(r'(\w+)\s+(\d+)', line)
        else: 
            yield ''

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
    def __init__(self, name, dim):
        # Create list of tuples of the form ('tensor_name', 'tensor_dim')
        tensor_dims = []
        with open("%s.mi_" % name, 'r' ) as source:
            for td in get_tensor_dim(source):
                tensor_dims += td

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
       
        # Create tensors; since tensors use compressed row storage for hermitian
        # matrices, we do not require the lower triangular half.
        tensors = {}
        for t in tensor_matrices:
            tensors[t] = cfl.Tensor(t, tensor_matrices[t])

        self.tensors = tensors

    def print_available_tensors(self):
        for t in self.tensors:
            print(t)

    def get_tensors(self):
        return self.tensors


