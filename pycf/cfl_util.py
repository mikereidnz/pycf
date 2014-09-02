#!/usr/bin/env python

from __future__ import division
import numpy as np
import re

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
        tensors = {}
        for td in tensor_dims:
            tensor_elements[td[0]] = data[i:i+int(td[1]), :]
            i += int(td[1])
            # Fill in zero matrix elements of final tensor dict.
            tensors[td[0]] = np.zeros((dim, dim), dtype=np.complex128)
        
        # Populate tensors with non-zero matrix elements.
        for t in tensors:
            for e in tensor_elements[t]:
                tensors[t][np.real(e[0])-1, np.real(e[1])-1] = e[2]

        tensors['MAGX'] = tensors['MAG11'] * 1/np.sqrt(2)
        tensors['MAGY'] = tensors['MAGX'] * np.complex(0, -1)
        tensors['MAGZ'] = tensors['MAG10']

        # Add Hermitian conjugate.
        for t in tensors:
            tensors[t] = tensors[t].conj().T - np.diag(np.diag(tensors[t])) + tensors[t]
        
        self.tensors = tensors

    def print_available_tensors(self):
        for t in self.tensors:
            print(t)

    def get_tensors(self):
        return self.tensors


