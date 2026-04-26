#!/usr/bin/env python
# Filename = import_sljm.py
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
import os
import re
from typing import Any, Generator, List, Optional

import numpy as np
from scipy.sparse import csr_matrix

import pycf.cfl as cfl
from pycf.cfl_util import term2L


def get_tensor_dim(source: Any) -> Generator[List[tuple], None, None]:
    "Generator for extracting tensor dimensions from ``*.mi_`` files."
    parse = False
    for line in source:
        if line.startswith("CREATED"):
            parse = True
            yield ""
        elif parse:
            yield re.findall(r"(\w+)\s+(\d+)", line)
        else:
            yield ""


def get_state_number(source: Any) -> Generator[List[int], None, None]:
    "Generator for extracting the number of states from a ``*.st_`` file."
    parse = False
    done = False
    for line in source:
        if line.startswith("CREATED"):
            parse = True
            yield [0]
        elif done:
            return
        elif parse:
            done = True
            yield re.findall(r"(\d+)\s+STATES", line)
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
    sl_name : string,optional
        When loading matrix elements for intensity calculations, the state label
        file is usually still the same as for cfl/cfit when following Mike's
        convention.  This optional argument allows one to specify the state
        label file with a different base name than the intensity matrix element
        file.  It should be specified without the extension, like name above.
    """

    def __init__(self, name: str, sl_name: Optional[str] = False) -> None:
        # Create list of tuples of the form ('tensor_name', 'tensor_dim')
        tensor_dims = []
        with open("%s.mi_" % name, "r") as f:
            for td in get_tensor_dim(f):
                tensor_dims += td
        if not sl_name:
            sl_name = name
        # Get the number of states and state labels from *.st file.
        # Initialise dim to 0 so it is defined even if the file is empty.
        dim = 0
        with open("%s.st_" % sl_name, "r") as f:
            for d in get_state_number(f):
                # Guard against empty match (regex didn't find STATES line).
                if d:
                    dim = int(d[0])
        if not dim:
            raise RuntimeError(
                "Could not find state count in %s.st_; "
                "STATES line missing or in unexpected format." % sl_name
            )
        with open("%s.st_" % sl_name, "r") as f:
            state_labels = re.findall(
                r"[^[]*\[(\(?2?F?\s?\)?)(\d+)(\w)(\d?)\s*(\d+)\s*([\d-]*),?\s*([\d-]*)[)>]",
                f.read(),
            )
        if not state_labels:
            raise RuntimeError(
                "No state labels found in %s.st_; "
                "file may be corrupt or the regex does not match its format." % sl_name
            )
        if dim != len(state_labels):
            raise RuntimeError(
                "Parsing state labels file %s.st_ failed.  This "
                "is indicative of either a limitation of the parsing regex,"
                " or a corrupt *.st_ file." % sl_name
            )
        # Index of regex group for each label type.
        gi = {"S": 1, "L": 2, "J": 4, "M": 5, "I": 6, "T": 3, "F": 0}
        label_key = ["L", "J", "M"]
        for state_label in state_labels:
            label_key += [k for k in gi if (state_label[gi[k]] and k not in label_key)]
        # FIXME: T, which was intended as 'seniority', is labeled as X in
        # Nielson and Koster; should adopt this, but make sure if I change it
        # here nothing else get's messed up.
        # Rearrange label key to cannonical order.
        sort_key = ["T", "F", "S", "L", "J", "M", "I"]
        label_key.sort(key=lambda k: sort_key.index(k))
        sl = []
        for state_label in state_labels:
            lk = []
            for k in label_key:
                # Convert total orbital angular momentum label to numerical
                # label.
                if k == "L":
                    label = term2L(state_label[gi[k]])
                elif k == "T":
                    # Set T labels to zero for states that don't specify it.
                    if not state_label[gi[k]]:
                        label = 0
                    else:
                        label = int(state_label[gi[k]])
                # Only 2 F states seem to be labeled for F->D, so we set those F
                # labels to 1, all others to 0.
                elif k == "F":
                    if state_label[gi[k]]:
                        label = 1
                    else:
                        label = 0
                else:
                    label = int(state_label[gi[k]])
                lk += [label]
            sl += [lk]
        label_key = "".join(label_key)
        # Generate a dictionary with keys for each tensor and lists of the form
        # [row, col, matel].  These are then used to create Scipy sparse CSR
        # matrices.
        filepath = "%s.txt" % name
        if not os.path.exists(filepath):
            raise IOError("Required data file not found: %s" % filepath)
        data = np.loadtxt(filepath, skiprows=2, ndmin=2)
        i = 0
        tensor_elements = {}
        tensor_matrices = {}
        for td in tensor_dims:
            tensor_elements[td[0]] = data[i : i + int(td[1]), :]
            i += int(td[1])
            tensor_matrices[td[0]] = csr_matrix(
                (
                    tensor_elements[td[0]][:, 2],
                    (
                        tensor_elements[td[0]][:, 0] - 1,
                        tensor_elements[td[0]][:, 1] - 1,
                    ),
                ),
                shape=(dim, dim),
                dtype=np.complex128,
            )
        # Create tensors; since tensors use hermitian matrix compressed row
        # storage we do not require the lower triangular half.
        sl = cfl.StateLabels(label_key, sl)
        tensors = {}
        for t in tensor_matrices:
            if tensor_matrices[t].nnz == 0:
                print("Warning: all matrix elements of %s are zero." % t)
            tensors[t] = cfl.Tensor(
                t,
                np.ascontiguousarray(tensor_matrices[t].indptr, dtype=np.intc),
                np.ascontiguousarray(tensor_matrices[t].indices, dtype=np.intc),
                np.ascontiguousarray(tensor_matrices[t].data),
                sl,
            )
        # Create convenience aliases only when the source tensors are available.
        if "MAG11" in tensors and "MAG10" in tensors:
            # MFR: Changed the signs for MAGX and MAGY to the standard definitions
            # of the spherical tensor components. This affects eigenvector phases
            # and therefore transition intensities, but not eigenvalues.
            tensors["MAGX"] = -1.0 / np.sqrt(2) * tensors["MAG11"]
            tensors["MAGX"].name = "MAGX"
            tensors["MAGY"] = complex(0, 1) / np.sqrt(2) * tensors["MAG11"]
            tensors["MAGY"].name = "MAGY"
            tensors["MAGZ"] = tensors["MAG10"]
            tensors["MAGZ"].name = "MAGZ"
        if "AHYP" in tensors and "BHYP" in tensors:
            tensors["HYP"] = tensors["AHYP"] - np.sqrt(10) * tensors["BHYP"]
            tensors["HYP"].name = "HYP"
        self.tensors = tensors
        self.__dict__.update(tensors)

    def __iter__(self) -> Generator[Any, None, None]:
        for t in self.tensors:
            yield self.__dict__.get(t)

    def print_names(self) -> None:
        r"""
        Print the names of all the tensors that have been loaded.
        """
        for t in self.tensors:
            print(t)
