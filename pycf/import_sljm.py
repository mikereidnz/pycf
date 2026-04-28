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
from collections.abc import Mapping as MappingABC
from typing import Any, Generator, List, Mapping, Optional

import numpy as np
from scipy.sparse import csr_matrix, issparse, triu

import pycf.cfl as cfl
from pycf.cfl_util import term2L

# Attribute names on ImportTensors / ImportSLJM that must not be shadowed by
# user-supplied tensor names when ``expose_attrs=True``.
_RESERVED_TENSOR_NAMES = frozenset(
    {
        "tensors",
        "states",
        "label_key",
        "print_names",
        "_wrapped",
    }
)


class ImportTensors(object):
    r"""
    Wrap pre-parsed states and matrix elements as :class:`cfl.Tensor` objects.

    This class decouples matrix-element ingestion from any particular file
    format. It accepts a state-label matrix and a dictionary of tensors as
    NumPy arrays or SciPy sparse matrices, and returns the same in-memory
    collection of :class:`cfl.Tensor` objects that :class:`ImportSLJM`
    produces from the legacy jmcalc text outputs.

    Parameters
    ----------
    label_key : str
        Canonical-order label key (one character per label column, e.g.
        ``"LJM"`` or ``"SLJMI"``). Half-integer quantum numbers are encoded
        as doubled integers in ``states``.
    states : array-like, shape (N, len(label_key))
        Integer state-label matrix. Rows are states, columns correspond to
        characters of ``label_key``. A 1-D input is allowed only when
        ``len(label_key) == 1``.
    tensors : Mapping[str, numpy.ndarray | scipy.sparse.spmatrix]
        Mapping from tensor name to an N x N matrix. Dense ndarrays and any
        SciPy sparse format are accepted; all are converted to
        upper-triangle Hermitian CSR with ``complex128`` data before being
        handed to :class:`cfl.Tensor`.
    storage : {"full", "upper"}, optional
        Declares the storage convention of the input matrices.

        - ``"full"`` (default): inputs are full Hermitian; the upper triangle
          is taken via :func:`scipy.sparse.triu`. Dense input is also
          validated for Hermiticity (subject to ``check_hermitian``).
        - ``"upper"``: caller promises the matrices already store only the
          upper triangle in Hermitian compressed-row form. The legacy
          :class:`ImportSLJM` path uses this because the ``*.txt`` files
          contain upper-triangle elements only.

        Crystal-field and spin-Hamiltonian operators are Hermitian, so the
        underlying C layer (``cfl.Tensor``) requires upper-triangle
        Hermitian compressed-row storage. The ``storage`` parameter exists
        to protect new callers who would otherwise pass a full sparse
        Hermitian matrix and silently double-count off-diagonal elements.
    add_aliases : bool, optional
        Default ``False``. When ``True`` and the corresponding source
        tensors are present, synthesise the rare-earth-specific convenience
        aliases ``MAGX``, ``MAGY``, ``MAGZ`` (from ``MAG10``/``MAG11``) and
        ``HYP`` (from ``AHYP``/``BHYP``). Raises :class:`ValueError` if any
        alias name collides with a user-supplied tensor.
        :class:`ImportSLJM` passes ``True``.
    expose_attrs : bool, optional
        Default ``True``. Mirror tensors as attributes on ``self`` (the
        legacy :class:`ImportSLJM` behaviour). Tensor names that collide
        with reserved attribute names raise :class:`ValueError` when this
        is enabled.
    check_hermitian : bool, optional
        Default ``True``. Validate dense input matrices are Hermitian using
        :func:`numpy.allclose`. Has no effect for sparse input.
    warn_zero : bool, optional
        Default ``True``. Print a warning if a supplied tensor has no
        non-zero elements (matches the legacy :class:`ImportSLJM`
        behaviour).
    """

    def __init__(
        self,
        label_key: str,
        states: Any,
        tensors: Mapping[str, Any],
        storage: str = "full",
        add_aliases: bool = False,
        expose_attrs: bool = True,
        check_hermitian: bool = True,
        warn_zero: bool = True,
    ) -> None:
        if not isinstance(label_key, str) or not label_key:
            raise ValueError("label_key must be a non-empty string")
        if storage not in ("full", "upper"):
            raise ValueError("storage must be 'full' or 'upper'")

        states_arr = np.asarray(states, dtype=np.int32)
        nkey = len(label_key)
        if states_arr.ndim == 1 and nkey == 1:
            states_arr = states_arr.reshape(-1, 1)
        if states_arr.ndim != 2:
            raise ValueError("states must be 2-D with shape (N, len(label_key))")
        if states_arr.shape[1] != nkey:
            raise ValueError(
                "states has %d columns but label_key has %d characters"
                % (states_arr.shape[1], nkey)
            )
        dim = states_arr.shape[0]
        if dim == 0:
            raise ValueError("states must contain at least one row")

        if not isinstance(tensors, MappingABC):
            raise TypeError("tensors must be a mapping of name -> matrix")

        # Reserved-name check is independent of expose_attrs because the
        # alias-collision check below would also be ambiguous if a user
        # supplied a tensor literally named "tensors".
        if expose_attrs:
            collisions = set(tensors).intersection(_RESERVED_TENSOR_NAMES)
            if collisions:
                raise ValueError(
                    "Tensor names collide with reserved attributes: %s" % sorted(collisions)
                )

        # Convert each matrix to upper-triangle CSR complex128.
        tensor_matrices = {}
        for name, mat in tensors.items():
            if not isinstance(name, str) or not name:
                raise ValueError("tensor names must be non-empty strings")
            tensor_matrices[name] = self._normalise_matrix(name, mat, dim, storage, check_hermitian)

        # Build StateLabels.
        sl_list = [list(row) for row in states_arr.tolist()]
        sl_obj = cfl.StateLabels(label_key, sl_list)

        # Build cfl.Tensor objects.
        cfl_tensors = {}
        for name, m in tensor_matrices.items():
            if warn_zero and m.nnz == 0:
                print("Warning: all matrix elements of %s are zero." % name)
            cfl_tensors[name] = cfl.Tensor(
                name,
                np.ascontiguousarray(m.indptr, dtype=np.intc),
                np.ascontiguousarray(m.indices, dtype=np.intc),
                np.ascontiguousarray(m.data),
                sl_obj,
            )

        if add_aliases:
            self._apply_aliases(cfl_tensors)

        self.label_key = label_key
        self.states = sl_obj
        self.tensors = cfl_tensors
        if expose_attrs:
            self.__dict__.update(cfl_tensors)

    @staticmethod
    def _normalise_matrix(
        name: str,
        mat: Any,
        dim: int,
        storage: str,
        check_hermitian: bool,
    ) -> "csr_matrix":
        """Validate, cast, and (if storage='full') upper-triangle a matrix."""
        if issparse(mat):
            sp = mat.tocsr().astype(np.complex128)
            if sp.shape != (dim, dim):
                raise ValueError(
                    "tensor %r has shape %s, expected (%d, %d)" % (name, sp.shape, dim, dim)
                )
            if storage == "full":
                sp = triu(sp, format="csr")
            return sp.tocsr()

        arr = np.asarray(mat)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("tensor %r must be a square 2-D matrix" % name)
        if arr.shape != (dim, dim):
            raise ValueError(
                "tensor %r has shape %s, expected (%d, %d)" % (name, arr.shape, dim, dim)
            )
        arr = arr.astype(np.complex128, copy=False)
        if check_hermitian and not np.allclose(arr, arr.conj().T):
            raise ValueError(
                "tensor %r is not Hermitian; pass check_hermitian=False to "
                "bypass, or set storage='upper' if it already stores only "
                "the upper triangle" % name
            )
        if storage == "full":
            sp = triu(csr_matrix(arr), format="csr")
        else:
            sp = csr_matrix(arr)
        return sp

    @staticmethod
    def _apply_aliases(tensors_dict: dict) -> None:
        """Synthesise MAGX/Y/Z and HYP convenience aliases in-place."""
        if "MAG11" in tensors_dict and "MAG10" in tensors_dict:
            collisions = {"MAGX", "MAGY", "MAGZ"} & set(tensors_dict)
            if collisions:
                raise ValueError(
                    "alias synthesis would overwrite supplied tensors: %s" % sorted(collisions)
                )
            # MFR: signs match the standard spherical tensor component
            # definitions; affects eigenvector phases (and therefore
            # transition intensities) but not eigenvalues.
            tensors_dict["MAGX"] = -1.0 / np.sqrt(2) * tensors_dict["MAG11"]
            tensors_dict["MAGX"].name = "MAGX"
            tensors_dict["MAGY"] = complex(0, 1) / np.sqrt(2) * tensors_dict["MAG11"]
            tensors_dict["MAGY"].name = "MAGY"
            tensors_dict["MAGZ"] = tensors_dict["MAG10"]
            tensors_dict["MAGZ"].name = "MAGZ"
        if "AHYP" in tensors_dict and "BHYP" in tensors_dict:
            if "HYP" in tensors_dict:
                raise ValueError("alias synthesis would overwrite supplied tensor: HYP")
            tensors_dict["HYP"] = tensors_dict["AHYP"] - np.sqrt(10) * tensors_dict["BHYP"]
            tensors_dict["HYP"].name = "HYP"

    def __iter__(self) -> Generator[Any, None, None]:
        for t in self.tensors:
            yield self.tensors[t]

    def print_names(self) -> None:
        r"""Print the names of all the tensors that have been loaded."""
        for t in self.tensors:
            print(t)


def get_tensor_dim(source: Any) -> Generator[List[tuple], None, None]:
    "Generator for extracting tensor dimensions from ``*.mi_`` files."
    parse = False
    for line in source:
        if line.startswith("CREATED"):
            parse = True
            yield []
        elif parse:
            yield re.findall(r"(\w+)\s+(\d+)", line)
        else:
            yield []


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

    def __init__(self, name: str, sl_name: Optional[str] = None) -> None:
        # Create list of tuples of the form ('tensor_name', 'tensor_dim')
        tensor_dims: List[tuple] = []
        with open("%s.mi_" % name, "r") as f:
            for td_chunk in get_tensor_dim(f):
                tensor_dims += td_chunk
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
        label_key_list = ["L", "J", "M"]
        for state_label in state_labels:
            label_key_list += [k for k in gi if (state_label[gi[k]] and k not in label_key_list)]
        # FIXME: T, which was intended as 'seniority', is labeled as X in
        # Nielson and Koster; should adopt this, but make sure if I change it
        # here nothing else get's messed up.
        # Rearrange label key to cannonical order.
        sort_key = ["T", "F", "S", "L", "J", "M", "I"]
        label_key_list.sort(key=lambda k: sort_key.index(k))
        sl = []
        for state_label in state_labels:
            lk = []
            for k in label_key_list:
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
        label_key = "".join(label_key_list)
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
        # Delegate state-label construction, Tensor wrapping, and alias
        # synthesis to ImportTensors. The *.txt files contain only the
        # upper triangle of each (Hermitian) tensor, so storage="upper".
        # check_hermitian=False because the input is already upper-triangle
        # only and would fail a Hermitian check on a dense round-trip.
        # expose_attrs=False so that the reserved-name guard does not fire
        # on legacy file inputs (the legacy path tolerated tensor names
        # that shadow attributes; we preserve that behaviour here).
        self._wrapped = ImportTensors(
            label_key,
            sl,
            tensor_matrices,
            storage="upper",
            add_aliases=True,
            expose_attrs=False,
            check_hermitian=False,
            warn_zero=True,
        )
        self.tensors = self._wrapped.tensors
        self.__dict__.update(self._wrapped.tensors)

    def __iter__(self) -> Generator[Any, None, None]:
        for t in self.tensors:
            yield self.__dict__.get(t)

    def print_names(self) -> None:
        r"""
        Print the names of all the tensors that have been loaded.
        """
        for t in self.tensors:
            print(t)
