#!/usr/bin/env python
# Filename = cfl_util.py
"""
Utility functions for crystal field calculations and data presentation.
This module provides formatting, summary, and analysis helpers for crystal field
Hamiltonians and experimental data. Key functions include:
- Energy level summaries and formatting
- State label parsing and manipulation
- Experimental data handling and normalization
- Transition grouping and analysis
- Result printing and visualization
Used throughout pycf for formatting output and presenting crystal field
calculation results to users.
"""

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
import inspect
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pycf.__version__ import __build_comment__, __build_timestamp__, __version__
except ImportError as e:
    logging.warning("Could not import build metadata from pycf.__version__: %s", e)
    __version__ = "unknown"
    __build_timestamp__ = "unknown"
    __build_comment__ = ""
from math import fsum

from scipy.special import factorial  # type: ignore[import-untyped]


def uline_char(s: str) -> str:
    """Underline all non-whitespace characters in a string, except for single
    spaces between non-whitespace characters."""
    ul = ""
    end = len(s) - 1 if s.endswith("\n") else len(s)
    for i in range(end):
        if s[i].isspace():
            # A space gets an underline dash if it sits between two non-space chars
            if i > 0 and i < end - 1 and not s[i - 1].isspace() and not s[i + 1].isspace():
                ul += "-"
            else:
                ul += " "
        else:
            ul += "-"
    if s.endswith("\n"):
        return s + ul + "\n"
    else:
        return s + ul


def term2L(c: str) -> int:
    "Convert an L quantum number term character to its numerical value."
    try:
        return "SPDFGHIKLMNOQRTUV".index(c)
    except ValueError:
        raise ValueError("Unsupported L quantum number: {}.".format(c))


def L2term(i: int) -> str:
    "Convert an L quantum number numerical value to its term character."
    if i < 0:
        raise ValueError("Unsupported L quantum number: {}.".format(i))
    try:
        return "SPDFGHIKLMNOQRTUV"[i]
    except IndexError:
        raise ValueError("Unsupported L quantum number: {}.".format(i))


def format_state_label(li: int, labels: List[Any], label_key: str) -> str:
    """
    Format a single state label using the label_key convention.
    
    Parameters
    ----------
    li : int
        Index into the labels list
    labels : list
        List of state labels (each is a tuple/list of quantum numbers)
    label_key : str
        String specifying the format: positions determine which quantum numbers
        are S, L, J, M, I, T, F, X. E.g., "SLJM" means labels[li] = (S, L, J, M)
    
    Returns
    -------
    str
        Formatted label string like "|2F 7,  1>" or "|1,5D 4, -2>"
    """
    label = "|"
    for i, l in enumerate(labels[li]):
        if label_key[i] == "X":
            label += "{:d},".format(l)
        elif label_key[i] == "F":
            if l:
                label += "(2F)"
            else:
                label += "    "
        elif label_key[i] == "S":
            label += "{:d}".format(l)
        elif label_key[i] == "L":
            label += L2term(l)
        elif label_key[i] == "J":
            label += "{: >2d},".format(l)
        elif i < len(label_key) - 1:
            label += "{: >3d},".format(l)
        else:
            label += "{: >3d}>".format(l)
    return label


def fmt_timestamp(timestamp: Optional[Union[datetime, str]] = None) -> str:
    if timestamp is None:
        timestamp = datetime.now()
    if isinstance(timestamp, str):
        return timestamp
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def gen_pycf_details(started_at: Optional[Union[datetime, str]] = None) -> str:
    r"""
    Generate the pycf metadata block for summaries and stdout.
    """
    s = "pycf details\n"
    s += "============\n\n"
    s += "pycf revision: {}  built at {}\n".format(__version__, __build_timestamp__)
    s += "Build comment: {}\n".format(__build_comment__)
    s += "Calculation started at: {}\n".format(fmt_timestamp(started_at))
    return s


def gen_pycf_summary(started_at: Optional[Union[datetime, str]] = None, suppress_input: bool = False) -> str:
    r"""
    Read input file and add to long string. Further, print the pycf version and
    date/time.
    
    Parameters
    ----------
    started_at : datetime or str, optional
        Starting timestamp for the fit
    suppress_input : bool, optional
        If True, omit the input file echo from the summary (default: False)
        Useful when running multiple fits to reduce verbose output
    """
    s = ""
    if not suppress_input:
        s = "\nInput file\n"
        s += "==========\n\n"
        try:
            filename = os.path.abspath(inspect.stack()[1][1])
            s += "File: {}\n\n".format(filename)
            with open(filename, "r") as f:
                s += f.read()
            s += "\n\n"
        except (OSError, IndexError) as e:
            # If we can't read the input file (e.g., called from interactive shell,
            # or file permissions issue), include a note and continue
            s += "*** ERROR: Unable to read input file ***\n"
            s += "Reason: {}\n".format(str(e))
            s += "\nThis typically happens when gen_pycf_summary() is called from:\n"
            s += "  - An interactive Python/IPython session\n"
            s += "  - A script in a location that can't be read\n"
            s += "  - Use suppress_input=True to skip input file echo\n\n"
    s += gen_pycf_details(started_at)
    return s


def print_pycf_details(started_at: Optional[Union[datetime, str]] = None) -> None:
    print(gen_pycf_details(started_at), end="")


def gen_completed_str(completed_at: Optional[Union[datetime, str]] = None) -> str:
    r"""
    Return string of fit completion time.
    """
    s = "Calculation completed at: {}\n\n".format(fmt_timestamp(completed_at))
    return s


def print_completed_str(completed_at: Optional[Union[datetime, str]] = None) -> None:
    print(gen_completed_str(completed_at), end="")


def ex_parse_abs(ex: Any, z: np.ndarray, labels: List[Any]) -> np.ndarray:
    r"""
    Helper function for extracting and formatting experimental energy level data
    from an ExData object for absolute energy level data.

    Parameters
    ----------
    ex : ExData
        The object to be parsed.
    z : np.ndarray
        Eigenvector array the principal components of which are used to sort
        state labels.
    labels : list
        A list of state labels.

    Returns
    -------
    parsed_ex : np.ndarray
        Two column array containing level indices starting at 1 in the zeroeth
        column and corresponding experimental energy levels in the first column.
        If the ExData object contains no absolute energy levels an empty array
        is returned.
    """
    if ex.n_a == 0:
        parsed_ex = np.array([])
    elif ex.sl_index:
        parsed_ex = np.zeros((ex.n_a, 2))
        parsed_ex[:, 1] = ex.e[: ex.n_a]
        # Determine the index of the principal component of each
        # eigenvector.
        pc = np.argmax(np.abs(z), axis=0)
        # Validate that pc indices are within bounds of labels
        if np.any(pc >= len(labels)):
            raise ValueError("Principal component index exceeds bounds of labels array")
        for i, r in enumerate(ex.a_states):
            # Find the index of the principal component of each state label.
            # Guard against a missing match: np.where returns an empty array
            # when no eigenvector has the requested label as its principal
            # component, which would produce a bare IndexError without context.
            idxs = np.where((np.array(labels)[pc] == r).all(axis=1))[0]
            if len(idxs) == 0:
                raise RuntimeError(
                    "Experimental state label {} not found in computed "
                    "eigenvectors; check that the label is correct and that "
                    "the basis is large enough.".format(r)
                )
            parsed_ex[i, 0] = idxs[0]
    else:
        parsed_ex = np.zeros((ex.n_a, 2))
        # Abs. energy values are ordered to preceed diff. values.
        parsed_ex[:, 1] = ex.e[: ex.n_a]
        parsed_ex[:, 0] = ex.la
        # Sort ex according to index column.
        parsed_ex = parsed_ex[np.argsort(parsed_ex[:, 0]), :]
    return parsed_ex


def ex_parse_diff(ex: Any, z: np.ndarray, labels: List[Any]) -> np.ndarray:
    r"""
    Helper function for extracting and formatting experimental energy level data
    from an ExData object for energy level differences.

    Parameters
    ----------
    ex : ExData
        The object to be parsed.
    z : np.ndarray
        Eigenvector array the principal components of which are used to sort
        state labels.
    labels : list
        A list of state labels.

    Returns
    -------
    parsed_ex : np.ndarray
        Three coloumn array containing initial level indices starting at 1 in
        the zeroeth column, final level indices starting at 1 in the first
        column, and corresponding experimental energy levels differences in the
        second column.  If the ExData object contains no difference energy
        levels an empty array is returned.
    """
    if ex.n_d == 0:
        parsed_ex = np.array([])
    elif ex.sl_index:
        parsed_ex = np.zeros((ex.n_d, 3))
        parsed_ex[:, 2] = ex.e[ex.n_a :]
        # Determine the index of the principal component of each
        # eigenvector.
        pc = np.argmax(np.abs(z), axis=0)
        # Find the index of the principal component of each state label.
        # Both loops guard against missing matches for the same reason as
        # ex_parse_abs: a bare IndexError gives no hint about which label
        # failed or why.
        for i, s in enumerate(ex.id_states):
            idxs = np.where((np.array(labels)[pc] == s).all(axis=1))[0]
            if len(idxs) == 0:
                raise RuntimeError(
                    "Initial-state label {} not found in computed "
                    "eigenvectors; check that the label is correct and that "
                    "the basis is large enough.".format(s)
                )
            parsed_ex[i, 0] = idxs[0]
        for i, s in enumerate(ex.fd_states):
            idxs = np.where((np.array(labels)[pc] == s).all(axis=1))[0]
            if len(idxs) == 0:
                raise RuntimeError(
                    "Final-state label {} not found in computed "
                    "eigenvectors; check that the label is correct and that "
                    "the basis is large enough.".format(s)
                )
            parsed_ex[i, 1] = idxs[0]
    else:
        parsed_ex = np.zeros((ex.n_d, 3))
        # Diff. energy values are ordered to come after abs. values.
        parsed_ex[:, 2] = ex.e[ex.n_a :]
        parsed_ex[:, 0] = ex.ild
        parsed_ex[:, 1] = ex.fld
        # Sort ex according to initial level (col 0), then final level (col 1).
        # np.lexsort sorts primarily by the *last* key in its tuple, so col 0
        # (initial level) must come last to be the primary sort key.  The
        # previous order had the keys reversed, making final level primary.
        parsed_ex = parsed_ex[np.lexsort((parsed_ex[:, 1], parsed_ex[:, 0])), :]
    return parsed_ex


_EDATA_DTYPE = np.dtype(
    [
        ("h_index", np.int32),
        ("h_label", object),
        ("kind", "U2"),
        ("i_lo", np.int32),
        ("i_hi", np.int32),
        ("e_calc", np.float64),
        ("e_obs", np.float64),
        ("weight", np.float64),
        ("residual", np.float64),
        ("wresidual", np.float64),
    ]
)


class EData:
    """A flat per-observation table of fit-residual data.

    `EData` is a thin wrapper around a NumPy structured array that holds
    one row per experimental observation seen by an :class:`EFit` /
    :class:`MHFit` objective evaluation.  It is normally produced by
    :py:meth:`EFit.get_edata` or :py:meth:`MHFit.get_edata`, not
    constructed directly by user code.

    Row order matches the order in which the C minimiser concatenates
    residuals, so row indices into ``arr`` align with column indices of
    the Jacobian returned by :py:meth:`fd_jacobian` and the Jacobian
    captured by ``gsl_nls`` (see :py:attr:`EFit.last_jacobian`).

    Parameters
    ----------
    arr : numpy.ndarray
        A structured array with dtype matching :py:attr:`EData.DTYPE`.

    Attributes
    ----------
    arr : numpy.ndarray
        The underlying structured array.  Modifying its values is
        supported but its dtype must not change.
    DTYPE : numpy.dtype
        The expected structured dtype.  Available on the class.

    See Also
    --------
    gen_edata_summary : pretty-printer for an :class:`EData` instance.
    """

    DTYPE = _EDATA_DTYPE

    def __init__(self, arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError("EData requires a NumPy structured array")
        if arr.dtype != self.DTYPE:
            raise TypeError(
                f"EData expects dtype matching EData.DTYPE; got {arr.dtype!r}"
            )
        if arr.ndim != 1:
            raise ValueError(
                f"EData expects a 1-D structured array; got ndim={arr.ndim}"
            )
        self.arr = arr

    @classmethod
    def empty(cls, n: int) -> "EData":
        """Allocate an EData with ``n`` zero-initialised rows."""
        if n < 0:
            raise ValueError("EData length must be non-negative")
        return cls(np.zeros(n, dtype=cls.DTYPE))

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return self.arr[idx]

    def chi2(self) -> float:
        r"""Return :math:`\sum_i w_i (e_{\mathrm{calc},i} - e_{\mathrm{obs},i})^2`.

        This matches the scalar minimised by the underlying C objective
        (modulo any sign/constant conventions handled inside the C code).
        For zero-length tables, returns ``0.0``.
        """
        if len(self) == 0:
            return 0.0
        return float(np.sum(self.arr["weight"] * self.arr["residual"] ** 2))

    def to_str(self, precision: int = 4, max_rows: Optional[int] = None) -> str:
        """Render the table as a labelled, fixed-width string.

        Parameters
        ----------
        precision : int
            Number of digits after the decimal point for floating-point
            columns.
        max_rows : int, optional
            If set, truncate the printed output to this many rows and
            append an ellipsis line.  ``None`` (the default) prints all
            rows.
        """
        n = len(self)
        if n == 0:
            return "EData (empty)"
        n_show = n if max_rows is None else min(n, max_rows)

        header = (
            f"{'idx':>4}  {'H':>3}  {'label':<14}  {'kind':<4}  "
            f"{'i_lo':>4}  {'i_hi':>4}  "
            f"{'e_calc':>12}  {'e_obs':>12}  "
            f"{'weight':>10}  {'residual':>12}  {'wresid':>12}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        fmt_e = f"{{:>12.{precision}f}}"
        fmt_w = f"{{:>10.{precision}f}}"
        for i in range(n_show):
            row = self.arr[i]
            lab = "" if row["h_label"] is None else str(row["h_label"])
            if len(lab) > 14:
                lab = lab[:13] + "…"
            lines.append(
                f"{i:>4}  {int(row['h_index']):>3}  {lab:<14}  "
                f"{str(row['kind']):<4}  "
                f"{int(row['i_lo']):>4}  {int(row['i_hi']):>4}  "
                + fmt_e.format(float(row["e_calc"]))
                + "  "
                + fmt_e.format(float(row["e_obs"]))
                + "  "
                + fmt_w.format(float(row["weight"]))
                + "  "
                + fmt_e.format(float(row["residual"]))
                + "  "
                + fmt_e.format(float(row["wresidual"]))
            )
        if n_show < n:
            lines.append(f"... ({n - n_show} more rows)")
        lines.append(sep)
        lines.append(f"chi2 = {self.chi2():.{precision}e}   N = {n}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EData(n={len(self)}, chi2={self.chi2():.6g})"


def gen_edata_summary(edata: EData, **kwargs: Any) -> str:
    """Return a pretty-printed summary of an :class:`EData` table.

    Thin wrapper around :py:meth:`EData.to_str` provided for symmetry
    with the other ``gen_*_summary`` helpers in this module.

    Parameters
    ----------
    edata : EData
        The table to render.
    **kwargs
        Forwarded to :py:meth:`EData.to_str` (``precision``, ``max_rows``).
    """
    if not isinstance(edata, EData):
        raise TypeError("gen_edata_summary requires an EData instance")
    return edata.to_str(**kwargs)


def calc_mu(m: int, minimum_q: int, half_integer_states: bool = False) -> int:
    r"""
    Calculate the folded magnetic quantum number (crystal quantum number).
    
    The folded magnetic quantum number mu wraps the magnetic quantum number m
    into a fundamental domain determined by the smallest non-zero q value in
    the crystal-field tensor expansion. This provides a robust identifier for
    state blocks that can be mixed by C_kq tensor terms.
    
    Parameters
    ----------
    m : int
        Magnetic quantum number from the principal eigenvector component.
    minimum_q : int
        Smallest non-zero q value across all C_kq tensors in Hamiltonian.
        Typical values: 2 (for C20, C22), 4 (for C40, C44), etc.
    half_integer_states : bool, optional
        If False (default): m values are actual integers (e.g., m ∈ {-3, -1, 1, 3} for J=3/2)
        and minimum_q is used as-is.
        If True: m values are half-integers stored as doubled integers (e.g., m ∈ {-3, -1, 1, 3}
        representing m ∈ {-3/2, -1/2, 1/2, 3/2}) and minimum_q must be doubled for folding.
    
    Returns
    -------
    mu : int
        Folded magnetic quantum number in range [0, minimum_q // 2].
    """
    # Adjust minimum_q based on how m values are stored:
    # If half_integer_states=False: m values are actual integers, use minimum_q as-is
    # If half_integer_states=True: m values are half-integers (stored doubled), double minimum_q
    min_q_eff = minimum_q * 2 if half_integer_states else minimum_q
    
    # Fold m into fundamental domain: |m| % min_q_eff
    mu = abs(m) % min_q_eff
    
    # Fold back if in upper half of period
    if mu > min_q_eff // 2:
        mu = min_q_eff - mu
    
    return mu


def mu_n_to_level(h: 'cfl.Hamiltonian', mu_n_array: np.ndarray, minimum_q: int,
                  half_integer_states: bool) -> np.ndarray:
    r"""
    Convert (mu, n) state pairs to energy level indices for a given Hamiltonian.
    
    This function computes mu/n for all eigenstates of the Hamiltonian and matches
    user-provided (mu, n) pairs to their corresponding level indices (1-based).
    
    Parameters
    ----------
    h : Hamiltonian
        Diagonalized Hamiltonian with current coefficients. Must have:
        - h.z (eigenvector matrix, shape n_states x n_basis)
        - h.tensors[0].states.labels (basis state SLJM labels, shape n_basis x 4)
    mu_n_array : ndarray
        Array of (mu, n) pairs, shape (N, 2), with mu in column 0, n in column 1.
        mu values are folded magnetic quantum numbers, n values are ordinal
        indices within each mu group (1-based).
    minimum_q : int
        Smallest non-zero q in the Hamiltonian expansion (for mu calculation).
    half_integer_states : bool
        Whether m quantum numbers are half-integers stored as doubled integers.
    
    Returns
    -------
    ndarray
        Array of level indices (1-based), shape (N,).
        
    Raises
    ------
    ValueError
        If any (mu, n) pair is not found in the current Hamiltonian.
    """
    # Get basis state labels
    if not h.tensors or not h.tensors[0].states:
        raise ValueError("Hamiltonian must have state labels (SLJM format)")
    
    state_labels_list = h.tensors[0].states.labels
    state_labels = np.array(state_labels_list, dtype=np.int32)
    
    # Get eigenvectors
    z = h.z
    if z is None:
        raise ValueError("Hamiltonian must be diagonalized (call h.diag() first)")
    
    n_states = len(state_labels)
    
    # For each eigenstate, compute its (mu, n) and track level indices
    mu_to_levels: Dict[int, List[int]] = {}
    
    for state_idx in range(n_states):
        # Find principal component
        row = z[state_idx, :]
        abs_row = np.abs(row)
        pc_idx = np.argmax(abs_row)
        
        # Extract m from principal component SLJM labels
        m_value = int(state_labels[pc_idx, 3])
        
        # Compute mu
        mu = calc_mu(m_value, minimum_q, half_integer_states)
        
        # Track level indices for this mu (level indices are 1-based)
        if mu not in mu_to_levels:
            mu_to_levels[mu] = []
        mu_to_levels[mu].append(state_idx + 1)
    
    # Match requested (mu, n) to level indices
    level_indices = np.zeros(len(mu_n_array), dtype=np.int32)
    for i in range(len(mu_n_array)):
        mu_req = int(mu_n_array[i, 0])
        n_req = int(mu_n_array[i, 1])
        
        if mu_req not in mu_to_levels or n_req > len(mu_to_levels[mu_req]):
            available = sorted([(m, len(lvls)) for m, lvls in mu_to_levels.items()])
            raise ValueError(
                f"No state found with (mu, n) = ({mu_req}, {n_req}). "
                f"Available: {available}")
        
        level_indices[i] = mu_to_levels[mu_req][n_req - 1]
    
    return level_indices


def gen_e_summary(w: np.ndarray, z: np.ndarray, labels: List[Any], label_key: str, **kwargs: Any
) -> str:
    r"""
    Generate energy level summary given eigenvalues and eigenvectors.

    Parameters
    ----------
    w : np.ndarray
        The eigenvalue vector, of length n.
    z : np.ndarray
        The eigenvectors in an n by n matrix.
    labels : list
        A list of state labels.
    label_key : str
       String identifying the type of label.  Valid characters are S, L, J, M,
       I, T, and F and their position in label_key specifies the location in
       each label.
    ex : np.ndarray or ExData, optional
        Either a 2 by n dimensional array or an ExData object. The two
        column case is used to specify only absolute energy levels.  In this
        instance, the first column contains energy level indices starting at 1,
        and the second column contains the absolute experimental energy of the
        corresponding level.  Other types of energy level data must be passed as
        an ExData object.
    nstates : int, optional
        The number of constituent states to display for mixed states.
    chi2 : float, optional
        The final chi2 value of the fit.
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : float, optional
        The weighting applied during the chi2 fit.  This should be set if ndof is set.
    e_shift : bool, optional
        Shift entire eigenvalue spectrum s.t. the first eigenvalue is zero.
    minimum_q : int, optional
        Smallest non-zero q value across all C_kq tensors in the Hamiltonian.
        If provided, add mu (folded magnetic quantum number) and n (ordinal index
        within each mu group) columns to the output table. Typical values: 2 for
        C20/C22 terms, 4 for C40/C44 terms, etc.
    half_integer_states : bool, optional
        If False (default): m values are actual integers, and minimum_q is used as-is.
        If True: m values are half-integers stored as doubled integers, and minimum_q
        is automatically doubled for folding. Only used if minimum_q is provided.
    """

    if "nstates" not in kwargs:
        nstates = 2
    else:
        nstates = kwargs["nstates"]
    if "ex" in kwargs:
        ex = kwargs["ex"]
        if isinstance(ex, np.ndarray):
            # Sort ex according to index column.
            ex = ex[np.argsort(ex[:, 0]), :]
            # Change to zero based indexing
            ex[:, 0] = ex[:, 0] - 1
        else:
            ex = ex_parse_abs(ex, z, labels)
        if len(ex[:, 0]) != len(set(ex[:, 0])):
            raise ValueError(
                "e_summary: ex input data contains duplicate entries in the index column."
            )
    else:
        ex = np.array([])
    if "e_shift" in kwargs:
        if kwargs["e_shift"]:
            e_shift = -np.min(w)
            w = w + e_shift
    s = "Energy level summary\n"
    s += "====================\n"
    if "h_label" in kwargs and kwargs["h_label"] is not None:
        s += "Hamiltonian: {}\n".format(kwargs["h_label"])
    s += "\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[:, i]))[::-1]]
    
    # Calculate mu and n if minimum_q is provided
    mu_values = None
    n_values = None
    if "minimum_q" in kwargs and kwargs["minimum_q"] is not None:
        minimum_q = kwargs["minimum_q"]
        half_integer = kwargs.get("half_integer_states", False)
        
        # Extract m values from principal component of each eigenvector
        m_values = []
        for i in range(len(z)):
            # Principal component is the largest amplitude in eigenvector i
            principal_idx = sort_list[i][0]
            m = labels[principal_idx][-1]  # m is the last element in the label
            m_values.append(m)
        
        # Calculate mu for each eigenvector
        mu_values = [calc_mu(m, minimum_q, half_integer) for m in m_values]
        
        # Calculate n (ordinal index within each mu group), sorted by energy
        n_values = [0] * len(mu_values)
        mu_groups = {}
        for i in range(len(mu_values)):
            mu = mu_values[i]
            if mu not in mu_groups:
                mu_groups[mu] = []
            mu_groups[mu].append((w[i], i))  # (energy, index)
        
        # Sort each group by energy and assign n values
        for mu, group in mu_groups.items():
            group.sort(key=lambda x: x[0])  # Sort by energy
            for n, (_, idx) in enumerate(group, start=1):
                n_values[idx] = n
    
    heading = (
        "Lev.  "
        + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4))
        * nstates
        + "       Theory"
    )
    
    # Insert mu and n columns if calculated
    if mu_values is not None:
        # Modify heading to include mu and n columns after "Lev."
        heading = "Lev.  mu   n     " + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4)) * nstates + "       Theory"
    if ex.size != 0:
        heading += "     Experiment    Difference\n"
    else:
        heading += " \n"
    s += uline_char(heading)
    ii = 0
    for i in range(len(z)):
        line = "{0:<6}".format(i + 1)
        # Add mu and n columns if calculated
        if mu_values is not None:
            line += "{0:>2} {1:>5} ".format(mu_values[i], n_values[i])
        N = np.sum(np.abs(z[:, i]) ** 2)
        for j in range(nstates):
            si = sort_list[i][j]
            line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(
                z[si, i], np.abs(z[si, i]) ** 2 / N, si + 1, format_state_label(si, labels, label_key)
            )
        s += line + " {: >12.4f}".format(w[i])
        if ex.size != 0:
            if ex[ii, 0] == i:
                s += "   {: >12.4f}  {: >12.4f}".format(ex[ii, 1], ex[ii, 1] - w[i]) + "\n"
                if ii != len(ex) - 1:
                    ii += 1
            else:
                s += "         --            --\n"
        else:
            s += "\n"
    s += "Label key: {}".format(label_key)
    if "minimum_q" in kwargs and kwargs["minimum_q"] is not None:
        minimum_q = kwargs["minimum_q"]
        half_integer = kwargs.get("half_integer_states", False)
        s += ", minimum_q={}, half_integer_states={}".format(minimum_q, half_integer)
    s += "\n"
    if "chi2" in kwargs:
        s += "weighted chi2 = {:.4f}\n".format(kwargs["chi2"])
        if "ndof" in kwargs:
            if "weighting" not in kwargs:
                raise ValueError("The weight argument needs to be provided if you provide ndof.")
            else:
                weighting = kwargs["weighting"]
            # Fix: sigma is an RMS quantity, so ndof belongs inside the square root.
            if kwargs["ndof"] == 0:
                s += "sigma = N/A (ndof=0)\n"
            else:
                s += "sigma = {:.4f}\n".format(
                    np.sqrt(kwargs["chi2"] / (weighting * kwargs["ndof"]))
                )
            if weighting != 1:
                s += "weighting factor = {:.2e}\n".format(weighting)
    s += "\n"
    if "e_shift" in kwargs:
        if kwargs["e_shift"]:
            s += "Energy level shift: {:.4f}\n".format(e_shift)
    return s


def gen_e_summary_trunc(
    w: np.ndarray,
    z: np.ndarray,
    labels: List[Any],
    label_key: str,
    ex: Any,
    name: str,
    **kwargs: Any,
) -> str:
    r"""
    Generate a truncated energy level summary displaying only levels for which
    experimental energy level data is provided.

    Parameters
    ----------
    w : np.ndarray
        The eigenvalue vector, of length n.
    z : np.ndarray
        The eigenvectors in an n by n matrix.
    labels : list
        A list of state labels.
    label_key : str
       String identifying the type of label.  Valid characters are S, L, J, M,
       I, T, and F and their position in label_key specifies the location in
       each label.
    ex : ExData
        The ExData object for which to generate the truncated energy level summary.
    name : str
        Name used in heading for this truncated summary.
    nstates : int, optional
        The number of constituent states to display for mixed states.
    chi2 : float, optional
        The final chi2 value of the fit.
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : float, optional
        The weighting applied during the chi2 fit.  This should be set if ndof is set.
    minimum_q : int, optional
        Smallest non-zero q value across all C_kq tensors in the Hamiltonian.
        If provided, add mu (folded magnetic quantum number) and n (ordinal index
        within each mu group) columns to the output table.
    half_integer_states : bool, optional
        If False (default): m values are actual integers, and minimum_q is used as-is.
        If True: m values are half-integers stored as doubled integers, and minimum_q
        is automatically doubled for folding. Only used if minimum_q is provided.
    """

    if "nstates" not in kwargs:
        nstates = 2
    else:
        nstates = kwargs["nstates"]
    if ex.n_a + ex.n_d == 0:
        return ""
    s = "{} summary\n".format(name)
    s += "=" * len(name) + "========\n\n"
    sort_list = []
    for i in range(len(z)):
        sort_list += [np.argsort(np.abs(z[:, i]))[::-1]]
    
    # Calculate mu and n if minimum_q is provided
    mu_values = None
    n_values = None
    if "minimum_q" in kwargs and kwargs["minimum_q"] is not None:
        minimum_q = kwargs["minimum_q"]
        half_integer = kwargs.get("half_integer_states", False)
        
        # Extract m values from principal component of each eigenvector
        m_values = []
        for i in range(len(z)):
            # Principal component is the largest amplitude in eigenvector i
            principal_idx = sort_list[i][0]
            m = labels[principal_idx][-1]  # m is the last element in the label
            m_values.append(m)
        
        # Calculate mu for each eigenvector
        mu_values = [calc_mu(m, minimum_q, half_integer) for m in m_values]
        
        # Calculate n (ordinal index within each mu group, sorted by energy)
        # Group eigenvectors by mu value
        from collections import defaultdict
        mu_groups: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for i in range(len(z)):
            mu_groups[mu_values[i]].append((w[i], i))
        
        # Sort each group by energy and assign ordinal indices
        n_values = [0] * len(z)
        for mu, group in mu_groups.items():
            # Sort by energy (first element of tuple)
            sorted_group = sorted(group, key=lambda x: x[0])
            for n, (_, idx) in enumerate(sorted_group, start=1):
                n_values[idx] = n
    if ex.n_a != 0:
        if ex.n_d != 0:
            s += uline_char("Absolute energy levels:\n")
        exa = ex_parse_abs(ex, z, labels)
        heading = (
            "Lev.  "
            + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4))
            * nstates
            + "       Theory"
        )
        # Insert mu and n columns if calculated
        if mu_values is not None:
            heading = "Lev.  mu   n     " + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4)) * nstates + "       Theory"
        heading += "     Experiment    Difference\n"
        s += uline_char(heading)
        for ii in range(ex.n_a):
            i = int(exa[ii, 0])
            line = "{0:<6}".format(i + 1)
            # Add mu and n columns if calculated
            if mu_values is not None:
                line += "{0:>2} {1:>5} ".format(mu_values[i], n_values[i])
            N = np.sum(np.abs(z[:, i]) ** 2)
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(
                    z[si, i], np.abs(z[si, i]) ** 2 / N, si + 1, format_state_label(si, labels, label_key)
                )
            s += line + " {: >12.4f}".format(w[i])
            s += "   {: >12.4f}  {: >12.4f}".format(exa[ii, 1], exa[ii, 1] - w[i]) + "\n"
        s += "\n"
    # Difference energy level summary.
    if ex.n_d != 0:
        if ex.n_a != 0:
            s += uline_char("Energy level differences:\n")
        exd = ex_parse_diff(ex, z, labels)
        heading = (
            "Lev.  "
            + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4))
            * nstates
            + "    Th. diff."
        )
        # Insert mu and n columns if calculated
        if mu_values is not None:
            heading = "Lev.  mu   n     " + ("Percentage                 " + "State" + " " * (len(format_state_label(0, labels, label_key)) - 4)) * nstates + "    Th. diff."
        heading += "     Exp. diff.    Diff. diff.\n"
        s += uline_char(heading)
        for ii in range(ex.n_d):
            i = int(exd[ii, 0])
            line = "{0:<6}".format(i + 1)
            # Add mu and n columns if calculated
            if mu_values is not None:
                line += "{0:>2} {1:>5} ".format(mu_values[i], n_values[i])
            N = np.sum(np.abs(z[:, i]) ** 2)
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(
                    z[si, i], np.abs(z[si, i]) ** 2 / N, si + 1, format_state_label(si, labels, label_key)
                )
            s += line + "\n"
            tmp_w = w[i]
            i = int(exd[ii, 1])
            line = "{0:<6}".format(i + 1)
            # Add mu and n columns if calculated
            if mu_values is not None:
                line += "{0:>2} {1:>5} ".format(mu_values[i], n_values[i])
            N = np.sum(np.abs(z[:, i]) ** 2)
            for j in range(nstates):
                si = sort_list[i][j]
                line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(
                    z[si, i], np.abs(z[si, i]) ** 2 / N, si + 1, format_state_label(si, labels, label_key)
                )
            tmp_w = w[i] - tmp_w
            s += line + " {: >12.4g}".format(tmp_w)
            s += "   {: >12.4g}  {: >12.4g}".format(exd[ii, 2], exd[ii, 2] - tmp_w) + "\n"
        s += "\n"
    s += "Label key: {}".format(label_key)
    if "minimum_q" in kwargs and kwargs["minimum_q"] is not None:
        minimum_q = kwargs["minimum_q"]
        half_integer = kwargs.get("half_integer_states", False)
        s += ", minimum_q={}, half_integer_states={}".format(minimum_q, half_integer)
    s += "\n"
    if "chi2" in kwargs:
        s += "weighted chi2 = {:.4f}\n".format(kwargs["chi2"])
        if "ndof" in kwargs:
            if "weighting" not in kwargs:
                raise ValueError("The weight argument needs to be provided if you provide ndof.")
            else:
                weighting = kwargs["weighting"]
            # Fix: sigma is an RMS quantity, so ndof belongs inside the square root.
            if kwargs["ndof"] == 0:
                s += "sigma = N/A (ndof=0)\n"
            else:
                s += "sigma = {:.4f}\n".format(
                    np.sqrt(kwargs["chi2"] / (weighting * kwargs["ndof"]))
                )
            if weighting != 1:
                s += "weighting factor = {:.2e}\n".format(weighting)
    s += "\n"
    return s


def gen_sh_summary(param: List[np.ndarray], sh: Any, **kwargs: Any) -> str:
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
    name : str, optional
        If provided, the summary heading uses the provided string instead of
        "Spin Hamiltonian".
    chi2 : np.ndarray, optional
        The final chi2 value of the fit for each spin Hamiltonian term.
    ndof : int, optional
        The number of degrees of freedom of the fit; that is, the number of
        observables minus the number of parameters.  If this is provided, along
        with chi2, then the standard deviation -- assuming a model fit -- will
        be shown.  See Chapter 15 (page 780) of Numerical Recipes, 3rd edition.
    weighting : dict, optional
        The weighting applied during the chi2 fit; one entry for each spin
        Hamiltonian term.  This should be set if ndof is set.
    """
    np.set_printoptions(
        formatter={"float": lambda x: "{:8.5f}".format(x)},  # type: ignore[arg-type]
    )
    if "name" in kwargs:
        s = "{} summary\n".format(kwargs["name"])
        s += "=" * len(kwargs["name"]) + "========\n\n"
    else:
        s = "Spin Hamiltonian summary\n"
        s += "========================\n\n"
    tmp_sigma = 0
    for i, inter in enumerate(sh.interactions):
        s += uline_char("%s interaction\n" % inter)
        if "shx" in kwargs:
            s += uline_char(
                "Theory (abs. value)           Experiment (abs. value)       Difference\n"
            )
        else:
            s += uline_char("Theory (abs. value)\n")
        for j in range(3):
            s += str(np.abs(np.real(param[i])).reshape(3, 3)[j, :])
            if "shx" in kwargs:
                shx = kwargs["shx"]
                s += (
                    "  "
                    + str(np.abs(shx[inter]).reshape(3, 3)[j, :])
                    + "  "
                    + str((np.abs(shx[inter]) - np.abs(np.real(param[i]))).reshape(3, 3)[j, :])
                    + "\n"
                )
            else:
                s += "\n"
        if "chi2" in kwargs:
            s += "weighted chi2 = {:.4f}\n".format(kwargs["chi2"][i])
            if "weighting" in kwargs:
                s += "weighting factor = {:.2e}\n".format(kwargs["weighting"][inter])
                tmp_sigma += kwargs["chi2"][i] / kwargs["weighting"][inter]
        s += "\n"
    if "chi2" in kwargs and "ndof" in kwargs:
        if "weighting" not in kwargs:
            raise ValueError("The weight argument needs to be provided if you provide ndof.")
        # Fix: the total chi-squared-like sum must be normalized by ndof
        # before taking the RMS square root.
        if kwargs["ndof"] == 0:
            s += "sigma = N/A (ndof=0)\n"
        else:
            s += "sigma = {:.4f}\n".format(np.sqrt(tmp_sigma / kwargs["ndof"]))
    return s


def gen_fit_summary(
    coeff: Dict[str, Any], fit_obj: Any, method: str, fmin: float, **kwargs: Any
) -> str:
    r"""
    Create a string summarizing a crystal-field Hamiltonian fitting run.

    Parameters
    ----------
    coeff : dict
        Contains the fitted interaction coefficients.
    fit_obj : EFitRunner, MHFitRunner, ESHFitRunner, or MESHFitRunner
        Must have __iter__ method that iterates over names of tensors.
    method : str
        The optimization algorithm used for the fit.
    kwargs: dict
        Additional, optimization algorithm specific, settings to print.
    """
    # Formatting definitions.  There are three parameter classes with different
    # formatting options: free-ion, crystal-field and hyperfine. Any param not
    # in CF or HYP is assumed to be in FI.
    cf_l = [
        "C20",
        "C21",
        "C22",
        "C40",
        "C41",
        "C42",
        "C43",
        "C44",
        "C60",
        "C61",
        "C62",
        "C63",
        "C64",
        "C65",
        "C66",
        "c2",
        "c4",
        "c6",
        "C2",
        "C4",
        "C6",
    ]
    hyp_l = ["HYP", "EQHYP", "NUCQUAD20", "NUCQUAD21", "NUCQUAD22"]
    # Param class specific print formats.
    fmt_coeff = {
        "FI": "{0: >19.2f} {1: >19.2f} {2: >19.2f}",
        "CF": "{0: >19.2f} {1: >19.2f} {2: >19.2f}",
        "HYP": "{0: >19.2g} {1: >19.2g} {2: >19.2g}",
    }
    fmt_bounds = {
        "FI": "{0: >15.0f} {1: >15.0f}",
        "CF": "{0: >15.0f} {1: >15.0f}",
        "HYP": "{0: >15.2g} {1: >15.2g}",
    }
    fmt_stepsize = {"FI": "{0: >15.0f}", "CF": "{0: >15.0f}", "HYP": "{0: >15.0f}"}
    fmt_scov = {"FI": "{0: >17.2g}", "CF": "{0: >17.2g}", "HYP": "{0: >17.2g}"}
    np.set_printoptions(formatter={"float": lambda x: "{:.3f}".format(x)})  # type: ignore[arg-type]
    cov = None
    s = "Fitting summary\n"
    s += "===============\n\n"
    heading = "Tensor name           Fitted coeff       Initial coeff          Difference"
    if "covar" in kwargs:
        cov = kwargs["covar"]
        heading += "      Uncertainty"
        kwargs["cov"] = True
    else:
        kwargs["cov"] = False
    if "bounds" in kwargs:
        heading += "   Lower bounds    Upper bounds"
    if "stepsize" in kwargs:
        heading += "       Stepsize"
    heading += "\n"
    s += uline_char(heading)
    ii = 0  # Index for covariance matrix; increments two for imaginary params.
    for i, p in enumerate(fit_obj):
        co = fit_obj.coeff[p]
        if p in cf_l:
            key = "CF"
        elif p in hyp_l:
            key = "HYP"
        else:
            key = "FI"
        if co.imag == 0:
            co = co.real
            if kwargs["cov"]:
                assert cov is not None
                scov = fmt_scov[key].format(np.sqrt(cov[ii, ii]))
            else:
                scov = ""
            ii += 1
        else:
            if kwargs["cov"]:
                assert cov is not None
                scov = fmt_scov[key].format(
                    complex(np.sqrt(cov[ii, ii]), np.sqrt(cov[ii + 1, ii + 1]))
                )
            else:
                scov = ""
            ii += 2
        s += "'{0:<12}: ".format(p + "'")
        s += fmt_coeff[key].format(coeff[p], co, coeff[p] - co)
        s += scov
        if "bounds" in kwargs:
            s += fmt_bounds[key].format(kwargs["bounds"][p][0], kwargs["bounds"][p][1])
        if "stepsize" in kwargs:
            s += fmt_stepsize[key].format(kwargs["stepsize"][p])
        s += "\n"
    if "bounds" in kwargs:
        del kwargs["bounds"]
    if "stepsize" in kwargs:
        del kwargs["stepsize"]
    np.set_printoptions(
        formatter={"float": lambda x: "{:11.2f}".format(x)},  # type: ignore[arg-type]
        linewidth=200,
    )
    if kwargs["cov"]:
        s += "\n" + uline_char("Covariance matrix:\n")
        s += str(cov) + "\n"
        del kwargs["covar"]
    del kwargs["cov"]
    s += "\nNumber of observables: {}\n".format(kwargs["n_obs"])
    s += "Number of real-valued parameters: {}\n".format(kwargs["n_param"])
    del kwargs["n_obs"]
    del kwargs["n_param"]
    if method == "basinhopping":
        kwargs["naccept"] = kwargs["retval"]
        del kwargs["retval"]
    s += "\n" + uline_char("Optimization routine details:\n")
    s += "{0:<20} {1: <}\n".format("fmin:", fmin)
    s += "{0:<20} {1: <}\n".format("method:", method)
    for k in kwargs:
        if k not in ["chi2accept", "xaccept", "covar", "jac"]:
            s += "{0:<20} {1: <}\n".format(k + ":", kwargs[k])
    return s


def print_as_fortran_array(a: np.ndarray) -> None:
    r"""
    Print a two dimensional numpy array in a form that makes it easy to include
    in a c program, using column major ordering.
    """
    s = "{"
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.real(a[i, j]) == 0:
                a_real = 0
            else:
                a_real = np.real(a[i, j])
            if np.imag(a[i, j]) > 0:
                s += "{0}+{1}*I".format(a_real, np.imag(a[i, j]))
            elif np.imag(a[i, j]) < 0:
                s += "{0}{1}*I".format(a_real, np.imag(a[i, j]))
            else:
                s += str(a_real)
            if (i * a.shape[1] + j) < (a.shape[0] * a.shape[1]) - 1:
                s += ", "
    s += "};"
    print(s)


def print_as_c_array(a: np.ndarray) -> None:
    r"""
    Print a two dimensional numpy array in a form that makes it easy to include
    in a c program, using row major ordering.
    """
    s = "{"
    for i in range(a.shape[0]):
        s += "{"
        for j in range(a.shape[1]):
            if np.real(a[i, j]) == 0:
                a_real = 0
            else:
                a_real = np.real(a[i, j])
            if np.imag(a[i, j]) > 0:
                s += "{0}+{1}*I".format(a_real, np.imag(a[i, j]))
            elif np.imag(a[i, j]) < 0:
                s += "{0}{1}*I".format(a_real, np.imag(a[i, j]))
            else:
                s += str(a_real)
            if j != a.shape[1] - 1:
                s += ","
        s += "}"
        if i != a.shape[0] - 1:
            s += ", "
    s += "};"
    print(s)


def MHz2cm1(val: float) -> float:
    r"Convert MHz to cm$^{-1}$."
    return (1.0 / 29979.2458) * val


def cm12MHz(val: float) -> float:
    r"Convert cm$^{-1}$ to MHz."
    return 29979.2458 * val


def bal_bounds(coeff: Dict[str, float], bounds: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    r"""
    Helper function for creating balanced bounds dictionary.  That is, the
    bounds are are some constant, symmetric, $\pm$ offset from the starting
    coefficient values.

    Parameters
    ----------
    coeff : dict
        Coefficient initial value dictionary.
    bounds : dict
        Dictionary of single bounds values for each parameter to be fit, which
        will be added/subtracted from the initial coeff value.

    Returns
    -------
    bal_bounds : dict
        The balanced bounds dictionary.
    """
    bal_b = {}
    for c in bounds:
        try:
            bal_b[c] = (coeff[c] - bounds[c], coeff[c] + bounds[c])
        except KeyError:
            pass
    return bal_b


def rJmmp(j: Union[int, float], m: Union[int, float], mp: Union[int, float], beta: float) -> float:
    r"""
    Equation (C.72) of Messiah.

    Parameters
    ----------
    j : half int or int
    m : half int or int
    mp : half int or int
    beta : float
        Angle in radians
    """
    if j >= abs(m) and j >= abs(mp):
        xi = np.cos(0.5 * beta)
        eta = np.sin(0.5 * beta)
        tmin = max([0, m - mp])
        tmax = min([j + m, j - mp])
        prefact = np.sqrt(
            factorial(j + m) * factorial(j - m) * factorial(j + mp) * factorial(j - mp)
        )
        tlist = np.arange(tmin, tmax + 1)
        r = fsum(
            (
                (
                    (-1) ** t
                    * 1
                    / (
                        factorial(j + m - t)
                        * factorial(j - mp - t)
                        * factorial(t)
                        * factorial(t - m + mp)
                    )
                    * xi ** (2 * j + m - mp - 2 * t)
                    * eta ** (2 * t - m + mp)
                )
                for t in tlist
            )
        )
        r *= prefact
    else:
        r = 0
    return r


def WignerR(
    j: Union[int, float],
    m: Union[int, float],
    mp: Union[int, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> complex:
    r"""
    Implement Wigner rotation of state vector; Eq. (C56) of Messiah.
    Angles in radians, no factors of 2 in angular momentum quantum numbers (that
    is, j, m, and mp are either integer or half integer).
    """
    r1 = np.exp(-1j * alpha * m)
    r2 = rJmmp(j, m, mp, beta)
    r3 = np.exp(-1j * gamma * mp)
    return r1 * r2 * r3


def rotate_cf_params(
    coeff: Dict[str, Any], alpha: float, beta: float, gamma: float
) -> Dict[str, Any]:
    r"""
    Rotate crystal-field parameters by angles alpha, beta, and gamma, using the
    Euler angle convention of Messiah (zyz').

    Parameters
    ----------
    coeff : dict
        Coefficient dictionary in the usual format. Only parametrs with names
        Ckq, where k and q are zero or positive integers, are touched. These
        will be rotated by the specified Euler angles.
    alpha : float
        First Euler angle using zyz' convention (in radians).
    beta : float
        Second Euler angle using zyz' convention (in radians).
    gamma : float
        Third Euler angle using zyz' convention (in radians).

    Returns
    -------
    rcoeff : dict
        A copy of coeff with all parameters of for Ckq transformed by the
        specified Euler rotation.
    """
    k_list = [2, 4, 6]
    p_list = []  # List of parameter lists (sublists indexed by k)
    for i, k in enumerate(k_list):
        p_list += [[0] * (2 * k + 1)]
        for q in range(k + 1):
            try:
                p_list[i][k + q] = coeff["C%i%i" % (k, q)]
                if q != 0:
                    # (Bkq)* = (-1)^q Bk-q
                    p_list[i][k - q] = (-1) ** q * np.conj(p_list[i][k + q])
            except KeyError:
                pass
    # Implement Eq. (C76) of Quantum Mechanics - Messiah.
    rp_list = []  # Rotated list of parameter lists
    for i, p in enumerate(p_list):
        # Fix: np.sum(p) can be zero even when individual elements are nonzero
        # (e.g. cancelling terms), so check that at least one element is nonzero.
        if np.any(np.array(p) != 0):
            rp = np.zeros(len(p), dtype=complex)
            j = k_list[i]
            for mi, m in enumerate(np.arange(-j, j + 1)):
                for mpi, mp in enumerate(np.arange(-j, j + 1)):
                    rp[mi] += WignerR(j, int(m), int(mp), alpha, beta, gamma) * p[mpi]
            rp_list += [rp]
        else:
            rp_list += [np.asarray(p)]
    rcoeff = coeff.copy()
    for i, k in enumerate(k_list):
        # Only want complex q >=0 parameters (negative q is implict for complex
        # valued Bkq given (Bkq)* = (-1)^q Bk-q).
        for ii, q in enumerate(range(0, k + 1)):
            ii += k
            if rp_list[i][ii] != 0:
                try:
                    rcoeff["C%i%i" % (k, q)] = rp_list[i][ii]
                except KeyError:
                    pass
    return rcoeff


def update_coeff(coeff: dict, updates: dict) -> dict:
    """
    Update a coefficient dictionary with new values from a fit result.

    Common pattern after fitting: merge fitted parameter updates into the full
    coefficient dictionary and pass to hamiltonian.set_coeff().

    Parameters
    ----------
    coeff : dict
        Base coefficient dictionary (e.g., from hamiltonian.coeff).
    updates : dict
        Dictionary of updated parameters (e.g., from fit result['coeff']).
        Any keys in updates that are not in coeff will be added to the result.
        This is by design: if fit results include new tensors, they are merged in.
        **Warning**: It is the user's responsibility to ensure all keys in updates
        are valid tensor names; otherwise h.set_coeff() will fail downstream.

    Returns
    -------
    dict
        New coefficient dictionary with updates applied.
        Keys from coeff are preserved; keys from updates override or are added.

    Example
    -------
    >>> coeff = {"EAVG": 1000, "ZETA": 600, "C20": 500}
    >>> fitcoeff = {"EAVG": 1010, "C20": 480}  # from fit result
    >>> new_coeff = update_coeff(coeff, fitcoeff)
    >>> # new_coeff is {"EAVG": 1010, "ZETA": 600, "C20": 480}
    >>> h.set_coeff(new_coeff)
    """
    result = coeff.copy()
    result.update(updates)
    return result
