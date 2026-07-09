#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pycf.cfl as cfl

try:
    from pycf.__version__ import __build_comment__, __build_timestamp__, __version__
except ImportError as e:
    logging.warning("Could not import build metadata from pycf.__version__: %s", e)
    __version__ = "unknown"
    __build_timestamp__ = "unknown"
    __build_comment__ = ""
from math import fsum

from scipy.special import factorial  # type: ignore[import-untyped]


def principal_components(z: np.ndarray) -> np.ndarray:
    """Return the basis-state index of the largest-magnitude entry in each
    eigenvector column.

    Parameters
    ----------
    z : np.ndarray
        Eigenvector matrix; columns are eigenvectors expressed in the basis
        indexed by rows.

    Returns
    -------
    np.ndarray
        1-D integer array of length ``z.shape[1]`` giving the principal-
        component row index for each column.
    """
    return np.argmax(np.abs(z), axis=0)


def _resolve_state_label_to_eigenstate(
    target_label: np.ndarray,
    pc: np.ndarray,
    labels_array: np.ndarray,
    label_kind: str = "State",
) -> int:
    """Return the eigenstate index whose principal-component basis-state
    label matches ``target_label``.

    Raises RuntimeError with a label-kind-specific message when no
    eigenvector has the requested label as its principal component.
    Used by :func:`ex_parse_abs` and :func:`ex_parse_diff` to resolve
    state-label experimental data against the diagonalised basis.
    """
    idxs = np.where((labels_array[pc] == target_label).all(axis=1))[0]
    if len(idxs) == 0:
        raise RuntimeError(
            "{} label {} not found in computed eigenvectors; check that "
            "the label is correct and that the basis is large enough.".format(
                label_kind, target_label
            )
        )
    return int(idxs[0])


def _resolve_mu_n_kwargs(ex: Any, kwargs: Dict[str, Any]) -> Optional[Tuple[Any, int, bool]]:
    """Extract ``(h, minimum_q, half_integer_states)`` from ``kwargs`` for
    the marker-column mu/n path.

    Returns ``None`` if no Hamiltonian was provided in ``kwargs``; in that
    case callers fall back to cached level indices on ``ex``.
    """
    if "h" not in kwargs:
        return None
    h = kwargs["h"]
    minimum_q = kwargs.get("minimum_q", h.minimum_q if h.minimum_q is not None else 2)
    half_integer_states = kwargs.get("half_integer_states", h.half_integer_states)
    return h, minimum_q, half_integer_states


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
        Formatted label string like ``"|2F 7,  1>"`` or ``"|1,5D 4, -2>"``
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
    r"""
    Format a timestamp for display in pycf summary headers.

    Parameters
    ----------
    timestamp : datetime or str, optional
        If a ``datetime``, formatted as ``"%Y-%m-%d %H:%M:%S"``.  If a string,
        returned unchanged.  If ``None`` (default), the current local time is
        used.

    Returns
    -------
    str
        Formatted timestamp string.
    """
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


def gen_pycf_summary(
    started_at: Optional[Union[datetime, str]] = None, suppress_input: bool = False
) -> str:
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
    r"""Print the pycf metadata block produced by ``gen_pycf_details``."""
    print(gen_pycf_details(started_at), end="")


def gen_completed_str(completed_at: Optional[Union[datetime, str]] = None) -> str:
    r"""
    Return string of fit completion time.
    """
    s = "Calculation completed at: {}\n\n".format(fmt_timestamp(completed_at))
    return s


def print_completed_str(completed_at: Optional[Union[datetime, str]] = None) -> None:
    r"""Print the completion-time string produced by ``gen_completed_str``."""
    print(gen_completed_str(completed_at), end="")


def ex_parse_abs(ex: Any, z: np.ndarray, labels: List[Any], **kwargs: Any) -> np.ndarray:
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
    h : Hamiltonian, optional
        Hamiltonian object needed for mu/n marker-column data. If provided with
        minimum_q and half_integer_states, enables dynamic mu_n_to_level computation.
    minimum_q : int, optional
        Smallest non-zero q value in the Hamiltonian expansion.
    half_integer_states : bool, optional
        Whether m values are half-integers.

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
        pc = principal_components(z)
        # Validate that pc indices are within bounds of labels
        if np.any(pc >= len(labels)):
            raise ValueError("Principal component index exceeds bounds of labels array")
        labels_array = np.array(labels)
        for i, r in enumerate(ex.a_states):
            parsed_ex[i, 0] = _resolve_state_label_to_eigenstate(
                r, pc, labels_array, "Experimental state"
            )
    else:
        parsed_ex = np.zeros((ex.n_a, 2))
        # Abs. energy values are ordered to preceed diff. values.
        parsed_ex[:, 1] = ex.e[: ex.n_a]

        # Check if marker-column mu/n data is present
        has_marker_mu_n = hasattr(ex, "mu_n_abs") and len(ex.mu_n_abs) > 0
        mu_n_kw = _resolve_mu_n_kwargs(ex, kwargs) if has_marker_mu_n else None

        if mu_n_kw is not None:
            # Marker-column mu/n data: compute eigenstate indices dynamically
            # This mirrors the sl_index approach: recompute for every summary call
            h, minimum_q, half_integer_states = mu_n_kw

            # Compute mu_n_to_level for all user-provided (mu, n) pairs
            level_indices = mu_n_to_level(h, ex.mu_n_abs, minimum_q, half_integer_states)

            # For mixed marker/regular data, fill in only the mu rows; others come from ex.la
            if hasattr(ex, "mu_row_indices") and len(ex.mu_row_indices) > 0:
                parsed_ex[:, 0] = ex.la
                for i, row_idx in enumerate(ex.mu_row_indices):
                    parsed_ex[row_idx, 0] = level_indices[i] - 1
            else:
                # Pure mu/n data: use all computed indices
                # BUT preserve user order by indexing ex.mu_n_abs
                for i in range(len(level_indices)):
                    parsed_ex[i, 0] = level_indices[i] - 1
            # NOTE: Don't sort marker-column mu/n data - user specified the order
        elif has_marker_mu_n:
            # Marker-column data but no Hamiltonian provided - use cached ex.la
            # Don't sort - user specified the order
            parsed_ex[:, 0] = ex.la
        else:
            # Regular level index data - sort for display
            parsed_ex[:, 0] = ex.la
            parsed_ex = parsed_ex[np.argsort(parsed_ex[:, 0]), :]
    return parsed_ex


def ex_parse_diff(ex: Any, z: np.ndarray, labels: List[Any], **kwargs: Any) -> np.ndarray:
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
    h : Hamiltonian, optional
        Hamiltonian object needed for mu/n marker-column data.
    minimum_q : int, optional
        Smallest non-zero q value in the Hamiltonian expansion.
    half_integer_states : bool, optional
        Whether m values are half-integers.

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
        pc = principal_components(z)
        labels_array = np.array(labels)
        for i, s in enumerate(ex.id_states):
            parsed_ex[i, 0] = _resolve_state_label_to_eigenstate(
                s, pc, labels_array, "Initial-state"
            )
        for i, s in enumerate(ex.fd_states):
            parsed_ex[i, 1] = _resolve_state_label_to_eigenstate(s, pc, labels_array, "Final-state")
    else:
        parsed_ex = np.zeros((ex.n_d, 3))
        # Diff. energy values are ordered to come after abs. values.
        parsed_ex[:, 2] = ex.e[ex.n_a :]

        # Check if marker-column mu/n data is present
        has_marker_mu_n = hasattr(ex, "mu_n_diff") and len(ex.mu_n_diff) > 0
        mu_n_kw = _resolve_mu_n_kwargs(ex, kwargs) if has_marker_mu_n else None

        if mu_n_kw is not None:
            # Marker-column mu/n data: compute eigenstate indices dynamically
            h, minimum_q, half_integer_states = mu_n_kw

            mu_n_initial = ex.mu_n_diff[:, :2]
            mu_n_final = ex.mu_n_diff[:, 2:4]

            initial_levels = mu_n_to_level(h, mu_n_initial, minimum_q, half_integer_states)
            final_levels = mu_n_to_level(h, mu_n_final, minimum_q, half_integer_states)

            # For mixed marker/regular data, fill appropriately
            if hasattr(ex, "mu_row_indices_d") and len(ex.mu_row_indices_d) > 0:
                parsed_ex[:, 0] = ex.ild
                parsed_ex[:, 1] = ex.fld
                for i, row_idx in enumerate(ex.mu_row_indices_d):
                    parsed_ex[row_idx, 0] = initial_levels[i] - 1
                    parsed_ex[row_idx, 1] = final_levels[i] - 1
            else:
                # Pure mu/n data
                parsed_ex[:, 0] = initial_levels - 1
                parsed_ex[:, 1] = final_levels - 1
            # NOTE: Don't sort marker-column mu/n data - user specified the order
        elif has_marker_mu_n:
            # Marker-column data but no Hamiltonian provided - use cached ex.ild/fld
            # Don't sort - user specified the order
            parsed_ex[:, 0] = ex.ild
            parsed_ex[:, 1] = ex.fld
        else:
            # Regular level index data - sort for display
            parsed_ex[:, 0] = ex.ild
            parsed_ex[:, 1] = ex.fld
            # Sort ex according to initial level (col 0), then final level (col 1).
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
            raise TypeError(f"EData expects dtype matching EData.DTYPE; got {arr.dtype!r}")
        if arr.ndim != 1:
            raise ValueError(f"EData expects a 1-D structured array; got ndim={arr.ndim}")
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


def get_eigenstate_mu_n(
    eigenstate_idx: int,
    z: np.ndarray,
    labels: List[Any],
    w: np.ndarray,
    minimum_q: int,
    half_integer_states: bool,
) -> Tuple[int, int]:
    r"""
    Calculate (mu, n) quantum numbers for a single eigenstate.

    This is the SINGLE SOURCE OF TRUTH for computing mu/n values.
    Used by both mu_n_to_level() and gen_e_summary_trunc().

    Parameters
    ----------
    eigenstate_idx : int
        Index of the eigenstate (0-based).
    z : np.ndarray
        Shape (n_basis_states, n_eigenstates). Each column is an eigenvector.
    labels : list
        SLJM labels for each basis state.
    w : np.ndarray
        Eigenvalues (energies).
    minimum_q : int
        Smallest non-zero q in crystal field expansion.
    half_integer_states : bool
        If True, m values are half-integers stored as doubled integers.

    Returns
    -------
    tuple
        (mu, n) where:
        - mu: folded magnetic quantum number (0 to minimum_q)
        - n: ordinal index within mu group, sorted by energy (1-based)
    """
    # Extract m from principal component of this eigenstate
    col = z[:, eigenstate_idx]
    abs_col = np.abs(col)
    pc_idx = np.argmax(abs_col)
    m_value = int(labels[pc_idx][-1])  # m is last element in label

    # Compute mu from m
    mu = calc_mu(m_value, minimum_q, half_integer_states)

    # Compute n: count how many eigenstates with same mu have energy <= this eigenstate
    # Need to group all eigenstates by mu and sort by energy
    n_eigenstates = z.shape[1]
    mu_to_levels: Dict[int, List[Tuple[float, int]]] = {}

    for idx in range(n_eigenstates):
        col = z[:, idx]
        abs_col = np.abs(col)
        pc_idx_temp = np.argmax(abs_col)
        m_temp = int(labels[pc_idx_temp][-1])
        mu_temp = calc_mu(m_temp, minimum_q, half_integer_states)

        if mu_temp not in mu_to_levels:
            mu_to_levels[mu_temp] = []
        mu_to_levels[mu_temp].append((w[idx], idx))

    # Sort each mu group by energy (with eigenstate index as tie-breaker for stability)
    for mu_key in mu_to_levels:
        mu_to_levels[mu_key].sort(key=lambda x: (x[0], x[1]))

    # Find the ordinal position (n) of this eigenstate in its mu group
    n: int | None = None
    for rank, (energy, idx) in enumerate(mu_to_levels[mu], start=1):
        if idx == eigenstate_idx:
            n = rank
            break

    return mu, n  # type: ignore[return-value]


def _build_mu_groups(
    h: "cfl.Hamiltonian", minimum_q: int, half_integer_states: bool
) -> Dict[int, List[Tuple[float, int]]]:
    """Group eigenstates by mu value (sorted within each mu by energy).

    Internal helper shared by :func:`mu_n_to_level` and the Cython hot
    loop in ``_update_exdata_mu_n_indices``. Callers needing multiple
    (mu, n) -> level lookups against the same Hamiltonian build the
    grouping once and pass it to :func:`_resolve_mu_n_to_levels`.

    Returns
    -------
    Dict[int, List[Tuple[float, int]]]
        Mapping ``mu -> [(energy, level_idx_1based), ...]`` sorted by
        ``(energy, level_idx)`` within each mu group.
    """
    if not h.tensors or not h.tensors[0].states:
        raise ValueError("Hamiltonian must have state labels (SLJM format)")
    state_labels_list = h.tensors[0].states.labels
    z = h.z
    if z is None:
        raise ValueError("Hamiltonian must be diagonalized (call h.diag() first)")
    if z.ndim != 2:
        raise ValueError(
            f"Eigenvector matrix must be 2D, got shape {z.shape}. "
            "This should not happen with a properly diagonalized Hamiltonian."
        )
    n_states = len(state_labels_list)
    if z.shape[0] != n_states:
        raise ValueError(
            f"Eigenvector matrix has {z.shape[0]} rows but state labels "
            f"has {n_states} entries. Hamiltonian must be properly initialized."
        )

    mu_to_levels: Dict[int, List[Tuple[float, int]]] = {}
    pc_idx = principal_components(z)
    w = h.w
    for eigenstate_idx in range(z.shape[1]):
        m_value = int(state_labels_list[pc_idx[eigenstate_idx]][-1])
        mu = calc_mu(m_value, minimum_q, half_integer_states)
        if mu not in mu_to_levels:
            mu_to_levels[mu] = []
        mu_to_levels[mu].append((w[eigenstate_idx], eigenstate_idx + 1))
    for mu in mu_to_levels:
        mu_to_levels[mu].sort(key=lambda x: (x[0], x[1]))
    return mu_to_levels


def _resolve_mu_n_to_levels(
    mu_to_levels: Dict[int, List[Tuple[float, int]]], mu_n_array: np.ndarray
) -> np.ndarray:
    """Resolve (mu, n) pairs to 1-based level indices using a prebuilt grouping."""
    level_indices = np.zeros(len(mu_n_array), dtype=np.int32)
    for i in range(len(mu_n_array)):
        mu_req = int(mu_n_array[i, 0])
        n_req = int(mu_n_array[i, 1])
        if mu_req not in mu_to_levels or n_req > len(mu_to_levels[mu_req]):
            available = sorted([(m, len(lvls)) for m, lvls in mu_to_levels.items()])
            raise ValueError(
                f"No state found with (mu, n) = ({mu_req}, {n_req}). " f"Available: {available}"
            )
        level_indices[i] = mu_to_levels[mu_req][n_req - 1][1]
    return level_indices


def mu_n_to_level(
    h: "cfl.Hamiltonian", mu_n_array: np.ndarray, minimum_q: int, half_integer_states: bool
) -> np.ndarray:
    r"""
    Convert (mu, n) state pairs to energy level indices for a given Hamiltonian.

    This function computes mu/n for all eigenstates of the Hamiltonian and matches
    user-provided (mu, n) pairs to their corresponding level indices (1-based).
    The (mu, n) parametrization is useful for systems with low-symmetry crystal fields
    where magnetic quantum numbers m are "folded" into an effective parameter mu based
    on the minimum q-value in the expansion (typically q=2 for C20/C22 terms).

    **Understanding mu and n:**

    - ``mu``: Folded magnetic quantum number, computed as ``mu = m * sign(q_min)``
      where ``q_min`` is the smallest non-zero q in your expansion.
    - ``n``: Ordinal index (1, 2, 3, ...) of eigenstates grouped by their mu value.
      The n-th eigenstate within a mu group.

    **Half-integer m values:**

    For systems with half-integer m (e.g., f-electrons with J=5/2), m values are stored
    as doubled integers (±1, ±3, ±5 representing ±1/2, ±3/2, ±5/2). Set ``half_integer_states=True``
    in these cases. The effective q used for folding is then ``q_min * 2``.

    **Usage Example:**

    For Ce:YLF (f-electrons, J=5/2, m ∈ {-5/2, -3/2, -1/2, 1/2, 3/2, 5/2}):

    .. code-block:: python

        import numpy as np
        import pycf

        # Setup Hamiltonian with Ce:YLF crystal field
        importer = pycf.ImportSLJM(...)  # Load from SLJM format
        h = pycf.cfl.Hamiltonian(importer.tensors)
        h.minimum_q = 2                  # C20, C22 lowest terms
        h.half_integer_states = True     # f-electrons have half-integer m
        h.set_coeff(coeffs)
        h.diag()

        # Map experimental data to levels
        mu_n_pairs = np.array([
            [2, 1],   # 1st eigenstate with mu=+2 (corresponds to m=±5/2)
            [2, 2],   # 2nd eigenstate with mu=+2
            [0, 1],   # 1st eigenstate with mu=0 (m=±1/2)
        ], dtype=np.int32)

        level_indices = pycf.cfl_util.mu_n_to_level(
            h, mu_n_pairs, minimum_q=2, half_integer_states=True
        )
        # Returns: [1, 2, 5] (1-based level indices)

    Parameters
    ----------
    h : Hamiltonian
        Diagonalized Hamiltonian with current coefficients. Must have ``h.z``
        (eigenvector matrix, shape ``(n_states, n_basis)``) and
        ``h.tensors[0].states.labels`` (basis state SLJM labels, shape
        ``(n_basis, 4)`` where each row is ``[S, L, J, M]``)
    mu_n_array : ndarray
        Array of (mu, n) pairs to convert, shape ``(N, 2)`` with dtype ``int32``.
        - Column 0: mu values (folded magnetic quantum numbers)
        - Column 1: n values (ordinal indices, 1-based)
    minimum_q : int
        Smallest non-zero q value in the Hamiltonian expansion. Common values:
        - ``2``: For C20, C22 (most common)
        - ``4``: For C40, C44, C60, C64 expansions
        - ``6``: For very high-order expansions
    half_integer_states : bool
        Whether the system has half-integer m quantum numbers (stored as doubled
        integers). Set to ``True`` for f-electrons (J=5/2, d=7/2, etc.) with m ∈
        {..., -3/2, -1/2, 1/2, 3/2, ...}, or ``False`` for integer m values
        (p, d-electrons) with m ∈ {..., -1, 0, 1, ...}.

    Returns
    -------
    ndarray
        Array of level indices (1-based), shape ``(N,)`` with dtype ``int32``.
        Index ``i`` in the returned array is the eigenstate level number
        corresponding to ``mu_n_array[i, :]``.

    Raises
    ------
    ValueError
        - If any (mu, n) pair is not found in the current eigenstate spectrum
        - If ``h.z`` is not 2D (Hamiltonian not properly diagonalized)
        - If eigenvector matrix dimensions don't match state labels

    Notes
    -----
    The principal component method is used: each eigenstate is matched to its
    largest component in the original basis. This works well for weakly-mixing
    crystal fields. For strongly-mixing systems, custom matching logic may be needed.

    **Important:** The (mu, n) → level index mapping is data-dependent and must be
    recomputed after every ``h.diag()`` call, since the eigenvector matrix changes when
    parameters change. Caching would produce stale results. The conversion is performed
    at fitting initialization time in :class:`cfl.EFit` and is negligible in cost.

    The conversion is performed at the time of fitting initialization in :class:`cfl.EFit`.
    Once converted to level indices, the fitting proceeds using the standard
    energy-level comparison workflow.
    """
    # Get basis state labels
    mu_to_levels = _build_mu_groups(h, minimum_q, half_integer_states)
    return _resolve_mu_n_to_levels(mu_to_levels, mu_n_array)


def _build_ex_parse_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ``h`` / ``minimum_q`` / ``half_integer_states`` from a summary
    kwargs dict for passing through to :func:`ex_parse_abs` /
    :func:`ex_parse_diff`. ``h`` and ``minimum_q`` are only included when not
    ``None``; ``half_integer_states`` is forwarded whenever it is present.
    """
    out: Dict[str, Any] = {}
    if kwargs.get("h") is not None:
        out["h"] = kwargs["h"]
    if kwargs.get("minimum_q") is not None:
        out["minimum_q"] = kwargs["minimum_q"]
    if "half_integer_states" in kwargs:
        out["half_integer_states"] = kwargs["half_integer_states"]
    return out


def _format_eigenstate_row(
    i: int,
    z: np.ndarray,
    labels: List[Any],
    label_key: str,
    sort_list: List[np.ndarray],
    nstates: int,
    mu_values: Optional[List[int]],
    n_values: Optional[List[int]],
) -> str:
    """Format the leading portion of one eigenstate row:
    ``"Lev   [mu n]  (amp) pct% idx label  (amp) ..."`` (no trailing energy or
    newline). Shared by :func:`gen_e_summary` and :func:`gen_e_summary_trunc`.
    """
    line = "{0:<6}".format(i + 1)
    if mu_values is not None:
        line += "{0:>2} {1:>5} ".format(mu_values[i], n_values[i])  # type: ignore[index]
    N = np.sum(np.abs(z[:, i]) ** 2)
    for j in range(nstates):
        si = sort_list[i][j]
        line += "({0: .2f}) {1:6.1%} {2:>5} {3} ".format(
            z[si, i],
            np.abs(z[si, i]) ** 2 / N,
            si + 1,
            format_state_label(si, labels, label_key),
        )
    return line


def _format_summary_footer(label_key: str, kwargs: Dict[str, Any]) -> str:
    """Format the ``Label key: ... / chi2 / sigma / weighting factor`` block
    that closes both energy-level summaries.
    """
    s = "Label key: {}".format(label_key)
    if kwargs.get("minimum_q") is not None:
        s += ", minimum_q={}, half_integer_states={}".format(
            kwargs["minimum_q"], kwargs.get("half_integer_states", False)
        )
    s += "\n"
    if "chi2" in kwargs:
        s += "weighted chi2 = {:.4f}\n".format(kwargs["chi2"])
        if "ndof" in kwargs:
            if "weighting" not in kwargs:
                raise ValueError("The weight argument needs to be provided if you provide ndof.")
            weighting = kwargs["weighting"]
            # sigma is an RMS quantity, so ndof belongs inside the square root.
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


def gen_e_summary(
    w: np.ndarray, z: np.ndarray, labels: List[Any], label_key: str, **kwargs: Any
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
    max_levels : int, optional
        Maximum number of energy levels to display in the output (display only;
        does not affect chi2 statistics or other calculations). Useful for systems
        with hundreds of levels where only the first N are of interest.
        Default: None (display all levels).
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

    # Handle max_levels parameter (display only, not for statistics)
    max_levels = kwargs.get("max_levels", None)
    if max_levels is not None and max_levels < 1:
        raise ValueError("max_levels must be >= 1 if specified")

    if "ex" in kwargs:
        ex = kwargs["ex"]
        if isinstance(ex, np.ndarray):
            # Sort ex according to index column.
            ex = ex[np.argsort(ex[:, 0]), :]
            # Change to zero based indexing
            ex[:, 0] = ex[:, 0] - 1
        else:
            ex = ex_parse_abs(ex, z, labels, **_build_ex_parse_kwargs(kwargs))
            # ex_parse_abs already returns 0-based indices (sorted for regular data,
            # in user-specified order for marker-column data). Do not subtract or sort here.
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
    mu_values: Optional[List[int]] = None
    n_values: Optional[List[int]] = None
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
        mu_groups: Dict[int, List[Tuple[float, int]]] = {}
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
        + (
            "Percentage                 "
            + "State"
            + " " * (len(format_state_label(0, labels, label_key)) - 4)
        )
        * nstates
        + "       Theory"
    )

    # Insert mu and n columns if calculated
    if mu_values is not None:
        # Modify heading to include mu and n columns after "Lev."
        heading = (
            "Lev.  mu   n     "
            + (
                "Percentage                 "
                + "State"
                + " " * (len(format_state_label(0, labels, label_key)) - 4)
            )
            * nstates
            + "       Theory"
        )
    if ex.size != 0:
        heading += "     Experiment    Difference\n"
    else:
        heading += " \n"
    s += uline_char(heading)
    # Build a dictionary for fast lookup of experimental data by eigenstate index
    ex_dict: Dict[int, Tuple[float, float]] = {}
    if ex.size != 0:
        for row in ex:
            eigenstate_idx = int(row[0])
            energy = row[1]
            ex_dict[eigenstate_idx] = (energy, energy - w[eigenstate_idx])
    # Determine the number of levels to display
    n_display = len(z) if max_levels is None else min(len(z), max_levels)

    # Optional multiplet boundaries (1-based inclusive level indices)
    multiplet_end_levels = kwargs.get("multiplet_end_levels", None)
    multiplet_start_by_end: Dict[int, int] = {}
    if multiplet_end_levels is not None:
        if not isinstance(multiplet_end_levels, (list, tuple, np.ndarray)):
            raise TypeError("multiplet_end_levels must be a sequence of 1-based level indices.")
        validated_ends: List[int] = []
        for x in multiplet_end_levels:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
                raise TypeError("multiplet_end_levels must contain integer values.")
            validated_ends.append(int(x))
        if any(x < 1 for x in validated_ends):
            raise ValueError("multiplet_end_levels entries must be >= 1.")
        if any(validated_ends[i] <= validated_ends[i - 1] for i in range(1, len(validated_ends))):
            raise ValueError("multiplet_end_levels must be strictly increasing with no duplicates.")
        if any(x > len(z) for x in validated_ends):
            raise ValueError("multiplet_end_levels entries must be <= number of levels.")

        # Respect max_levels: only print multiplet diagnostics for displayed boundaries.
        display_multiplet_ends = [x for x in validated_ends if x <= n_display]
        prev_end = 0
        for end_level in display_multiplet_ends:
            multiplet_start_by_end[end_level] = prev_end + 1
            prev_end = end_level
    for i in range(n_display):
        line = _format_eigenstate_row(
            i, z, labels, label_key, sort_list, nstates, mu_values, n_values
        )
        s += line + " {: >12.4f}".format(w[i])
        if ex.size != 0:
            if i in ex_dict:
                s += "   {: >12.4f}  {: >12.4f}".format(ex_dict[i][0], ex_dict[i][1]) + "\n"
            else:
                s += "         --            --\n"
        else:
            s += "\n"

        # Multiplet diagnostics line printed after each configured end level.
        end_level = i + 1  # 1-based
        if end_level in multiplet_start_by_end:
            start_level = multiplet_start_by_end[end_level]
            start_idx = start_level - 1
            end_idx = end_level - 1
            c_block = np.asarray(w[start_idx : end_idx + 1], dtype=float)
            barycenter = float(np.mean(c_block))
            diag_line = (
                f"      [Multiplet {start_level:>3}-{end_level:>3}] "
                f"barycenter = {barycenter:>10.4f}"
            )

            # Absolute-energy residuals only (already represented in ex_dict).
            residuals_list = [
                float(ex_dict[idx][1]) for idx in range(start_idx, end_idx + 1) if idx in ex_dict
            ]
            if residuals_list:
                residuals = np.asarray(residuals_list, dtype=float)
                barycenter_shift = float(np.mean(residuals))
                sigma_total = float(np.sqrt(np.mean(residuals**2)))
                sigma_crystal_field = float(np.sqrt(np.mean((residuals - barycenter_shift) ** 2)))
                diag_line += (
                    f"  shift = {barycenter_shift:>9.4f}"
                    f"  sigma_total = {sigma_total:>9.4f}"
                    f"  sigma_crystal_field = {sigma_crystal_field:>9.4f}"
                )
            s += diag_line + "\n"
    s += _format_summary_footer(label_key, kwargs)
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
        labels_list = list(labels) if not isinstance(labels, list) else labels

        # Build mu grouping once (O(n) instead of O(n²))
        mu_to_levels: Dict[int, List[Tuple[float, int]]] = {}
        for idx in range(len(z)):
            col = z[:, idx]
            abs_col = np.abs(col)
            pc_idx = np.argmax(abs_col)
            m_value = int(labels_list[pc_idx][-1])
            mu = calc_mu(m_value, minimum_q, half_integer)

            if mu not in mu_to_levels:
                mu_to_levels[mu] = []
            mu_to_levels[mu].append((w[idx], idx))

        # Sort each mu group by energy (with eigenstate index as tie-breaker for stability)
        for mu_key in mu_to_levels:
            mu_to_levels[mu_key].sort(key=lambda x: (x[0], x[1]))

        # Now compute (mu, n) for each eigenstate
        mu_values = [0] * len(z)
        n_values = [0] * len(z)
        for eigenstate_idx in range(len(z)):
            col = z[:, eigenstate_idx]
            abs_col = np.abs(col)
            pc_idx = np.argmax(abs_col)
            m_value = int(labels_list[pc_idx][-1])
            mu = calc_mu(m_value, minimum_q, half_integer)

            # Find n: ordinal position in this mu group
            n = None
            for rank, (energy, idx) in enumerate(mu_to_levels[mu], start=1):
                if idx == eigenstate_idx:
                    n = rank
                    break

            mu_values[eigenstate_idx] = mu
            n_values[eigenstate_idx] = 0 if n is None else n
    if ex.n_a != 0:
        if ex.n_d != 0:
            s += uline_char("Absolute energy levels:\n")
        exa = ex_parse_abs(ex, z, labels, **_build_ex_parse_kwargs(kwargs))

        heading = (
            "Lev.  "
            + (
                "Percentage                 "
                + "State"
                + " " * (len(format_state_label(0, labels, label_key)) - 4)
            )
            * nstates
            + "       Theory"
        )
        heading += "     Experiment    Difference\n"
        s += uline_char(heading)
        for ii in range(ex.n_a):
            i = int(exa[ii, 0])
            line = _format_eigenstate_row(
                i, z, labels, label_key, sort_list, nstates, mu_values, n_values
            )
            s += line + " {: >12.4f}".format(w[i])
            s += "   {: >12.4f}  {: >12.4f}".format(exa[ii, 1], exa[ii, 1] - w[i]) + "\n"
        s += "\n"
    # Difference energy level summary.
    if ex.n_d != 0:
        if ex.n_a != 0:
            s += uline_char("Energy level differences:\n")
        exd = ex_parse_diff(ex, z, labels, **_build_ex_parse_kwargs(kwargs))
        heading = (
            "Lev.  "
            + (
                "Percentage                 "
                + "State"
                + " " * (len(format_state_label(0, labels, label_key)) - 4)
            )
            * nstates
            + "    Th. diff."
        )
        # Insert mu and n columns if calculated
        if mu_values is not None:
            heading = (
                "Lev.  mu   n     "
                + (
                    "Percentage                 "
                    + "State"
                    + " " * (len(format_state_label(0, labels, label_key)) - 4)
                )
                * nstates
                + "    Th. diff."
            )
        heading += "     Exp. diff.    Diff. diff.\n"
        s += uline_char(heading)
        for ii in range(ex.n_d):
            i = int(exd[ii, 0])
            line = _format_eigenstate_row(
                i, z, labels, label_key, sort_list, nstates, mu_values, n_values
            )
            s += line + "\n"
            tmp_w = w[i]
            i = int(exd[ii, 1])
            line = _format_eigenstate_row(
                i, z, labels, label_key, sort_list, nstates, mu_values, n_values
            )
            tmp_w = w[i] - tmp_w
            s += line + " {: >12.4g}".format(tmp_w)
            s += "   {: >12.4g}  {: >12.4g}".format(exd[ii, 2], exd[ii, 2] - tmp_w) + "\n"
        s += "\n"
    s += _format_summary_footer(label_key, kwargs)
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
    coeff: Dict[str, Any],
    fit_obj: Any,
    method: str,
    fmin: float,
    initial_coeff: Dict[str, Any] | None = None,
    include_covariance_matrix: bool = True,
    **kwargs: Any,
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
    initial_coeff : dict, optional
        Pre-fit coefficient dictionary for display of "Initial coeff".
        If omitted, fit_obj.coeff is used.
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
        co = initial_coeff[p] if initial_coeff is not None else fit_obj.coeff[p]
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
    if kwargs["cov"] and include_covariance_matrix:
        s += "\n" + uline_char("Covariance matrix:\n")
        s += str(cov) + "\n"
        del kwargs["covar"]
    elif kwargs["cov"]:
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
    if "jacobian_diagnostics" in kwargs:
        jd = kwargs.pop("jacobian_diagnostics")
        if isinstance(jd, dict) and jd:
            n_rows = jd.get("n_rows", "?")
            n_params = jd.get("n_params", "?")
            rank = jd.get("rank", "?")
            cond = jd.get("condition_jtj", "?")
            s += "{0:<20} {1: <}\n".format("jacobian shape:", f"({n_rows}, {n_params})")
            s += "{0:<20} {1: <}\n".format("jacobian rank:", f"{rank}/{n_params}")
            s += "{0:<20} {1: <}\n".format("cond(J^T J):", cond)
    for k in kwargs:
        if k not in ["chi2accept", "xaccept", "covar", "jac"]:
            s += "{0:<20} {1: <}\n".format(k + ":", kwargs[k])
    return s


def map_sigma_by_parameter(fit_obj: Any, sigma_vector: np.ndarray) -> Dict[str, Any]:
    """Map real-valued sigma vector entries back to parameter names."""
    sigma_by_param: Dict[str, Any] = {}
    ii = 0
    for p in fit_obj:
        ptype = fit_obj.param_types[p]
        if ptype == "c":
            sigma_by_param[p] = complex(float(sigma_vector[ii]), float(sigma_vector[ii + 1]))
            ii += 2
        else:
            sigma_by_param[p] = float(sigma_vector[ii])
            ii += 1
    return sigma_by_param


def jacobian_diagnostics(jacobian: Optional[np.ndarray], n_params: int) -> Dict[str, Any]:
    """Return basic Jacobian conditioning diagnostics."""
    if jacobian is None:
        return {}
    j = np.asarray(jacobian, dtype=np.float64)
    if j.size == 0:
        return {
            "rank": 0,
            "n_params": int(n_params),
            "n_rows": int(j.shape[0]),
            "condition_jtj": np.inf,
        }
    rank = int(np.linalg.matrix_rank(j))
    jtj = j.T @ j
    try:
        cond = float(np.linalg.cond(jtj))
    except np.linalg.LinAlgError:
        cond = np.inf
    return {
        "rank": rank,
        "n_params": int(n_params),
        "n_rows": int(j.shape[0]),
        "condition_jtj": cond,
        "well_conditioned": bool(np.isfinite(cond) and cond < 1e12 and rank == int(n_params)),
    }


def gen_all_coeff_summary(
    all_coeff: Dict[str, Any],
    fitted_coeff: Optional[Dict[str, Any]] = None,
    sigma_by_param: Optional[Dict[str, Any]] = None,
    name: str = "All Hamiltonian parameters",
) -> str:
    """Create a compact table showing all coefficients and fitted/sigma status."""

    def _natural_sort_key(text: str) -> Tuple[Any, ...]:
        return tuple(int(tok) if tok.isdigit() else tok for tok in re.findall(r"\d+|[^\d]+", text))

    def _all_coeff_sort_key(param: str) -> Tuple[int, int, Tuple[Any, ...], str]:
        u = str(param).upper()

        if u == "EAVG":
            return (0, 0, (), u)

        f_priority = {"F2": 0, "F4": 1, "F6": 2}
        if u in f_priority:
            return (1, f_priority[u], (), u)
        if u == "FTOT":
            return (2, 0, (), u)
        if u.startswith("F"):
            return (3, 0, _natural_sort_key(u), u)

        if u == "ALPHA":
            return (4, 0, (), u)
        if u == "BETA":
            return (5, 0, (), u)
        if u == "GAMMA":
            return (6, 0, (), u)

        t_priority = {"T2": 0, "T3": 1, "T4": 2, "T6": 3, "T7": 4, "T8": 5}
        if u in t_priority:
            return (7, t_priority[u], (), u)

        if u == "ZETA":
            return (8, 0, (), u)
        if u == "MTOT":
            return (9, 0, (), u)
        if u == "PTOT":
            return (10, 0, (), u)

        if u.startswith("C"):
            return (11, 0, _natural_sort_key(u), u)

        m_priority = {"MX": 0, "MY": 1, "MZ": 2}
        if u in m_priority:
            return (12, m_priority[u], (), u)
        if u.startswith("M"):
            return (13, 0, _natural_sort_key(u), u)

        if u == "A":
            return (14, 0, (), u)
        if u.startswith("A"):
            return (15, 0, _natural_sort_key(u), u)

        if u == "Q":
            return (16, 0, (), u)
        if u.startswith("Q"):
            return (17, 0, _natural_sort_key(u), u)

        return (99, 0, _natural_sort_key(u), u)

    def _fmt(v: Any) -> str:
        if isinstance(v, complex):
            return f"{float(v.real):13.5f}{float(v.imag):+13.5f}j"
        if isinstance(v, (int, float, np.floating)):
            return f"{float(v):13.5f}"
        return str(v)

    fitted_coeff = fitted_coeff or {}
    sigma_by_param = sigma_by_param or {}
    s = f"{name}\n"
    s += "=" * len(name) + "\n"
    s += "{:<14} {:>30} {:>10} {:>30}\n".format("Parameter", "Value", "Status", "Sigma")
    s += "{}\n".format("-" * 90)
    for p in sorted(all_coeff.keys(), key=lambda x: _all_coeff_sort_key(str(x))):
        val = _fmt(all_coeff[p])
        status = "fitted" if p in fitted_coeff else "fixed"
        sig = _fmt(sigma_by_param[p]) if p in sigma_by_param else "n/a"
        s += "{:<14} {:>30} {:>10} {:>30}\n".format(str(p), str(val), status, str(sig))
    return s + "\n"


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
            v = rp_list[i][ii]
            if q == 0:
                # q=0 components are physically real; drop numerical imag roundoff.
                v = np.real(v)
            if v != 0:
                try:
                    rcoeff["C%i%i" % (k, q)] = v
                except KeyError:
                    pass
    return rcoeff


def conjugate_cf_params(coeff: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Complex-conjugate crystal-field parameters in a coefficient dictionary.

    Parameters
    ----------
    coeff : dict
        Coefficient dictionary in the usual format. Only parameters with names
        Ckq, where k and q are zero or positive integers, are touched.

    Returns
    -------
    rcoeff : dict
        A copy of coeff with all Ckq parameters complex conjugated.
    """
    rcoeff = coeff.copy()
    for key, value in coeff.items():
        if re.fullmatch(r"C\d{2,}", key):
            rcoeff[key] = np.conj(value)
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


def compute_chi2_numpy(efit: "cfl.EFit") -> float:
    r"""Compute chi² from fitted eigenvalues and experimental data.

    Vectorized NumPy computation matching C objective: sum(w * residual²).
    Used by both marker-column fits (via Cython objective wrapper) and
    display functions. Shared source of truth for chi² calculation.

    Parameters
    ----------
    efit : cfl.EFit or object with h and ex attributes
        EFit instance (or similar) with diagonalized Hamiltonian and
        experimental data.
        Must have: ``efit.h`` (Hamiltonian with eigenvalues w) and
        ``efit.ex`` (ExData with la, e, w).

    Returns
    -------
    float
        Sum of weighted squared residuals: ``sum(w_i * (E_fitted - E_expt)²)``

    Raises
    ------
    ValueError
        If any experimental level index (la) is invalid (<0).
    """
    evals = efit.h.w
    ex = efit.ex
    n_a = getattr(ex, "n_a", len(ex.la) if ex.la is not None else 0)
    n_d = getattr(ex, "n_d", 0)

    chi2 = 0.0

    # Absolute energies: sum_i w_i * (eval[la_i] - e_i)^2
    if n_a > 0:
        if n_d == 0:
            la_a = ex.la
            e_a = ex.e
            w_a = ex.w
        else:
            la_a = ex.la[:n_a]
            e_a = ex.e[:n_a]
            w_a = ex.w[:n_a]
        if (la_a >= 0).all():
            r = evals[la_a] - e_a
            chi2 += float(np.sum(w_a * r * r))
        else:
            valid = la_a >= 0
            if valid.any():
                r = evals[la_a[valid]] - e_a[valid]
                chi2 += float(np.sum(w_a[valid] * r * r))

    # Difference energies: sum_i w_i * (|eval[fld_i] - eval[ild_i]| - dE_i)^2
    if n_d > 0:
        ild = ex.ild[:n_d]
        fld = ex.fld[:n_d]
        e_d = ex.e[n_a : n_a + n_d]
        w_d = ex.w[n_a : n_a + n_d]
        if (ild >= 0).all() and (fld >= 0).all():
            r = np.abs(evals[fld] - evals[ild]) - e_d
            chi2 += float(np.sum(w_d * r * r))
        else:
            valid = (ild >= 0) & (fld >= 0)
            if valid.any():
                r = np.abs(evals[fld[valid]] - evals[ild[valid]]) - e_d[valid]
                chi2 += float(np.sum(w_d[valid] * r * r))

    return chi2
