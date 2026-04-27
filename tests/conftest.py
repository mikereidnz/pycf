#!/usr/bin/env python3
"""
Pytest fixtures for pycf tests.
This module provides reusable fixtures for:
- ImportSLJM tensor loading from test data directories
- Hamiltonian construction with standard coefficients
- ExData (experimental data) setup for different test scenarios
- Common test parameters and data paths
"""

from pathlib import Path

import numpy as np
import pytest

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM


# ============================================================================
# Path Fixtures
# ============================================================================
@pytest.fixture
def ceylf_matel_path():
    """Path to Ce:YLF crystal field matrix elements directory."""
    return Path(__file__).resolve().parent / "integration" / "ceylf" / "matel" / "f1cf"


@pytest.fixture
def ceylf_inten_path():
    """Path to Ce:YLF intensity matrix elements directory."""
    return Path(__file__).resolve().parent / "integration" / "inten" / "matel" / "f1int"


@pytest.fixture
def inten_c1_matel_path():
    """Path to C1 intensity test matrix elements directory."""
    return Path(__file__).resolve().parent / "integration" / "inten" / "matel" / "f1cf"


@pytest.fixture
def inten_c3_matel_path():
    """Path to C3 intensity test matrix elements directory."""
    return Path(__file__).resolve().parent / "integration" / "inten" / "matel" / "f1cf"


@pytest.fixture
def eryso_path():
    """Path to Er:YSO example data directory."""
    return Path(__file__).resolve().parent / "integration" / "eryso"


# ============================================================================
# ImportSLJM Fixtures (Tensor Loading)
# ============================================================================
@pytest.fixture
def ceylf_tensors(ceylf_matel_path):
    """Load Ce:YLF crystal field tensors from standard test data."""
    return ImportSLJM(str(ceylf_matel_path))


@pytest.fixture
def ceylf_inten_tensors(ceylf_inten_path, ceylf_matel_path):
    """Load Ce:YLF intensity tensors, aligned with crystal field basis."""
    return ImportSLJM(str(ceylf_inten_path), sl_name=str(ceylf_matel_path))


@pytest.fixture
def inten_c1_tensors(inten_c1_matel_path):
    """Load C1 intensity test tensors."""
    return ImportSLJM(str(inten_c1_matel_path))


@pytest.fixture
def inten_c3_tensors(inten_c3_matel_path):
    """Load C3 intensity test tensors."""
    return ImportSLJM(str(inten_c3_matel_path))


# ============================================================================
# Hamiltonian Fixtures
# ============================================================================
@pytest.fixture
def ceylf_hamiltonian(ceylf_tensors):
    """
    Create Ce:YLF Hamiltonian with crystal field tensors.
    Returns unconfigured Hamiltonian; use set_coeff() to add parameters.
    """
    return cfl.Hamiltonian(
        [
            ceylf_tensors.EAVG,
            ceylf_tensors.ZETA,
            ceylf_tensors.C20,
            ceylf_tensors.C40,
            ceylf_tensors.C44,
            ceylf_tensors.C60,
            ceylf_tensors.C64,
        ]
    )


@pytest.fixture
def ceylf_hamiltonian_configured(ceylf_hamiltonian):
    """
    Create Ce:YLF Hamiltonian with standard fitted parameters.
    Parameters from 10.1016/j.optmat.2015.06.046 (Ce:YLF experiment).
    """
    coeff = {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
    }
    ceylf_hamiltonian.set_coeff(coeff)
    return ceylf_hamiltonian


@pytest.fixture
def ceylf_diagonalized(ceylf_hamiltonian_configured):
    """
    Ce:YLF Hamiltonian eigenvalues and eigenvectors.
    Returns (eigenvalues, eigenvectors) with eigenvalues shifted to zero-based.
    """
    w, z = ceylf_hamiltonian_configured.diag()
    w = w - np.min(w)  # Shift to ground state at zero energy
    return w, z


# ============================================================================
# ExData Fixtures (Experimental Data)
# ============================================================================
@pytest.fixture
def ceylf_exdata_abs():
    """
    Ce:YLF experimental data: absolute energy level values.
    Returns ExData object with 'A' (absolute) mode.
    """
    data = np.array(
        [
            [2, 0],
            [3, 216],
            [8, 2216],
            [9, 2312.8],
            [10, 2314],
            [11, 2396],
            [12, 3158.5],
            [13, 3158.7],
            [14, 3240.3],
        ]
    )
    return cfl.ExData(data, "A")


@pytest.fixture
def ceylf_exdata_abs_diff():
    """
    Ce:YLF experimental data: absolute values + differences.
    Returns ExData object with 'AD' (absolute + differences) mode.
    """
    data = np.array(
        [
            [2, 0],
            [3, 216],
            [8, 2216],
            [9, 2312.8],
            [3, 2, 216],  # Difference: level 3 - level 2
            [8, 2, 2216],  # Difference: level 8 - level 2
            [9, 8, 96.8],  # Difference: level 9 - level 8
        ]
    )
    return cfl.ExData((data[:4], data[4:]), ("A", "D"))


@pytest.fixture
def ceylf_exdata_sl_diff():
    """
    Ce:YLF experimental data: state label differences.
    Returns ExData object with 'SD' (state label + differences) mode.
    """
    ex_asl = np.array(
        [
            [1, 2, 0],  # State label difference
            [1, 3, 216],
            [1, 8, 2216],
            [1, 9, 2312.8],
        ]
    )
    ex_dsl = np.array([[3, 2, 216], [8, 2, 2216], [9, 8, 96.8]])
    return cfl.ExData((ex_asl, ex_dsl), ("AS", "DS"), label_key="SLJM")


# ============================================================================
# Minimizer Fixtures (if needed in future)
# ============================================================================
# These are placeholder fixtures for potential future minimizer testing.
# Uncomment and extend as needed.
# @pytest.fixture
# def ceylf_minimizer(ceylf_hamiltonian):
#     """Create a minimizer instance for Ce:YLF fitting."""
#     from pycf.minimizer import Minimizer
#     return Minimizer(ceylf_hamiltonian)
# ============================================================================
# Utility Fixtures
# ============================================================================
@pytest.fixture
def standard_ceylf_coefficients():
    """
    Standard Ce:YLF crystal field parameters.
    From 10.1016/j.optmat.2015.06.046 (experimental fit).
    """
    return {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
    }


@pytest.fixture
def inten_c1_coefficients():
    """
    Standard C1 intensity test coefficients.
    Parameters are tuned for demonstrating intensity calculations
    in C1 symmetry with split Kramers doublets.
    """
    return {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,
        "C60": 0,
    }


@pytest.fixture
def inten_c3_coefficients():
    """
    Standard C3 intensity test coefficients.
    Parameters are tuned for demonstrating intensity calculations
    in C3 symmetry.
    """
    return {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": -5500,
        "C60": 0,
    }
