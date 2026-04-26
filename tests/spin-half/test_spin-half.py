#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pytest
import pycf
import pycf.cfl as cfl
from pycf.cfl_util import *
from pycf.import_sljm import ImportSLJM
def print_calc():
    """Print the parameters and summary"""
    print("\nParameters")
    print("==========")
    print(h.coeff_dict, "\n")
    print(h.gen_summary())
def eig(H):
    """Calculate eigenvalues and eigenvectors of a Hamiltonian matrix H,
    and sort them in ascending order of energy."""
    w, z = np.linalg.eig(H)
    w = w.real
    idx = w.argsort()
    w = w[idx]
    z = z[:, idx]
    return w, z
def norm_eig(z):
    """Make largest component positive and real"""
    z_n = np.zeros_like(z)
    for j in range(z.shape[1]):
        col = z[:, j]
        y = col[np.argmax(abs(col))]
        z_n[:, j] = np.conj(y) * z[:, j] / abs(y)
    return z_n
def norm_eig_top(z):
    """make first component positive and real"""
    z_n = np.zeros_like(z)
    # print('\nnorm_eig_top')
    for j in range(z.shape[1]):
        col = z[:, j]
        # print(col)
        y = col[0]
        z_n[:, j] = np.conj(y) * z[:, j] / abs(y)
    return z_n
#### 1S example spin-half state
MATEL_BASE = Path(__file__).resolve().parent / "matel" / "s1cf"
t = ImportSLJM(str(MATEL_BASE))
h = cfl.Hamiltonian([t.EAVG, t.MAGX, t.MAGY, t.MAGZ])
# For running as part of a test suite from repo root:
#  python -m pytest tests
@pytest.mark.parametrize("data_sel", ["real", "imag", "complex"])
def test_spin_half(data_sel) -> None:
    #### Testing a spin-half system, so that we can compare pycf eigenvectors to numpy.linalg.eig() eigenvectors, which are complex.  This is a test of the handling of complex numbers in pycf.
    print("\nRunning a spin-half test:\n")
    # Define coeff locally so parametrized variants don't share mutable state.
    coeff = {
        "EAVG": 0.0,
        "MAGX": 0.0,
        "MAGY": 0.0,
        "MAGZ": 0.0,
    }
    if data_sel == "real":
        print("data_sel is real")
        coeff["MAGX"] = 1.0
        coeff["MAGY"] = 0.0
    elif data_sel == "imag":
        print("data_sel is imag")
        coeff["MAGX"] = 0.0
        coeff["MAGY"] = 1
    elif data_sel == "complex":
        print("data_sel is complex")
        coeff["MAGX"] = 1.0
        coeff["MAGY"] = 1.0
    print("STATES")
    for i, state in enumerate(h.tensors[0].states.labels):
        print(i, state)
    # print('TENSORS')
    # for label, tensor in t.tensors.items():
    #    print(label, '\n',
    #        tensor.get_matel())
    MAGX = t.MAGX.get_matel()
    print("MAGX")
    print(MAGX)
    MAGY = t.MAGY.get_matel()
    print("MAGY")
    print(MAGY)
    MX = MAGX - np.tril(MAGX, k=-1)  # subtract the lower triangle
    print("MX upper triangle only")
    print(MX)
    MY = MAGY - np.tril(MAGY, k=-1)  # subtract the lower triangle
    print("MAGY upper triangle only")
    print(MY)
    M = coeff["MAGX"] * MX + coeff["MAGY"] * MY
    print("M")
    print(M)
    M = M + np.tril(M.conj().T, k=-1)
    print("M made Hermitian")
    print(M)
    print("\nCALCULATE EIGENVALUES AND EIGENVECTORS OF M\n")
    h.set_coeff(coeff)
    w, z = h.diag()
    # print_calc()
    print("\npycf h.diag()")
    print("Energies")
    print(w)
    print("Eigenvectors")
    print(z)
    print("Eigenvectors with first real and positive")
    print(norm_eig_top(z))
    print("\npython np.linalg.eig()")
    w_p, z_p = eig(M)
    print("Energies")
    print(w_p)
    print("Eigenvectors")
    print(z_p)
    print("Eigenvectors with first real and positive")
    print(norm_eig_top(z_p))
    print("\nDifference between numpy and pycf")
    print(norm_eig_top(z_p) - norm_eig_top(z))
    abs_diff = np.max(abs(norm_eig_top(z_p) - norm_eig_top(z)))
    print("Max absolute difference between numpy and pycf eigenvectors is", abs_diff)
    tolerance = 1e-10
    assert (
        abs_diff < tolerance
    ), f"Max absolute difference between numpy and pycf eigenvectors is {abs_diff}, \
    #which is greater than the tolerance of {tolerance}"
if __name__ == "__main__":
    # for running from spyder or as a stand-alone file
    pycf.pycf_info()
    print("\nSpin-half tests\n")
    data_sel_list = [
        "real",
        "imag",
        "complex",
    ]
    for data_sel in data_sel_list:
        test_spin_half(data_sel)
