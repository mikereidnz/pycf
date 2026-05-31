#!/usr/bin/env python3
# file: test_inten_c1.py
# This is a test of the intensity calculation for Ce3+ in C1 symmetry.
# We apply a magnetic field along Z to split the Kramers doublets,
# and then calculate the intensities of the transitions between the split levels.
# A small magnetic field is applied along X to achieve C1 symmetry.
from pathlib import Path

import pycf
import pycf.cfl as cfl
from pycf.cfl_util import *
from pycf.import_sljm import ImportSLJM
from pycf.inten import *


def test_inten_c1() -> None:
    """Test the intensity calculation for Ce3+ in C1 symmetry.
    We apply a magnetic field along Z to split the Kramers doublets,
    and then calculate the intensities of the transitions between the split levels.
    A small magnetic field is applied along X to achieve C1 symmetry.
    """
    # Read the tensors for the crystal-field calculation and the intensity calculation.
    MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"
    INTEN_BASE = Path(__file__).resolve().parent / "matel" / "f1int"
    # for crystal-field calcualtion:
    t = ImportSLJM(MATEL_BASE)
    # for intensity calculation:
    t_int = ImportSLJM(INTEN_BASE, sl_name=MATEL_BASE)
    coeff = {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,  # what goes into CFIT
        # 'C43'   :     200-100j, # complex conjugate of what went into CFIT
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 1e-10,
        "MY": 0,
        "MZ": 1,
    }
    # Bohr magnetion in cm-1/T.
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C43, t.C60, t.C63, t.C66, MX, MY, MZ])
    h.set_coeff(coeff)
    ex = np.array(
        [
            [2, 0],
            [4, 1],
            [6, 2],
        ]
    )
    weights = np.ones(len(ex))
    exdata = cfl.ExData(ex, "A", weights=weights)
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True)
    param = ["EAVG"]
    res = cfl.e_fit(param, h, exdata, cfl_min)
    print(res["summary"])
    """
    Copy the fitted coefficients back.
    """
    fitcoeff = res["coeff"]
    for p in fitcoeff:
        coeff[p] = fitcoeff[p]
    # calculate the eigenvalues and eigenvectors
    h.set_coeff(coeff)
    w, z = h.diag()
    E = w
    V = z
    #################################################
    print("\nState label_key:", h.tensors[0].states.label_key)
    # print('\nState labels:\n', h.tensors[0].states.labels)
    print("\nState labels:\n")
    for i, state in enumerate(h.tensors[0].states.labels):
        print(i, state)
    print("\n")
    print("\ncoeff:", coeff)
    # print('\nE:\n', E, '\nV:\n', np.real(V), '\n')  # omit imaginary part for clarity
    """
    Now transform tensors to the eigenbasis.
    the vtrans function uses symmetry to generate the -q tensors.
    """
    tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
    tensor_dict = vtrans(tensors, V)
    # print('\ntensor_dict')
    # print('M1-1')
    # print(tensor_dict['M1-1'][np.ix_([0,1,6,7,8,9], [0,1,6,7,8,9])])
    """
    dipole strengths
    """
    Altp = [["A210", 1e-10], ["A230", -1e-10], ["A233", 1e-10 + 2e-10j]]
    print("Altp")
    for A in Altp:
        print(A)
    i_range = [1, 2]  # Z1 (1-based)
    f_range = [7, 8, 9, 10]  # Y1 + Y2 (1-based)
    lrange = [i_range, f_range]
    print("lrange", lrange)
    trs = dipole_str(i_range, f_range, tensor_dict, h, E, V, ed=True, Altp=Altp)
    # sort the dictionary
    from operator import itemgetter

    trs.sort(key=itemgetter("e"))
    labels = h.tensors[0].states.labels
    print("\ntrs\n")
    # From Pascal calculation (cfit/vtrans/inten)
    pascal_calculation = [
        {"isotropic": 1.71098e-05, "axial": 0, "sigma": 4.22286e-05, "pi": 9.10073e-06},
        {
            "isotropic": 6.32926e-03,
            "axial": 9.49388e-03,
            "sigma": 5.70304e-05,
            "pi": 9.43685e-03,
        },
        {
            "isotropic": 6.33295e-03,
            "axial": 9.49943e-03,
            "sigma": 5.74526e-05,
            "pi": 9.44198e-03,
        },
        {"isotropic": 1.78411e-05, "axial": 0, "sigma": 4.35112e-05, "pi": 1.00121e-05},
        {
            "isotropic": 1.66891e-04,
            "axial": 2.50337e-04,
            "sigma": 2.43952e-04,
            "pi": 6.38448e-06,
        },
        {"isotropic": 5.35368e-03, "axial": 0, "sigma": 4.38493e-03, "pi": 1.16761e-02},
        {"isotropic": 5.34163e-03, "axial": 0, "sigma": 4.38983e-03, "pi": 1.16351e-02},
        {
            "isotropic": 1.57487e-04,
            "axial": 2.36230e-04,
            "sigma": 2.30619e-04,
            "pi": 5.61136e-06,
        },
    ]
    print("INTENSITIES")
    for i, line in enumerate(trs):
        print(
            "\n",
            i,
            ":",
            line["i"],
            line["ei"],
            labels[line["pc_i"]],
            "\t->\t",
            line["f"],
            line["ef"],
            labels[line["pc_f"]],
            "\tEnergy:",
            line["e"],
        )
        print(
            "\tD_ISO:",
            "\tED:",
            (abs(line["ed_-1"]) ** 2 + abs(line["ed_0"]) ** 2 + abs(line["ed_+1"]) ** 2) / 3,
            "\tMD:",
            (abs(line["md_-1"]) ** 2 + abs(line["md_0"]) ** 2 + abs(line["md_+1"]) ** 2) / 3,
            "\tTOT:",
            line["isotropic"],
        )
        print(
            "ED",
            "\t-1:",
            line["ed_-1"],
            abs(line["ed_-1"]),
            "\t 0:",
            line["ed_0"],
            abs(line["ed_0"]),
            "\t+1:",
            line["ed_+1"],
            abs(line["ed_+1"]),
            "\nMD",
            "\t-1:",
            line["md_-1"],
            abs(line["md_-1"]),
            "\t 0:",
            line["md_0"],
            abs(line["md_0"]),
            "\t+1:",
            line["md_+1"],
            abs(line["md_+1"]),
        )
        # print('\n', line)
    print("\nCompare to Pascal calculation")
    tolerance = 1e-6
    for i, line in enumerate(trs):
        for key in ["isotropic", "axial", "sigma", "pi"]:
            print(i, key, line[key], pascal_calculation[i][key])
            abs_diff = abs(line[key] - pascal_calculation[i][key])
            assert (
                abs_diff < tolerance
            ), f"Absolute difference between pascal and pycf caculation is {abs_diff}, \
                which is greater than the tolerance of {tolerance}"
    print("\nCreate transition groups")
    # group the transitions by energy.
    groups = group_transitions(trs, tol=1e-3)
    # print(groups)
    for i, group in enumerate(groups):
        print(
            "Group",
            i,
            "Energy:",
            group["Energy"],
            "g_i",
            group["g_i"],
            "g_f",
            group["g_f"],
            "e_i",
            group["e_i"],
            "e_f",
            group["e_f"],
        )
        for line in group["t_list"]:
            print("\t", line["i"], line["f"], line["e"], line["isotropic"])
    """
    Plot data: skip old inten() function test, as inten_plot() replaces it
    """
    # inten() has been replaced by the more versatile inten_plot()
    # This test demonstrates that the intensity calculation completes without error
    print("\n Intensity calculation complete (old inten() function replaced by inten_plot())")


if __name__ == "__main__":
    # for running from spyder or as a stand-alone file
    pycf.pycf_info()
    print("\nIntensity tests\n")
    test_inten_c1()
