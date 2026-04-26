#!/usr/bin/env python3
# file: test_inten_c3.py
# This is a test of the intensity calculation for Ce3+ in C3 symmetry.
# It calculates transition intensities between the split levels without
# applying a symmetry-lowering magnetic field.
from pathlib import Path

import numpy as np
import pytest

import pycf
import pycf.cfl as cfl
from pycf.cfl_util import *
from pycf.import_sljm import ImportSLJM
from pycf.inten import *


# @pytest.mark.slow  # mark this test as slow, so it can be skipped by default
def test_inten_c3() -> None:
    """Test the intensity calculation for Ce3+ in C3 symmetry.
    This computes transition intensities between the split levels without
    applying a symmetry-lowering magnetic field.
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
        #'C43'   :     200-100j, # complex conjugate of what went into CFIT
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 0,
        #'MX'    :     1e-10,
        "MY": 0,
        "MZ": 0,
        #'MZ'    :     1,
    }
    # Bohr magnetion in cm-1/T.
    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"
    h = cfl.Hamiltonian(
        [t.EAVG, t.ZETA, t.C20, t.C40, t.C43, t.C60, t.C63, t.C66, MX, MY, MZ]
    )
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
    lrange = [[0, 1], [6, 7, 8, 9]]  # Z1 to Y1 + Y2
    print("lrange", lrange)
    trs = dipole_str(lrange, tensor_dict, h, E, V, ed=True, Altp=Altp)
    # sort the dictionary
    from operator import itemgetter
    trs.sort(key=itemgetter("e"))
    labels = h.tensors[0].states.labels
    print("\ntrs\n")
    print("INTENSITIES")
    for i, line in enumerate(trs):
        print(
            "\n",
            i,
            ":",
            line["i"],
            line["ei"],
            labels[line["pci"]],
            "\t->\t",
            line["f"],
            line["ef"],
            labels[line["pcf"]],
            "\tEnergy:",
            line["e"],
        )
        print(
            "\tD_ISO:",
            "\tED:",
            (abs(line["ed_-1"]) ** 2 + abs(line["ed_0"]) ** 2 + abs(line["ed_+1"]) ** 2)
            / 3,
            "\tMD:",
            (abs(line["md_-1"]) ** 2 + abs(line["md_0"]) ** 2 + abs(line["md_+1"]) ** 2)
            / 3,
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
    print("\nCreate transition groups")
    # group the transitions by energy.
    groups = group_transitions(trs, tol=1e-3)
    add_oscillator_strengths_and_A_coefficients(groups)
    # print(groups)
    print("\nCompare oscillator strengths to Pascal calculation")
    tolerance = 1e-6
    # From Pascal calculation (cfit/vtrans/inten)
    pascal_f = [4.482614e-08, 4.148602e-08]
    # pascal_f =[4.482614e-08,4.14e-08]# deliberately less precise value for the second transition to test the assertion
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
        print("\tA:", group["A"], "f:", group["f"], "Compare to Pascal f:", pascal_f[i])
        assert group["f"] == pytest.approx(
            pascal_f[i], rel=tolerance
        ), f'Group {i} oscillator strength \
        {group["f"]} differs from Pascal calculation {pascal_f[i]} by more than the tolerance of {tolerance}'
    # Nolrange = [[0, 1], [6, 7, 8, 9]]  # Z1 to Y1 + Y2
    print("lrange", lrange)
    trs = dipole_str(lrange, tensor_dict, h, E, V, ed=True, Altp=Altp)
    # sort the dictionary
    from operator import itemgetter
    trs.sort(key=itemgetter("e"))
    labels = h.tensors[0].states.labels
    print("\ntrs\n")
    print("INTENSITIES")
    for i, line in enumerate(trs):
        print(
            "\n",
            i,
            ":",
            line["i"],
            line["ei"],
            labels[line["pci"]],
            "\t->\t",
            line["f"],
            line["ef"],
            labels[line["pcf"]],
            "\tEnergy:",
            line["e"],
        )
        print(
            "\tD_ISO:",
            "\tED:",
            (abs(line["ed_-1"]) ** 2 + abs(line["ed_0"]) ** 2 + abs(line["ed_+1"]) ** 2)
            / 3,
            "\tMD:",
            (abs(line["md_-1"]) ** 2 + abs(line["md_0"]) ** 2 + abs(line["md_+1"]) ** 2)
            / 3,
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
    print("\nCreate transition groups")
    # group the transitions by energy.
    groups = group_transitions(trs, tol=1e-3)
    add_oscillator_strengths_and_A_coefficients(groups)
    # print(groups)
    print("\nCompare oscillator strengths to Pascal calculation")
    tolerance = 1e-6
    # From Pascal calculation (cfit/vtrans/inten)
    pascal_f = [4.482614e-08, 4.148602e-08]
    # pascal_f =[4.482614e-08,4.14e-08]# deliberately less precise value for the second transition to test the assertion
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
        print("\tA:", group["A"], "f:", group["f"], "Compare to Pascal f:", pascal_f[i])
        assert group["f"] == pytest.approx(
            pascal_f[i], rel=tolerance
        ), f'Group {i} oscillator strength \
        {group["f"]} differs from Pascal calculation {pascal_f[i]} by more than the tolerance of {tolerance}'
    # Now do emission from Y1 and Y2 to Z
    print("\nNow do emission from Y1 and Y2 to Z\n")
    lrange = [[6, 7], [0, 1, 2, 3, 4, 5]]  # Y1, Y2 to Z
    print("lrange", lrange)
    trs_em = dipole_str(lrange, tensor_dict, h, E, V, ed=True, Altp=Altp)
    # sort the dictionary
    from operator import itemgetter
    trs_em.sort(key=itemgetter("e"))
    labels = h.tensors[0].states.labels
    # print('\ntrs_em\n')
    print("\nCreate transition groups")
    # group the transitions by energy.
    groups_em = group_transitions(trs_em, tol=1e-3)
    add_oscillator_strengths_and_A_coefficients(groups_em)
    # print(groups_em)
    print("\nCompare oscillator strengths to Pascal calculation")
    tolerance = 1e-6
    # From Pascal calculation (cfit/vtrans/inten)
    pascal_f = [4.482614e-08, 4.148602e-08]
    # pascal_f =[4.482614e-08,4.14e-08]# deliberately less precise value for the second transition to test the assertion
    for i, group in enumerate(groups_em):
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
        print("\tA:", group["A"], "f:", group["f"])
        #'Compare to Pascal f:', pascal_f[i])
        # assert group['f'] == pytest.approx(pascal_f[i], rel=tolerance), f'Group {i} oscillator strength \
        # {group["f"]} differs from Pascal calculation {pascal_f[i]} by more than the tolerance of {tolerance}'
    A_total = sum(group["A"] for group in groups_em)
    print("\nTotal A coefficient for all transitions:", A_total)
    print(
        "Lifetime corresponding to total A coefficient:",
        1 / A_total,
        "seconds",
        "or",
        1e3 / A_total,
        "milliseconds",
    )
    pascal_A = 3.193699e-01
    tolerance = 1e-5
    print("Pascal calculation A coefficient:", pascal_A)
    assert A_total == pytest.approx(
        pascal_A, rel=tolerance
    ), f"Atotal \
        {A_total} differs from Pascal calculation {pascal_A} by more than the tolerance of {tolerance}"
if __name__ == "__main__":
    # for running from spyder or as a stand-alone file
    pycf.pycf_info()
    print("\nIntensity tests\n")
    test_inten_c3()
