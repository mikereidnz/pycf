from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pycf.cfl import Hamiltonian
from pycf.cfl_util import printeigenmatrix, printeigenvectors, printmatrix, printstates
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum


class _Tensor:
    def __init__(self, name, matrix, states):
        self.name = name
        self._matrix = np.asarray(matrix, dtype=np.complex128)
        self.states = states

    def get_matel(self):
        return self._matrix.copy()


def _hamiltonian(tensors):
    states = tensors[0].states
    size = len(states.labels)
    w, z = np.arange(size, dtype=float), np.eye(size, dtype=np.complex128)
    return SimpleNamespace(
        tensors=tensors,
        has_diagonalization=True,
        w=w,
        z=z,
        diag=Mock(return_value=(w, z)),
    )


def _states():
    return SimpleNamespace(labels=[(0, 0), (1, 1)], label_key="JM")


def test_printstates_lists_original_basis_indices(capsys):
    states = _states()
    hamiltonian = _hamiltonian([_Tensor("C20", np.eye(2), states)])

    printstates(hamiltonian)

    output = capsys.readouterr().out
    assert "Basis index" in output
    assert "| 0,  0>" in output
    assert "| 1,  1>" in output


def test_printmatrix_completes_q0_hermitian_lower_triangle(capsys):
    states = _states()
    tensor = _Tensor("C20", [[1, 2 + 3j], [0, 4]], states)
    hamiltonian = _hamiltonian([tensor])

    printmatrix(hamiltonian, "C20", [2], [1])

    output = capsys.readouterr().out
    assert "-3.000000e+00j" in output
    assert "2.000000e+00" in output


@pytest.mark.parametrize("eigenbasis", [False, True])
def test_matrix_column_blocks_repeat_row_and_column_labels(capsys, eigenbasis):
    states = _states()
    tensor = _Tensor("C20", [[1, 2 + 3j], [0, 4]], states)
    hamiltonian = _hamiltonian([tensor])
    printer = printeigenmatrix if eigenbasis else printmatrix

    printer(hamiltonian, "C20", [1, 2], [2, 1], width=48)

    first, second = capsys.readouterr().out.split("\n", 1)[1].strip("\n").split("\n\n")
    for block, index, imaginary in [(first, 2, "+"), (second, 1, "-")]:
        header, row1, row2 = block.splitlines()
        assert header.endswith(f"{index} ({index - 1:.2f})" if eigenbasis else str(index))
        assert row1.lstrip().startswith("1 ")
        assert row2.lstrip().startswith("2 ")
        complex_row = row1 if index == 2 else row2
        assert complex_row.endswith(f"2.000000e+00{imaginary}3.000000e+00j")
        assert len(header) == len(row1) == len(row2) <= 48


@pytest.mark.parametrize("eigenbasis", [False, True])
def test_element_list_filters_by_magnitude_and_keeps_both_directions(capsys, eigenbasis):
    states = _states()
    tensor = _Tensor("C20", [[0, 0.3 + 0.4j], [0, 0.4]], states)
    hamiltonian = _hamiltonian([tensor])
    printer = printeigenmatrix if eigenbasis else printmatrix
    printer(hamiltonian, "C20", [1, 2], [1, 2], format="elements", cutoff=0.4, width=70)
    entries = [line for line in capsys.readouterr().out.splitlines() if line.startswith("<")]
    assert len(entries) == 2
    assert all(len(line) <= 70 and line.count(" = ") == 1 for line in entries)
    left, right = ("1 (0.00)", "2 (1.00)") if eigenbasis else ("1:  0,  0", "2:  1,  1")
    assert f"<{left} | C20 | {right}> =  3.000000e-01+4.000000e-01j" in entries
    assert f"<{right} | C20 | {left}> =  3.000000e-01-4.000000e-01j" in entries


def test_element_list_aligns_slots_with_mixed_signs_and_label_lengths(capsys):
    states = SimpleNamespace(labels=[(0, 0)] * 10, label_key="JM")
    matrix = np.zeros((10, 10))
    matrix[0, 0], matrix[0, 9], matrix[9, 9] = 1, -2, 3
    hamiltonian = _hamiltonian([_Tensor("C20", matrix, states)])

    printmatrix(hamiltonian, "C20", [1, 10], [1, 10], format="elements", width=120)

    first, second = [line for line in capsys.readouterr().out.splitlines() if line.startswith("<")]
    assert first.index("<", 1) == second.index("<", 1)
    assert [i for i, char in enumerate(first) if char == "="] == [
        i for i, char in enumerate(second) if char == "="
    ]
    assert first.index("1.000000e+00") == second.index("2.000000e+00")
    assert max(len(first), len(second)) <= 120


def test_element_list_zero_cutoff_preserves_small_imaginary_entries(capsys):
    states = _states()
    tensor = _Tensor("U21", [[0, 1e-16j], [0, 0]], states)
    hamiltonian = _hamiltonian([tensor])

    printmatrix(hamiltonian, "U21", [1, 2], [1, 2], format="elements")
    output = capsys.readouterr().out
    assert output.count(" = ") == 1
    assert "0.000000e+00+1.000000e-16j" in output

    printmatrix(hamiltonian, "U21", [1, 2], [1, 2], format="elements", cutoff=1e-8)
    assert "No matrix elements exceed the cutoff." in capsys.readouterr().out


def test_printeigenvectors_prints_every_basis_component(capsys):
    states = SimpleNamespace(labels=[(0, 0), (1, 1), (2, 2)], label_key="JM")
    tensor = _Tensor("C20", np.eye(3), states)
    hamiltonian = _hamiltonian([tensor])

    printeigenvectors(hamiltonian, [2])

    output = capsys.readouterr().out
    assert output.splitlines()[2].split()[:2] == ["2", "1.0000"]
    fields = output.splitlines()[2].split()[2:]
    fields += " ".join(output.splitlines()[3:]).split()
    assert fields[0:3] == ["0", "0.0%", "1"]
    assert fields[6:9] == ["1", "100.0%", "2"]
    assert fields[12:15] == ["0", "0.0%", "3"]
    hamiltonian.diag.assert_not_called()


def test_printeigenvectors_filters_by_modulus_and_wraps(capsys):
    states = _states()
    hamiltonian = _hamiltonian([_Tensor("C20", np.eye(2), states)])
    hamiltonian.z[:, 0] = [0.5j, np.sqrt(0.75)]
    printeigenvectors(hamiltonian, [1], cutoff=0.5, width=55)
    first, second = capsys.readouterr().out.splitlines()[2:]
    assert first.split()[2:5] == ["0+0.5j", "25.0%", "1"]
    assert second.split()[:3] == ["0.866025", "75.0%", "2"]
    assert first.index("25.0%") == second.index("75.0%")
    assert first.index("|") == second.index("|")
    assert max(len(first), len(second)) <= 55

    printeigenvectors(hamiltonian, [1], cutoff=0.9)
    assert "no components meet cutoff" in capsys.readouterr().out


def test_hamiltonian_requires_current_diagonalization():
    base = Path(__file__).resolve().parents[1] / "integration/inten/matel/f1cf"
    tensor = ImportSLJM(base).C20
    hamiltonian = Hamiltonian([tensor])
    for set_coeff in (hamiltonian.set_coeff, hamiltonian.update_coeff):
        set_coeff({"C20": 1})
        with pytest.raises(ValueError, match=r"h\.diag\(\)"):
            printeigenvectors(hamiltonian, [1])
        hamiltonian.diag()
        printeigenmatrix(hamiltonian, "C20", [1], [1])


def test_printeigenmatrix_prints_ed_parameter_moments_by_q(capsys):
    states = _states()
    hamiltonian_tensor = _Tensor("C20", np.eye(2), states)
    hamiltonian = _hamiltonian([hamiltonian_tensor])
    unit_tensors = [
        _Tensor("U20", np.zeros((2, 2)), states),
        _Tensor("U21", [[0, 1], [0, 0]], states),
    ]
    spectrum = Spectrum(
        hamiltonian, "ED test", [1], [2], unit_tensors, altp={"A210": 1e-10}, md=False, ed=True
    )
    spectrum.calculate_intensities()
    spectrum.update_altp(A210=2e-10)

    printeigenmatrix(spectrum, "A210", [1], [2])

    output = capsys.readouterr().out
    _, header, row = output.splitlines()
    assert header.split()[:5] == ["initial", "E_i", "final", "E_f", "DeltaE"]
    assert row.split()[:5] == ["1", "0.000", "2", "1.000", "1.000"]
    assert "q=-1" in output
    assert "q=0" in output
    assert "q=+1" in output
    assert "A210 electric-dipole moment contribution" in output
    assert "5.477226e-01" in output
    hamiltonian.diag.assert_called_once()


def test_spectrum_unit_tensor_in_original_and_cached_eigenbasis(capsys):
    states = _states()
    hamiltonian = _hamiltonian([_Tensor("C20", np.eye(2), states)])
    unit_tensor = _Tensor("U20", [[1, 2], [0, 3]], states)
    spectrum = Spectrum(hamiltonian, "Unit tensor test", [1], [2], [unit_tensor], md=False)
    printmatrix(spectrum, "U20", [2], [1])
    assert "2.000000e+00" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"calculate_intensities\(\)"):
        printeigenmatrix(spectrum, "U20", [1], [2])

    spectrum.calculate_intensities()
    hamiltonian.z[:] = np.fliplr(np.eye(2))
    printeigenvectors(spectrum, [1], cutoff=0.5)
    assert capsys.readouterr().out.splitlines()[-1].split()[2:5] == ["1", "100.0%", "1"]
    printeigenmatrix(spectrum, "U20", [1], [2])

    output = capsys.readouterr().out
    assert "U20 matrix in the eigenbasis" in output
    assert "2.000000e+00" in output
    hamiltonian.diag.assert_called_once()

    unit_tensor.states = SimpleNamespace(labels=states.labels[::-1], label_key=states.label_key)
    with pytest.raises(ValueError, match="does not match the Hamiltonian basis"):
        printmatrix(spectrum, "U20", [1], [2])
