"""Regression tests for complex crystal-field spin-Hamiltonian projections."""

from types import SimpleNamespace

import numpy as np
import pytest

from pycf import cfl
from pycf.import_sljm import ImportTensors


def _make_model(nuclear_spin=None):
    rng = np.random.default_rng(20261110)
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    S = [cfl.matel(axis, 0.5) for axis in ("jx", "jy", "jz")]
    g = np.array([[2.0, 0.4, 0.7], [-0.3, 1.5, 0.9], [0.6, 0.8, 3.0]])
    A = np.array([[0.03, 0.004, -0.002], [0.004, 0.04, 0.003], [-0.002, 0.003, 0.05]])
    Q = np.array([[0.006, -0.001, 0.002], [-0.001, -0.004, 0.0015], [0.002, 0.0015, -0.002]])
    h0 = U @ np.diag([0.0, 0.0, 10.0, 10.0]) @ U.conj().T
    operators = {
        name: U @ np.kron(np.eye(2), sum(g[i, j] * S[j] for j in range(3))) @ U.conj().T
        for i, name in enumerate(("MAGX", "MAGY", "MAGZ"))
    }
    labels = [[1, -1], [1, 1], [3, -1], [3, 1]]
    couplings = {}
    I = None
    nI = 1
    if nuclear_spin is not None:
        I = [cfl.matel(axis, nuclear_spin) for axis in ("jx", "jy", "jz")]
        nI = len(I[0])
        electronic_spin = [U @ np.kron(np.eye(2), spin) @ U.conj().T for spin in S]
        operators = {name: np.kron(op, np.eye(nI)) for name, op in operators.items()}
        operators["HYP"] = sum(
            A[i, j] * np.kron(electronic_spin[j], I[i]) for i in range(3) for j in range(3)
        )
        operators["EQHYP"] = np.kron(
            np.eye(4), sum(Q[i, j] * I[i] @ I[j] for i in range(3) for j in range(3))
        )
        h0 = np.kron(h0, np.eye(nI))
        labels = [[*label, int(round(2 * m))] for label in labels for m in np.diag(I[2]).real]
        couplings = {"HYP": 0.7, "EQHYP": -0.4}
    tensors = ImportTensors(
        "JM" if nuclear_spin is None else "JMI",
        np.array(labels, dtype=np.int32),
        dict(operators, H0=h0, MAGZS=1e-6 * operators["MAGZ"]),
        warn_zero=False,
    )
    h = cfl.Hamiltonian([tensors.H0, tensors.MAGZS])
    h.set_coeff({"H0": 1.0, "MAGZS": 1.0})
    interactions = ["zeeman"]
    kwargs = {"S": 0.5, "level": 1}
    if nuclear_spin is not None:
        interactions += ["hyperfine", "quadrupole"]
        kwargs["I"] = nuclear_spin
    sh = cfl.SpinHamiltonian(interactions, **kwargs)
    sh.set_pro_data([tensors.tensors[name] for name in operators], couplings)
    values, blocks = sh.calc_param(h, matel=True)
    return SimpleNamespace(
        h=h,
        sh=sh,
        tensors=tensors,
        h0=h0,
        operators=operators,
        couplings=couplings,
        S=S,
        I=I,
        nI=nI,
        g=g,
        A=A,
        Q=Q,
        values=values,
        blocks=blocks,
    )


@pytest.fixture(params=[None, 1.0, 3.5], ids=["electronic", "I=1", "I=7/2"])
def model(request):
    return _make_model(request.param)


def test_projected_matrix_elements_match_numpy(model):
    V = model.h.z[:, : 2 * model.nI]
    for name, operator in model.operators.items():
        expected = V.conj().T @ operator @ V
        if name.startswith("MAG"):
            indices = [0, model.nI]
            expected = expected[np.ix_(indices, indices)]
            key = name.lower()
        elif name == "HYP":
            key = "hyperfine"
        else:
            expected = expected[: model.nI, : model.nI]
            key = "quadrupole"
        expected *= model.couplings.get(name, 1.0)
        np.testing.assert_allclose(model.blocks[key], expected, atol=1e-12, err_msg=name)


def test_extracted_g_preserves_known_magnitude_and_handedness(model):
    g = model.values[0]
    np.testing.assert_allclose(g @ g.T, model.g @ model.g.T, atol=1e-11)
    assert np.linalg.det(g) == pytest.approx(np.linalg.det(model.g), abs=1e-11)
    symmetric_g = model.sh.calc_param(model.h, svd_sym=True)[0]
    np.testing.assert_allclose(symmetric_g, symmetric_g.T, atol=1e-12)
    np.testing.assert_allclose(symmetric_g @ symmetric_g.T, model.g @ model.g.T, atol=1e-11)


def test_extracted_tensors_reproduce_full_mixed_field_spectra(model):
    g = model.values[0]
    if model.I is not None:
        A, Q = model.values[1:]
        np.testing.assert_allclose(Q, model.couplings["EQHYP"] * model.Q, atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.svd(A, compute_uv=False),
            np.linalg.svd(model.couplings["HYP"] * model.A, compute_uv=False),
            atol=1e-12,
        )
    for field in (np.array([0.0003, -0.0005, 0.0007]), np.array([0.0003, 0.0005, 0.0007])):
        field = 0.466860 * field
        full = model.h0 + sum(
            field[i] * model.operators[name] for i, name in enumerate(("MAGX", "MAGY", "MAGZ"))
        )
        effective = sum(
            field[i] * g[i, j] * np.kron(model.S[j], np.eye(model.nI))
            for i in range(3)
            for j in range(3)
        )
        if model.I is not None:
            full += sum(
                coupling * model.operators[name] for name, coupling in model.couplings.items()
            )
            effective += sum(
                A[i, j] * np.kron(model.S[j], model.I[i])
                + Q[i, j] * np.kron(np.eye(2), model.I[i] @ model.I[j])
                for i in range(3)
                for j in range(3)
            )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(effective),
            np.linalg.eigvalsh(full)[: 2 * model.nI],
            atol=1e-11,
        )


@pytest.mark.parametrize("multiple", [False, True], ids=["ESHFit", "MESHFit"])
def test_tensor_fit_objective_matches_known_complex_model(multiple):
    model = _make_model()
    U, s, _ = np.linalg.svd(model.g)
    reference = {"zeeman": U @ np.diag(s) @ U.T}
    ex = np.empty((0, 2))
    if multiple:
        fit = cfl.MESHFit(
            ["H0"],
            [
                {
                    "h": model.h,
                    "sh": model.sh,
                    "ex": ex,
                    "shx": reference,
                    "weights": {},
                    "svd_sym": True,
                }
            ],
        )
    else:
        fit = cfl.ESHFit(["H0"], model.h, model.sh, ex, reference, {}, svd_sym=True)
    np.testing.assert_allclose(fit.eval({"H0": 1.0}), 0.0, atol=1e-20)
