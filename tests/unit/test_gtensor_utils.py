"""Unit tests for pycf.gtensor_utils numerical behavior."""

import numpy as np

from pycf.gtensor_utils import gtensor_calc


class _MockHamiltonian:
    """Minimal Hamiltonian stub returning synthetic Zeeman-split energies."""

    def __init__(self, g_matrix, b0, mu_b):
        self._g = np.asarray(g_matrix, dtype=float).reshape(3, 3)
        self._b0 = float(b0)
        self._mu_b = float(mu_b)
        self._coeff = {"MX": 0.0, "MY": 0.0, "MZ": 0.0}

    def set_coeff(self, coeff):
        self._coeff = dict(coeff)

    def diag(self):
        mx, my, mz = self._coeff["MX"], self._coeff["MY"], self._coeff["MZ"]
        b_vec = np.array([mx, my, mz], dtype=float)
        b_hat = b_vec / self._b0

        # Directional g-value from g @ g.T, matching gtensor_calc assumptions.
        g_dir = np.sqrt(b_hat @ (self._g @ self._g.T) @ b_hat)
        split = self._mu_b * self._b0 * g_dir

        # Inject tiny deterministic asymmetry to emulate diagonalization noise.
        eps = 1e-14
        e0 = -0.5 * split + eps
        e1 = +0.5 * split - eps

        energies = np.array([e0, e1], dtype=float)
        vecs = np.eye(2)
        return energies, vecs


def test_gtensor_calc_zeroes_roundoff_offdiagonals_for_diagonal_tensor():
    """Off-diagonals should be exactly zero after numerical cleanup."""
    b0 = 0.01
    mu_b = 1.0
    g_true = np.diag([1.7, 2.3, 3.1])

    ham = _MockHamiltonian(g_true, b0=b0, mu_b=mu_b)
    coeff = {"MX": 0.0, "MY": 0.0, "MZ": 0.0}

    out = gtensor_calc(maxlev=1, h=ham, coeff=coeff, B0=b0, mu_b=mu_b)
    g_rec = out[0].reshape(3, 3)

    # Exact zeros are expected for symmetry-forbidden terms.
    assert g_rec[0, 1] == 0.0
    assert g_rec[1, 0] == 0.0
    assert g_rec[0, 2] == 0.0
    assert g_rec[2, 0] == 0.0
    assert g_rec[1, 2] == 0.0
    assert g_rec[2, 1] == 0.0

    # Diagonal values remain accurate.
    np.testing.assert_allclose(np.diag(g_rec), np.diag(g_true), rtol=0.0, atol=1e-10)
