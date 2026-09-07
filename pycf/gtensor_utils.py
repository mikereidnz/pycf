#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: pycf/gtensor_utils.py
"""
Calculation and printing of g-tensors by explicitly calculating the Zeeman
splitting of the energy levels.
"""

import numpy as np


# gtensor calculation
def gtensor_calc(maxlev, h, coeff, B0, mu_b):
    """Compute g-tensor matrices up to ``maxlev``.

    The function evaluates Zeeman splittings along 9 field directions, reconstructs
    the symmetric matrix ``G = g @ g.T`` for each Kramers doublet, and then takes
    a numerically stable symmetric square root to obtain ``g``.

    Parameters
    ----------
    maxlev : int
        Highest level index included in the splitting analysis.
    h : object
        Hamiltonian-like object providing ``set_coeff(dict)`` and ``diag()``.
    coeff : dict
        Coefficient dictionary containing ``MX``, ``MY``, and ``MZ`` entries
        used for Zeeman field components.
    B0 : float
        Field magnitude used for directional splitting calculations.
    mu_b : float
        Bohr magneton scaling used to convert splittings to g-values.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_doublets, 9)`` where each row is a flattened ``3x3``
        g-tensor matrix.
    """

    # field directions used in the calcuation
    bhat = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
        "xpy": np.array([1, 1, 0]) / np.sqrt(2),
        "ypz": np.array([0, 1, 1]) / np.sqrt(2),
        "zpx": np.array([1, 0, 1]) / np.sqrt(2),
        "xmy": np.array([1, -1, 0]) / np.sqrt(2),
        "ymz": np.array([0, 1, -1]) / np.sqrt(2),
        "zmx": np.array([-1, 0, 1]) / np.sqrt(2),
    }

    # calculate the gtensors by calculating splittings along 9 directions.
    # Numerics summary (for easier future maintenance):
    # 1) Off-diagonals are formed from paired +/-45 degree directions.
    # 2) G is explicitly symmetrized before decomposition.
    # 3) Small eigenvalues are zeroed/clipped before sqrt.
    # 4) Tiny residual entries in g are hard-zeroed.
    g_dir = {}
    gtensor_rows = []
    bhat_list = list(bhat.keys())
    for direction in bhat_list:
        # print(direction)
        coeff["MX"], coeff["MY"], coeff["MZ"] = tuple(B0 * bhat[direction])
        # print(coeff)
        h.set_coeff(coeff)
        E, V = h.diag()
        # print(E)
        # print(V)
        E0 = E[0 : maxlev + 1 : 2]
        E1 = E[1 : maxlev + 1 : 2]
        g_dir[direction] = (E1 - E0) / B0 / mu_b
    # print(g_dir)
    x, y, z = 0, 1, 2
    for i in range(len(g_dir["x"])):
        G = np.zeros((3, 3))
        G[x, x] = g_dir["x"][i] ** 2
        G[y, y] = g_dir["y"][i] ** 2
        G[z, z] = g_dir["z"][i] ** 2
        # Use paired +/-45deg directions directly for off-diagonals:
        # g(u)^2 = u^T G u, so with u=(a+b)/sqrt(2), v=(a-b)/sqrt(2),
        # G_ab = (g(u)^2 - g(v)^2) / 2.
        G[x, y] = (g_dir["xpy"][i] ** 2 - g_dir["xmy"][i] ** 2) / 2
        G[y, x] = G[x, y]
        G[x, z] = (g_dir["zpx"][i] ** 2 - g_dir["zmx"][i] ** 2) / 2
        G[z, x] = G[x, z]
        G[y, z] = (g_dir["ypz"][i] ** 2 - g_dir["ymz"][i] ** 2) / 2
        G[z, y] = G[y, z]
        # Enforce symmetry and compute a real symmetric square root in a numerically
        # stable way to suppress tiny roundoff-induced imaginary/asymmetric parts.
        G = 0.5 * (G + G.T)
        evals, evecs = np.linalg.eigh(G)
        eval_scale = max(1.0, np.max(np.abs(evals)))
        eval_tol = 1e-12 * eval_scale
        evals = np.where(np.abs(evals) < eval_tol, 0.0, evals)
        evals = np.clip(evals, 0.0, None)

        gtensor_i = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        # Clean tiny residuals from finite-precision arithmetic.
        g_scale = max(1.0, np.max(np.abs(gtensor_i)))
        g_tol = 1e-12 * g_scale
        gtensor_i[np.abs(gtensor_i) < g_tol] = 0.0

        gtensor_rows.append(gtensor_i.flatten())

    coeff["MX"], coeff["MY"], coeff["MZ"] = 0, 0, 0
    return np.array(gtensor_rows)


def ordered_eig(H):
    """Return eigenpairs sorted by ascending real eigenvalue."""
    w, z = np.linalg.eig(H)
    w = w.real
    idx = w.argsort()
    w = w[idx]
    z = z[:, idx]
    return w, z


def print_gtensor(g):
    """Print one g-tensor with aligned matrix, principal values, and axes."""
    g = np.asarray(g, dtype=float).reshape(3, 3)
    print("g tensor:")
    for row in g:
        print("  " + " ".join(f"{val:10.6f}" for val in row))

    w, z = ordered_eig(g)
    print("principal g values:")
    print("  " + " ".join(f"{val:10.6f}" for val in w))

    print("orientations (columns are principal axes):")
    for i in range(3):
        # Prints the i-th Cartesian component (x, y, or z) for all three vectors
        print(f"  {z[i, 0]:10.6f} {z[i, 1]:10.6f} {z[i, 2]:10.6f}")


def gtensor_print_all(gtensor):
    """Print all flattened g-tensors in ``gtensor`` level-by-level."""
    print("gtensors:")
    for i in range(gtensor.shape[0]):
        print(
            "level ", i * 2 + 1
        )  # to match the level numbering in print_energies_g, which starts at 1.
        g = gtensor[i].reshape(3, 3)
        print_gtensor(g)
