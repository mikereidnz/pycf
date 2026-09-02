#!/usr/bin/env python3
"""Regression guard for the mu/n + mh_fit code path.

This is the *only* test that exercises the `mu_n_mhfit_obj` Cython hot loop
in pycf/cfl.pyx (the per-Hamiltonian iteration that:
  - updates each H's coefficients
  - diagonalises
  - re-resolves dynamic mu/n eigenstate indices
  - accumulates chi² via compute_chi2_numpy)

All other mu/n tests use the single-Hamiltonian e_fit objective
(`mu_n_efit_obj`), and all other mh_fit tests use state-label or
absolute-level ExData. Without this file there is no automated coverage
for the intersection.

The test re-uses the Ce:YLF ground-multiplet mu/n data from
`test_exdata.py` (data_sel == "mu") and fits two identical
Hamiltonians simultaneously. With identical H's and identical ExData
the converged coefficients must match the single-H mu fit; the chi²
must also be deterministic across runs.
"""

from pathlib import Path

import numpy as np
import pytest

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM

MATEL_BASE = Path(__file__).resolve().parent / "matel" / "f1cf"


def _build_mu_exdata() -> cfl.ExData:
    """Build the mu/n ExData used by the single-H reference fit."""
    ex_a = [
        ["mu", 1, 1, 0],
        ["mu", 1, 3, 216],
        ["mu", 1, 7, 2216.10],
        ["mu", 1, 8, 2312.80],
    ]
    ex_d = [
        ["mu", 1, 8, 1, 12, 116.0],
        ["mu", 1, 12, 1, 14, 729.0],
    ]
    return cfl.ExData((ex_a, ex_d), ("A", "D"), label_key="MuN")


def _build_hamiltonian() -> cfl.Hamiltonian:
    t = ImportSLJM(str(MATEL_BASE))
    coeff = {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
    }
    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64])
    h.set_coeff(coeff)
    h.minimum_q = 2
    h.half_integer_states = True
    return h


def test_mh_fit_mu_n_two_identical_hamiltonians() -> None:
    """Fit two identical Hamiltonians against the same mu/n ExData.

    Asserts that:
      1. mh_fit completes (no exception escaping the mu/n hot loop).
      2. The converged coefficients match the single-H mu fit reference
         (same expected values, same loose tolerance as test_exdata[mu]).
      3. fmin > 0 and finite (the inf-on-error path is not silently active).
    """
    h1 = _build_hamiltonian()
    h2 = _build_hamiltonian()
    exdata = _build_mu_exdata()

    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6)
    param = ["EAVG", "C20", "C40", "C44"]
    h_list = [h1, h2]
    weights_list = [1.0, 1.0]
    ex_list = [exdata, exdata]

    res = cfl.mh_fit(param, h_list, weights_list, ex_list, cfl_min)

    fit_coeff = res["coeff"]
    fmin = res["fmin"]

    print("\n[two-H mu/n mh_fit] fmin =", fmin)
    print("[two-H mu/n mh_fit] fitted coefficients:")
    for label in param:
        print(f"    {label:<6s} = {fit_coeff[label]:>20.10f}")

    assert np.isfinite(fmin), f"fmin should be finite, got {fmin}"
    assert fmin >= 0.0, f"fmin should be non-negative, got {fmin}"

    expected = {
        "EAVG": 1535.048512090352,
        "C20": 301.21004426854967,
        "C40": -1323.988805632041,
        "C44": -1283.570122479815,
    }
    # Same loose tolerance as the single-H mu test: the mu/n subset
    # produces a flatter optimisation landscape than absolute energies.
    tolerance = 0.6
    for label, expected_value in expected.items():
        assert label in fit_coeff
        assert fit_coeff[label] == pytest.approx(expected_value, rel=tolerance)


def test_mh_fit_mu_n_matches_single_h_efit() -> None:
    """Two-H mu/n mh_fit with identical inputs must converge to the
    same coefficients (and same chi²) as a single-H e_fit.

    This is the strongest regression guard: any change that perturbs
    `mu_n_mhfit_obj` (allocation pattern, type coercion, loop order,
    chi² accumulation) but leaves `mu_n_efit_obj` alone will break
    this equivalence.
    """
    h_single = _build_hamiltonian()
    ex_single = _build_mu_exdata()
    cfl_min_a = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6)
    param = ["EAVG", "C20", "C40", "C44"]

    res_single = cfl.e_fit(param, h_single, ex_single, cfl_min_a)

    h_a = _build_hamiltonian()
    h_b = _build_hamiltonian()
    ex_a = _build_mu_exdata()
    ex_b = _build_mu_exdata()
    cfl_min_b = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6)

    res_multi = cfl.mh_fit(param, [h_a, h_b], [1.0, 1.0], [ex_a, ex_b], cfl_min_b)

    print("\n[single-H vs two-H mu/n equivalence]")
    print(f"    single-H fmin = {res_single['fmin']:.10f}")
    print(f"    two-H    fmin = {res_multi['fmin']:.10f}")
    print(f"    ratio (expect ~2.0) = {res_multi['fmin'] / res_single['fmin']:.6f}")
    print("    coefficient comparison (single vs two-H):")
    for label in param:
        s = res_single["coeff"][label]
        m = res_multi["coeff"][label]
        print(f"        {label:<6s} {s:>20.10f}  vs  {m:>20.10f}   (Δ={m - s:+.3e})")

    # With identical H's and identical ExData, summed chi² is 2 * single chi².
    # Allow a small relative slack: the two optimisers may take different
    # numbers of evaluations and stop at slightly different points within
    # xtol of the same minimum.
    assert res_multi["fmin"] == pytest.approx(2.0 * res_single["fmin"], rel=1e-3)

    for label in ["EAVG", "C20", "C40", "C44"]:
        assert res_multi["coeff"][label] == pytest.approx(res_single["coeff"][label], rel=1e-3)


def test_mh_fit_mu_n_is_deterministic() -> None:
    """Running the same mu/n mh_fit twice must produce bit-identical fmin
    and coefficients. Detects any non-deterministic allocation/caching
    bug introduced into the hot loop.
    """

    def _run() -> dict:
        h_a = _build_hamiltonian()
        h_b = _build_hamiltonian()
        ex_a = _build_mu_exdata()
        ex_b = _build_mu_exdata()
        cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6)
        return cfl.mh_fit(
            ["EAVG", "C20", "C40", "C44"],
            [h_a, h_b],
            [1.0, 1.0],
            [ex_a, ex_b],
            cfl_min,
        )

    res1 = _run()
    res2 = _run()
    print("\n[determinism] run-1 fmin =", repr(res1["fmin"]))
    print("[determinism] run-2 fmin =", repr(res2["fmin"]))
    print("[determinism] coefficients (run-1 == run-2 required):")
    for label in res1["coeff"]:
        print(f"    {label:<6s} {res1['coeff'][label]!r:>30s}  " f"{res2['coeff'][label]!r:>30s}")
    assert res1["fmin"] == res2["fmin"]
    for label in res1["coeff"]:
        assert res1["coeff"][label] == res2["coeff"][label], (
            f"Non-deterministic fit result for {label}: "
            f"{res1['coeff'][label]} vs {res2['coeff'][label]}"
        )


def _build_hamiltonian_with_field(t, magz: float) -> cfl.Hamiltonian:
    """Same as ``_build_hamiltonian`` but adds a MAGZ (Zeeman field) tensor.

    ``magz`` is a *fixed* (non-fitted) coefficient: it is set once via
    ``set_coeff`` and must never be varied by the fit or by any other
    Hamiltonian sharing the same tensor name.
    """
    coeff = {
        "EAVG": 1035.1277,
        "ZETA": 625.6990,
        "C20": 297.8906,
        "C40": -1328.1522,
        "C44": -1282.4766,
        "C60": -191.5100,
        "C64": -1743.1424 + 692.8662j,
        "MAGZ": magz,
    }
    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64, t.MAGZ])
    h.set_coeff(coeff)
    h.minimum_q = 2
    h.half_integer_states = True
    return h


def test_mh_fit_mu_n_does_not_clobber_fixed_coeff_across_hamiltonians() -> None:
    """Regression test for a bug where a fixed (non-fitted) coefficient
    that is shared *by name* but differs in value between Hamiltonians
    (e.g. a Zeeman field ``MAGZ`` that is 0 for a zero-field energy-level
    Hamiltonian but nonzero for a field-on g-value Hamiltonian) got
    silently overwritten during ``mh_fit``.

    Root cause: the mu/n hot loop (``mu_n_mhfit_obj`` in cfl.pyx) called
    ``h.set_coeff(mhfit.coeff)`` for *every* Hamiltonian on every
    objective-function evaluation, where ``mhfit.coeff`` is a single dict
    merged (via ``dict.update``) across *all* Hamiltonians in ``h_list``.
    Any coefficient name shared between Hamiltonians with different fixed
    values (like ``MAGZ``) would end up with the *last* Hamiltonian's
    value on every one of them, regardless of what was actually fitted.

    This test builds a zero-field Hamiltonian and a field-on Hamiltonian
    that share the ``MAGZ`` tensor name, runs ``mh_fit``, and asserts that
    each Hamiltonian's own fixed ``MAGZ`` value survives the fit
    unchanged.
    """
    t = ImportSLJM(str(MATEL_BASE))
    h_energy = _build_hamiltonian_with_field(t, magz=0.0)
    h_field = _build_hamiltonian_with_field(t, magz=0.05)

    ex_energy = _build_mu_exdata()
    # A second, distinct mu/n data set for the field-on Hamiltonian so the
    # two Hamiltonians are not literally identical (mirrors the real
    # energy-levels + g-values fit structure).
    ex_field = cfl.ExData(
        ([["mu", 1, 1, 0], ["mu", 1, 2, 0.1]],), ("A",), label_key="MuN"
    )

    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, maxeval=25)
    param = ["EAVG", "C20"]
    res = cfl.mh_fit(
        param,
        [h_energy, h_field],
        [1.0, 1.0],
        [ex_energy, ex_field],
        cfl_min,
        suppress_input=True,
    )

    assert np.isfinite(res["fmin"])

    # Each Hamiltonian's own MAGZ coefficient must be exactly what it was
    # set to, independent of the other Hamiltonian's field value and
    # independent of which parameters were actually fitted.
    assert h_energy.coeff_dict["MAGZ"] == 0.0, (
        "Zero-field Hamiltonian's MAGZ was clobbered by mh_fit "
        f"(got {h_energy.coeff_dict['MAGZ']!r}); the fixed coefficient "
        "of one Hamiltonian must not leak into another."
    )
    assert h_field.coeff_dict["MAGZ"] == 0.05, (
        "Field-on Hamiltonian's MAGZ was corrupted by mh_fit "
        f"(got {h_field.coeff_dict['MAGZ']!r})."
    )

    # The "all_coeff" reporting dict (used to build fit summaries) is
    # derived from h_list[0], so it must also reflect the zero field.
    assert res["all_coeff"]["MAGZ"] == 0.0


if __name__ == "__main__":
    test_mh_fit_mu_n_two_identical_hamiltonians()
    test_mh_fit_mu_n_matches_single_h_efit()
    test_mh_fit_mu_n_is_deterministic()
    test_mh_fit_mu_n_does_not_clobber_fixed_coeff_across_hamiltonians()
    print("All mu/n mh_fit regression tests passed.")
