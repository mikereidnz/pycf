#!/usr/bin/env python3
"""Integration tests for multi-Hamiltonian Altp fitting via multiple Spectrum objects."""

from pathlib import Path

import pytest

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, fit_altp


def _build_hamiltonian(t, coeff_overrides):
    coeff = {
        "EAVG": 1035 + 361.3287 + 6.326681621113494,
        "ZETA": 626,
        "C20": 500,
        "C40": 0,
        "C43": 200 + 100j,
        "C60": 0,
        "C63": 0,
        "C66": 0,
        "MX": 0,
        "MY": 0,
        "MZ": 0,
    }
    coeff.update(coeff_overrides)

    mu_b = 0.466860
    MX = mu_b * t.MAGX
    MY = mu_b * t.MAGY
    MZ = mu_b * t.MAGZ
    MX.name = "MX"
    MY.name = "MY"
    MZ.name = "MZ"

    h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C43, t.C60, t.C63, t.C66, MX, MY, MZ])
    h.set_coeff(coeff)
    return h


def _build_spectrum(h, intensity_tensors, altp, name):
    return Spectrum(
        hamiltonian=h,
        name=name,
        i_range=[1, 2],
        f_range=[7, 8, 9, 10],
        intensity_tensors=intensity_tensors,
        altp=altp,
        group_tol=1e-3,
        md=True,
        ed=True,
    )


def _target_map(spec):
    spec.calculate_intensities(polarization="isotropic")
    return {idx: group.get("f", 0.0) for idx, group in enumerate(spec.groups, start=1)}


def test_fit_altp_multi_spectrum_two_spectra():
    """Fit shared Altp parameters across two different Hamiltonians."""
    matel_base = Path(__file__).resolve().parent / "matel" / "f1cf"
    inten_base = Path(__file__).resolve().parent / "matel" / "f1int"
    t = ImportSLJM(matel_base)
    t_int = ImportSLJM(inten_base, sl_name=matel_base)
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]

    # Two distinct Hamiltonians.
    h1 = _build_hamiltonian(t, {"MZ": 0.0})
    h2 = _build_hamiltonian(t, {"MZ": 0.8})

    known_altp = {"A210": 1.0e-10, "A230": -1.0e-10}
    initial_altp = {"A210": 4.5e-11, "A230": -4.0e-11}

    # Build synthetic targets from known parameters.
    target_spec1 = _build_spectrum(h1, intensity_tensors, known_altp, "target-h1")
    target_spec2 = _build_spectrum(h2, intensity_tensors, known_altp, "target-h2")
    target1 = _target_map(target_spec1)
    target2 = _target_map(target_spec2)

    # Fit both spectra jointly from perturbed initial values.
    fit_spec1 = _build_spectrum(h1, intensity_tensors, initial_altp, "fit-h1")
    fit_spec2 = _build_spectrum(h2, intensity_tensors, initial_altp, "fit-h2")

    dry = fit_altp(["A210", "A230"], [fit_spec1, fit_spec2], [target1, target2], dry_run=True)
    result = fit_altp(
        ["A210", "A230"],
        [fit_spec1, fit_spec2],
        [target1, target2],
        method="Nelder-Mead",
        options={"maxiter": 2500, "xatol": 1e-10, "fatol": 1e-12},
    )

    assert result["n_spectra"] == 2
    assert len(result["chi2_by_spectrum"]) == 2
    assert result["chi2"] < 1e-8
    assert result["chi2"] < dry["chi2"]
    assert result["fitted_params"]["A210"] > 0
    assert result["fitted_params"]["A230"] < 0


def test_fit_altp_multi_spectrum_dry_run():
    """dry_run returns current-parameter chi² without optimization for multi-spectrum fits."""
    matel_base = Path(__file__).resolve().parent / "matel" / "f1cf"
    inten_base = Path(__file__).resolve().parent / "matel" / "f1int"
    t = ImportSLJM(matel_base)
    t_int = ImportSLJM(inten_base, sl_name=matel_base)
    intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]

    h1 = _build_hamiltonian(t, {"MZ": 0.0})
    h2 = _build_hamiltonian(t, {"MZ": 0.8})

    known_altp = {"A210": 1.0e-10, "A230": -1.0e-10}
    initial_altp = {"A210": 4.5e-11, "A230": -4.0e-11}

    target_spec1 = _build_spectrum(h1, intensity_tensors, known_altp, "target-h1")
    target_spec2 = _build_spectrum(h2, intensity_tensors, known_altp, "target-h2")
    target1 = _target_map(target_spec1)
    target2 = _target_map(target_spec2)

    fit_spec1 = _build_spectrum(h1, intensity_tensors, initial_altp, "fit-h1")
    fit_spec2 = _build_spectrum(h2, intensity_tensors, initial_altp, "fit-h2")

    # Provide expt_data and rely on target_intensities=None path.
    fit_spec1.set_expt_data([[idx, value] for idx, value in sorted(target1.items())])
    fit_spec2.set_expt_data([[idx, value] for idx, value in sorted(target2.items())])

    result = fit_altp(["A210", "A230"], [fit_spec1, fit_spec2], dry_run=True)

    assert result["dry_run"] is True
    assert result["n_spectra"] == 2
    assert len(result["chi2_by_spectrum"]) == 2
    assert result["chi2"] > 0.0
    assert result["fitted_params"]["A210"] == pytest.approx(initial_altp["A210"])
    assert result["fitted_params"]["A230"] == pytest.approx(initial_altp["A230"])
    assert result["uncertainties"] == {}
