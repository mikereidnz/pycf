#!/usr/bin/env python3
"""
Minimal multi-spectrum intensity fit example (fit_altp workflow).

This script shows how to fit shared Altp parameters across two Spectrum objects
that belong to different Hamiltonians.

NOTE:
The synthetic-data block is marked and can be deleted for real experimental use.
"""

from pathlib import Path

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, fit_altp, inten_print

print("\n" + "=" * 76)
print("MINIMAL MULTI-SPECTRUM INTENSITY FIT EXAMPLE")
print("=" * 76)

matel_dir = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
)
matel_cf = matel_dir / "f1cf"
matel_int = matel_dir / "f1int"
t_cf = ImportSLJM(str(matel_cf))
t_int = ImportSLJM(str(matel_int), sl_name=str(matel_cf))

mu_b = 0.466860
MX = mu_b * t_cf.MAGX
MY = mu_b * t_cf.MAGY
MZ = mu_b * t_cf.MAGZ
MX.name, MY.name, MZ.name = "MX", "MY", "MZ"

base_coeff = {
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

coeff1 = dict(base_coeff)
coeff2 = dict(base_coeff)
coeff2["MZ"] = 0.8

h1 = cfl.Hamiltonian(
    [t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ]
)
h2 = cfl.Hamiltonian(
    [t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ]
)
h1.set_coeff(coeff1)
h2.set_coeff(coeff2)
h1.diag()
h2.diag()

intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
initial_altp = {"A210": 4.5e-11, "A230": -4.0e-11}

spec1 = Spectrum(
    hamiltonian=h1,
    name="H1 (zero field)",
    i_range=[1, 2],
    f_range=[7, 8, 9, 10],
    intensity_tensors=intensity_tensors,
    altp=initial_altp,
    group_tol=1e-3,
    md=True,
    ed=True,
)
spec2 = Spectrum(
    hamiltonian=h2,
    name="H2 (Bz field)",
    i_range=[1, 2],
    f_range=[7, 8, 9, 10],
    intensity_tensors=intensity_tensors,
    altp=initial_altp,
    group_tol=1e-3,
    md=True,
    ed=True,
)

spec1.calculate_intensities()
spec2.calculate_intensities()

print("\nBefore fit:")
inten_print([spec1, spec2])

# ---------------------------------------------------------------------------
# SYNTHETIC DATA BLOCK (DELETE THIS BLOCK FOR REAL EXPERIMENTAL DATA)
# In real use, replace this with measured datasets and call:
#   spec.set_expt_data([[group_idx, measured_value], ...])
# ---------------------------------------------------------------------------
known_altp = {"A210": 1.0e-10, "A230": -1.0e-10}
spec1.set_altp(known_altp)
spec2.set_altp(known_altp)
spec1.recalculate()
spec2.recalculate()
spec1.set_expt_data([[idx, group.get("f", 0.0)] for idx, group in enumerate(spec1.groups, start=1)])
spec2.set_expt_data([[idx, group.get("f", 0.0)] for idx, group in enumerate(spec2.groups, start=1)])
spec1.set_altp(initial_altp)
spec2.set_altp(initial_altp)
spec1.recalculate()
spec2.recalculate()
# ---------------------------------------------------------------------------

# fit_altp handles both single and multi-spectrum inputs
result = fit_altp(
    ["A210", "A230"],
    [spec1, spec2],
    dry_run=False,
    method="Nelder-Mead",
    options={"maxiter": 2500, "xatol": 1e-10, "fatol": 1e-12},
)
