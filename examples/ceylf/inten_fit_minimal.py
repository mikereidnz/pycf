#!/usr/bin/env python3
"""
Minimal intensity-fit example (linear workflow, no helper functions).

This script demonstrates the intended Phase 8 user flow:
1. Build Hamiltonian and Spectrum once
2. Reuse Spectrum via set_altp()/set_expt_data()/recalculate()
3. Fit Altp parameters with fit_altp(spec, ...)

NOTE:
The synthetic-data block is clearly marked and can be deleted in real use.
In real workflows, replace it with your measured expt_data.
"""

from pathlib import Path

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, fit_altp, gen_inten_summary


print("\n" + "=" * 72)
print("MINIMAL INTENSITY FIT EXAMPLE (CEYLF)")
print("=" * 72)

# ---------------------------------------------------------------------------
# 1) Load matrix elements
# ---------------------------------------------------------------------------
matel_dir = Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
matel_cf = matel_dir / "f1cf"
matel_int = matel_dir / "f1int"

t_cf = ImportSLJM(str(matel_cf))
t_int = ImportSLJM(str(matel_int), sl_name=str(matel_cf))

# ---------------------------------------------------------------------------
# 2) Build Hamiltonian (same setup used in inten examples/tests)
# ---------------------------------------------------------------------------
coeff = {
    "EAVG": 1402.6553816211135,
    "ZETA": 626,
    "C20": 500,
    "C40": 0,
    "C43": (200 + 100j),
    "C60": 0,
    "C63": 0,
    "C66": 0,
    "MX": 1e-10,
    "MY": 0,
    "MZ": 1,
}

mu_b = 0.466860
MX = mu_b * t_cf.MAGX
MY = mu_b * t_cf.MAGY
MZ = mu_b * t_cf.MAGZ
MX.name, MY.name, MZ.name = "MX", "MY", "MZ"

h = cfl.Hamiltonian([t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ])
h.set_coeff(coeff)
h.diag()

# ---------------------------------------------------------------------------
# 3) Build one reusable Spectrum object with an initial Altp guess
# ---------------------------------------------------------------------------
intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]

initial_altp = {
    "A210": 5e-11,
    "A230": -5e-11,
    "A233": 5e-11 + 1e-10j,
}

spec = Spectrum(
    hamiltonian=h,
    name="Minimal fit spectrum",
    i_range=[1, 2],
    f_range=[7, 8, 9, 10],
    intensity_tensors=intensity_tensors,
    altp=initial_altp,
    group_tol=1e-3,
    nrefractive=1.0,
    md=True,
    ed=True,
)

spec.calculate_intensities(polarization="isotropic")

print("\nBefore fit (initial Altp):")
print(gen_inten_summary(spec, format="brief"))

# ---------------------------------------------------------------------------
# 4) SYNTHETIC DATA BLOCK (DELETE THIS BLOCK FOR REAL EXPERIMENTAL DATA)
# ---------------------------------------------------------------------------
# In real use:
#   spec.set_expt_data([[group_idx, measured_value], ...])
# and skip directly to the fitting block below.
known_altp_for_synthetic_data = {
    "A210": 1e-10,
    "A230": -1e-10,
    "A233": 1e-10 + 2e-10j,
}

spec.set_altp(known_altp_for_synthetic_data)
spec.recalculate(polarization="isotropic")
synthetic_expt_data = [[group_idx, group.get("f", 0.0)] for group_idx, group in enumerate(spec.groups, start=1)]

# Restore initial guess before fit.
spec.set_altp(initial_altp)
spec.recalculate(polarization="isotropic")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5) Attach expt_data and fit (no Spectrum rebuild needed)
# ---------------------------------------------------------------------------
spec.set_expt_data(synthetic_expt_data)

result = fit_altp(
    ["A210", "A230", "A233"],
    spec,
    method="Nelder-Mead",
    options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
)

print("\nFit summary:")
print(result["summary"])

print("\nAfter fit:")
print(gen_inten_summary(spec, format="brief"))

