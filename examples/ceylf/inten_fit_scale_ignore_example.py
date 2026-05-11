#!/usr/bin/env python3
"""
Minimal CE:YLF intensity-fit example using scale_to() and ignored groups.

This demonstrates Phase 9 fit controls for relative intensity data:
1. Attach experimental data with arbitrary overall scale.
2. Anchor that scale to one transition group with ``spec.scale_to(group_idx)``.
3. Exclude selected groups from chi-square with ``spec.set_ignored_groups([...])``.

The synthetic-data block is clearly marked and can be replaced with measured data.
"""

from pathlib import Path

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.inten import Spectrum, fit_altp, gen_inten_summary

print("\n" + "=" * 78)
print("CE:YLF INTENSITY FIT EXAMPLE (SCALE + IGNORE)")
print("=" * 78)

matel_dir = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "integration" / "inten" / "matel"
)
matel_cf = matel_dir / "f1cf"
matel_int = matel_dir / "f1int"

t_cf = ImportSLJM(str(matel_cf))
t_int = ImportSLJM(str(matel_int), sl_name=str(matel_cf))

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

h = cfl.Hamiltonian(
    [t_cf.EAVG, t_cf.ZETA, t_cf.C20, t_cf.C40, t_cf.C43, t_cf.C60, t_cf.C63, t_cf.C66, MX, MY, MZ]
)
h.set_coeff(coeff)
h.diag()

intensity_tensors = [t_int.M11, t_int.M10, t_int.U20, t_int.U21, t_int.U22]
initial_altp = {"A210": 5e-11, "A230": -5e-11}

spec = Spectrum(
    hamiltonian=h,
    name="Relative-data demo",
    i_range=[1, 2],
    f_range=[7, 8, 9, 10],
    intensity_tensors=intensity_tensors,
    altp=initial_altp,
    group_tol=1e-3,
    md=True,
    ed=True,
)
spec.calculate_intensities()

# ---------------------------------------------------------------------------
# SYNTHETIC DATA BLOCK (replace with measured values in real workflows)
# ---------------------------------------------------------------------------
known_altp = {"A210": 1.0e-10, "A230": -1.0e-10}
spec.set_altp(known_altp)
spec.recalculate()
synthetic_expt = [
    {
        "group": idx,
        "intensity": 0.4 * group.get("f", 0.0),
        "energy": abs(group.get("Energy", 0.0)),
    }
    for idx, group in enumerate(spec.groups, start=1)
]

# Restore initial guess before fitting.
spec.set_altp(initial_altp)
spec.recalculate()
# ---------------------------------------------------------------------------

spec.set_expt_data(synthetic_expt)

# Scale all expt points to calculated group 1; exclude group 2 from chi-square.
spec.scale_to(1)
spec.set_ignored_groups([2])

result = fit_altp(
    ["A210", "A230"],
    spec,
    method="Nelder-Mead",
    options={"maxiter": 2500, "xatol": 1e-9, "fatol": 1e-11},
)

print("\nFit summary:")
print(result["summary"])

print("\nBrief spectrum summary (shows scaled values and notes):")
print(gen_inten_summary(spec, format="brief"))
