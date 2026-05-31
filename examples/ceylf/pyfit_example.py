#!/usr/bin/env python3
"""PyFit fitting example for Ce:YLF crystal field parameters.

This example mirrors the flow used by ``exdata_example.py`` and
``mhfit_example.py``, but runs the optimization through
``pycf.pyfit.PyFit.fit_res`` so the result payload and summary are
consistent with ``cfl.e_fit`` / ``cfl.mh_fit``.
"""

from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.import_sljm import ImportSLJM
from pycf.pyfit import PyFit

t = ImportSLJM(str(Path(__file__).parent / "matel" / "f1cf"))

coeff = {
    "EAVG": 1035.1277,
    "ZETA": 625.6990,
    "C20": 297.8906,
    "C40": -1328.1522,
    "C44": -1282.4766,
    "C60": -191.5100,
    "C64": -1743.1424 + 692.8662j,
}
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64], label="Ce:YLF")

# Perturb the starting point so the fit has visible movement.
h.set_coeff(
    {
        "EAVG": coeff["EAVG"] * 0.95,
        "ZETA": coeff["ZETA"],
        "C20": coeff["C20"] * 0.85,
        "C40": coeff["C40"] * 0.92,
        "C44": coeff["C44"] * 1.07,
        "C60": coeff["C60"],
        "C64": coeff["C64"],
    }
)

ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])
exdata = cfl.ExData(ex, "A")
params = ["EAVG", "C20", "C40", "C44"]

efit = cfl.EFit(params, h, exdata)
py = PyFit(efit)

res = py.fit_res(
    method="lm",
    jac="pycf",
    max_levels=65,
    calculate_sigma=True,
    include_covariance=False,
    include_jacobian=True,
)

print(res["summary"])

fitcoeff = res["coeff"]
allcoeff = res["all_coeff"]
sigma = res["sigma"]
jacdiag = res["jacobian_diagnostics"]
