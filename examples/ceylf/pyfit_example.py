#!/usr/bin/env python3
"""
PyFit demo: drive a Ce:YLF crystal-field fit with scipy.optimize.

This example mirrors ``exdata_example.py`` but replaces the
``cfl.e_fit`` driver (which uses the C/NLopt minimizer) with the new
:class:`pycf.pyfit.PyFit` wrapper around ``scipy.optimize.least_squares``.

Why use this?

* PyFit is a pure-Python wrapper that exposes the same residual vector
  the C objective minimises, so any scipy least-squares method
  (``lm``, ``trf``, ``dogbox``) becomes available — including bounds
  and custom Jacobians.
* It is a stepping stone for irrep-aware residuals or custom loss
  functions that don't yet exist in the C code.
* For high-symmetry materials the cost of evaluating residuals in
  Python is negligible compared to one C-side iteration.

Data source: 10.1016/j.optmat.2015.06.046

Prerequisites:
- Ce:YLF matrix element data in ``matel/f1cf/``
- numpy, scipy, pycf
"""

from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.cfl_util import gen_edata_summary
from pycf.import_sljm import ImportSLJM
from pycf.pyfit import PyFit

t = ImportSLJM(str(Path(__file__).parent / "matel" / "f1cf"))

# Initial coefficients from the literature; we will perturb a few of
# them and let PyFit drive them back.
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
h.set_coeff(coeff)

# Absolute energies used by data_sel == "abs" in exdata_example.py.
ex = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])
exdata = cfl.ExData(ex, "A")

# Fit the energy zero-point alongside the four real crystal-field
# strengths.  EAVG sets the absolute origin of the spectrum; without it
# the residuals are dominated by a global shift and the optimiser
# stalls far from the literature values.
params = ["EAVG", "C20", "C40", "C44"]

# Perturb the starting point by a few percent so the optimiser has
# something non-trivial to do.
h.update_coeff(
    {
        "EAVG": coeff["EAVG"] * 0.95,
        "C20": coeff["C20"] * 0.85,
        "C40": coeff["C40"] * 0.92,
        "C44": coeff["C44"] * 1.07,
    }
)

efit = cfl.EFit(params, h, exdata)
py = PyFit(efit)

print("=" * 72)
print("Initial chi2 (perturbed start):")
print("  pyfit:        {:14.6f}".format(py.chi2(py.x0)))
print("  efit.eval:    {:14.6f}".format(float(efit.eval({})[0])))
print("=" * 72)

result = py.fit_(method="lm", jac="pycf", xtol=1e-10, ftol=1e-10, verbose=1)

print()
print("scipy.optimize.least_squares result:")
print("  status:   ", result.status)
print("  message:  ", result.message)
print("  nfev:     ", result.nfev)
print("  cost:     {:.6e}  (== 0.5 * chi2)".format(result.cost))
print("  chi2:     {:.6e}".format(2.0 * result.cost))

print()
print("Optimised parameters (with one-sigma uncertainties):")
sigma = py.stderr()
for name, value, err in zip(params, result.x, sigma):
    print("  {:6s} = {:14.6f}  +/- {:10.6f}".format(name, value, err))

# Reuse PyFit's residual vector to populate an EData snapshot at the
# optimum and pretty-print it.
from pycf.cfl import _temporary_x  # noqa: E402

with _temporary_x(efit, result.x):
    edata = efit.get_edata()
print()
print("EData snapshot at optimum:")
print(gen_edata_summary(edata))
