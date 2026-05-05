#!/usr/bin/env python3
"""Post-fit diagnostics example for the Ce:YLF multi-Hamiltonian fit.

Demonstrates the data accessors that complement ``cfl.mh_fit`` /
``cfl.MHFit``:

* :py:attr:`pycf.cfl.Hamiltonian.label` and the matching ``Hamiltonian:``
  heading rendered by :py:func:`pycf.cfl_util.gen_e_summary` and
  :py:meth:`Hamiltonian.gen_summary`.
* :py:meth:`MHFit.get_edata` returning a tabular
  :class:`pycf.cfl_util.EData` view of every observation, with the
  per-Hamiltonian scalar weight already folded into the ``weight`` column.
* :py:meth:`MHFit.fd_jacobian` (finite-difference Jacobian at the current
  point) and :py:attr:`MHFit.last_jacobian` (the GSL-converged Jacobian
  captured during ``fit()``).
* :py:meth:`MHFit.covariance` and the matching parameter 1-sigma
  uncertainties.

Run from this directory so the relative ``matel/`` path resolves::

    cd examples/ceylf
    python mhfit_diagnostics_example.py
"""

from pathlib import Path

import numpy as np

import pycf.cfl as cfl
from pycf.cfl_util import gen_edata_summary
from pycf.import_sljm import ImportSLJM

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
h = cfl.Hamiltonian([t.EAVG, t.ZETA, t.C20, t.C40, t.C44, t.C60, t.C64])
h.set_coeff(coeff)
h.label = "Ce:YLF crystal field"

exdata = np.array([[2, 0], [3, 216], [8, 2216], [9, 2312.8], [12, 2428.8], [14, 3157.8]])

cfl_min = cfl.CFLMin("gsl_nls", niter=1)
param = ["EAVG", "C20", "C40", "C44"]

h_list = [h, h]
weights_list = [1.0, 0.5]
exdata_list = [exdata, exdata]

fit = cfl.MHFit(param, h_list, weights_list, exdata_list)
(x, fmin) = fit.fit(cfl_min)

print("=" * 72)
print("MHFit.get_edata() - aggregated observation table for all Hamiltonians")
print("=" * 72)
ed = fit.get_edata()
print(gen_edata_summary(ed))
print(f"\nchi2 (combined) = {ed.chi2():.6g}")
print(
    "\nThe 'H' column distinguishes rows from h_list[0] (h_weight=1.0) from"
    "\nh_list[1] (h_weight=0.5).  Each row's 'weight' is ex.w * h_weight, so"
    "\nthe second block's weight column is halved relative to the first."
)

print("\n" + "=" * 72)
print("MHFit.fd_jacobian() - finite-difference Jacobian at the fit minimum")
print("=" * 72)
J = fit.fd_jacobian()
print(f"shape = {J.shape}    (rows = total residuals, cols = {len(param)} params)")
print(f"params: {param}")
print(np.array2string(J, precision=4, suppress_small=True))

print("\n" + "=" * 72)
print("MHFit.last_jacobian - GSL's converged Jacobian, captured during fit()")
print("=" * 72)
print(f"shape = {fit.last_jacobian.shape}")
print(np.array2string(fit.last_jacobian, precision=4, suppress_small=True))

print("\n" + "=" * 72)
print("MHFit.covariance() - (J^T W J)^-1, default scale='reduced_chi2'")
print("=" * 72)
cov, sigma, _ = fit.covariance()
print("covariance:")
print(np.array2string(cov, precision=4, suppress_small=True))
print("\nParameter 1-sigma uncertainties:")
for name, sig in zip(param, sigma):
    print(f"  {name:>6s}  +/- {sig:.4g}")

print("\n" + "=" * 72)
print("Hamiltonian.label is auto-included in h.gen_summary() heading")
print("=" * 72)
h.update_coeff(x)
(h.w, h.z) = h.diag()
print(h.gen_summary())
