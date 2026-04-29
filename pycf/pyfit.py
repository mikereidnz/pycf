"""Pure-Python fitting wrapper around :mod:`scipy.optimize.least_squares`.

:class:`PyFit` wraps an existing :class:`pycf.cfl.EFit` or
:class:`pycf.cfl.MHFit` instance and exposes its residual vector
(``sqrt(w_i) * (e_calc_i - e_obs_i)`` for each EData row) so that any
SciPy least-squares method can be used to drive the fit.

Why a separate Python wrapper?

* It does not replace the C minimizer — it complements it.  For
  high-symmetry materials where the cost of a residual evaluation is
  small, a pure-Python loop over scipy methods (``lm``, ``trf``,
  ``dogbox``) gives easy access to bounds, custom Jacobians, and
  alternative regularisation strategies.
* It is a reference implementation that is easy to extend — for
  example with state-label / irrep-aware residuals, alternative
  loss functions, or pre-conditioning — without touching the C code.
* It re-uses the existing parameter-handling machinery
  (:func:`pycf.cfl._temporary_x`, :func:`pycf.cfl._x_to_coeff_dict`)
  so complex parameters, parameter sharing, and multi-Hamiltonian
  fits work transparently.

Example
-------
>>> from pycf.pyfit import PyFit
>>> py = PyFit(efit)         # efit is an EFit or MHFit
>>> result = py.fit(method='lm')
>>> result.x                 # final parameter vector
>>> py.chi2(result.x)        # matches efit.eval(...) at the same x
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pycf.cfl import _temporary_x

__all__ = ["PyFit"]


class PyFit:
    """Wrap an :class:`EFit`/:class:`MHFit` for use with SciPy least-squares.

    Parameters
    ----------
    fit
        An :class:`pycf.cfl.EFit` or :class:`pycf.cfl.MHFit` instance.
        The wrapper does not modify the underlying object's persistent
        state during evaluation: parameter perturbations are applied
        through a context manager that restores the original
        coefficients on exit.

    Attributes
    ----------
    fit
        The wrapped fit object.
    x0 : numpy.ndarray
        A copy of the initial parameter vector at construction time.
        ``len(x0) == fit.n_p_real``.
    """

    def __init__(self, fit: Any) -> None:
        if not hasattr(fit, "n_p_real") or not hasattr(fit, "x0"):
            raise TypeError(
                "PyFit requires an EFit or MHFit instance "
                "(missing 'n_p_real' or 'x0')."
            )
        if not hasattr(fit, "get_edata"):
            raise TypeError(
                "PyFit requires the fit to expose get_edata(); upgrade "
                "to a recent pycf with the Hamiltonian data accessors."
            )
        self.fit = fit
        self.x0 = np.asarray(fit.x0, dtype=np.float64).copy()

    @property
    def n_p_real(self) -> int:
        """Number of real-valued fit parameters."""
        return int(self.fit.n_p_real)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        r"""Return the weighted residual vector at parameter point ``x``.

        Each entry is :math:`\sqrt{w_i}\,(e_{\mathrm{calc},i} -
        e_{\mathrm{obs},i})`, matching the structured-array column
        ``wresidual`` of :class:`pycf.cfl_util.EData`.  The squared sum
        of this vector equals the C objective at ``x`` (modulo
        per-Hamiltonian global weights, which are already baked into
        ``weight``).

        Parameters
        ----------
        x : array_like
            Real-valued parameter vector of length ``n_p_real``.

        Returns
        -------
        r : numpy.ndarray
            One-dimensional float64 array of length ``n_obs_total``.
        """
        x = np.asarray(x, dtype=np.float64)
        with _temporary_x(self.fit, x):
            edata = self.fit.get_edata()
        return np.asarray(edata.arr["wresidual"], dtype=np.float64)

    def chi2(self, x: np.ndarray) -> float:
        r"""Return :math:`\chi^2 = \sum_i w_i\,(e_{\mathrm{calc},i} -
        e_{\mathrm{obs},i})^2` at parameter point ``x``."""
        r = self.residuals(x)
        return float(np.dot(r, r))

    def fit_(
        self,
        x0: Optional[np.ndarray] = None,
        *,
        method: str = "lm",
        bounds: Any = None,
        jac: Any = "2-point",
        **kwargs: Any,
    ) -> Any:
        """Run :func:`scipy.optimize.least_squares` on :meth:`residuals`.

        The trailing underscore in the method name avoids a clash with
        the ``fit`` attribute (the wrapped EFit/MHFit instance).

        Parameters
        ----------
        x0 : array_like, optional
            Initial parameter vector.  Defaults to ``self.x0``.
        method : {'lm', 'trf', 'dogbox'}, optional
            SciPy method.  ``'lm'`` is the closest analogue to the C
            GSL non-linear least-squares minimiser.  ``'trf'`` and
            ``'dogbox'`` support bounds.
        bounds : 2-tuple of array_like, optional
            ``(lower, upper)`` bounds; passed straight to SciPy.  Only
            valid with ``method != 'lm'``.
        jac : str or callable, optional
            Jacobian specification.  Defaults to numerical differences
            via SciPy.  Pass ``self.fit.fd_jacobian`` (suitably
            wrapped) to use the pycf finite-difference helper, or your
            own callable returning shape ``(n_obs, n_p_real)``.
        **kwargs
            Any other keyword argument accepted by
            :func:`scipy.optimize.least_squares`
            (``ftol``, ``xtol``, ``gtol``, ``max_nfev``, ``verbose``...).

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The full SciPy result object (``x``, ``cost``, ``fun``,
            ``jac``, ``nfev``, ``status``, ``message``, ...).

        Notes
        -----
        Note that ``result.cost = 0.5 * chi2``; SciPy minimises
        ``0.5 * sum(residuals**2)`` while pycf reports the unscaled
        :math:`\\chi^2`.  Use :meth:`chi2` if you need the latter.
        """
        # Imported lazily so importing pycf.pyfit doesn't pay for SciPy
        # at module-load time when the user only wants residuals/chi2.
        from scipy.optimize import least_squares  # type: ignore[import-untyped]

        if x0 is None:
            x0 = self.x0
        x0 = np.asarray(x0, dtype=np.float64)

        kwargs.setdefault("method", method)
        if bounds is not None:
            kwargs["bounds"] = bounds
        kwargs["jac"] = jac

        return least_squares(self.residuals, x0, **kwargs)
