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
>>> result = py.fit(method='lm')  # Run the fit with SciPy
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
    efit
        An :class:`pycf.cfl.EFit` or :class:`pycf.cfl.MHFit` instance.
        The wrapper does not modify the underlying object's persistent
        state during evaluation: parameter perturbations are applied
        through a context manager that restores the original
        coefficients on exit.

    Attributes
    ----------
    efit
        The wrapped EFit or MHFit object.
    x0 : numpy.ndarray
        A copy of the initial parameter vector at construction time.
        ``len(x0) == efit.n_p_real``.
    """

    def __init__(self, efit: Any) -> None:
        if not hasattr(efit, "n_p_real") or not hasattr(efit, "x0"):
            raise TypeError(
                "PyFit requires an EFit or MHFit instance " "(missing 'n_p_real' or 'x0')."
            )
        if not hasattr(efit, "get_edata"):
            raise TypeError(
                "PyFit requires the fit to expose get_edata(); upgrade "
                "to a recent pycf with the Hamiltonian data accessors."
            )
        self.efit = efit
        self.x0 = np.asarray(efit.x0, dtype=np.float64).copy()
        # The most recent scipy.optimize.OptimizeResult, populated by
        # :meth:`fit`.  Used by :meth:`covariance` and :meth:`stderr`
        # when no explicit ``x`` is supplied.
        self.last_result: Any = None

    @property
    def n_p_real(self) -> int:
        """Number of real-valued fit parameters."""
        return int(self.efit.n_p_real)

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
        with _temporary_x(self.efit, x):
            edata = self.efit.get_edata()
        return np.asarray(edata.arr["wresidual"], dtype=np.float64)

    def chi2(self, x: np.ndarray) -> float:
        r"""Return :math:`\chi^2 = \sum_i w_i\,(e_{\mathrm{calc},i} -
        e_{\mathrm{obs},i})^2` at parameter point ``x``."""
        r = self.residuals(x)
        return float(np.dot(r, r))

    def jacobian(self, x: np.ndarray, **fd_kwargs: Any) -> np.ndarray:
        r"""Weighted residual Jacobian at parameter point ``x``.

        Wraps :py:meth:`pycf.cfl.EFit.fd_jacobian` (or the equivalent
        MHFit method), then multiplies row :math:`i` by
        :math:`\sqrt{w_i}` so the returned matrix is
        :math:`\partial r_i / \partial x_\alpha` for the same residuals
        ``r`` returned by :meth:`residuals`.  Suitable for passing to
        ``scipy.optimize.least_squares`` via ``jac=PyFit.jacobian``.

        Parameters
        ----------
        x : array_like
            Parameter vector of length ``n_p_real``.
        **fd_kwargs
            Forwarded to ``efit.fd_jacobian`` (``delta``, ``rel_delta``,
            ``atol``, ``check_swaps``).

        Returns
        -------
        J : numpy.ndarray, shape ``(n_obs, n_p_real)``
        """
        x = np.asarray(x, dtype=np.float64)
        # fd_jacobian internally restores parameter state on exit.
        J_E = np.asarray(self.efit.fd_jacobian(x, **fd_kwargs), dtype=np.float64)
        # Weight rows by sqrt(w_i) so that J matches d(residuals)/dx.
        with _temporary_x(self.efit, x):
            edata = self.efit.get_edata()
        weights = np.asarray(edata.arr["weight"], dtype=np.float64)
        sqrtw = np.sqrt(np.maximum(weights, 0.0))
        return sqrtw[:, None] * J_E

    def fit(
        self,
        x0: Optional[np.ndarray] = None,
        *,
        method: str = "lm",
        bounds: Any = None,
        jac: Any = "2-point",
        **kwargs: Any,
    ) -> Any:
        """Run :func:`scipy.optimize.least_squares` on :meth:`residuals`.

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
            via SciPy (``'2-point'``).  Pass the string ``'pycf'`` to
            use :meth:`jacobian` (pycf's own central-difference helper
            with the same step convention as
            ``efit.covariance()``), or any callable returning shape
            ``(n_obs, n_p_real)``.
        **kwargs
            Any other keyword argument accepted by
            :func:`scipy.optimize.least_squares`
            (``ftol``, ``xtol``, ``gtol``, ``max_nfev``, ``verbose``...).

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The full SciPy result object (``x``, ``cost``, ``fun``,
            ``jac``, ``nfev``, ``status``, ``message``, ...).  Also
            cached on :attr:`last_result` for use by :meth:`covariance`.

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

        if jac == "pycf":
            jac = self.jacobian

        kwargs.setdefault("method", method)
        if bounds is not None:
            kwargs["bounds"] = bounds
        kwargs["jac"] = jac

        result = least_squares(self.residuals, x0, **kwargs)
        self.last_result = result
        return result

    def covariance(
        self,
        x: Optional[np.ndarray] = None,
        *,
        scale: str = "reduced_chi2",
        **fd_kwargs: Any,
    ) -> Any:
        r"""Variance-covariance matrix of the fit parameters.

        Delegates to ``self.efit.covariance(...)`` (which builds
        :math:`(J_E^T W J_E)^+` and applies the requested scale).  If
        ``x`` is omitted and a previous :meth:`fit` call has populated
        :attr:`last_result`, the optimum from that result is used.

        Parameters
        ----------
        x : array_like, optional
            Parameter vector at which to evaluate.  Defaults to
            ``self.last_result.x`` if available, else the underlying
            efit's current ``x0``.
        scale : {"reduced_chi2", "unscaled"}, optional
            Forwarded to :py:meth:`pycf.cfl.EFit.covariance`.
        **fd_kwargs
            Forwarded to ``efit.fd_jacobian`` when an FD Jacobian is
            recomputed.

        Returns
        -------
        cov : numpy.ndarray, shape ``(n_p_real, n_p_real)``
        sigma : numpy.ndarray, shape ``(n_p_real,)``
            ``sqrt(diag(cov))`` (clipped at 0).
        edata : EData
            Snapshot used to weight the normal matrix.
        """
        if x is None and self.last_result is not None:
            x = np.asarray(self.last_result.x, dtype=np.float64)
        return self.efit.covariance(x=x, scale=scale, **fd_kwargs)

    def stderr(
        self,
        x: Optional[np.ndarray] = None,
        *,
        scale: str = "reduced_chi2",
        **fd_kwargs: Any,
    ) -> np.ndarray:
        """One-sigma parameter uncertainties.

        Convenience wrapper around :meth:`covariance` that returns
        only the ``sigma`` vector (``sqrt(diag(cov))``).
        """
        _, sigma, _ = self.covariance(x=x, scale=scale, **fd_kwargs)
        return np.asarray(sigma, dtype=np.float64)
