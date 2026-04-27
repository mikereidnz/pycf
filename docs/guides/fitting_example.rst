=================
Fitting Guide
=================

This guide demonstrates how to fit crystal field parameters to experimental data.

Basic Fitting Workflow
======================

1. Load experimental data
2. Define residual function
3. Optimize parameters
4. Validate results

Example: CeYLF
==============

Fit the crystal field parameters for Ce³⁺ in YLiF₄.

.. code-block:: python

    import numpy as np
    from scipy.optimize import minimize
    from pycf.import_sljm import ImportSLJM
    from pycf import cfl, inten

    # Load SLJM tensors
    importer = ImportSLJM('ceylf/matel')

    # Experimental data (energy in cm⁻¹, intensity in arbitrary units)
    exp_energies = np.array([0.0, 120.5, 250.3, 375.1])
    exp_intensities = np.array([1.0, 0.8, 0.6, 0.4])

    def calculate_spectrum(cf_param):
        """Calculate spectrum for given CF parameter."""
        h = cfl.Hamiltonian()
        h.add_term(importer.CF, cf_param)
        h.diag()

        # Calculate transition intensities
        spec = inten.inten(h, importer.MAG, hwhm=50, temp=300)
        return spec

    def residual(params):
        """Residual between calculated and experimental spectrum."""
        calc = calculate_spectrum(params[0])

        # Simple residual: sum of squared differences at experimental points
        residual = 0.0
        for energy, intensity in zip(exp_energies, exp_intensities):
            idx = int(energy / 0.5)  # Assuming 0.5 cm⁻¹ resolution
            if idx < len(calc):
                residual += (calc[idx] - intensity)**2

        return residual

    # Fit the parameter
    result = minimize(residual, x0=[1.0], method='Nelder-Mead')
    print(f"Best CF parameter: {result.x[0]:.4f}")
    print(f"Residual: {result.fun:.6f}")

    # Verify result
    best_h = cfl.Hamiltonian()
    best_h.add_term(importer.CF, result.x[0])
    best_h.diag()
    print("Eigenvalues (cm⁻¹):")
    print(best_h.eigenvalues())

Multiple Parameter Fitting
===========================

Fit multiple CF parameters (e.g., B⁰₂ and B⁰₄):

.. code-block:: python

    from pycf import spinh, cfl_util

    # Load separate B20 and B40 tensors
    # (Assuming SLJM provides these separately)

    def residual_multi(params):
        """Residual for multiple CF parameters."""
        h = cfl.Hamiltonian()
        h.add_term(importer.B20, params[0])
        h.add_term(importer.B40, params[1])
        h.diag()

        # Calculate spectrum and compare with experiment
        # ... same as before ...
        return residual

    # Fit multiple parameters
    result = minimize(residual_multi, x0=[1.0, 0.1], method='Nelder-Mead')
    print(f"B20 = {result.x[0]:.4f}, B40 = {result.x[1]:.4f}")

Temperature-Dependent Fitting
==============================

Fit parameters to spectra at multiple temperatures:

.. code-block:: python

    def residual_temperature(params):
        """Fit to spectra at multiple temperatures."""
        total_residual = 0.0

        for temperature in [10, 50, 100, 300]:  # Kelvin
            h = cfl.Hamiltonian()
            h.add_term(importer.CF, params[0])
            h.diag()

            spec = inten.inten(h, importer.MAG, hwhm=50, temp=temperature)

            # Compare with experimental spectrum at this temperature
            exp_spectrum = load_experimental(temperature)
            total_residual += np.sum((spec - exp_spectrum)**2)

        return total_residual

Advanced: Constrained Fitting
=============================

Fit with constraints using ``scipy.optimize.minimize`` with constraints:

.. code-block:: python

    from scipy.optimize import minimize, LinearConstraint, Bounds

    # Bounds: B20 in [0.5, 2.0], B40 in [-0.5, 0.5]
    bounds = Bounds([0.5, -0.5], [2.0, 0.5])

    result = minimize(
        residual_multi,
        x0=[1.0, 0.1],
        method='L-BFGS-B',  # Supports bounds
        bounds=bounds
    )

Validation
==========

After fitting, validate by comparing predictions:

.. code-block:: python

    # Get best-fit Hamiltonian
    best_h = cfl.Hamiltonian()
    best_h.add_term(importer.CF, result.x[0])
    best_h.diag()

    # Calculate spectrum
    calc_spec = inten.inten(best_h, importer.MAG, hwhm=50, temp=300)

    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(energies, exp_spectrum, 'ko-', label='Experiment')
    plt.plot(energies, calc_spec, 'r-', label='Fit')
    plt.legend()
    plt.xlabel('Energy (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.show()

References
==========

- scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- Example scripts in ``examples/`` directory
