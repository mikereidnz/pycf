====================
Spin Hamiltonian
====================

Extract spin-Hamiltonian parameters from crystal field calculations.

What is a Spin Hamiltonian?
===========================

A spin Hamiltonian provides an effective description of the ground electronic state,
replacing the full electronic Hamiltonian with a much simpler effective Hamiltonian
acting on spin degrees of freedom alone.

For rare-earth ions, the effective spin Hamiltonian typically includes:

- **g-tensor**: Magnetic moment scaling
- **Anisotropy**: Uniaxial or rhombic
- **Hyperfine coupling**: Coupling to nuclear spin

Basic Extraction
================

Extract spin-Hamiltonian from crystal field calculation:

.. code-block:: python

    from pycf.import_sljm import ImportSLJM
    from pycf import cfl, spinh
    
    # Build and diagonalize crystal field Hamiltonian
    importer = ImportSLJM('ceylf/matel')
    h = cfl.Hamiltonian()
    h.add_term(importer.CF, 1.0)
    h.diag()
    
    # Extract spin-Hamiltonian for ground state doublet
    # For Ce³⁺: ⁴f¹, J = 5/2 (6 levels → 3 Kramers doublets)
    spin_h = spinh.extract_spin_hamiltonian(
        h,
        J=2.5,  # Half-integer J
        num_doublets=1  # Extract ground state only
    )
    
    print("Spin-Hamiltonian parameters:")
    print(spin_h)

G-Tensor
========

Extract the g-tensor from magnetic moment tensors:

.. code-block:: python

    # Magnetic moment operator: μ = g * μ_B * J
    # For rare-earth: μ = μ_B * sqrt(J(J+1)) * g_J
    
    # Get magnetic moment eigenvector basis
    ground_state = h.eigenvectors()[:, 0]  # Ground state
    
    # Project magnetic tensor to ground state
    mu_x = importer.MAGX
    mu_y = importer.MAGY
    mu_z = importer.MAGZ
    
    # Effective g-values
    g_x = np.abs(np.conj(ground_state) @ mu_x.to_dense() @ ground_state)
    g_y = np.abs(np.conj(ground_state) @ mu_y.to_dense() @ ground_state)
    g_z = np.abs(np.conj(ground_state) @ mu_z.to_dense() @ ground_state)
    
    print(f"g-tensor: ({g_x:.4f}, {g_y:.4f}, {g_z:.4f})")

Anisotropy
==========

For uniaxial (z-axis quantization):

.. code-block:: python

    # Energy splitting in magnetic field
    # E = g_z * μ_B * B_z for B along z
    # E = g_perp * μ_B * B_perp for B perpendicular
    
    # Anisotropy parameter
    D = 0.5 * (g_z**2 - g_perp**2)  # Uniaxial
    
    # The spin-Hamiltonian is:
    # H_s = D * (S_z**2) + g_z * μ_B * B_z * S_z

Temperature-Dependent G-Factor
==============================

For systems with excited states close to ground:

.. code-block:: python

    def effective_g_tensor(h, temperature=10):
        """Calculate temperature-dependent effective g-tensor."""
        evals = h.eigenvalues()
        evecs = h.eigenvectors()
        
        # Boltzmann weights
        k_B = 0.69504  # cm⁻¹/K
        energies = evals - evals[0]  # Relative to ground
        weights = np.exp(-energies / (k_B * temperature))
        weights /= weights.sum()
        
        # Thermal average of g-tensor
        g_eff = np.zeros((3, 3))
        for i, (w, E) in enumerate(zip(weights, evals)):
            state = evecs[:, i]
            # ... calculate g-tensor for this state ...
            # g_eff += w * g_i
        
        return g_eff

Hyperfine Coupling
==================

Extract hyperfine coupling constants:

.. code-block:: python

    # Hyperfine tensor (if available in SLJM)
    hyp_tensor = importer.HYP
    
    # Project to ground state
    ground_state = h.eigenvectors()[:, 0]
    
    # Effective hyperfine coupling
    A = np.conj(ground_state) @ hyp_tensor.to_dense() @ ground_state
    
    print(f"Hyperfine coupling: {A:.4f} cm⁻¹")

Validation
==========

Validate spin-Hamiltonian by comparing with experiment:

.. code-block:: python

    # Apply external magnetic field and diagonalize
    B_field = 0.1  # Tesla
    h_mag = cfl.Hamiltonian()
    h_mag.add_term(importer.CF, 1.0)
    h_mag.add_term(importer.MAG, g_eff * B_field)  # Zeeman term
    h_mag.diag()
    
    # Transition energies should match electron paramagnetic resonance (EPR)
    evals = h_mag.eigenvalues()
    transitions = evals[1:] - evals[0]
    
    print("Predicted EPR transitions:")
    print(transitions)

Advanced: Kramers Doublet Projection
====================================

For half-integer J, each Kramers doublet can be described as an effective spin-1/2:

.. code-block:: python

    def project_to_spin_half(h, doublet_index=0):
        """Project Kramers doublet to effective spin-1/2."""
        evals = h.eigenvalues()
        evecs = h.eigenvectors()
        
        # Get two states in the doublet
        i1 = 2 * doublet_index
        i2 = 2 * doublet_index + 1
        
        state1 = evecs[:, i1]
        state2 = evecs[:, i2]
        
        # Gap between states
        delta_E = evals[i2] - evals[i1]
        
        # Extract effective spin-1/2 parameters
        # ...
        
        return delta_E, g_eff, etc.

References
==========

- Golding, R. M., & Halley, M. J. (1984). Spin-Hamiltonian and its Application to the Fine Structure of Rare-Earth Ions. Physical Review B, 30(8), 4661.
- Pilbrow, J. R. (1990). Transition metal ions in crystals. Clarendon Press.
- See spinh module for implementation details
