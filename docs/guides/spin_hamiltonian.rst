================
Spin Hamiltonian
================

This guide explains how to extract and print calculated g, A, and Q tensors
from crystal-field calculations, using the Er:YSO ``site1pycf`` calculations
as an example.

Two Different Spin-Hamiltonian APIs
===================================

``pycf.cfl.SpinHamiltonian`` projects crystal-field wavefunctions into an
effective spin space. Its ``calc_param`` method extracts calculated parameter
tensors.

``pycf.spinh.SpinH`` constructs an effective spin-Hamiltonian matrix from
parameter tensors, or inverts supplied interaction matrices. In the
``site1pycf`` scripts, it constructs reference spectra from experimental
``gg``, ``Ag``, and ``Qg`` tensors. These reference energy differences are
then used as targets for fitting the crystal-field parameters.

Thus, using experimental spin-Hamiltonian tensors as fitting input does not
itself print the calculated tensors. After fitting, use ``calc_param`` with
the fitted crystal-field parameters.

.. warning::

   Earlier affected builds mishandled complex matrix layouts during
   projection. Recalculate their ``calc_param`` results and repeat direct
   tensor fits using ``esh_fit`` or ``mesh_fit``. Full-Hamiltonian spectra
   and spectral ``mh_fit`` objectives are unchanged by this correction.
   See :ref:`projection-layout-validation` below.

Effective Hamiltonian and Units
===============================

For an isolated Kramers doublet with effective electronic spin S = 1/2,
the interactions considered here are

.. math::

   H_\mathrm{eff}
   = \mu_B \mathbf{B}^{\mathsf{T}} g \mathbf{S}
   + \mathbf{I}^{\mathsf{T}} A \mathbf{S}
   + \mathbf{I}^{\mathsf{T}} Q \mathbf{I}.

Here, S and I denote spin operators. For the Er isotope considered in the
hyperfine example, I = 7/2.

* g is dimensionless.
* A and Q have units of energy: cm\ :sup:`-1` when the supplied coupling
  constants use the crystal-field calculation's cm\ :sup:`-1` convention.
* ``pycf.cfl_util.cm12MHz`` converts these energies to MHz using
  29979.2458 MHz per cm\ :sup:`-1`.
* ``SpinH`` uses magnetic fields in tesla and energies in MHz, inserting its
  own magnetic conversion factor for its Zeeman term.

The older ``site1pycf`` scripts use ``29.9702547e3`` for some MHz conversions.
That differs from the current helper's ``29979.2458``. Keep this distinction
in mind when reproducing old fits.

Extracting Ground-State g, A, and Q
===================================

The following code uses variables already defined in the ``site1pycf``
calculation scripts:

* ``thfs`` is the ``ImportSLJM`` collection containing electronic and nuclear
  matrix elements.
* ``thfs_list`` contains the Hamiltonian tensors in that same basis,
  including the named ``MTOT`` and ``PTOT`` combinations.
* ``coeff`` contains the crystal-field parameters and the nuclear coupling
  constants ``HYP`` and ``EQHYP``.

To inspect the fitted result, run this code after copying ``res["coeff"]``
into ``coeff``. To inspect the starting model, use the initial ``coeff``
instead. Do not mix tensors from the electronic-only and electronic+nuclear
datasets in one Hamiltonian.

.. code-block:: python

    import numpy as np

    from pycf import cfl
    from pycf.cfl_util import cm12MHz

    exclude = {"MX", "MY", "MZ", "HYP", "EQHYP"}
    h0 = cfl.Hamiltonian([
        tensor for tensor in thfs_list
        if tensor.get_name() not in exclude
    ])
    h0.set_coeff(coeff)

    sh = cfl.SpinHamiltonian(
        ["zeeman", "hyperfine", "quadrupole"],
        S=0.5, I=3.5, level=1,
    )
    sh.set_pro_data(
        [thfs.MAGX, thfs.MAGY, thfs.MAGZ, thfs.HYP, thfs.EQHYP],
        {"HYP": coeff["HYP"], "EQHYP": coeff["EQHYP"]},
    )

    g, A, Q = sh.calc_param(h0, svd_sym=False)

    np.set_printoptions(precision=5, suppress=True)
    print("Calculated g:\n", g)
    print("Calculated A (MHz):\n", cm12MHz(A))
    print("Calculated Q (MHz):\n", cm12MHz(Q))

The Hamiltonian ``h0`` contains the free-ion and crystal-field interactions
but excludes the magnetic-field, hyperfine, and quadrupole terms. Its
wavefunctions provide the basis into which the interaction operators are
projected. ``calc_param`` performs the diagonalization itself and introduces
a small magnetic term internally to order the states.

The nuclear coupling constants are applied through ``set_pro_data``.
Use their current fitted values, not separate hard-coded starting values.
The results are returned in the order of the requested interactions.

.. important::

   Supply the unscaled ``thfs.MAGX``, ``thfs.MAGY``, and ``thfs.MAGZ``
   operators. Do not supply the ``mu_b``-scaled ``hfsMX``, ``hfsMY``, or
   ``hfsMZ`` tensors used for finite-field calculations. Scaling the
   projection operators by the Bohr magneton would incorrectly scale g.

The exclusion names above are specific to the ``site1pycf`` scripts. If
adapting this example, exclude all physical magnetic-field and hyperfine
interactions under the names used in your own Hamiltonian.

Level Selection and Electronic-Only Extraction
==============================================

``level`` is one-based and selects the first state of the subspace to
project. ``level=1`` selects the ground doublet and, in the hyperfine basis,
its nuclear-spin states.

For g alone, use the electronic-only dataset and a Hamiltonian constructed
from tensors in that dataset. For example, with the ``site1pycf`` variables:

.. code-block:: python

    h_electronic = cfl.Hamiltonian([
        tensor for tensor in t_list
        if tensor.get_name() not in {"MX", "MY", "MZ"}
    ])
    h_electronic.set_coeff(coeff)

    sh_g = cfl.SpinHamiltonian("zeeman", S=0.5, level=1)
    sh_g.set_pro_data([t.MAGX, t.MAGY, t.MAGZ])
    g_electronic = sh_g.calc_param(h_electronic, svd_sym=False)[0]
    print("Electronic-only g:\n", g_electronic)

For the first doublet of the excited 4I13/2 multiplet in these
electronic-only Er calculations, use ``level=17`` instead. Do not transfer
this level number unchanged to a hyperfine-expanded basis.

The electronic and hyperfine examples use different truncations
(``er_trunc40`` and ``erhfs_trunc2``), so even their calculated ground-state
g tensors need not agree exactly.

Spin-Basis Conventions and Symmetric Tensors
============================================

The effective electronic spin axes depend on the chosen doublet basis.
Consequently, the raw g and A tensors need not be symmetric or resemble
the experimental matrices element by element.

To express both tensors in a symmetric-g convention, apply the same
electronic-spin transformation to both:

.. code-block:: python

    U, singular_values, Vt = np.linalg.svd(g)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        raise ValueError(
            "A signed-g convention is needed to preserve spin handedness."
        )

    g_report = g @ R
    A_report = A @ R

    print("Symmetric g:\n", g_report)
    print("A in the SAME spin convention (MHz):\n", cm12MHz(A_report))
    print("Q (MHz):\n", cm12MHz(Q))

For a proper rotation R, this preserves the combined effective-model
spectrum. Q is unchanged because only the electronic spin axes are being
transformed, not the nuclear or crystal axes. A need not become exactly
symmetric in this convention.

The determinant check matters: an improper rotation cannot be implemented
as an ordinary electronic-spin basis rotation. In that case, retain the
raw tensors or choose a signed symmetric-g convention explicitly rather
than silently changing handedness.

``calc_param(..., svd_sym=True)`` also provides symmetric tensors, but
symmetrizes g and A independently. It can be useful for individual tensor
comparisons; it does not generally preserve their common spin convention
for constructing one combined Hamiltonian.

Before comparing off-diagonal signs with experiment, also check the
crystal-axis ordering and orientations. A change of crystal coordinates
is different from the electronic-spin basis transformation above.

Checked Er:YSO Example
======================

Using ``erhfs_trunc2`` and the starting coefficients from
``eryso_site1_bobyqa_4_Nick_Final_Thesis_no_variation.py``, rotated once by
pi about Z, gives the following results in the common symmetric-g convention.
This is the frame used by ``site1_abinitio_test`` with the corrected
``MAGX``/``MAGY`` definitions.

Starting from the original, unrotated thesis coefficients, apply this
before constructing the projection Hamiltonian:

.. code-block:: python

    from pycf.cfl_util import rotate_cf_params

    coeff = rotate_cf_params(coeff, np.pi, 0.0, 0.0)

Do not apply this again to coefficients already rotated by the comparison
driver. The projection correction does not replace or undo this physical
parameter rotation.

.. code-block:: text

    g:
    [[ 3.0262 -3.0005 -3.4414]
     [-3.0005  8.7471  5.7361]
     [-3.4414  5.7361  4.9228]]

    A (MHz):
    [[ 316.9431 -314.6890 -361.4575]
     [-314.5566  916.6520  601.9719]
     [-361.2742  601.5293  515.6720]]

    Q (MHz):
    [[ 4.8883  3.9468  4.2297]
     [ 3.9468 -4.2217 -5.3833]
     [ 4.2297 -5.3833 -0.6666]]

These are starting-model results, not a new fit or an extraction of the
saved fitted parameters. The common g/A transformation was also checked
to preserve the effective-model spectrum at a nonzero magnetic field.
The corrected symmetric g has the same off-diagonal sign pattern as the
experimental input in this frame. This does not imply equally good
agreement for A or Q; in particular, Q is only a first-order projection.
Fitting weights remain user inputs, separate from the projection correction.

.. _projection-layout-validation:

Projection Correction and Independent Checks
============================================

The projected operator must be

.. math::

   O_\mathrm{eff} = V^\dagger O V,

where the columns of V are the selected crystal-field eigenvectors.
The affected implementation instead produced
:math:`(V^\dagger O^* V)^\mathsf{T}`. Both the dense operator layout and the
projected-matrix readout have been corrected. With complex wavefunctions,
the error can change magnitudes as well as signs; it is not generally
removable by changing spin axes.

Rebuild the extension after updating the C source:

.. code-block:: bash

    python setup.py build_ext --inplace

For a direct NumPy check using the hyperfine example above, supply an
explicit small ordering operator so that ``h_check.z`` contains the exact
eigenvectors used by ``calc_param``:

.. code-block:: python

    magzs = 1e-6 * thfs.MAGZ
    magzs.name = "MAGZS"
    h_check = cfl.Hamiltonian(h0.tensors + [magzs])
    h_check.set_coeff(dict(coeff, MAGZS=1.0))
    parameters, blocks = sh.calc_param(h_check, matel=True, svd_sym=False)

    nI = 8  # 2*I + 1 for I = 7/2
    V = h_check.z[:, :2*nI]
    upper = np.triu(thfs.MAGX.get_matel())
    operator = upper + np.triu(upper, 1).conj().T
    projected = V.conj().T @ operator @ V
    electronic_indices = [0, nI]
    expected = projected[np.ix_(electronic_indices, electronic_indices)]
    np.testing.assert_allclose(blocks["magx"], expected, atol=1e-10)

``Tensor.get_matel()`` exposes the stored upper triangle, so Hermitian
completion is intentional here for the physical magnetic operator. Do not
apply that completion indiscriminately to individual crystal-field tensors.
The projection resets the ordering coefficient to zero afterward; the
stored eigenvectors still correspond to the ordering-field calculation.

Independent checks of the rebuilt Er:YSO calculation found agreement with
NumPy for all five projected operators to about 2e-15 in their native units.
All 35 full-Hamiltonian spectra and their spectral-fit objective were
unchanged. Mixed-direction fields are important: tests along crystal axes
alone can miss erroneous off-diagonal signs.

For the electronic-only ground doublet at 1e-4 T, the effective g magnitudes
are:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Field direction
     - Full CF splitting / (mu_B B)
     - Corrected projected g
   * - (x + y) / sqrt(2)
     - 4.45037748
     - 4.45037796
   * - (y + z) / sqrt(2)
     - 13.55429285
     - 13.55429318
   * - (x + z) / sqrt(2)
     - 2.14754094
     - 2.14754100

The combined projected g/A/Q Hamiltonian also reproduced the independently
projected interaction spectrum to 2.3e-9 cm\ :sup:`-1`. The regression tests
add complex synthetic models with known tensors, including I = 1 and
I = 7/2, and check both direct tensor-fitting objective paths.

First-Order Projection Versus a Full Hyperfine Calculation
==========================================================

``calc_param`` projects each interaction operator into the selected
crystal-field subspace. This is a first-order effective Hamiltonian:

.. math::

   H_\mathrm{eff}^{(1)} = P H_\mathrm{interaction} P.

It does not automatically include second-order mixing with other
crystal-field doublets, including pseudoquadrupole contributions.

By contrast, diagonalizing the full crystal-field-plus-hyperfine
Hamiltonian includes mixing within its retained basis. A spin Hamiltonian
fitted to those spectra can therefore give different effective parameters,
particularly Q. The distinction is important in the ``site1pycf`` workflow,
which fits full-model energy differences to spectra generated from
experimental spin-Hamiltonian parameters.

To validate a projected model, compare its transition energies against
the full calculation at zero field and at small fields in several
directions, using consistent units and including the same interactions.
The ``bi`` nuclear Zeeman term used by the experimental reference model is
not one of the three tensors extracted here.

For an isolated S = 1/2 doublet there is no nontrivial electronic
zero-field-splitting D term: a symmetric quadratic electronic-spin term
is proportional to the identity. More elaborate effective models require
an appropriate subspace and operator expansion, not a formula inferred
from the g values alone.

Related Code
============

* ``pycf/cfl.pyx``: ``SpinHamiltonian``, ``set_pro_data``, and ``calc_param``.
* ``pycf/spinh.py``: effective-model construction using ``SpinH``.
* ``examples/ceylf/shfit_example.py``: electronic-only g extraction.
* ``examples/eryso/mesh_fit.py``: electronic and hyperfine projection setup.
* :doc:`../api/cfl` and :doc:`../api/spinh`: API references.
