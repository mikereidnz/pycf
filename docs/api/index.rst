===
API
===

Complete API reference for PyCF modules.

Core Modules
============

.. toctree::
   :maxdepth: 2

   cfl
   import_sljm
   inten
   paramcalc

Utility Modules
===============

.. toctree::
   :maxdepth: 2

   spinh
   pyemp
   cfl_util
   njsymbols
   matel

Quick Reference
================

**Hamiltonian Construction**

.. code-block:: python

    from pycf import cfl
    h = cfl.Hamiltonian()
    h.add_term(tensor, coefficient)
    h.diag()
    eigenvalues = h.eigenvalues()

**Data Import**

.. code-block:: python

    from pycf.import_sljm import ImportSLJM
    importer = ImportSLJM('path/to/matel')
    cf_tensor = importer.CF

**Intensity Calculations**

.. code-block:: python

    from pycf import inten
    dipole_strength = inten.dipole_str(tensor)
    spectrum = inten.inten(h, tensor, hwhm=50, temp=300)

**Parameter Calculations**

.. code-block:: python

    from pycf import paramcalc
    xi = paramcalc.Xi_val(J, L, S)
    c_k_q = paramcalc.Ckq(k, q, J, L, S)

**Spin Hamiltonian**

.. code-block:: python

    from pycf import spinh
    spin_h = spinh.extract_spin_hamiltonian(cf_hamiltonian, J)

Classes
=======

Hamiltonian
-----------

Main class for crystal field calculations.

.. autoclass:: pycf.cfl.Hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Tensor
------

Sparse matrix representation of operators.

.. autoclass:: pycf.cfl.Tensor
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ExData
------

Experimental data container.

.. autoclass:: pycf.cfl.ExData
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ImportSLJM
----------

SLJM file parser and tensor loader.

.. autoclass:: pycf.import_sljm.ImportSLJM
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Key Functions
=============

Crystal Field
-------------

.. autosummary::

   pycf.cfl.Hamiltonian.diag
   pycf.cfl.Hamiltonian.set_coeff
   pycf.cfl.Hamiltonian.update_coeff
   pycf.cfl.Hamiltonian.gen_summary

Intensity
---------

.. autosummary::

   pycf.inten.dipole_str
   pycf.inten.group_transitions
   pycf.inten.Spectrum

Parameters
----------

.. autosummary::

   pycf.paramcalc.Xi_val
   pycf.paramcalc.RInt4f
   pycf.paramcalc.Ckq

Wigner Symbols
--------------

.. autosummary::

   pycf.njsymbols.wigner_3j
   pycf.njsymbols.wigner_6j
   pycf.njsymbols.wigner_9j

Matrix Elements
---------------

.. autosummary::

   pycf.matel.matel
   pycf.matel.t_q

Data Import
-----------

.. autosummary::

   pycf.import_sljm.ImportSLJM.print_names

See individual module pages for detailed documentation.
