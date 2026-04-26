=======================
PyCF Documentation
=======================

PyCF is a Python package for crystal field calculations on rare-earth ions, combining
density-functional-theory (DFT) computations with experimental fitting.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   quickstart
   api/index
   guides/index

Overview
========

Crystal field calculations are essential for understanding the electronic structure
and spectroscopic properties of rare-earth ions in crystals and solutions. PyCF provides
a comprehensive toolkit for:

- Constructing and diagonalizing crystal field Hamiltonians
- Calculating transition intensities and spectra
- Fitting crystal field parameters to experimental data
- Extracting spin-Hamiltonian descriptions

Key Features
~~~~~~~~~~~~

- **Performance-critical C library** with 10-100x speedup for large matrices
- **Comprehensive Python API** with type hints and detailed documentation
- **BLAS/LAPACK integration** with optional Intel MKL support
- **Flexible tensor framework** for arbitrary rare-earth ions
- **Experimental data handling** with ExData class supporting multiple modes

Getting Started
===============

For a quick introduction, see the :doc:`quickstart` guide.

For detailed API reference, visit the :doc:`api/index`.

For installation instructions, see :doc:`installation`.

Requirements
~~~~~~~~~~~~

- Python 3.8+
- NumPy, SciPy
- C compiler (gcc, clang, or Intel ICC)
- BLAS/LAPACK (ATLAS, OpenBLAS, MKL, or system LAPACK)

Installation
============

Quick install::

    python setup.py install --prefix=/usr/local

For detailed instructions including system-specific requirements,
see :doc:`installation`.

Basic Usage
===========

Load crystal field tensors and build a Hamiltonian::

    from pycf.import_sljm import ImportSLJM
    from pycf import cfl

    # Load SLJM output
    importer = ImportSLJM('path/to/sljm/files')

    # Create Hamiltonian
    h = cfl.Hamiltonian()
    h.add_term(importer.CF, 1.0)  # Add crystal field term

    # Diagonalize
    h.diag()

    # Get eigenvalues
    eigenvalues = h.eigenvalues()

For more examples, see the :doc:`guides/index`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. [1] Gómez-Herrero, J. C., & Sanchez-Dehesa, J. (1988). Crystal Field Theory and Crystal Field Effects. In Handbook of the Physics and Chemistry of Rare Earths.

.. [2] Krupke, W. F. (1966). Induced-emission cross sections for laser transitions in Nd: YAG. IEEE Journal of Quantum Electronics, 7(4), 153-159.

.. [3] Reid, M. F. (1997). Parameterization of the Nd³⁺ Free Ion and Crystal Field Interaction in NdAl₃(BO₃)₄. The Journal of Physical Chemistry A, 101(36), 6773-6781.

.. [4] Rasch, M. J., & Yu, A. C. H. (2003). Efficient Computation of the Wigner 3j, 6j and 9j Symbols. SIAM Journal on Scientific Computing, 25(4), 1416-1428.

.. [5] Golding, R. M., & Halley, M. J. (1984). Spin-Hamiltonian and its Application to the Fine Structure of Rare-Earth Ions. Physical Review B, 30(8), 4661.

License
=======

PyCF is distributed under the GNU General Public License v3.
See LICENSE file for details.

Contributing
============

Contributions are welcome! Please submit issues and pull requests on GitHub.

Contact
=======

For questions and support, please contact the maintainers through GitHub issues.
