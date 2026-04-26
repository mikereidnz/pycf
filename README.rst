==========
About pycf
==========

pycf is a collection of Python modules for crystal field theory and spin Hamiltonian calculations. It provides tools for setting up and fitting energy levels of rare-earth ions using crystal-field Hamiltonians, as well as utilities for working with spin Hamiltonians and transition intensities. The core library is implemented in C99 for performance, with Python/Cython wrappers for ease of use.

**Status and Quality**

.. image:: https://github.com/mikereidnz/pycf/workflows/CI/badge.svg
   :target: https://github.com/mikereidnz/pycf/actions
   :alt: CI Status

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python Version

.. image:: https://img.shields.io/badge/test%20status-106%20passing-brightgreen
   :alt: Tests Passing

**For detailed installation instructions, see** `INSTALL.md <INSTALL.md>`_.

**Quick Start (Development)**

::

  git clone https://github.com/mikereidnz/pycf.git
  cd pycf
  python3 -m venv env
  source env/bin/activate
  pip install -e ".[test,examples]"
  python -m pytest tests/ -q

**Project Structure**

- ``cfl/`` — C99 core library for setting up and fitting Hamiltonians. 
- ``pycf/`` — Python/Cython wrappers
- ``examples/`` — End-to-end workflows for real materials
- ``tests/`` — Python and C unit tests

**Documentation**

- `INSTALL.md <INSTALL.md>`_ — Installation guide with platform-specific instructions
- `docs/legacy/ <docs/legacy/>`_ — Legacy technical documentation preserved for reference
- `examples/ <examples/>`_ — Material-specific usage examples

**Authorship**

Originally developed by Sebastian Horvath (sebastian.horvath@gmail.com). Currently maintained by Mike Reid (mike.reid@canterbury.ac.nz).

A legacy version is available at https://bitbucket.org/sebastianhorvath/pycf/ and also on the "legacy" branch of this repository.

**License**

GNU General Public License v3 (GPLv3)
