==========
About pycf
==========

pycf is a collection of Python modules for crystal field theory and spin Hamiltonian calculations.

**For detailed installation instructions, see** `INSTALL.md <INSTALL.md>`_.

**Quick Start (Development)**

::

  git clone https://github.com/mikereidnz/pycf.git
  cd pycf
  python3 -m venv env
  source env/bin/activate
  pip install -e ".[dev,examples]"
  python -m pytest tests/ -q

**Project Structure**

- ``cfl/`` — C99 core library for Hamiltonian assembly and fitting
- ``pycf/`` — Python/Cython wrappers
- ``examples/`` — End-to-end workflows for real materials
- ``tests/`` — Python and C unit tests

**Documentation**

- `INSTALL.md <INSTALL.md>`_ — Installation guide with platform-specific instructions
- `doc/ <doc/>`_ — Design notes and technical documentation
- `examples/ <examples/>`_ — Material-specific usage examples

**Authorship**

Originally developed by Sebastian Horvath (sebastian.horvath@gmail.com). Currently maintained by Mike Reid (mike.reid@canterbury.ac.nz).

A legacy version is available at https://bitbucket.org/sebastianhorvath/pycf/.

**License**

GNU General Public License v3 (GPLv3)
