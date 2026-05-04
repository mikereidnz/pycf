# Contributing to PyCF

Thank you for your interest in contributing to PyCF! This document provides guidelines and instructions for contributing code, documentation, and bug reports.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and adhere to our Code of Conduct.

## Ways to Contribute

- **Report bugs**: File issues on GitHub for any problems you find
- **Suggest features**: Open a GitHub issue to discuss new functionality
- **Fix bugs**: Submit pull requests to fix known issues
- **Improve documentation**: Fix typos, improve clarity, add examples
- **Add tests**: Increase test coverage for existing code
- **Optimize performance**: Profile and improve slow code paths

## Getting Started

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mikereidnz/pycf.git
   cd pycf
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .  # Editable install for development
   pip install pytest numpy scipy cython
   ```

4. **Build the C extension** (if modifying C code):
   ```bash
   python setup.py build_ext --inplace
   ```

5. **Run tests to verify setup**:
   ```bash
   python -m pytest tests/ -v
   make -C cfl test  # For C tests
   ```

## Development Workflow

### Before Starting

1. Check if an issue exists on GitHub for what you want to work on
2. If not, create one to discuss your proposed changes
3. Wait for feedback before starting significant work

### Making Changes

1. **Create a feature branch** from `devel`:
   ```bash
   git checkout devel
   git pull origin devel
   git checkout -b feature/my-feature-name
   ```

2. **Follow the code style**:
   - Python: Follow PEP 8
   - Use type hints for all new functions (see existing code for examples)
   - Add docstrings to all public functions and classes
   - Use Google-style docstrings with proper formatting

3. **Add tests for your changes**:
   - For Python: Add module-level unit tests to `tests/unit/test_*.py` (or to a new example-based subdirectory of `tests/` for integration tests)
   - For C: Add tests to `cfl/tests/*_test.c`
   - Aim for high coverage of new code
   - Test both happy path and error cases

4. **Update documentation**:
   - Update `docs/*.rst` files if adding new features
   - Update module docstrings
   - Add examples if appropriate

5. **Commit with clear messages**:
   ```bash
   git commit -m "Brief description (50 chars max)

   Longer explanation if needed. Reference issues with #123.
   Co-authored-by: Your Name <your.email@example.com>"
   ```

### Testing

Before submitting a PR, verify:

```bash
# Python tests (should pass)
python -m pytest tests/ -q

# C tests (should pass)
make -C cfl clean
make -C cfl test

# Type checking (non-blocking but good to fix warnings)
mypy pycf/ --ignore-missing-imports

# Build documentation (requires the [docs] extra: pip install -e ".[docs]")
cd docs && sphinx-build -b html . _build/html

# Verify no deprecation warnings
python -m pytest tests/ -v 2>&1 | grep -i deprecat
```

## Code Quality Standards

### Type Hints

All new functions must have type hints:

```python
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

def my_function(param1: int, param2: Optional[str] = None) -> Tuple[float, np.ndarray]:
    """Description of function."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(a: np.ndarray, b: float) -> Dict[str, Any]:
    """Brief one-line description.

    Longer description explaining the function, its purpose,
    and any important notes about usage.

    Parameters
    ----------
    a : np.ndarray
        Description of parameter a.
    b : float
        Description of parameter b.

    Returns
    -------
    dict
        Dictionary with keys 'result' and 'status'.

    Raises
    ------
    ValueError
        If a is empty or has incompatible shape.
    """
    pass
```

### Error Handling

- Use specific exception types (not bare `except:`)
- Provide helpful error messages
- Validate input parameters early
- Check array dimensions and types

Example:

```python
def process_tensor(tensor, scale: float) -> Tensor:
    if not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be numeric, got {type(scale)}")
    if tensor is None:
        raise ValueError("tensor cannot be None")
    if tensor.size == 0:
        raise ValueError("tensor cannot be empty")

    # ... process tensor ...
    return result
```

### Parameter Validation for (mu, n) Fitting

When adding or modifying code that uses the (mu, n) experimental data format
for fitting, ensure proper validation of `Hamiltonian` parameters:

**Required parameters for (mu, n) fitting:**

- `h.minimum_q` (int): Smallest non-zero q in crystal field expansion (typically 2)
- `h.half_integer_states` (bool): Whether m values are half-integers stored as doubled integers

**Validation checklist:**

1. In `Hamiltonian` class initialization:
   - `minimum_q` defaults to `None` (no default value — must be explicitly set)
   - `half_integer_states` defaults to `False` (reasonable default for integer m)

2. In fitting code (`EFit.__init__`):
   ```python
   if self.h.minimum_q is None:
       raise ValueError("Hamiltonian.minimum_q must be set before fitting with (mu, n) data...")
   if not isinstance(self.h.half_integer_states, bool):
       raise ValueError("Hamiltonian.half_integer_states must be a bool...")
   ```

3. In conversion functions (`mu_n_to_level`):
   - Validate eigenvector matrix shape matches state labels
   - Provide clear error messages if (mu, n) pairs don't exist in spectrum

**User guidance:**

- Always set `minimum_q` and `half_integer_states` explicitly before using (mu, n) format
- For f-electrons (J=5/2, 7/2): use `half_integer_states=True`
- For d-electrons with integer m: use `half_integer_states=False`
- See EXAMPLES.md section "Using (mu, n) Format for Low-Symmetry Crystal Fields" for workflow

**Related code:**

- `pycf/cfl_util.py::mu_n_to_level()` — Conversion implementation and detailed documentation
- `pycf/cfl.pyx::Hamiltonian` — Parameter definition
- `pycf/cfl.pyx::EFit.__init__()` — Validation enforcement
- `tests/integration/ceylf/test_exdata.py` — Test examples

### Legacy modules

Some files are kept for backwards compatibility but are not actively
maintained (for example, thin wrappers around external legacy
executables).  These modules are flagged with the `LEGACY:` convention
and receive only minimal audit attention.

A module is considered **legacy** when its module-level docstring's
first line begins with the literal token `LEGACY:`, e.g.:

```python
"""LEGACY: short summary of the module's historical purpose.

Longer description...
"""
```

For legacy modules:

- Only **security-critical** issues (e.g. command injection, arbitrary
  file write, credential leakage) and import-time crashes are fixed.
- Type hints, coverage gaps, lint/style warnings, and dead-code clean-up
  are **not** addressed; mypy / coverage / bandit are configured to skip
  these files (see `pyproject.toml`, `pytest.ini`, `.bandit`).
- New code **must not** add dependencies on legacy modules.  Prefer the
  actively maintained alternatives (`pycf.cfl`, `pycf.import_sljm`,
  etc.).
- When auditing the codebase, treat the contents of `LEGACY:` modules
  as out-of-scope unless a finding is `critical` severity.

Currently flagged as legacy:

- `pycf/pyemp.py` — Python wrapper around Michael F. Reid's external
  EMP executables (`cfit`, `inten`, `vtrans`, `spectrum`).

To **add** a new module to the legacy list:

1. Prepend `LEGACY:` to the first line of its docstring with a brief
   reason.
2. Add an entry to `[[tool.mypy.overrides]]` in `pyproject.toml` with
   `ignore_errors = true`.
3. Add the path to the `omit` list in `pytest.ini` `[coverage:run]`.
4. Add the path to `exclude_dirs` in `.bandit`.
5. Update the list above and note the addition in `CHANGELOG.md`.

To **remove** a module from the legacy list, reverse those steps and
fix any issues that the audit tools subsequently report.

### Preserving legacy data files byte-faithfully

Some files must stay byte-identical to the producing tool: the JMCALC/
SLJM matrix-element fixtures, sample output from legacy pascal programs
kept for comparison, etc. **Place any such file under a `matel/`
subdirectory anywhere in the tree.** All `matel/` directories are
excluded from every formatter, whitespace fixer, and lint hook in
`.pre-commit-config.yaml`. The same convention applies to test
fixtures (`tests/integration/<dir>/matel/`) and example data
(`examples/<dir>/matel/`).

The historical Sphinx documentation under `docs/legacy/` is also kept
as a record and is excluded from the Sphinx build (via
`exclude_patterns` in `docs/conf.py`) and from pre-commit hooks.

## Submitting a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/my-feature-name
   ```

2. **Create a pull request** on GitHub:
   - Title: Brief description (50 chars max)
   - Description: Explain what, why, and how
   - Reference related issues with `Fixes #123`
   - Link to any discussion issues

3. **PR checklist**:
   - [ ] Tests pass locally
   - [ ] Type checking passes
   - [ ] Documentation updated
   - [ ] Commit messages are clear
   - [ ] No breaking changes to public API

4. **Respond to feedback**:
   - All feedback is constructive and in good faith
   - Push changes to the same branch
   - Maintain conversation in the PR

## C Code Contributions

For C code changes:

1. **Follow C99 standard** (not C11)
2. **Memory management**:
   - Always check malloc/realloc return values
   - Free allocated memory in reverse order of allocation
   - Use consistent error handling patterns
3. **Testing**:
   - Add tests to `cfl/tests/*_test.c`
   - Verify with `make -C cfl test`
   - Run with AddressSanitizer: `-fsanitize=address`

Example C test:

```c
void test_function_name() {
    // Arrange
    int expected = 42;

    // Act
    int result = my_function();

    // Assert
    if (result != expected) {
        printf("fail: expected %d, got %d\n", expected, result);
        return;
    }
    printf("pass\n");
}
```

## Documentation Contributions

- Update `.rst` files in `docs/` directory
- Install the docs dependencies: `pip install -e ".[docs]"`
- Build with `sphinx-build -b html docs/ docs/_build/html`
- Check HTML output at `docs/_build/html/index.html`
- For API docs, docstrings are auto-extracted via Sphinx autodoc

## Reporting Bugs

When filing a bug report, include:

1. **Description**: What did you expect vs what happened?
2. **Reproduction steps**: Minimal example to reproduce the issue
3. **Environment**: Python version, OS, pycf version
4. **Traceback**: Full error message and stack trace
5. **Additional context**: Any relevant environment variables, build flags, etc.

Example issue:

```
## Bug: Crystal field calculation gives wrong eigenvalues

### Description
When fitting Ce³⁺ in YLF, eigenvalues are incorrect.

### Steps to Reproduce
1. Load ceylf/matel files
2. Create Hamiltonian with CF term scaled to 1.5
3. Call h.diag()
4. Print eigenvalues - they don't match experiment

### Expected
Eigenvalues should be close to experimental values.

### Actual
Values differ by > 100 cm⁻¹.

### Environment
- Python 3.11
- numpy 1.24
- scipy 1.10
- pycf dev (commit abc123)
- Linux 5.14.0

### Traceback
(if applicable)
```

## Seeking Help

- **Questions**: Open a GitHub discussion or issue
- **Architecture**: Check `docs/overview.rst` for high-level design
- **Examples**: See `examples/` directory for working code
- **API reference**: See `docs/api/` or generated HTML docs

## License

By contributing to PyCF, you agree that your contributions will be licensed
under the same license as the project (GNU GPL v3 for most code, MIT for
specific modules as noted in file headers).

## Additional Resources

- [GitHub Issues](https://github.com/mikereidnz/pycf/issues)
- [GitHub Discussions](https://github.com/mikereidnz/pycf/discussions)
- [Documentation](https://github.com/mikereidnz/pycf/blob/devel/docs/index.rst)
- [Installation Guide](https://github.com/mikereidnz/pycf/blob/devel/docs/installation.rst)
- [API Reference](https://github.com/mikereidnz/pycf/blob/devel/docs/api/index.rst)

## Thank You!

Thank you for contributing to PyCF! Your efforts help make crystal field
calculations more accessible and robust for the research community.
