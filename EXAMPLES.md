# Examples Guide

This guide describes the example workflows in `pycf` and how to run them.

## Prerequisites

Install pycf with examples dependencies:
```bash
pip install -e ".[examples]"
```

This installs additional packages like `pymatgen` for crystal structure handling.

## Running Examples

All examples should be run from their own directory (examples use relative paths for data):

```bash
cd examples/ceylf/
python exdata_example.py

cd ../eryso/
python mesh_fit.py
```

## Example Descriptions

### Ce:YLF Examples

Ce:YLF (Cerium:Yttrium Lithium Fluoride) is a well-studied laser material. These examples demonstrate the main features of pycf.

#### 1. `ceylf/exdata_example.py` - Experimental Data Fitting

**What it demonstrates:**
- Load crystal field matrix elements from SLJM format
- Define experimental energy levels (absolute and relative)
- Fit crystal field parameters to experimental data
- Analyze residuals and parameter quality

**Prerequisites:**
- Test data in `tests/ceylf/matel/f1cf/` (included)

**Expected output:**
- Fitted crystal field parameters (Ckq values)
- Energy level comparison (calculated vs. experimental)
- Fit quality metrics (rms, parameter uncertainty)

**Key concepts:**
- `ImportSLJM`: Load tensor data from electronic structure codes
- `ExData`: Represent experimental energy level measurements
- `cfl.Hamiltonian`: Assemble crystal field matrix
- Least-squares fitting workflow

**Typical runtime:** < 1 second

---

#### 2. `ceylf/mhfit_example.py` - Multi-Hamiltonian Fitting

**What it demonstrates:**
- Fit multiple physical properties simultaneously
- Handle competing parameter constraints
- Use tensor arithmetic for derived Hamiltonians
- Multi-property optimization

**Prerequisites:**
- Test data in `tests/ceylf/matel/f1cf/` (included)

**Expected output:**
- Joint fit results with multiple property weights
- Individual property residuals
- Trade-offs between fitting different observables

**Key concepts:**
- Multiple tensor definitions for different observables
- Weighted fitting (balance multiple properties)
- Optimization convergence analysis

**Typical runtime:** 1-5 seconds

---

#### 3. `ceylf/shfit_example.py` - Spin Hamiltonian Extraction

**What it demonstrates:**
- Extract low-energy Kramers doublet physics
- Map crystal field Hamiltonian to spin Hamiltonian
- Compare theory to spin-Hamiltonian parameters

**Prerequisites:**
- Test data in `tests/ceylf/matel/f1cf/` (included)

**Expected output:**
- Spin-Hamiltonian parameters (g-factors, D, E, A)
- Effective spin quantum numbers
- Ground state composition

**Key concepts:**
- `pycf.spinh`: Spin-Hamiltonian theory
- Projection of crystal field eigenvalues to effective spins
- Kramers doublet isolation

**Typical runtime:** < 1 second

**Reference:**
- 10.1016/j.optmat.2015.06.046 (Ce:YLF experiment)

---

### Er:YSO Examples

Er:YSO (Erbium:Yttrium Silicate) is a key material for quantum memory. These examples show advanced fitting strategies.

#### 4. `eryso/mesh_fit.py` - Grid Search Fitting

**What it demonstrates:**
- Large-scale parameter space exploration
- Grid (mesh) search over parameter ranges
- Identify promising initial conditions
- Parameter correlation analysis

**Prerequisites:**
- Test data in `tests/eryso/` (included)
- pymatgen library for crystal structure analysis

**Expected output:**
- Grid search results (parameter values vs. fit quality)
- Best-fit parameters from grid
- Parameter correlation heat maps
- 2D/3D visualization of fit landscape

**Key concepts:**
- `numpy` grid generation for parameter sweeps
- Batch fitting across parameter space
- Visualization of high-dimensional optimization landscape

**Typical runtime:** 10-30 seconds (varies with grid resolution)

---

#### 5. `eryso/mesh_fit_original_g.py` and `mesh_fit_transformed_g.py`

**What they demonstrate:**
- Alternative parameterizations of the same physics
- Impact of parameter choice on fitting
- Trade-offs in parameter space sampling

**Prerequisites:**
- Same as `mesh_fit.py`

**Expected output:**
- Similar results to `mesh_fit.py` but with different parameter distributions

**Key concepts:**
- Parameter transformations (g-tensor rotations, scaling)
- Equivalent physics with different mathematical representations
- Choosing optimal parameterization for fitting

**Typical runtime:** 10-30 seconds each

---

#### 6. `eryso/mhfit_siman.py` - Stochastic Optimization (Simulated Annealing)

**What it demonstrates:**
- Stochastic optimization algorithms (simulated annealing/MCMC)
- Escape local minima through probabilistic jumps
- Monte Carlo parameter exploration
- Uncertainty quantification from ensemble

**Prerequisites:**
- Test data in `tests/eryso/` (included)
- scipy for optimization algorithms

**Expected output:**
- Optimized parameters from stochastic fit
- Parameter distribution (posterior)
- Convergence history
- Correlation analysis of fitted parameters

**Key concepts:**
- `scipy.optimize.minimize` with different algorithms
- Ensemble fitting for uncertainty estimation
- MCMC for posterior sampling

**Typical runtime:** 20-60 seconds

---

#### 7. `eryso/mcmc_analysis.py` - Post-Processing MCMC Results

**What it demonstrates:**
- Analyze MCMC chains after optimization
- Compute parameter statistics (mean, std, correlation)
- Visualize posterior distributions
- Identify parameter correlations and degeneracies

**Prerequisites:**
- MCMC results file (generated by previous examples)
- matplotlib for visualization

**Expected output:**
- Parameter histograms and statistics
- Correlation scatter plots
- Chain convergence diagnostics
- Joint posterior contours

**Key concepts:**
- Statistical analysis of MCMC ensembles
- Credible intervals and confidence regions
- Identifying degenerate parameters

**Typical runtime:** 1-5 seconds

---

#### 8. `eryso/spinh_example.py` - Spin Hamiltonian Construction

**What it demonstrates:**
- Direct spin-Hamiltonian definition (without crystal field codes)
- Diagonalize effective spin Hamiltonian
- Calculate properties in low-energy manifold

**Prerequisites:**
- None (pure pycf spinh module)

**Expected output:**
- Spin Hamiltonian eigenvalues and eigenvectors
- Effective magnetic field effects on energy levels
- Ground state properties

**Key concepts:**
- `pycf.spinh` module for spin-Hamiltonian theory
- Direct construction without SLJM data
- Comparison to effective-spin theories

**Typical runtime:** < 1 second

---

### Sm:NaCaF₂ Example

#### 9. `smnacaf2/smnacaf2.py` - Full Material Analysis Workflow

**What it demonstrates:**
- End-to-end analysis of new material
- Crystal structure to physical parameters
- Multi-scale fitting (SLJM → crystal field → effective spins)
- Material characterization and documentation

**Prerequisites:**
- Crystal structure file (included)
- pymatgen for structure analysis
- Full pycf installation

**Expected output:**
- Crystal field parameters for Sm:NaCaF₂
- Comparison to literature values
- Energy level diagrams
- Temperature-dependent properties

**Key concepts:**
- Integrated workflow combining structure + fitting
- Multi-scale physics (ab initio → crystal field → spins)
- Material characterization pipeline

**Typical runtime:** 5-10 seconds

---

### Er:CaWO₄ Example

#### 10. `Altp_ercawo4/ercawo4_altp_calc.py` - Altp Coordination Parameter Calculation

**What it demonstrates:**
- Calculate Altp (apparent coordination number) from crystal structure
- Derive crystallographic parameters for use in fitting
- Connect pymatgen structures to crystal field models

**Prerequisites:**
- Crystal structure file (included)
- pymatgen library

**Expected output:**
- Altp parameter value
- Coordination geometry analysis
- Crystal field parameter estimates

**Key concepts:**
- Coordination analysis using pymatgen
- Structure-parameter relationships
- Bridging computational chemistry to crystal field theory

**Typical runtime:** < 1 second

---

### Advanced Fitting Techniques

#### 11. Using (mu, n) Format for Low-Symmetry Crystal Fields

**What it demonstrates:**
- Fit using folded magnetic quantum numbers (mu, n) instead of individual m values
- Useful when symmetry breaks and individual |m⟩ states become ambiguous
- Proper handling of half-integer magnetic quantum numbers (f-electrons)
- Validation of mu/n fitting parameters

**When to use (mu, n) format:**

The (mu, n) parametrization is essential when:
- Crystal field term strengths are large (strong mixing of |m⟩ states)
- Magnetic decoherence makes individual m values physically meaningless
- You have f-electrons with half-integer m (e.g., Ce:YLF with J=5/2)
- Experimental data is naturally grouped by "folded" states

**Prerequisites:**
- Ce:YLF test data in `tests/ceylf/matel/f1cf/` (included)
- Basic understanding of crystal field theory and energy level assignments

**Key parameters:**

```python
h.minimum_q = 2              # Smallest non-zero q in expansion
h.half_integer_states = True # For f-electrons: m = ±1/2, ±3/2, ±5/2
```

See `examples/ceylf/mu_exdata_example.py` for a complete workflow.

**Expected output:**
- Fitted crystal field parameters (Ckq values)
- Energy level comparison using (mu, n) assignments
- Parameter uncertainties
- Residual analysis

**Key concepts:**
- `mu = m * sign(minimum_q)` for effective folding
- For `half_integer_states=True`, effective q doubles: `mu = m * sign(2*minimum_q)`
- Validation catches missing or inconsistent parameters
- (mu, n) → level index conversion via :func:`cfl_util.mu_n_to_level`

**Typical runtime:** < 1 second

**Example usage:**

.. code-block:: python

    import pycf
    import numpy as np
    
    # Load crystal field data
    importer = pycf.ImportSLJM(...)
    h = pycf.cfl.Hamiltonian(importer.tensors)
    
    # Set mu/n parameters (REQUIRED for this format)
    h.minimum_q = 2              # C20, C22 expansion
    h.half_integer_states = True # Ce has f-electrons (J=5/2)
    
    # Create experimental data in (mu, n) format
    mu_n_data = np.array([
        [2, 1],    # 1st state with mu=+2
        [2, 2],    # 2nd state with mu=+2
        [0, 1],    # 1st state with mu=0
        [-2, 1],   # 1st state with mu=-2
    ], dtype=np.int32)
    experimental_energies = np.array([0.0, 45.2, 156.8, 234.5])
    
    exdata = pycf.cfl.ExData(
        (mu_n_data, experimental_energies),
        key=('mu', 'n', 'energy'),
        label_key='mu'
    )
    
    # Fit using (mu, n) format
    fit = pycf.cfl.EFit(h, exdata)
    fit.fit_cmplx(coefficients)

**Common pitfalls:**

1. **Forgetting to set minimum_q**: Raises ``ValueError: Hamiltonian.minimum_q must be set...``
   - **Fix**: Set `h.minimum_q` to your expansion's smallest q value (usually 2)

2. **Wrong half_integer_states for system**: Silent bug producing wrong mu values
   - **Rule**: `True` for f-electrons (J=5/2, 7/2, ...), `False` for d-electrons with integer m
   - **Check**: Verify against your material's ionic configuration

3. **Mismatch between (mu, n) and eigenstate spectrum**: Raises ``ValueError: (mu, n) pair not found...``
   - **Fix**: Use only (mu, n) pairs that actually exist for your Hamiltonian
   - **Tip**: Print the full mu/n spectrum first: use `gen_e_summary()` after diagonalization

---

## Data Organization

Each example accesses data via relative paths. The structure is:

```
examples/
├── ceylf/
│   ├── exdata_example.py
│   ├── mhfit_example.py
│   ├── shfit_example.py
│   └── (data in tests/ceylf/matel/ — accessed via relative imports)
│
├── eryso/
│   ├── mesh_fit.py
│   ├── mesh_fit_*.py
│   ├── mhfit_siman.py
│   ├── mcmc_analysis.py
│   ├── spinh_example.py
│   └── (data in tests/eryso/ — accessed via relative imports)
│
├── smnacaf2/
│   ├── smnacaf2.py
│   └── (data files in same directory)
│
└── Altp_ercawo4/
    ├── ercawo4_altp_calc.py
    └── (structure file in same directory)
```

## Understanding Example Code

### Typical Structure

Most examples follow this pattern:

```python
#!/usr/bin/env python3
"""Docstring describing the example."""

# 1. Load data
importer = ImportSLJM("path/to/matrix/elements")

# 2. Set up physics (crystal field)
hamiltonian = cfl.Hamiltonian([importer.tensor1, importer.tensor2, ...])

# 3. Define experimental data (optional)
exdata = cfl.ExData(measured_levels, "A")

# 4. Fit or analyze
result = fit_hamiltonian(hamiltonian, exdata)

# 5. Extract results and visualize
print_results(result)
plot_results(result)
```

### Data Flow

```
[SLJM Matrix Elements]
         ↓
    ImportSLJM
         ↓
    [Tensor Objects]
         ↓
    Hamiltonian
         ↓
    [Eigenvalues/Eigenvectors]
         ↓
    Experimental Comparison
         ↓
    [Fitted Parameters]
```

## Common Tasks

### Adapt Example to New Material

1. **Prepare SLJM input** (or generate from other software)
   - Need LS coupling basis matrix elements
   - Can use Cowan code, FAC, or similar

2. **Create ImportSLJM instance**
   ```python
   importer = ImportSLJM("/path/to/matrix/elements")
   ```

3. **Select relevant tensors**
   - Start with energy, spin-orbit coupling, crystal field
   - Add more tensors if needed

4. **Define experimental data**
   ```python
   data = np.array([[level, energy], ...])
   exdata = cfl.ExData(data, "A")
   ```

5. **Fit parameters**
   ```python
   result = fit_hamiltonian(hamiltonian, exdata)
   ```

### Compare to Theory

1. Load theoretical parameters
2. Set Hamiltonian coefficients
3. Diagonalize
4. Compare eigenvalues to experimental levels
5. Compute residuals and correlation analysis

### Extract Effective Spins

```python
from pycf.spinh import build_spin_hamiltonian

# Project ground Kramers doublet to effective S=1/2
g_tensor, D, E, A = build_spin_hamiltonian(
    ham_cf,  # Crystal field Hamiltonian
    n_cf_states=2,  # Kramers doublet
)
```

## Troubleshooting

### Import Error: "No module named pymatgen"

Some examples need extra dependencies:
```bash
pip install ".[examples]"  # Install pymatgen and similar
```

### "Cannot find matel directory"

Examples use relative paths. Run from the example directory:
```bash
cd examples/ceylf/
python exdata_example.py  # ✓ Works

python examples/ceylf/exdata_example.py  # ✗ May fail
```

### SLJM Data Not Found

Ensure test data is present:
```bash
ls tests/ceylf/matel/f1cf/  # Should list .txt, .mi_, .st_ files
```

If missing, reinstall from source or clone repository.

### Fitting Does Not Converge

- Start with finer grid search to find good initial conditions
- Reduce parameter ranges
- Check experimental data for outliers
- Try different optimization algorithms (scipy, nlopt)

## Performance Tips

### Speed Up Grid Search

```python
# Use multiprocessing
from multiprocessing import Pool

with Pool(n_workers=4) as pool:
    results = pool.map(evaluate_fit, parameter_grid)
```

### Cache Hamiltonian Matrix

```python
# Build matrix once, fit multiple times
matrix = hamiltonian.matrix()
for params in parameter_sweep:
    eigvals = diag_cached(matrix)
```

### Parallel Fitting

```python
# Use scipy.optimize.minimize with n_jobs>1
result = minimize(objective, x0, method="Powell", options={"n_jobs": 4})
```

## Further Reading

- **INSTALL.md** - Installation and dependency management
- **CONTRIBUTING.md** - Contributing improvements to examples
- **pycf.import_sljm** - API for loading matrix elements
- **pycf.cfl** - Hamiltonian construction and fitting
- **pycf.spinh** - Spin-Hamiltonian theory
- **tests/** - Additional examples in test fixtures

## References

- Ce:YLF: 10.1016/j.optmat.2015.06.046
- Er:YSO: Examples follow PhD research by original authors
- Sm:NaCaF₂: Material characterization example
- General crystal field theory: Textbooks by Carnall, Reid
