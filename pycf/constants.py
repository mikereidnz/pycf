"""
Physical and mathematical constants used in crystal field calculations.
All constants are in SI units unless otherwise noted.
References:
    - CODATA 2014 recommended values
    - NIST Physical Constant Database
"""
# ============================================================================
# Fundamental Physical Constants (SI)
# ============================================================================
# Electron mass (kg)
ELECTRON_MASS = 9.1093897e-31
# Elementary charge (C)
ELEMENTARY_CHARGE = 1.60217733e-19
# Permittivity of free space (F/m or NA^-2)
EPSILON_0 = 8.8541878e-12
# Reduced Planck constant (J·s)
HBAR = 1.05457266e-34
# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT = 2.997924580e8
# ============================================================================
# Useful Derived Constants (for spectroscopic calculations)
# ============================================================================
# Boltzmann constant conversion factor for wavenumbers
# Used in: exp(-E / (k_B * T)) where E is in cm^-1 and T in K
# Value = k_B * c * 100 in cm^-1 / K units (≈ 0.6952 cm^-1 / K)
BOLTZMANN_CM_INVERSE = 0.6952
# Bohr radius (m)
# Used in parameterized electron density calculations
BOHR_RADIUS = 0.529177210903
# ============================================================================
# Radial Integral Parameters
#
# From: Freeman and Watson, Phys. Rev. 127, 2058 (1962)
#       Parametrized electron density for calculating <r^k>
#
# These are used in paramcalc.py for calculating radial integrals
# for 3d^n and 4f^n configurations.
#
# Format: [R0, R1, R2] parameters for each configuration
# ============================================================================
# 3d^1 radial integral parameters
RADIAL_PARAMS_3D1 = [0.883, 1.897, 8.775]
# 3d^2 radial integral parameters
RADIAL_PARAMS_3D2 = [0.938, 2.273, 11.670]
# 3d^3 radial integral parameters
RADIAL_PARAMS_3D3 = [0.726, 1.322, 5.102]
# 3d^4 radial integral parameters
RADIAL_PARAMS_3D4 = [0.666, 1.126, 3.978]
# 3d^5 radial integral parameters
RADIAL_PARAMS_3D5 = [0.613, 0.960, 3.104]
# All 3d radial parameters (indexed by d-electron count: 1-5)
RADIAL_PARAMS_3D = {
    1: RADIAL_PARAMS_3D1,
    2: RADIAL_PARAMS_3D2,
    3: RADIAL_PARAMS_3D3,
    4: RADIAL_PARAMS_3D4,
    5: RADIAL_PARAMS_3D5,
}
# ============================================================================
# Numerical Tolerances
# ============================================================================
# Default tolerance for floating-point comparisons
DEFAULT_TOLERANCE = 1e-10
# Default tolerance for zero testing in optimization
ZERO_THRESHOLD = 1e-15
# Tolerance for test comparisons (wavenumbers in cm^-1)
TEST_TOLERANCE = 1e-2
# ============================================================================
# Configuration Constants for Crystal Field Calculations
# ============================================================================
# Maximum number of states typically handled
MAX_STATES = 1000
# Factor for prefixing dipole moments (10^-10 cm unit)
DIPOLE_MOMENT_UNIT = 1e10
# ============================================================================
# Refractive Index Constants
# ============================================================================
# Default refractive index (vacuum)
REFRACTIVE_INDEX_VACUUM = 1.0
# Typical values for common media
REFRACTIVE_INDEX_COMMON = {
    "vacuum": 1.0,
    "air": 1.0003,
    "glass": 1.5,
    "yttrium_aluminum_garnet": 1.82,
}
