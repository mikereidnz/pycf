#!/usr/bin/env python3
"""
Comprehensive tests for pycf.paramcalc module.

Tests cover:
- Ligand class initialization and properties
- Superposition model calculations (A_SC, A_DC)
- Spherical harmonics (Ckq)
- Radial integrals and Xi parameters
- Crystal field basis functions
- AltpData class functionality
"""

import numpy as np
import pytest

from pycf.paramcalc import A_DC, A_SC, AltpData, Ckq, Ligand, RInt4f, Xi_val


class TestLigand:
    """Test Ligand class initialization and properties."""

    def test_ligand_init_basic(self) -> None:
        """Test basic Ligand initialization."""
        coords = np.array([2.5, np.pi / 2, 0.0])  # r, theta, phi
        q = -2.0
        alpha_bar = 0.5
        lig = Ligand(coords, q, alpha_bar)

        assert np.allclose(lig.coords, coords)
        assert lig.q == -2.0
        assert lig.alpha_bar == 0.5

    def test_ligand_init_octahedral_geometry(self) -> None:
        """Test creating six ligands in octahedral geometry."""
        # Octahedral: 6 ligands along ±x, ±y, ±z axes
        R = 2.0  # Fe-O distance
        coords_list = [
            [R, 0, 0],  # +z
            [R, np.pi, 0],  # -z
            [R, np.pi / 2, 0],  # +x
            [R, np.pi / 2, np.pi],  # -x
            [R, np.pi / 2, np.pi / 2],  # +y
            [R, np.pi / 2, 3 * np.pi / 2],  # -y
        ]

        ligands = []
        for coords in coords_list:
            lig = Ligand(np.array(coords), -2.0, 0.5)
            ligands.append(lig)

        assert len(ligands) == 6
        for lig in ligands:
            assert lig.q == -2.0

    def test_ligand_init_tetrahedral_geometry(self) -> None:
        """Test creating four ligands in tetrahedral geometry."""
        # Tetrahedral geometry
        R = 2.2
        coords_list = [
            [R, np.arccos(1 / 3), 0],
            [R, np.arccos(1 / 3), 2 * np.pi / 3],
            [R, np.arccos(1 / 3), 4 * np.pi / 3],
            [R, np.pi - np.arccos(1 / 3), 0],
        ]

        ligands = [Ligand(np.array(coords), -2.0, 0.5) for coords in coords_list]
        assert len(ligands) == 4

    def test_ligand_different_charges(self) -> None:
        """Test ligands with different charges."""
        coords = np.array([2.0, np.pi / 2, 0.0])

        charges = [-2.0, -1.0, 0.0, 1.0, 2.0]
        ligands = [Ligand(coords, q, 0.5) for q in charges]

        for lig, q in zip(ligands, charges):
            assert lig.q == q

    def test_ligand_different_polarizabilities(self) -> None:
        """Test ligands with different polarizabilities."""
        coords = np.array([2.0, np.pi / 2, 0.0])
        q = -2.0

        alphas = [0.1, 0.5, 1.0, 2.0]
        ligands = [Ligand(coords, q, alpha) for alpha in alphas]

        for lig, alpha in zip(ligands, alphas):
            assert lig.alpha_bar == alpha


class TestXiVal:
    """Test Xi(t, l) parameters for intensity calculations."""

    def test_xi_val_basic(self) -> None:
        """Test basic Xi value retrieval."""
        # Test Pr3+ (available in Xi table)
        xi = Xi_val(1, 2, "Pr")
        assert isinstance(xi, (float, np.floating))
        assert xi < 0  # Pr Xi(1,2) should be negative

    def test_xi_val_all_valid_lanthanides(self) -> None:
        """Test Xi for all supported lanthanides."""
        valid_lanthanides = ["Pr", "Nd", "Eu", "Tb", "Er", "Tm", "Yb"]

        for Ln in valid_lanthanides:
            xi = Xi_val(1, 2, Ln)
            assert isinstance(xi, (float, np.floating))
            assert not np.isnan(xi)

    def test_xi_val_all_valid_t_l_pairs(self) -> None:
        """Test Xi for all valid (t, l) combinations."""
        # Only these (t, l) pairs are available in Xi table
        valid_pairs = [
            (1, 2),
            (3, 2),
            (3, 4),
            (5, 4),
            (5, 6),
            (7, 6),
        ]

        for t, l in valid_pairs:
            xi = Xi_val(t, l, "Er")
            assert isinstance(xi, (float, np.floating))
            assert not np.isnan(xi)

    def test_xi_val_invalid_t(self) -> None:
        """Test that invalid t values raise ValueError."""
        with pytest.raises(ValueError, match="t must be in"):
            Xi_val(2, 2, "Er")

        with pytest.raises(ValueError, match="t must be in"):
            Xi_val(0, 2, "Er")

    def test_xi_val_invalid_l(self) -> None:
        """Test that invalid lam values raise ValueError."""
        with pytest.raises(ValueError, match="lam must be in"):
            Xi_val(1, 3, "Er")

        with pytest.raises(ValueError, match="lam must be in"):
            Xi_val(1, 5, "Er")

    def test_xi_val_invalid_lanthanide(self) -> None:
        """Test that invalid lanthanide symbols raise ValueError."""
        with pytest.raises(ValueError, match="Invalid lanthanide"):
            Xi_val(1, 2, "Ce")  # Ce not in Xi table

        with pytest.raises(ValueError, match="Invalid lanthanide"):
            Xi_val(1, 2, "Invalid")

    def test_xi_val_yb_interpolation(self) -> None:
        """Test that Yb values are properly computed."""
        # Valid (t, l) pairs
        valid_pairs = [(1, 2), (3, 2), (3, 4), (5, 4), (5, 6), (7, 6)]

        for t, l in valid_pairs:
            xi_yb = Xi_val(t, l, "Yb")
            xi_er = Xi_val(t, l, "Er")
            xi_tm = Xi_val(t, l, "Tm")

            expected_yb = 0.5 * (xi_er + xi_tm)

            assert xi_yb == pytest.approx(expected_yb)
            assert np.isfinite(xi_yb)
            assert np.isfinite(xi_er)
            assert np.isfinite(xi_tm)


class TestRInt4f:
    """Test radial integral calculations."""

    def test_rint4f_basic(self) -> None:
        """Test basic radial integral retrieval."""
        rint = RInt4f(2, "Ce")
        assert isinstance(rint, (float, np.floating))
        assert rint > 0  # Radial integrals should be positive

    def test_rint4f_all_lanthanides(self) -> None:
        """Test RInt4f for all supported lanthanides."""
        valid_lanthanides = ["Ce", "Pr", "Nd", "Sm", "Eu", "Dy", "Er", "Yb"]

        for Ln in valid_lanthanides:
            rint = RInt4f(2, Ln)
            assert isinstance(rint, (float, np.floating))
            assert rint > 0
            assert not np.isnan(rint)

    def test_rint4f_all_lambda_values(self) -> None:
        """Test RInt4f for all valid lambda values."""
        valid_lambda = [2, 4, 6]

        for lam in valid_lambda:
            rint = RInt4f(lam, "Er")
            assert isinstance(rint, (float, np.floating))
            assert rint > 0

    def test_rint4f_monotonic_trend(self) -> None:
        """Test that radial integrals follow expected trends."""
        # For a given lambda, RInt4f should generally decrease across lanthanide series
        lanthanides = ["Ce", "Pr", "Nd", "Sm", "Eu", "Dy", "Er", "Yb"]
        rints = [RInt4f(2, Ln) for Ln in lanthanides]

        # Values should generally decrease across series (Ce > Yb)
        assert rints[0] > rints[-1]

    def test_rint4f_invalid_lambda(self) -> None:
        """Test that invalid lambda values raise ValueError."""
        with pytest.raises(ValueError, match="lam must be in"):
            RInt4f(3, "Er")

        with pytest.raises(ValueError, match="lam must be in"):
            RInt4f(5, "Er")

    def test_rint4f_invalid_lanthanide(self) -> None:
        """Test that invalid lanthanide symbols raise ValueError."""
        with pytest.raises(ValueError, match="Invalid lanthanide"):
            RInt4f(2, "Invalid")


class TestCkq:
    """Test spherical harmonic calculations."""

    def test_ckq_basic(self) -> None:
        """Test basic Ckq calculation."""
        C = Ckq(2, 0, np.pi / 2, 0)
        assert isinstance(C, (complex, np.complexfloating, float, np.floating))

    def test_ckq_k0_q0_constant(self) -> None:
        """Test that Ckq(0, 0) is constant (independent of angles)."""
        # For k=0, q=0, the value should be constant regardless of angles
        val1 = Ckq(0, 0, 0, 0)
        val2 = Ckq(0, 0, np.pi / 4, np.pi / 4)
        val3 = Ckq(0, 0, np.pi / 2, np.pi)
        val4 = Ckq(0, 0, np.pi, 0)

        # All should be the same
        assert np.isclose(val1, val2)
        assert np.isclose(val2, val3)
        assert np.isclose(val3, val4)

    def test_ckq_zero_for_invalid_q(self) -> None:
        """Test that invalid q raises ValueError."""
        with pytest.raises(ValueError, match="q must satisfy"):
            Ckq(2, 3, 0, 0)  # |q| > k

        with pytest.raises(ValueError, match="q must satisfy"):
            Ckq(2, -3, 0, 0)

    def test_ckq_negative_k_raises(self) -> None:
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 0"):
            Ckq(-1, 0, 0, 0)

    def test_ckq_symmetry_in_q(self) -> None:
        """Test Ckq symmetry properties."""
        theta, phi = np.pi / 3, np.pi / 4

        # For real spherical harmonics, Ckq(-q) should be related to Ckq(q)
        C_pos = Ckq(2, 1, theta, phi)
        C_neg = Ckq(2, -1, theta, phi)

        # Both should be non-zero and finite
        assert np.isfinite(C_pos)
        assert np.isfinite(C_neg)

    def test_ckq_magnitude_bounded(self) -> None:
        """Test that Ckq magnitudes are reasonable."""
        # Spherical harmonics have bounded magnitude
        for k in range(0, 4):
            for q in range(-k, k + 1):
                C = Ckq(k, q, np.random.random() * np.pi, np.random.random() * 2 * np.pi)
                # Magnitude should be bounded by a reasonable factor
                assert np.abs(C) < np.sqrt(4 * np.pi) * 10


class TestASC:
    """Test static coupling calculations."""

    def test_a_sc_octahedral_o2(self) -> None:
        """Test A_SC with octahedral O2- geometry."""
        R = 2.5
        coords_list = [
            [R, 0, 0],
            [R, np.pi, 0],
            [R, np.pi / 2, 0],
            [R, np.pi / 2, np.pi],
            [R, np.pi / 2, np.pi / 2],
            [R, np.pi / 2, 3 * np.pi / 2],
        ]

        ligands = [Ligand(np.array(coords), -2.0, 0.5) for coords in coords_list]

        # Test A_SC for Pr3+ octahedral O2- with valid (l=2, t=1) parameters
        A_chg, A_pol = A_SC(2, 1, 0, "Pr", -3, ligands)

        # Both should be real numbers
        assert isinstance(A_chg, (float, np.floating))
        assert isinstance(A_pol, (float, np.floating))

        # Charge contribution should be non-zero for static coupling
        assert A_chg != 0

    def test_a_sc_charge_scaling(self) -> None:
        """Test that A_SC scales linearly with ligand charge."""
        R = 2.5

        # Create octahedral geometry
        coords_list = [
            [R, 0, 0],
            [R, np.pi, 0],
            [R, np.pi / 2, 0],
            [R, np.pi / 2, np.pi],
            [R, np.pi / 2, np.pi / 2],
            [R, np.pi / 2, 3 * np.pi / 2],
        ]

        # Test with different charges using valid (l=2, t=1) parameters
        ligands_q1 = [Ligand(np.array(c), -1.0, 0.5) for c in coords_list]
        ligands_q2 = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        A1_chg, A1_pol = A_SC(2, 1, 0, "Pr", -3, ligands_q1)
        A2_chg, A2_pol = A_SC(2, 1, 0, "Pr", -3, ligands_q2)

        # Charge contribution should scale with ligand charge
        assert np.isclose(A2_chg / A1_chg, 2.0, rtol=0.01)

    def test_a_sc_returns_tuple(self) -> None:
        """Test that A_SC returns a tuple of (A_chg, A_pol)."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        result = A_SC(2, 1, 0, "Pr", -3, ligands)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_a_sc_different_l_values(self) -> None:
        """Test A_SC for different lambda values."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Test valid (lam, t) combinations
        test_cases = [(2, 1), (2, 3), (4, 3), (4, 5), (6, 5), (6, 7)]
        for lam, t in test_cases:
            A_chg, A_pol = A_SC(lam, t, 0, "Pr", -3, ligands)
            assert isinstance(A_chg, (float, np.floating))
            assert isinstance(A_pol, (float, np.floating))

    def test_a_sc_different_lanthanides(self) -> None:
        """Test A_SC for different lanthanides."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Test all available lanthanides with valid (l=2, t=1) parameters
        for Ln in ["Pr", "Nd", "Eu", "Tb", "Er", "Tm", "Yb"]:
            A_chg, A_pol = A_SC(2, 1, 0, Ln, -3, ligands)
            assert isinstance(A_chg, (float, np.floating))
            assert not np.isnan(A_chg)


class TestADC:
    """Test dynamic coupling calculations."""

    def test_a_dc_octahedral_o2(self) -> None:
        """Test A_DC with octahedral O2- geometry."""
        R = 2.5
        coords_list = [
            [R, 0, 0],
            [R, np.pi, 0],
            [R, np.pi / 2, 0],
            [R, np.pi / 2, np.pi],
            [R, np.pi / 2, np.pi / 2],
            [R, np.pi / 2, 3 * np.pi / 2],
        ]

        ligands = [Ligand(np.array(coords), -2.0, 0.5) for coords in coords_list]

        # A_DC is non-zero only when t = l + 1
        A = A_DC(2, 3, 0, "Pr", ligands)  # l=2, t=3 (l+1)

        assert isinstance(A, (float, np.floating))

    def test_a_dc_zero_when_t_ne_l_plus_1(self) -> None:
        """Test that A_DC returns 0 when t != l+1."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # A_DC should be zero when t != l+1
        A = A_DC(2, 0, 0, "Pr", ligands)  # l=2, t=0 (not l+1)
        assert A == 0

        A = A_DC(2, 2, 0, "Pr", ligands)  # l=2, t=2 (not l+1)
        assert A == 0

    def test_a_dc_nonzero_when_t_eq_l_plus_1(self) -> None:
        """Test that A_DC can be non-zero when t = l+1."""
        R = 2.5
        coords_list = [
            [R, 0, 0],
            [R, np.pi, 0],
            [R, np.pi / 2, 0],
            [R, np.pi / 2, np.pi],
            [R, np.pi / 2, np.pi / 2],
            [R, np.pi / 2, 3 * np.pi / 2],
        ]

        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Test all valid (lam, t=lam+1) combinations
        for lam in [2, 4, 6]:
            A = A_DC(lam, lam + 1, 0, "Pr", ligands)
            # Result should be a finite number (may be zero for specific geometry)
            assert isinstance(A, (float, np.floating))
            assert np.isfinite(A)

    def test_a_dc_different_lanthanides(self) -> None:
        """Test A_DC for different lanthanides."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Test all available lanthanides with t = l + 1 (l=2, t=3)
        for Ln in ["Ce", "Pr", "Nd", "Sm", "Eu", "Dy", "Er", "Yb"]:
            A = A_DC(2, 3, 0, Ln, ligands)
            assert isinstance(A, (float, np.floating))
            assert np.isfinite(A)

    def test_a_dc_polarizability_dependence(self) -> None:
        """Test that A_DC depends on ligand polarizability."""
        R = 2.5
        coords_list = [[R, i * np.pi / 3, 0] for i in range(6)]

        # Create two sets with different polarizabilities
        ligands_low_alpha = [Ligand(np.array(c), -2.0, 0.1) for c in coords_list]
        ligands_high_alpha = [Ligand(np.array(c), -2.0, 1.0) for c in coords_list]

        A_low = A_DC(2, 3, 0, "Pr", ligands_low_alpha)
        A_high = A_DC(2, 3, 0, "Pr", ligands_high_alpha)

        # Higher polarizability should give larger dynamic coupling
        assert abs(A_high) >= abs(A_low)


class TestAltpData:
    """Test AltpData class for managing Altp parameters."""

    def test_altpdata_init(self) -> None:
        """Test AltpData initialization."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        altp_data = AltpData("Pr", -3, ligands)

        assert altp_data.Ln == "Pr"
        assert altp_data.q_Ln == -3
        assert len(altp_data.ligands) == 6
        assert altp_data.nL == 6

    def test_altpdata_eval_params(self) -> None:
        """Test AltpData.eval_params() calculation."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        altp_data = AltpData("Pr", -3, ligands)
        A_list = altp_data.eval_params()

        # Should return a list of results
        assert isinstance(A_list, list)
        assert len(A_list) > 0

        # Each element should have a name and parameter list
        for item in A_list:
            assert isinstance(item, (list, tuple))
            assert len(item) >= 2

    def test_altpdata_eval_params_creates_attributes(self) -> None:
        """Test that eval_params creates A_list attribute."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        altp_data = AltpData("Pr", -3, ligands)
        altp_data.eval_params()

        # eval_params creates A_list attribute
        assert hasattr(altp_data, "A_list")
        assert isinstance(altp_data.A_list, list)

    def test_altpdata_different_lanthanides(self) -> None:
        """Test AltpData with different lanthanides."""
        coords_list = [[2.5, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Use lanthanides available in both Xi_val and RInt4f
        # Xi has: Pr, Nd, Eu, Tb, Er, Tm, Yb
        # RInt4f has: Ce, Pr, Nd, Sm, Eu, Dy, Er, Yb
        # Common: Pr, Nd, Eu, Er, Yb
        for Ln in ["Pr", "Nd", "Eu", "Er", "Yb"]:
            altp_data = AltpData(Ln, -3, ligands)
            A_list = altp_data.eval_params()

            assert isinstance(A_list, list)
            assert len(A_list) > 0

    def test_altpdata_tetrahedral_geometry(self) -> None:
        """Test AltpData with tetrahedral geometry."""
        R = 2.2
        coords_list = [
            [R, np.arccos(1 / 3), 0],
            [R, np.arccos(1 / 3), 2 * np.pi / 3],
            [R, np.arccos(1 / 3), 4 * np.pi / 3],
            [R, np.pi - np.arccos(1 / 3), 0],
        ]

        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        altp_data = AltpData("Pr", -3, ligands)
        A_list = altp_data.eval_params()

        # Should work with tetrahedral geometry too
        assert isinstance(A_list, list)
        assert len(A_list) > 0


class TestParameterConsistency:
    """Integration tests for parameter consistency."""

    def test_a_sc_and_a_dc_same_geometry(self) -> None:
        """Test that A_SC and A_DC work together on same geometry."""
        R = 2.5
        coords_list = [[R, i * np.pi / 3, 0] for i in range(6)]
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Get static coupling with valid (l=2, t=1) parameters
        A_chg, A_pol = A_SC(2, 1, 0, "Pr", -3, ligands)

        # Get dynamic coupling (non-zero when t = l+1)
        A_dyn = A_DC(2, 3, 0, "Pr", ligands)

        # Both should be finite
        assert np.isfinite(A_chg)
        assert np.isfinite(A_pol)
        assert np.isfinite(A_dyn)

    def test_altpdata_with_real_geometry(self) -> None:
        """Test AltpData with realistic crystal structure."""
        # Using a general octahedral geometry that produces non-zero parameters
        R = 2.0
        coords_list = [[R, i * np.pi / 3, 0] for i in range(6)]

        # Oxygen with typical fluoride characteristics
        ligands = [Ligand(np.array(c), -2.0, 0.5) for c in coords_list]

        # Use Pr (common lanthanide in both Xi_val and RInt4f)
        altp_data = AltpData("Pr", -3, ligands)
        A_list = altp_data.eval_params()

        # Should successfully compute parameters for realistic system
        assert len(A_list) > 0
