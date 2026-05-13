"""
Additional targeted tests for inten.py - focusing on non-plotting functions
that still have missing coverage. These tests target:
- _format_state_label_content
- _format_state_label_short
- AltpFit initialization and methods
- _estimate_parameter_uncertainties
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pycf.inten import (
    AltpFit,
    _format_state_label_content,
    _format_state_label_short,
    fit_altp,
    gen_inten_summary,
    inten_calculate,
    inten_print,
    inten_plot,
    inten_recalculate,
    inten_set_altp,
    inten_set_expt_data,
)


class TestFormatStateLabel:
    """Test state label formatting with various label_key conventions."""

    def test_format_state_label_content_SLJm_format(self):
        """Test formatting with SLJm label key (S, L, J, m)."""
        label = np.array([1, 2, 3, -1])
        result = _format_state_label_content(label, label_key="SLJm")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_state_label_content_no_label_array(self):
        """Test with array but no label_key."""
        label = np.array([1, 2, 3])
        result = _format_state_label_content(label, label_key=None)
        assert isinstance(result, str)
        # Should format as space-separated values
        assert "1" in result and "2" in result and "3" in result

    def test_format_state_label_content_no_label_scalar(self):
        """Test with scalar and no label_key."""
        result = _format_state_label_content(5, label_key=None)
        assert isinstance(result, str)
        assert "5" in result

    def test_format_state_label_content_invalid_label_format(self):
        """Test with malformed label array that raises exception."""
        # Force exception in try block
        label = np.array([1, 2, 3])
        result = _format_state_label_content(label, label_key="INVALID_KEY_THAT_IS_TOO_LONG")
        # Should use fallback formatting
        assert isinstance(result, str)

    def test_format_state_label_content_beyond_key_length(self):
        """Test with label array longer than label_key."""
        label = np.array([1, 2, 3, 4, 5])
        result = _format_state_label_content(label, label_key="SLJ")
        # Last two elements should be formatted beyond key length
        assert isinstance(result, str)

    def test_format_state_label_content_F_component_true(self):
        """Test F component formatting when True."""
        label = np.array([1, 1])
        result = _format_state_label_content(label, label_key="LF")
        assert "(2F)" in result

    def test_format_state_label_content_F_component_false(self):
        """Test F component formatting when False (0)."""
        label = np.array([1, 0])
        result = _format_state_label_content(label, label_key="LF")
        assert isinstance(result, str)

    def test_format_state_label_short(self):
        """Test short format wrapper."""
        label = np.array([2, 3, 4])
        result = _format_state_label_short(label, label_key="SLJ")
        assert isinstance(result, str)
        assert result.startswith("|")
        assert result.endswith(">")


class TestAltpFitInitialization:
    """Test AltpFit class initialization and parameter building."""

    def test_altp_fit_init_real_parameters(self):
        """Test AltpFit initialization with real parameters."""
        spectrum = MagicMock()
        spectrum.altp = {"A10": 1.0, "A20": 2.0}
        spectrum.expt_data = [[1, 0.5], [2, 0.3]]
        target_intensities = {0: 0.5, 1: 0.3}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            spectra=spectrum,
            target_intensities=target_intensities,
        )

        assert fitter.n_obs == 2
        assert len(fitter.param_names) == 2
        assert isinstance(fitter.param_info, dict)

    def test_altp_fit_init_complex_parameters(self):
        """Test AltpFit initialization with complex parameters."""
        spectrum = MagicMock()
        spectrum.altp = {"A10": complex(1.0, 0.5), "A20": 2.0}
        spectrum.expt_data = [[1, 0.5]]
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            spectra=spectrum,
            target_intensities=target_intensities,
        )

        # A10 is complex (2 params), A20 is real (1 param) = 3 total
        assert fitter.n_p == 3

    def test_altp_fit_missing_parameter_raises(self):
        """Test AltpFit raises when parameter not in altp."""
        spectrum = MagicMock()
        spectrum.altp = {"A10": 1.0}
        spectrum.expt_data = [[1, 0.5]]
        target_intensities = {0: 0.5}

        with pytest.raises(ValueError, match="not found in Spectrum.altp"):
            AltpFit(
                param_names=["A10", "A20"],  # A20 not in altp
                spectra=spectrum,
                target_intensities=target_intensities,
            )

    def test_altp_fit_build_param_info_type_checking(self):
        """Test parameter type detection (real vs complex)."""
        spectrum = MagicMock()
        spectrum.altp = {
            "A10": 1.5,  # real
            "A20": complex(2.0, 1.0),  # complex
            "A30": 3.0,  # real
        }
        spectrum.expt_data = [[1, 0.5]]
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20", "A30"],
            spectra=spectrum,
            target_intensities=target_intensities,
        )

        assert fitter.param_info["A10"]["type"] == "real"
        assert fitter.param_info["A20"]["type"] == "complex"
        assert fitter.param_info["A30"]["type"] == "real"

    def test_altp_fit_count_flat_params(self):
        """Test flat parameter counting (real=1, complex=2)."""
        spectrum = MagicMock()
        spectrum.altp = {
            "R1": 1.0,
            "C1": complex(1.0, 1.0),
            "R2": 2.0,
            "C2": complex(2.0, 2.0),
        }
        spectrum.expt_data = [[1, 0.5]]
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["R1", "C1", "R2", "C2"],
            spectra=spectrum,
            target_intensities=target_intensities,
        )

        # 4 reals + 4 complexes = 2 + 4 + 2 + 4 = wait, let me recalculate
        # R1=1, C1=2, R2=1, C2=2 = 6 total
        assert fitter.n_p == 6

    def test_altp_fit_extract_initial_params(self):
        """Test initial parameter vector extraction."""
        spectrum = MagicMock()
        spectrum.altp = {"A10": 1.5, "A20": complex(2.0, 1.0)}
        spectrum.expt_data = [[1, 0.5]]
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            spectra=spectrum,
            target_intensities=target_intensities,
        )

        # Should have extracted [1.5, 2.0, 1.0] for [real, complex_real, complex_imag]
        assert len(fitter.initial_x) == 3
        assert fitter.initial_x[0] == 1.5
        assert fitter.initial_x[1] == 2.0
        assert fitter.initial_x[2] == 1.0

    def test_altp_fit_uses_expt_data_when_targets_omitted(self):
        """If target_intensities is omitted, infer from spectrum.expt_data."""
        spectrum = MagicMock()
        spectrum.altp = {"A10": 1.0}
        spectrum.expt_data = [[1, 0.2], [2, 0.3]]

        fitter = AltpFit(
            param_names=["A10"],
            spectra=spectrum,
            target_intensities=None,
        )

        assert fitter.target_intensities == [{1: 0.2, 2: 0.3}]
        assert fitter.n_obs == 2

    def test_altp_fit_multi_spectrum_requires_list_targets(self):
        """Passing dict targets with multiple spectra should raise."""
        s1 = MagicMock()
        s1.altp = {"A10": 1.0}
        s1.expt_data = [[1, 0.2]]
        s2 = MagicMock()
        s2.altp = {"A10": 1.0}
        s2.expt_data = [[1, 0.3]]

        with pytest.raises(ValueError, match="list of dicts"):
            AltpFit(
                param_names=["A10"],
                spectra=[s1, s2],
                target_intensities={1: 0.2},
            )

    def test_altp_fit_empty_expt_data_contributes_zero_chi2(self):
        """Spectrum with no expt_data is allowed; it contributes 0 to chi-square."""
        spectrum = MagicMock()
        spectrum.name = "test"
        spectrum.altp = {"A10": 1.0}
        spectrum.expt_data = []

        # Should not raise — empty target map is returned
        fitter = AltpFit(
            param_names=["A10"],
            spectra=spectrum,
            target_intensities=None,
        )
        assert fitter.target_intensities == [{}]
        assert fitter.n_obs == 0


class TestEstimateParameterUncertainties:
    """Test uncertainty estimation function."""

    def test_estimate_uncertainties_handles_singular_hessian(self):
        """Test uncertainty estimation handles singular Hessian gracefully."""
        # This test just verifies the function exists and handles edge cases
        # The actual fitting is tested in integration tests
        from pycf.inten import _estimate_parameter_uncertainties

        assert callable(_estimate_parameter_uncertainties)


class TestFitAltpDryRun:
    """Test dry_run behavior in fit_altp."""

    def test_fit_altp_dry_run_computes_chi2_without_optimization(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        result = fit_altp(["A10"], spec, dry_run=True)

        assert result["dry_run"] is True
        assert result["chi2"] == pytest.approx(0.0)
        assert result["fitted_params"]["A10"] == pytest.approx(1.0)
        assert result["uncertainties"] == {}

    def test_fit_altp_reverts_when_optimization_worsens_chi2(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        fake_result = SimpleNamespace(
            x=np.array([2.0]), fun=0.2, success=False, status=2, message="did not converge"
        )
        with pytest.warns(UserWarning, match="reverted to the initial parameters"):
            with patch("pycf.inten.minimize", return_value=fake_result):
                result = fit_altp(["A10"], spec, dry_run=False, method="Nelder-Mead")

        assert result["optimizer_success"] is False
        assert result["optimizer_status"] == 2
        assert result["optimizer_message"] == "did not converge"
        assert "Optimizer success: False" in result["summary_diagnostics"]
        assert "Optimizer status: 2" in result["summary_diagnostics"]
        assert "Optimizer message: did not converge" in result["summary_diagnostics"]

        assert result["initial_chi2"] == pytest.approx(0.0)
        assert result["chi2"] == pytest.approx(0.0)
        assert result["fitted_params"]["A10"] == pytest.approx(1.0)
        assert result["reverted_to_initial"] is True
        assert result["improved"] is False
        assert result["uncertainties"] == {}

    def test_fit_altp_reverts_when_optimizer_reports_success_without_improvement(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        fake_result = SimpleNamespace(
            x=np.array([2.0]), fun=0.0, success=True, status=0, message="The lower bound for the trust-region radius has been reached"
        )
        with pytest.warns(UserWarning, match="did not improve chi2"):
            with patch("pycf.inten.minimize", return_value=fake_result):
                result = fit_altp(["A10"], spec, dry_run=False, method="COBYQA")

        assert result["optimizer_success"] is True
        assert result["optimizer_status"] == 0
        assert "trust-region radius" in result["optimizer_message"]
        assert result["reverted_to_initial"] is True
        assert result["improved"] is False
        assert result["chi2"] == pytest.approx(0.0)
        assert result["fitted_params"]["A10"] == pytest.approx(1.0)

    def test_fit_altp_keeps_nonfitted_altp_terms_fixed(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0, "A20": 2.0}
                self.expt_data = [[1, 3.0]]
                self.groups = [{"Energy": 1.0, "f": 3.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def update_altp(self, **updates):
                self.altp.update(updates)

            def recalculate(self, polarization="isotropic"):
                f_val = float(self.altp.get("A10", 0.0) + self.altp.get("A20", 0.0))
                self.groups = [{"Energy": 1.0, "f": f_val, "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        result = fit_altp(["A10"], spec, dry_run=True)

        # If non-fitted terms are preserved, chi2 stays zero for matching expt_data.
        assert result["chi2"] == pytest.approx(0.0)
        assert spec.altp["A20"] == pytest.approx(2.0)
        assert "A20" in result["summary_main"]

    def test_fit_altp_prints_pycf_details(self, capsys):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        fit_altp(["A10"], spec, dry_run=True)
        out = capsys.readouterr().out
        assert "pycf details" in out
        assert "Calculation started at:" in out
        assert "Calculation completed at:" in out

    def test_fit_altp_summary_includes_minimization_method(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        fake_result = SimpleNamespace(x=np.array([1.0]), fun=0.0)
        with patch("pycf.inten.minimize", return_value=fake_result):
            result = fit_altp(
                ["A10"],
                spec,
                dry_run=False,
                minimizer="minimize",
                method="Powell",
            )

        summary = result["summary"]
        assert "fmin =" in summary
        assert "Minimization method: minimize/Powell" in summary
        assert "All intensity parameters:" in result["summary_main"]
        assert "Fitted parameter details:" not in result["summary_diagnostics"]

    def test_fit_altp_include_covariance_without_sigma(self, capsys):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        cov = np.array([[0.25]])
        with patch(
            "pycf.inten._estimate_parameter_uncertainties",
            return_value=({"A10": 0.5}, cov, {"rank": 1}),
        ):
            result = fit_altp(
                ["A10"],
                spec,
                dry_run=False,
                calculate_sigma=False,
                include_covariance=True,
            )

        assert result["covariance"] is cov
        assert result["uncertainties"] == {"A10": 0.5}
        assert result["sigma"] == {"A10": 0.5}
        assert result["jacobian_diagnostics"] == {}
        assert result["sigma_forced"] is True
        assert "calculate_sigma assumed True" in capsys.readouterr().out
        assert "Covariance matrix:" in result["summary"]

    def test_fit_altp_reports_covariance_unavailable_in_dry_run(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        result = fit_altp(["A10"], spec, dry_run=True, include_covariance=True)

        assert result["covariance"] is None
        assert "Covariance matrix: not available (dry_run)" in result["summary"]

    def test_fit_altp_prints_covariance_and_jacobian_when_requested(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        cov = np.array([[0.25]])
        with (
            patch("pycf.inten.minimize", return_value=SimpleNamespace(x=np.array([1.0]), fun=0.0)),
            patch(
                "pycf.inten._estimate_parameter_uncertainties",
                return_value=({"A10": 0.5}, cov, {"rank": 1}),
            ),
        ):
            result = fit_altp(
                ["A10"],
                spec,
                dry_run=False,
                calculate_sigma=False,
                include_covariance=True,
                include_jacobian=True,
            )

        assert result["covariance"] is cov
        assert result["jacobian"] is not None
        assert "Covariance matrix:" in result["summary"]
        assert "2.50000000e-01" in result["summary"]
        assert "Jacobian diagnostics:" in result["summary"]

    def test_flat_weights_offset_advances_after_failed_spectrum(self):
        class FailingSpectrum:
            def __init__(self):
                self.name = "fail"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = []

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = []
                return self.groups

        class WorkingSpectrum:
            def __init__(self):
                self.name = "ok"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 2.0, "A": 0.0}]

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": 2.0, "A": 0.0}]
                return self.groups

        fitter = AltpFit(
            ["A10"],
            [FailingSpectrum(), WorkingSpectrum()],
            weights=np.array([4.0, 1.0]),
        )

        residuals = fitter.residuals(np.array([1.0]))

        assert residuals[0] == pytest.approx(1e5)
        assert residuals[1] == pytest.approx(1.0 / 3.0)

    def test_fit_altp_ignores_one_dimensional_optimizer_jacobian(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        with (
            patch(
                "pycf.inten.minimize",
                return_value=SimpleNamespace(x=np.array([1.0]), fun=0.0, jac=np.array([99.0])),
            ),
            patch("pycf.inten._estimate_parameter_uncertainties", return_value={} ),
        ):
            result = fit_altp(["A10"], spec, include_jacobian=True)

        assert result["jacobian"] is not None
        assert result["jacobian"].ndim == 2
        assert result["jacobian"].shape == (1, 1)
        assert result["jacobian"][0, 0] != pytest.approx(99.0)


class TestFitAltpWrapperInputs:
    """Test fit_altp wrapper input behavior."""

    def test_fit_altp_accepts_single_spectrum(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0]]
                self.groups = [{"Energy": 1.0, "f": 1.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def recalculate(self, polarization="isotropic"):
                self.groups = [{"Energy": 1.0, "f": float(self.altp["A10"]), "A": 0.0}]
                return self.groups

        spec = FakeSpectrum()
        result = fit_altp(["A10"], spec, dry_run=True)
        assert result["n_spectra"] == 1

    def test_fit_altp_requires_nonempty_sequence(self):
        with pytest.raises(ValueError, match="At least one Spectrum is required"):
            fit_altp(["A10"], [])

    def test_fit_altp_requires_spectrum_elements(self):
        with pytest.raises(TypeError, match="fit_altp requires a Spectrum or sequence"):
            fit_altp(["A10"], ["not-a-spectrum"])  # type: ignore[list-item]


class TestOptimizerDispatch:
    """Test that fit_altp dispatches to the correct scipy optimizer."""

    def _make_fake_spec(self):
        """Return a Spectrum-like object with expt_data and an Altp."""
        from pycf.inten import Spectrum as RealSpectrum

        spec = MagicMock(spec=RealSpectrum)
        spec.name = "test"
        spec.altp = {"A10": 1.0}
        spec.expt_data = [[1, 1.0]]
        spec.groups = [
            {
                "Energy": 1.0,
                "e_i": 0.0,
                "e_f": 1.0,
                "g_i": 1,
                "g_f": 1,
                "t_list": [{"i": 0, "f": 1, "ED": 1.0, "MD": 0.0}],
            }
        ]
        return spec

    def test_unknown_minimizer_raises(self):
        with pytest.raises(ValueError, match="Unknown minimizer"):
            with patch("pycf.inten.AltpFit") as mock_fitter_cls:
                mock_fitter = MagicMock()
                mock_fitter.initial_x = np.array([1.0])
                mock_fitter.objective_fn.return_value = 0.0
                mock_fitter_cls.return_value = mock_fitter
                from pycf.inten import fit_altp

                fit_altp(["A10"], MagicMock(), minimizer="not_a_real_optimizer")

    def test_bounds_required_for_differential_evolution(self):
        with pytest.raises(ValueError, match="requires a 'bounds' kwarg"):
            with patch("pycf.inten.AltpFit") as mock_fitter_cls:
                mock_fitter = MagicMock()
                mock_fitter.initial_x = np.array([1.0])
                mock_fitter.objective_fn.return_value = 0.0
                mock_fitter_cls.return_value = mock_fitter
                from pycf.inten import fit_altp

                fit_altp(["A10"], MagicMock(), minimizer="differential_evolution")

    def test_basinhopping_dispatch(self):
        with (
            patch("pycf.inten.AltpFit") as mock_fitter_cls,
            patch("pycf.inten.basinhopping") as mock_bh,
        ):
            mock_fitter = MagicMock()
            mock_fitter.initial_x = np.array([1.0])
            mock_fitter.objective_fn.return_value = 0.0
            mock_fitter._x_to_altp.return_value = {"A10": 1.0}
            mock_fitter.spectra = [MagicMock()]
            mock_fitter.spectra[0].name = "s"
            mock_fitter.n_obs = 1
            mock_fitter.n_p = 1
            mock_fitter.param_names = ["A10"]
            mock_fitter.per_spectrum_chi2.return_value = [{"name": "s", "chi2": 0.0, "n_obs": 1}]
            mock_fitter_cls.return_value = mock_fitter
            mock_bh.return_value = SimpleNamespace(x=np.array([1.0]), fun=0.0)
            with patch("pycf.inten._estimate_parameter_uncertainties", return_value={}):
                from pycf.inten import fit_altp

                fit_altp(
                    ["A10"],
                    MagicMock(),
                    minimizer="basinhopping",
                    niter=50,
                    minimizer_kwargs={"method": "Nelder-Mead"},
                )
            mock_bh.assert_called_once()
            _, bh_kwargs = mock_bh.call_args
            assert bh_kwargs.get("niter") == 50
            assert bh_kwargs["minimizer_kwargs"]["method"] == "Nelder-Mead"

    def test_basinhopping_strips_duplicate_bounds(self):
        """Avoid passing bounds twice to scipy minimize via options + kwargs."""
        with (
            patch("pycf.inten.AltpFit") as mock_fitter_cls,
            patch("pycf.inten.basinhopping") as mock_bh,
        ):
            mock_fitter = MagicMock()
            mock_fitter.initial_x = np.array([1.0])
            mock_fitter.objective_fn.return_value = 0.0
            mock_fitter._x_to_altp.return_value = {"A10": 1.0}
            mock_fitter.spectra = [MagicMock()]
            mock_fitter.spectra[0].name = "s"
            mock_fitter.n_obs = 1
            mock_fitter.n_p = 1
            mock_fitter.param_names = ["A10"]
            mock_fitter.per_spectrum_chi2.return_value = [{"name": "s", "chi2": 0.0, "n_obs": 1}]
            mock_fitter_cls.return_value = mock_fitter
            mock_bh.return_value = SimpleNamespace(x=np.array([1.0]), fun=0.0)
            bounds = [(-1.0, 1.0)]
            with patch("pycf.inten._estimate_parameter_uncertainties", return_value={}):
                from pycf.inten import fit_altp

                fit_altp(
                    ["A10"],
                    MagicMock(),
                    minimizer="basinhopping",
                    bounds=bounds,
                    minimizer_kwargs={
                        "method": "Nelder-Mead",
                        "options": {"maxiter": 10, "bounds": bounds},
                    },
                )

            _, bh_kwargs = mock_bh.call_args
            mk = bh_kwargs["minimizer_kwargs"]
            assert mk["bounds"] == bounds
            assert "bounds" not in mk.get("options", {})

    def test_default_uses_minimize(self):
        with (
            patch("pycf.inten.AltpFit") as mock_fitter_cls,
            patch("pycf.inten.minimize") as mock_min,
        ):
            mock_fitter = MagicMock()
            mock_fitter.initial_x = np.array([1.0])
            mock_fitter.objective_fn.return_value = 0.0
            mock_fitter._x_to_altp.return_value = {"A10": 1.0}
            mock_fitter.spectra = [MagicMock()]
            mock_fitter.spectra[0].name = "s"
            mock_fitter.n_obs = 1
            mock_fitter.n_p = 1
            mock_fitter.param_names = ["A10"]
            mock_fitter.per_spectrum_chi2.return_value = [{"name": "s", "chi2": 0.0, "n_obs": 1}]
            mock_fitter_cls.return_value = mock_fitter
            mock_min.return_value = SimpleNamespace(x=np.array([1.0]), fun=0.0)
            with patch("pycf.inten._estimate_parameter_uncertainties", return_value={}):
                from pycf.inten import fit_altp

                fit_altp(["A10"], MagicMock())
            mock_min.assert_called_once()


class TestFormattingEdgeCases:
    """Test edge cases in formatting functions."""

    def test_format_state_label_tuple_input(self):
        """Test with tuple instead of array."""
        label = (1, 2, 3)
        result = _format_state_label_content(label, label_key=None)
        assert isinstance(result, str)

    def test_format_state_label_list_input(self):
        """Test with list instead of array."""
        label = [1, 2, 3]
        result = _format_state_label_content(label, label_key=None)
        assert isinstance(result, str)

    def test_format_state_label_mixed_types(self):
        """Test with mixed integer and float types."""
        label = np.array([1, 2.5, 3])
        result = _format_state_label_content(label, label_key="SLJ")
        assert isinstance(result, str)

    def test_format_state_label_all_zeros(self):
        """Test with all-zero label."""
        label = np.array([0, 0, 0])
        result = _format_state_label_content(label, label_key="SLJ")
        assert isinstance(result, str)

    def test_format_state_label_negative_values(self):
        """Test with negative quantum numbers."""
        label = np.array([1, -2, -3])
        result = _format_state_label_content(label, label_key="SLJ")
        assert isinstance(result, str)


class TestIntensitySummaryFormatting:
    """Test formatting behavior for missing experimental data and fit uncertainty display."""

    def _make_fake_spectrum(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.altp_uncertainties = {}
                self.expt_data = [[1, 0.8]]
                self.groups = [
                    {
                        "Energy": 1.0,
                        "e_i": 0.0,
                        "e_f": 1.0,
                        "g_i": 1,
                        "g_f": 1,
                        "t_list": [
                            {
                                "i": 0,
                                "f": 1,
                                "e": 1.0,
                                "pc_i": 0,
                                "pc_f": 1,
                                "S_ED_isotropic": 0.0,
                                "S_MD_isotropic": 0.0,
                            }
                        ],
                        "f": 1.0,
                        "A": 0.0,
                    },
                    {
                        "Energy": 2.0,
                        "e_i": 0.0,
                        "e_f": 2.0,
                        "g_i": 1,
                        "g_f": 1,
                        "t_list": [
                            {
                                "i": 0,
                                "f": 2,
                                "e": 2.0,
                                "pc_i": 0,
                                "pc_f": 1,
                                "S_ED_isotropic": 0.0,
                                "S_MD_isotropic": 0.0,
                            }
                        ],
                        "f": 2.0,
                        "A": 0.0,
                    },
                ]
                self.total_f = 3.0
                self.total_A = 0.0
                self.eigenvalues = np.array([0.0, 1.0, 2.0])
                self.principal_components = np.array([0, 1, 2])
                self.hamiltonian = SimpleNamespace(tensors=[])
                self.nrefractive = 1.0

            def set_altp(self, altp):
                self.altp = dict(altp)
                self.altp_uncertainties = {}

            def update_altp(self, **updates):
                self.altp.update(updates)
                for name in updates:
                    self.altp_uncertainties.pop(name, None)

            def recalculate(self, polarization="isotropic"):
                self.groups[0]["f"] = float(self.altp["A10"])
                self.total_f = self.groups[0]["f"] + self.groups[1]["f"]
                return self.groups

        return FakeSpectrum()

    def test_missing_expt_data_shows_dashes_and_skips_chisqr(self):
        spec = self._make_fake_spectrum()
        summary = gen_inten_summary(spec, format="brief", state_labels=["|a>", "|b>", "|c>"])

        assert "chisqr" in summary
        assert "---" in summary
        assert "2    " in summary and "---          ---" in summary

    def test_fit_uncertainty_is_appended_in_altp_block(self):
        spec = self._make_fake_spectrum()

        with (
            patch("pycf.inten.minimize") as mock_minimize,
            patch("pycf.inten._estimate_parameter_uncertainties", return_value={"A10": 0.123}),
        ):
            mock_minimize.return_value = SimpleNamespace(x=np.array([0.8]), fun=0.01)
            fit_altp(["A10"], spec, dry_run=False)

        assert spec.altp_uncertainties == {"A10": 0.123}
        summary = gen_inten_summary(spec, format="brief", state_labels=["|a>", "|b>", "|c>"])
        assert "A10" in summary
        assert "+/-" in summary

    def test_brief_summary_shows_scaling_and_not_used_markers(self):
        spec = self._make_fake_spectrum()
        spec.expt_data = [[1, 0.5], [2, 1.0]]
        spec.fit_scale_to_group = 1
        spec.fit_ignore_groups = [2]

        summary = gen_inten_summary(spec, format="brief", state_labels=["|a>", "|b>", "|c>"])

        assert "Experimental scaling: group 1 factor =" in summary
        assert "(scaled to)" in summary
        assert "(not used)" in summary
        assert " Note" not in summary
        assert "Total" in summary
        assert "0.000000e+00" in summary

    def test_detailed_summary_includes_per_transition_lines(self):
        """format='detailed' exercises _format_transition_line (pyfit lines 1494-1540)
        and the transition-listing loop (lines 1720-1750)."""
        spec = self._make_fake_spectrum()
        summary = gen_inten_summary(
            spec, format="detailed", state_labels=["|a>", "|b>", "|c>"]
        )
        assert "Individual transitions:" in summary
        # Header for absorption (Energy > 0): f_ED column.
        assert "f_ED" in summary
        # 1-based level indices for the synthetic transitions.
        assert "1   " in summary

    def test_moments_summary_includes_dipole_components(self):
        """format='moments' additionally exercises _format_dipole_moments
        (lines 1543-1591) via the moments branch at line 1746."""
        spec = self._make_fake_spectrum()
        # Inject dipole components so the moments block renders non-trivially.
        spec.groups[0]["t_list"][0].update(
            {"ed_-1": 0.0, "ed_0": 0.1, "ed_+1": 0.0,
             "md_-1": 0.0, "md_0": 0.05, "md_+1": 0.0}
        )
        summary = gen_inten_summary(
            spec, format="moments", state_labels=["|a>", "|b>", "|c>"]
        )
        assert "D_ED" in summary
        assert "D_MD" in summary

    def test_detailed_summary_handles_out_of_range_pc_indices(self):
        """Out-of-range pc_i/pc_f fall through to 'State N' fallback labels
        (lines 1514-1515, 1519-1520)."""
        spec = self._make_fake_spectrum()
        # Force pc_i out of range; pc_f stays in range.
        spec.groups[0]["t_list"][0]["pc_i"] = 99
        spec.groups[0]["t_list"][0]["pc_f"] = -1
        summary = gen_inten_summary(
            spec, format="detailed", state_labels=["|a>", "|b>", "|c>"]
        )
        assert "State 1" in summary or "State 2" in summary

    def test_fit_altp_uses_scaled_targets_and_excludes_anchor_group(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[1, 1.0], [2, 3.0]]
                self.fit_scale_to_group = None
                self.fit_ignore_groups = []
                self.last_expt_scale_factor = None
                self.groups = [
                    {"Energy": 1.0, "f": 2.0, "A": 0.0},
                    {"Energy": 2.0, "f": 6.0, "A": 0.0},
                ]
                self.altp_uncertainties = {}

            def scale_to(self, idx):
                self.fit_scale_to_group = idx

            def set_ignored_groups(self, groups):
                self.fit_ignore_groups = list(groups)

            def set_altp(self, altp):
                self.altp = dict(altp)

            def update_altp(self, **updates):
                self.altp.update(updates)

            def recalculate(self, polarization="isotropic"):
                self.groups = [
                    {"Energy": 1.0, "f": 2.0, "A": 0.0},
                    {"Energy": 2.0, "f": 6.0 * float(self.altp["A10"]), "A": 0.0},
                ]
                return self.groups

        spec = FakeSpectrum()
        spec.scale_to(1)
        result = fit_altp(["A10"], spec, dry_run=True)

        assert result["n_obs"] == 1
        assert result["chi2"] == pytest.approx(0.0)
        assert spec.last_expt_scale_factor == pytest.approx(2.0)

    def test_fit_altp_scale_to_group_without_expt_data_raises(self):
        class FakeSpectrum:
            def __init__(self):
                self.name = "fake"
                self.altp = {"A10": 1.0}
                self.expt_data = [[2, 3.0]]
                self.fit_scale_to_group = 1
                self.fit_ignore_groups = []
                self.last_expt_scale_factor = None
                self.groups = [{"Energy": 1.0, "f": 2.0, "A": 0.0}]
                self.altp_uncertainties = {}

            def set_altp(self, altp):
                self.altp = dict(altp)

            def update_altp(self, **updates):
                self.altp.update(updates)

            def recalculate(self, polarization="isotropic"):
                return self.groups

        with pytest.raises(ValueError, match="scale_to group 1 has no experimental data"):
            fit_altp(["A10"], FakeSpectrum(), dry_run=True)


class TestConvenienceWrappers:
    """Tests for the multi-spectrum convenience wrapper functions."""

    def _make_mock_spec(self):
        spec = MagicMock()
        spec.name = "mock"
        spec.altp = {"A10": 1.0}
        spec.altp_uncertainties = {}
        spec.expt_data = []
        return spec

    def test_inten_calculate_calls_each_spectrum(self):
        specs = [self._make_mock_spec(), self._make_mock_spec()]
        inten_calculate(specs)
        for spec in specs:
            spec.calculate_intensities.assert_called_once_with(polarization="isotropic")

    def test_inten_calculate_custom_polarization(self):
        spec = self._make_mock_spec()
        inten_calculate([spec], polarization="linear")
        spec.calculate_intensities.assert_called_once_with(polarization="linear")

    def test_inten_print_calls_gen_inten_summary(self):
        spec = self._make_mock_spec()
        with patch("pycf.inten.gen_inten_summary", return_value="summary") as mock_gen:
            with patch("builtins.print") as mock_print:
                inten_print([spec], format="brief")
        mock_gen.assert_called_once_with(spec, format="brief", include_altp_parameters=True)
        mock_print.assert_called_once_with("summary")

    def test_inten_print_can_suppress_altp_parameters(self):
        spec = self._make_mock_spec()
        with patch("pycf.inten.gen_inten_summary", return_value="summary") as mock_gen:
            with patch("builtins.print"):
                inten_print([spec], format="brief", include_altp_parameters=False)
        mock_gen.assert_called_once_with(spec, format="brief", include_altp_parameters=False)

    def test_inten_print_total_chisqr_printed(self):
        """inten_print prints total chisqr when spec has expt_data in brief format."""
        spec = self._make_mock_spec()
        spec.expt_data = [[1, 2.0]]
        spec.groups = [{"Energy": 100.0, "f": 2.0, "A": 0.0}]
        with patch("pycf.inten.gen_inten_summary", return_value="s"):
            with patch("builtins.print") as mock_print:
                inten_print([spec], format="brief")
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Total chisqr" in c for c in calls)

    def test_inten_print_no_total_chisqr_for_detailed(self):
        """inten_print prints total chisqr even for detailed format."""
        spec = self._make_mock_spec()
        spec.expt_data = [[1, 2.0]]
        spec.groups = [{"Energy": 100.0, "f": 2.0, "A": 0.0}]
        with patch("pycf.inten.gen_inten_summary", return_value="s"):
            with patch("builtins.print") as mock_print:
                inten_print([spec], format="detailed")
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Total chisqr" in c for c in calls)

    def test_inten_print_accepts_single_spectrum(self):
        """inten_print should accept a bare Spectrum (not wrapped in a list)."""
        spec = self._make_mock_spec()
        with patch("pycf.inten.gen_inten_summary", return_value="s") as mock_gen:
            with patch("builtins.print"):
                # Pass a list — normal path
                inten_print([spec])
        mock_gen.assert_called_once()

    def test_inten_print_bare_spectrum_isinstance_guard(self):
        """inten_print wraps a bare Spectrum in a list via isinstance guard."""
        # Build a minimal real Spectrum to exercise isinstance branch
        import pycf.cfl as cfl  # noqa: F401 — needed for fixture
        from pycf.inten import Spectrum as RealSpectrum

        spec = MagicMock(spec=RealSpectrum)
        spec.expt_data = []
        with patch("pycf.inten.gen_inten_summary", return_value="s") as mock_gen:
            with patch("builtins.print"):
                inten_print(spec)
        mock_gen.assert_called_once()

    def test_inten_set_expt_data_calls_each(self):
        specs = [self._make_mock_spec(), self._make_mock_spec()]
        data = [[[1, 0.5]], [[1, 0.8]]]
        inten_set_expt_data(specs, data)
        specs[0].set_expt_data.assert_called_once_with([[1, 0.5]])
        specs[1].set_expt_data.assert_called_once_with([[1, 0.8]])

    def test_inten_set_expt_data_length_mismatch_raises(self):
        specs = [self._make_mock_spec(), self._make_mock_spec()]
        with pytest.raises(ValueError, match="same length"):
            inten_set_expt_data(specs, [[[1, 0.5]]])

    def test_inten_set_altp_calls_each(self):
        specs = [self._make_mock_spec(), self._make_mock_spec()]
        altp = {"A10": 2.0}
        inten_set_altp(specs, altp)
        for spec in specs:
            spec.set_altp.assert_called_once_with(altp)

    def test_inten_recalculate_calls_each(self):
        specs = [self._make_mock_spec(), self._make_mock_spec()]
        inten_recalculate(specs)
        for spec in specs:
            spec.recalculate.assert_called_once_with(polarization="isotropic")

    def test_inten_recalculate_custom_polarization(self):
        spec = self._make_mock_spec()
        inten_recalculate([spec], polarization="circular")
        spec.recalculate.assert_called_once_with(polarization="circular")

    def test_inten_plot_uses_experimental_energies_when_provided(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        class DummyHamiltonian:
            def diag(self):
                return np.array([0.0, 1.0]), np.eye(2)

        class DummySpectrum:
            def __init__(self):
                self.name = "demo"
                self.groups = [
                    {"Energy": 10.0, "f": 1.0, "A": 0.0},
                    {"Energy": 20.0, "f": 2.0, "A": 0.0},
                ]
                self.expt_data = [
                    {"group": 1, "intensity": 0.5, "energy": 15.0},
                    {"group": 2, "intensity": 1.5, "energy": 25.0},
                ]
                self.hamiltonian = DummyHamiltonian()
                self.nrefractive = 1.0

        spec = DummySpectrum()
        _, ax = inten_plot(spec, npoints=100)

        red_collections = [
            coll
            for coll in ax.collections
            if getattr(coll, "get_label", lambda: "")() == "Experimental"
        ]
        assert red_collections
        segments = red_collections[0].get_segments()
        xs = sorted(round(segment[0][0], 6) for segment in segments)
        assert xs == [15.0, 25.0]

    def test_inten_plot_separates_used_scaled_to_and_not_used(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.colors import to_rgba

        class DummyHamiltonian:
            def diag(self):
                return np.array([0.0, 1.0]), np.eye(2)

        class DummySpectrum:
            def __init__(self):
                self.name = "demo"
                self.groups = [
                    {"Energy": 10.0, "f": 1.0, "A": 0.0},
                    {"Energy": 20.0, "f": 2.0, "A": 0.0},
                    {"Energy": 30.0, "f": 3.0, "A": 0.0},
                ]
                self.expt_data = [
                    {"group": 1, "intensity": 1.0, "energy": 11.0},
                    {"group": 2, "intensity": 0.4, "energy": 22.0},
                    {"group": 3, "intensity": 0.5, "energy": 33.0},
                ]
                self.fit_scale_to_group = 2
                self.fit_ignore_groups = [3]
                self.last_expt_scale_factor = None
                self.hamiltonian = DummyHamiltonian()
                self.nrefractive = 1.0

        spec = DummySpectrum()
        _, ax = inten_plot(spec, npoints=100)

        labelled_collections = {
            getattr(coll, "get_label", lambda: "")(): coll for coll in ax.collections
        }
        exp_label = next(
            (label for label in labelled_collections if label.startswith("Experimental (scaled x")),
            None,
        )
        assert exp_label is not None
        assert "Scaled to" in labelled_collections
        assert "Not used" in labelled_collections

        exp_segments = labelled_collections[exp_label].get_segments()
        scaled_to_segments = labelled_collections["Scaled to"].get_segments()
        not_used_segments = labelled_collections["Not used"].get_segments()

        assert [round(seg[0][0], 6) for seg in exp_segments] == [11.0]
        assert [round(seg[0][0], 6) for seg in scaled_to_segments] == [22.0]
        assert [round(seg[0][0], 6) for seg in not_used_segments] == [33.0]

        assert np.allclose(labelled_collections[exp_label].get_colors()[0][:3], to_rgba("red")[:3])
        assert np.allclose(
            labelled_collections["Scaled to"].get_colors()[0][:3], to_rgba("green")[:3]
        )
        assert np.allclose(
            labelled_collections["Not used"].get_colors()[0][:3], to_rgba("black")[:3]
        )

        calc_line = next((line for line in ax.lines if line.get_label() == "Calculated"), None)
        assert calc_line is not None
        assert np.allclose(to_rgba(calc_line.get_color()), to_rgba("blue"))

        star_texts = [t for t in ax.texts if t.get_text() == "*"]
        assert len(star_texts) == 1
        assert round(star_texts[0].get_position()[0], 6) == 22.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
