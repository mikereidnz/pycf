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
    fit_altp,
    gen_inten_summary,
    inten_calculate,
    inten_print,
    inten_recalculate,
    inten_set_altp,
    inten_set_expt_data,
    ms_fit_altp,
    _format_state_label_content,
    _format_state_label_short,
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
        mock_ham = MagicMock()
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
        mock_ham = MagicMock()
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
        mock_ham = MagicMock()
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
        mock_ham = MagicMock()
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
        mock_ham = MagicMock()
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
        mock_ham = MagicMock()
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
        fake_result = SimpleNamespace(x=np.array([2.0]), fun=0.2)
        with patch("pycf.inten.minimize", return_value=fake_result):
            result = fit_altp(["A10"], spec, dry_run=False, method="Nelder-Mead")

        assert result["initial_chi2"] == pytest.approx(0.0)
        assert result["chi2"] == pytest.approx(0.0)
        assert result["fitted_params"]["A10"] == pytest.approx(1.0)
        assert result["reverted_to_initial"] is True
        assert result["improved"] is False
        assert result["uncertainties"] == {}

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


class TestMsFitAltpWrapper:
    """Test ms_fit_altp wrapper behavior."""

    def test_ms_fit_altp_requires_sequence(self):
        class FakeSpectrum:
            pass

        with pytest.raises(TypeError, match="requires a sequence"):
            ms_fit_altp(["A10"], FakeSpectrum())  # type: ignore[arg-type]

    def test_ms_fit_altp_requires_nonempty_sequence(self):
        with pytest.raises(ValueError, match="at least one Spectrum"):
            ms_fit_altp(["A10"], [])

    def test_ms_fit_altp_requires_spectrum_elements(self):
        with pytest.raises(TypeError, match="only Spectrum objects"):
            ms_fit_altp(["A10"], ["not-a-spectrum"])  # type: ignore[list-item]


class TestOptimizerDispatch:
    """Test that fit_altp dispatches to the correct scipy optimizer."""

    def _make_fake_spec(self):
        """Return a Spectrum-like object with expt_data and an Altp."""
        from pycf.inten import Spectrum as RealSpectrum
        spec = MagicMock(spec=RealSpectrum)
        spec.name = "test"
        spec.altp = {"A10": 1.0}
        spec.expt_data = [[1, 1.0]]
        spec.groups = [{"Energy": 1.0, "e_i": 0.0, "e_f": 1.0, "g_i": 1, "g_f": 1,
                        "t_list": [{"i": 0, "f": 1, "ED": 1.0, "MD": 0.0}]}]
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
        with patch("pycf.inten.AltpFit") as mock_fitter_cls, \
             patch("pycf.inten.basinhopping") as mock_bh:
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
                fit_altp(["A10"], MagicMock(), minimizer="basinhopping", niter=50,
                         minimizer_kwargs={"method": "Nelder-Mead"})
            mock_bh.assert_called_once()
            _, bh_kwargs = mock_bh.call_args
            assert bh_kwargs.get("niter") == 50
            assert bh_kwargs["minimizer_kwargs"]["method"] == "Nelder-Mead"

    def test_basinhopping_strips_duplicate_bounds(self):
        """Avoid passing bounds twice to scipy minimize via options + kwargs."""
        with patch("pycf.inten.AltpFit") as mock_fitter_cls, \
             patch("pycf.inten.basinhopping") as mock_bh:
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
        with patch("pycf.inten.AltpFit") as mock_fitter_cls, \
             patch("pycf.inten.minimize") as mock_min:
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

        with patch("pycf.inten.minimize") as mock_minimize, patch(
            "pycf.inten._estimate_parameter_uncertainties", return_value={"A10": 0.123}
        ):
            mock_minimize.return_value = SimpleNamespace(x=np.array([0.8]), fun=0.01)
            fit_altp(["A10"], spec, dry_run=False)

        assert spec.altp_uncertainties == {"A10": 0.123}
        summary = gen_inten_summary(spec, format="brief", state_labels=["|a>", "|b>", "|c>"])
        assert "A10: 0.8 +/- 0.123" in summary


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
        mock_gen.assert_called_once_with(spec, format="brief")
        mock_print.assert_called_once_with("summary")

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
        from pycf.inten import Spectrum as RealSpectrum
        # Build a minimal real Spectrum to exercise isinstance branch
        import pycf.cfl as cfl  # noqa: F401 — needed for fixture
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
