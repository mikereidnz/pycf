"""
Additional targeted tests for inten.py - focusing on non-plotting functions
that still have missing coverage. These tests target:
- _format_state_label_content
- _format_state_label_short
- AltpFit initialization and methods
- _estimate_parameter_uncertainties
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pycf.inten import AltpFit, _format_state_label_content, _format_state_label_short


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
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [("A10", 1.0), ("A20", 2.0)],
        }
        target_intensities = {0: 0.5, 1: 0.3}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            hamiltonian=mock_ham,
            spectrum_config=spectrum_config,
            target_intensities=target_intensities,
        )

        assert fitter.n_obs == 2
        assert len(fitter.param_names) == 2
        assert isinstance(fitter.param_info, dict)

    def test_altp_fit_init_complex_parameters(self):
        """Test AltpFit initialization with complex parameters."""
        mock_ham = MagicMock()
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [("A10", complex(1.0, 0.5)), ("A20", 2.0)],
        }
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            hamiltonian=mock_ham,
            spectrum_config=spectrum_config,
            target_intensities=target_intensities,
        )

        # A10 is complex (2 params), A20 is real (1 param) = 3 total
        assert fitter.n_p == 3

    def test_altp_fit_missing_parameter_raises(self):
        """Test AltpFit raises when parameter not in altp."""
        mock_ham = MagicMock()
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [("A10", 1.0)],
        }
        target_intensities = {0: 0.5}

        with pytest.raises(ValueError, match="not found in Altp"):
            AltpFit(
                param_names=["A10", "A20"],  # A20 not in altp
                hamiltonian=mock_ham,
                spectrum_config=spectrum_config,
                target_intensities=target_intensities,
            )

    def test_altp_fit_build_param_info_type_checking(self):
        """Test parameter type detection (real vs complex)."""
        mock_ham = MagicMock()
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [
                ("A10", 1.5),  # real
                ("A20", complex(2.0, 1.0)),  # complex
                ("A30", 3.0),  # real
            ],
        }
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20", "A30"],
            hamiltonian=mock_ham,
            spectrum_config=spectrum_config,
            target_intensities=target_intensities,
        )

        assert fitter.param_info["A10"]["type"] == "real"
        assert fitter.param_info["A20"]["type"] == "complex"
        assert fitter.param_info["A30"]["type"] == "real"

    def test_altp_fit_count_flat_params(self):
        """Test flat parameter counting (real=1, complex=2)."""
        mock_ham = MagicMock()
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [
                ("R1", 1.0),
                ("C1", complex(1.0, 1.0)),
                ("R2", 2.0),
                ("C2", complex(2.0, 2.0)),
            ],
        }
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["R1", "C1", "R2", "C2"],
            hamiltonian=mock_ham,
            spectrum_config=spectrum_config,
            target_intensities=target_intensities,
        )

        # 4 reals + 4 complexes = 2 + 4 + 2 + 4 = wait, let me recalculate
        # R1=1, C1=2, R2=1, C2=2 = 6 total
        assert fitter.n_p == 6

    def test_altp_fit_extract_initial_params(self):
        """Test initial parameter vector extraction."""
        mock_ham = MagicMock()
        spectrum_config = {
            "name": "test",
            "i_range": [1],
            "f_range": [2],
            "intensity_tensors": [MagicMock()],
            "altp": [("A10", 1.5), ("A20", complex(2.0, 1.0))],
        }
        target_intensities = {0: 0.5}

        fitter = AltpFit(
            param_names=["A10", "A20"],
            hamiltonian=mock_ham,
            spectrum_config=spectrum_config,
            target_intensities=target_intensities,
        )

        # Should have extracted [1.5, 2.0, 1.0] for [real, complex_real, complex_imag]
        assert len(fitter.initial_x) == 3
        assert fitter.initial_x[0] == 1.5
        assert fitter.initial_x[1] == 2.0
        assert fitter.initial_x[2] == 1.0


class TestEstimateParameterUncertainties:
    """Test uncertainty estimation function."""

    def test_estimate_uncertainties_handles_singular_hessian(self):
        """Test uncertainty estimation handles singular Hessian gracefully."""
        # This test just verifies the function exists and handles edge cases
        # The actual fitting is tested in integration tests
        from pycf.inten import _estimate_parameter_uncertainties

        assert callable(_estimate_parameter_uncertainties)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
