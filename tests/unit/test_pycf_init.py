"""Tests for top-level :mod:`pycf` package metadata and lazy attributes."""

import io
from datetime import datetime

import pycf


def test_version_attributes_present():
    """Build metadata exported from pycf is non-empty."""
    assert isinstance(pycf.__version__, str) and pycf.__version__
    assert isinstance(pycf.__build_timestamp__, str) and pycf.__build_timestamp__
    assert isinstance(pycf.__build_comment__, str)


def test_pycf_info_returns_string_and_writes_to_stream():
    stream = io.StringIO()
    out = pycf.pycf_info(stream=stream)
    assert isinstance(out, str)
    assert "pycf details" in out
    assert "pycf revision:" in out
    # The same content must have been written to the stream.
    assert "pycf details" in stream.getvalue()


def test_pycf_info_accepts_datetime():
    stamp = datetime(2024, 1, 2, 3, 4, 5)
    out = pycf.pycf_info(current_time=stamp, stream=io.StringIO())
    assert "2024-01-02 03:04:05" in out


def test_pycf_info_accepts_string_time():
    out = pycf.pycf_info(current_time="custom-time", stream=io.StringIO())
    assert "custom-time" in out


def test_lazy_cfl_attribute():
    """Accessing pycf.cfl triggers lazy import of the extension module."""
    cfl_mod = pycf.cfl
    assert hasattr(cfl_mod, "Hamiltonian")
    assert hasattr(cfl_mod, "set_error_handler")


def test_lazy_import_classes():
    """pycf.ImportSLJM and pycf.ImportTensors should resolve via __getattr__."""
    from pycf import import_sljm

    assert pycf.ImportSLJM is import_sljm.ImportSLJM
    assert pycf.ImportTensors is import_sljm.ImportTensors


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        _ = pycf.does_not_exist
