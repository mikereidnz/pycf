"""Tests for pyemp optional dependency behavior."""

import builtins
import importlib
import sys

import pytest


def test_pyemp_imports_without_matplotlib(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop("pycf.pyemp", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        module = importlib.import_module("pycf.pyemp")

        assert module.plt is None
        with pytest.raises(ImportError, match="requires matplotlib"):
            module.SpectrumAxes()
    finally:
        sys.modules.pop("pycf.pyemp", None)
