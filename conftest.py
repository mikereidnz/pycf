"""Pytest collection controls for optional, slow, and benchmark tests.

Examples:
    python -m pytest
    python -m pytest --runslow
    python -m pytest --runslow --runlong
    python -m pytest tests/test_benchmarks.py --benchmark-only
"""

import importlib.util

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--runlong",
        action="store_true",
        default=False,
        help="run long-running tests; use with --runslow",
    )


def pytest_collection_modifyitems(config, items):
    """Handle optional dependencies, benchmark tests, and slow tests."""
    skip_hypothesis = pytest.mark.skip(
        reason="hypothesis not installed - run: pip install hypothesis"
    )
    skip_benchmark = pytest.mark.skip(
        reason="pytest-benchmark not installed - run: pip install pytest-benchmark"
    )
    skip_benchmark_default = pytest.mark.skip(
        reason="benchmark tests run only with --benchmark-only"
    )
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_long = pytest.mark.skip(reason="need --runlong option to run")

    has_hypothesis = importlib.util.find_spec("hypothesis") is not None
    has_benchmark = importlib.util.find_spec("pytest_benchmark") is not None
    benchmark_only = getattr(config.option, "benchmark_only", False)

    for item in items:
        item_path = str(item.fspath)
        if not has_hypothesis and "test_properties" in item_path:
            item.add_marker(skip_hypothesis)
        elif "test_benchmarks" in item_path:
            if not has_benchmark:
                item.add_marker(skip_benchmark)
            elif not benchmark_only:
                item.add_marker(skip_benchmark_default)

        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "long_running" in item.keywords and not config.getoption("--runlong"):
            item.add_marker(skip_long)
