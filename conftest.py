# conftest.py
import pytest

"""
This file defines pytest fixtures and hooks for the test suite. 
In particular, it implements a command-line option "--runslow" to allow users to include or exclude slow tests. 
To mark a test as slow, simply decorate it with @pytest.mark.slow.
Example usage:
    python -m pytest            # to skip slow tests 
    python -m pytest --runslow  # to run slow tests  

It also handles optional test dependencies gracefully:
- If hypothesis is not installed, property-based tests are skipped
- If pytest-benchmark is not installed, benchmark tests are skipped
"""

import sys

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_configure(config):
    """Check for optional dependencies and warn if missing."""
    missing_deps = []
    
    try:
        import hypothesis
    except ImportError:
        missing_deps.append("hypothesis (property-based tests will be skipped)")
    
    try:
        import pytest_benchmark
    except ImportError:
        missing_deps.append("pytest-benchmark (performance benchmarks will be skipped)")
    
    if missing_deps:
        print(f"\n⚠️  Optional test dependencies not installed:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print(f"\nTo install: pip install -e '.[dev]'")
        print()


def pytest_collection_modifyitems(config, items):
    """Handle missing optional dependencies."""
    skip_hypothesis = pytest.mark.skip(reason="hypothesis not installed - run: pip install hypothesis")
    skip_benchmark = pytest.mark.skip(reason="pytest-benchmark not installed - run: pip install pytest-benchmark")
    
    has_hypothesis = 'hypothesis' in sys.modules
    has_benchmark = 'pytest_benchmark' in sys.modules
    
    for item in items:
        if not has_hypothesis and 'test_properties' in str(item.fspath):
            item.add_marker(skip_hypothesis)
        elif not has_benchmark and 'test_benchmarks' in str(item.fspath):
            item.add_marker(skip_benchmark)
    
    # Also handle slow tests
    if config.getoption("--runslow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
