# conftest.py
import pytest

"""
This file defines pytest fixtures and hooks for the test suite. 
In particular, it implements a command-line option "--runslow" to allow users to include or exclude slow tests. 
To mark a test as slow, simply decorate it with @pytest.mark.slow.
Example usage:
    python -m pytest            # to skip slow tests 
    python -m pytest --runslow  # to run slow tests  
"""


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
