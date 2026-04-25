#!/bin/bash
# Run the slow tests, which are marked with the "slow" pytest marker.
# This script is intended to be run from the root of the repository.
# Usage: ./run_slow_tests.sh

log_file="tests/logs/slow_tests_$(date +%Y%m%d_%H%M%S).log"

# Run pytest with full verbosity; tee to log file while filtering terminal
# output to: test name + result, section headers, and final summary line.
pytest -m slow --runslow -vv -rA --tb=short --no-header --color=no \
    | tee "$log_file" \
    | grep --line-buffered -E "^(PASSED|FAILED|ERROR|tests/|=====|WARNINGS)"

exit_code=${PIPESTATUS[0]}

if [[ $exit_code -eq 0 ]]; then
    echo "All slow tests passed. Full log: $log_file"
else
    echo ""
    echo "Some tests failed. Tracebacks from $log_file:"
    echo ""
    grep -A 40 "^_ \|^FAILED\|^E " "$log_file" | head -200
    echo ""
    echo "Full log: $log_file"
fi

exit $exit_code

