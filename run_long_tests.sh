#!/bin/bash
# Run the long tests, which are marked with the "long_running" pytest marker.
# This script is intended to be run from the root of the repository.
# Usage: ./run_long_tests.sh

log_file="tests/logs/long_tests_$(date +%Y%m%d_%H%M%S).log"

# Run pytest with full verbosity; tee to log file while filtering terminal
# output to: test name + result, section headers, and final summary line.
pytest -m long_running --runslow --runlong -vv -rA --tb=short --no-header --color=no \
    | tee "$log_file" \
    | grep --line-buffered -E "^(PASSED|FAILED|ERROR|tests/|=====|WARNINGS)"

exit_code=${PIPESTATUS[0]}

if [[ $exit_code -eq 0 ]]; then
    echo "All long tests passed. Full log: $log_file"
else
    echo ""
    echo "Some tests failed. Tracebacks from $log_file:"
    echo ""
    grep -A 40 "^_ \|^FAILED\|^E " "$log_file" | head -200
    echo ""
    echo "Full log: $log_file"
fi

exit $exit_code
