echo "Executing cfl tests:"
echo "--------------------"

failed=0
for f in *_test
do
    # opt_test has a known hang issue in the energy level fitting routine.
    # Apply a 60-second timeout to prevent CI from hanging indefinitely.
    # This is tracked as a v0.1.1 issue to investigate.
    if [ "$f" = "opt_test" ]; then
        output=$(timeout 60 "./$f" 2>&1)
        status=$?
        # Convert timeout signal (124) to a non-zero status for failure detection
        if [ $status -eq 124 ]; then
            echo "TIMEOUT: opt_test exceeded 60 seconds"
            failed=1
            continue
        fi
    else
        output=$("./$f")
        status=$?
    fi
    echo "$output"
    # Detect both non-zero exit codes and any "fail" line in output, since
    # the test binaries currently exit 0 even when an assertion fails.
    if [ $status -ne 0 ] || echo "$output" | grep -q "fail"; then
        echo "FAILED: $f"
        failed=1
    fi
done

if [ $failed -ne 0 ]; then
    echo "One or more cfl tests FAILED"
    exit 1
fi

echo 
