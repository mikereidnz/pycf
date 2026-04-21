echo "Executing cfl tests:"
echo "--------------------"

failed=0
for f in *_test
do
    output=$("./$f")
    status=$?
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
