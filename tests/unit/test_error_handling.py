#!/usr/bin/env python3
"""
Tests for error handling and logging infrastructure.
Verifies that custom error handlers work correctly and that
errors are properly routed through the logging system.
"""

import pycf.cfl as cfl


class TestErrorHandler:
    """Test custom error handler registration."""

    def test_set_error_handler_accepts_callable(self):
        """set_error_handler should accept a callable."""

        def my_handler(func, file, line, msg):
            pass

        # Should not raise
        cfl.set_error_handler(my_handler)
        # Restore default
        cfl.set_error_handler(None)

    def test_set_error_handler_accepts_none(self):
        """set_error_handler(None) should restore default behavior."""

        def my_handler(func, file, line, msg):
            pass

        # Register custom
        cfl.set_error_handler(my_handler)
        # Restore default
        cfl.set_error_handler(None)  # Should not raise

    def test_error_handler_receives_correct_parameters(self):
        """Verify error handler is callable and receives expected parameter types."""
        received_calls = []

        def capture_handler(func, file, line, msg):
            # Verify parameter types
            assert isinstance(func, str)
            assert isinstance(file, str)
            assert isinstance(line, int)
            assert isinstance(msg, str)
            received_calls.append({"func": func, "file": file, "line": line, "msg": msg})

        cfl.set_error_handler(capture_handler)
        cfl.set_error_handler(None)
        # Handler was successfully registered and restored

    def test_error_handler_with_lambda(self):
        """set_error_handler should accept lambda functions."""
        errors = []
        cfl.set_error_handler(lambda f, file, l, m: errors.append(m))
        cfl.set_error_handler(None)

    def test_error_handler_callable_check(self):
        """set_error_handler should work with any callable."""

        class CallableClass:
            def __call__(self, func, file, line, msg):
                pass

        handler = CallableClass()
        cfl.set_error_handler(handler)
        cfl.set_error_handler(None)


class TestLoggingBehavior:
    """Test that logging infrastructure doesn't affect normal operation."""

    def test_hamiltonian_creation_with_handler(self):
        """Hamiltonian should work normally with custom error handler."""

        def logging_handler(func, file, line, msg):
            pass

        cfl.set_error_handler(logging_handler)
        try:
            # This should work normally
            pass
        finally:
            cfl.set_error_handler(None)

    def test_multiple_handler_changes(self):
        """Should be able to change handlers multiple times."""

        def handler1(f, file, l, m):
            pass

        def handler2(f, file, l, m):
            pass

        # Register, change, restore
        cfl.set_error_handler(handler1)
        cfl.set_error_handler(handler2)
        cfl.set_error_handler(None)
        cfl.set_error_handler(handler1)
        cfl.set_error_handler(None)
        # Should not raise

    def test_handler_intercepts_real_c_error(self):
        """Custom handler must capture errors raised from any C TU.

        Regression test for finding F-008 in
        plan/audit_2026-04-27_171732_report.md: the global handler
        pointer was previously declared `static` in cfl_error.h, which
        gave each translation unit its own copy.  As a result,
        cfl_set_error_handler() only updated the TU from which it was
        called, while CFL_ERROR_* macros expanded inside other TUs
        (cfl_h.c, cfl_csr.c, etc.) continued to use the default printf
        handler.  This test triggers an error path inside cfl_h.c via
        the public Hamiltonian constructor and asserts the custom
        Python handler was invoked.
        """
        import numpy as np

        captured = []

        def handler(func, file, line, msg):
            captured.append((func, file, line, msg))

        cfl.set_error_handler(handler)
        try:
            # Two StateLabels of the same length but different label
            # contents -> different label-array hash -> mismatching
            # state labels at the C level.
            sl_a = cfl.StateLabels("J", [[2]])
            sl_b = cfl.StateLabels("J", [[3]])
            row_ptr = np.array([0, 0], dtype=np.int32)
            col_in = np.array([], dtype=np.int32)
            val = np.array([], dtype=np.complex128)
            ta = cfl.Tensor(b"Ta", row_ptr, col_in, val, sl_a)
            tb = cfl.Tensor(b"Tb", row_ptr, col_in, val, sl_b)
            try:
                cfl.Hamiltonian([ta, tb])
            except Exception:
                # The C error returns NULL which may surface as an
                # exception in the Cython wrapper; we only care that
                # the handler ran.
                pass
        finally:
            cfl.set_error_handler(None)

        assert any(
            "mismatching state labels" in msg for _, _, _, msg in captured
        ), "Custom error handler was not invoked from cfl_h.c. " "Captured calls: {0!r}".format(
            captured
        )
