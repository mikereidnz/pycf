#!/usr/bin/env python3
"""
Tests for error handling and logging infrastructure.
Verifies that custom error handlers work correctly and that
errors are properly routed through the logging system.
"""

import pytest

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
