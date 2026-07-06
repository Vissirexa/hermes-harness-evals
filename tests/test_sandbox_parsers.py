"""Tests for sandbox.py parser functions and edge cases."""

import subprocess
import sys
import time

import pytest

from local_bench.sandbox import (
    ExecutionResult,
    _parse_pytest_output,
    _parse_vitest_output,
    run_tests,
)


# --------------------------------------------------------------------------- #
# pytest output parsing
# --------------------------------------------------------------------------- #


class TestParsePytestOutput:
    def test_standard_summary_all_pass(self):
        output = "================ 5 passed in 0.12s ================"
        p, f, e, t = _parse_pytest_output(output, 0)
        assert p == 5 and f == 0 and e == 0 and t == 5

    def test_standard_summary_mixed(self):
        output = "==== 3 passed, 1 failed, 1 error in 1.23s ===="
        p, f, e, t = _parse_pytest_output(output, 1)
        assert p == 3 and f == 1 and e == 1 and t == 5

    def test_verbose_output_fallback(self):
        output = (
            "test_foo.py::test_one PASSED\n"
            "test_foo.py::test_two FAILED\n"
            "test_foo.py::test_three ERROR [1 import error]\n"
        )
        p, f, e, t = _parse_pytest_output(output, 1)
        assert p == 1 and f == 1 and e == 1 and t == 3

    def test_malformed_falls_back_to_verbose(self):
        output = "some random garbled output with no summary line"
        p, f, e, t = _parse_pytest_output(output, 0)
        assert t == 0

    def test_nonzero_rc_no_counts_gives_error(self):
        output = ""
        p, f, e, t = _parse_pytest_output(output, 1)
        assert e == 1 and t == 1

    def test_zero_rc_no_counts_no_error(self):
        output = ""
        p, f, e, t = _parse_pytest_output(output, 0)
        assert e == 0 and t == 0


# --------------------------------------------------------------------------- #
# vitest output parsing
# --------------------------------------------------------------------------- #


class TestParseVitestOutput:
    # Fixtures mirror real vitest summaries, which print BOTH a per-file and a
    # per-test line. The counts deliberately differ between the two lines so
    # these tests also prove the parser reads `Tests` and never `Test Files`.
    def test_standard_summary(self):
        output = (
            " Test Files  1 failed | 1 passed (2)\n"
            "      Tests  1 failed | 6 passed (7)\n"
            "   Duration  1.23s\n"
        )
        p, f, e, t = _parse_vitest_output(output, 1)
        assert p == 6 and f == 1 and e == 0 and t == 7

    def test_no_failed(self):
        output = (
            " Test Files  1 passed (1)\n"
            "      Tests  3 passed (3)\n"
        )
        p, f, e, t = _parse_vitest_output(output, 0)
        assert p == 3 and f == 0 and e == 0 and t == 3

    def test_compile_error_falls_back(self):
        output = "ERROR  Failed to compile"
        p, f, e, t = _parse_vitest_output(output, 1)
        assert e == 1 and t == 1

    def test_zero_rc_no_counts_no_error(self):
        output = ""
        p, f, e, t = _parse_vitest_output(output, 0)
        assert e == 0 and t == 0


# --------------------------------------------------------------------------- #
# ExecutionResult helpers
# --------------------------------------------------------------------------- #


class TestExecutionResult:
    def test_all_passed_true(self):
        r = ExecutionResult(3, 0, 0, 3, "ok", 0)
        assert r.all_passed is True

    def test_all_passed_false_on_fail(self):
        r = ExecutionResult(2, 1, 0, 3, "fail", 1)
        assert r.all_passed is False

    def test_all_passed_false_on_error(self):
        r = ExecutionResult(2, 0, 1, 3, "error", 1)
        assert r.all_passed is False

    def test_all_passed_false_on_zero_total(self):
        r = ExecutionResult(0, 0, 0, 0, "empty", 0)
        assert r.all_passed is False

    def test_pass_rate(self):
        r = ExecutionResult(3, 1, 0, 4, "ok", 0)
        assert r.pass_rate == 0.75

    def test_pass_rate_zero_total(self):
        r = ExecutionResult(0, 0, 0, 0, "empty", 0)
        assert r.pass_rate == 0.0


# --------------------------------------------------------------------------- #
# Timeout path (integration-ish, no model needed)
# --------------------------------------------------------------------------- #


class TestTimeoutPath:
    def test_pytest_timeout(self):
        """A sleep-based test should hit the timeout and return an error."""
        result = run_tests(
            code="",
            test_code="import time\ndef test_sleep():\n    time.sleep(60)",
            timeout=1,
            language="python",
        )
        assert result.errors == 1
        assert "TIMEOUT" in result.output
