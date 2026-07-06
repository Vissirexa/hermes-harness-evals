import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    passed: int
    failed: int
    errors: int
    total: int
    output: str
    return_code: int

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0 and self.total > 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #

def run_tests(
    code: str,
    test_code: str,
    solution_file: str = "solution.py",
    context_files: dict[str, str] | None = None,
    timeout: int = 30,
    language: str = "python",
) -> ExecutionResult:
    """Run a candidate solution against its test suite, dispatched by language."""
    if language == "typescript":
        return _run_vitest(code, test_code, solution_file, context_files, timeout)
    return _run_pytest(code, test_code, solution_file, context_files, timeout)


# --------------------------------------------------------------------------- #
# Python / pytest
# --------------------------------------------------------------------------- #

def _parse_pytest_output(output: str, return_code: int) -> tuple[int, int, int, int]:
    passed = failed = errors = 0

    summary_match = re.search(r"=+ ([\w\d ,]+) in [\d.]+s =+", output)
    if summary_match:
        for m in re.finditer(r"(\d+) (passed|failed|error)", summary_match.group(1)):
            n, kind = int(m.group(1)), m.group(2)
            if kind == "passed":
                passed = n
            elif kind == "failed":
                failed = n
            elif kind == "error":
                errors = n
    else:
        passed = len(re.findall(r"\bPASSED\b", output))
        failed = len(re.findall(r"\bFAILED\b", output))
        errors = len(re.findall(r"\bERROR\b", output))

    if return_code != 0 and failed == 0 and errors == 0 and passed == 0:
        errors = 1

    total = passed + failed + errors
    return passed, failed, errors, total


def _run_pytest(
    code: str,
    test_code: str,
    solution_file: str,
    context_files: dict[str, str] | None,
    timeout: int,
) -> ExecutionResult:
    tmpdir = tempfile.mkdtemp()
    try:
        (Path(tmpdir) / solution_file).write_text(code)
        if context_files:
            for filename, content in context_files.items():
                (Path(tmpdir) / filename).write_text(content)
        (Path(tmpdir) / "test_solution.py").write_text(test_code)

        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", "test_solution.py",
                "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider",
            ],
            capture_output=True, text=True, cwd=tmpdir, timeout=timeout,
        )
        output = result.stdout + result.stderr
        passed, failed, errors, total = _parse_pytest_output(output, result.returncode)
        return ExecutionResult(passed, failed, errors, total, output, result.returncode)
    except subprocess.TimeoutExpired:
        return ExecutionResult(0, 0, 1, 1, "TIMEOUT: test execution exceeded time limit", 1)
    except Exception as e:
        return ExecutionResult(0, 0, 1, 1, f"ERROR: {e}", 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# TypeScript / Vitest
# --------------------------------------------------------------------------- #

def _ts_runner_dir() -> Path | None:
    """Locate the shared Vitest harness (with node_modules installed once).

    Order: $HERMES_EVALS_TS_RUNNER, then `<repo>/ts_runner`. Returns None if its
    node_modules is missing (so the caller can report a clean setup error rather
    than spawning npm per task).
    """
    env = os.environ.get("HERMES_EVALS_TS_RUNNER")
    candidates = [Path(env)] if env else []
    candidates.append(Path(__file__).resolve().parent.parent / "ts_runner")
    for c in candidates:
        if (c / "node_modules" / ".bin" / "vitest").exists():
            return c
    return None


def _parse_vitest_output(output: str, return_code: int) -> tuple[int, int, int, int]:
    """Parse the Vitest text summary line: `Tests  2 failed | 6 passed (8)`."""
    passed = failed = errors = 0
    m = re.search(r"Tests\s+(.+?)\((\d+)\)", output)
    if m:
        body = m.group(1)
        pm = re.search(r"(\d+)\s+passed", body)
        fm = re.search(r"(\d+)\s+failed", body)
        passed = int(pm.group(1)) if pm else 0
        failed = int(fm.group(1)) if fm else 0
    # A compile/collection error means no test ran; surface it as an error.
    if return_code != 0 and passed == 0 and failed == 0:
        errors = 1
    total = passed + failed + errors
    return passed, failed, errors, total


def _run_vitest(
    code: str,
    test_code: str,
    solution_file: str,
    context_files: dict[str, str] | None,
    timeout: int,
) -> ExecutionResult:
    runner = _ts_runner_dir()
    if runner is None:
        return ExecutionResult(
            0, 0, 1, 1,
            "SETUP ERROR: TypeScript runner not installed. Run `npm install` in "
            "ts_runner/ (or set $HERMES_EVALS_TS_RUNNER).",
            1,
        )

    tmpdir = tempfile.mkdtemp()
    try:
        sol = solution_file if solution_file.endswith((".ts", ".tsx")) else "solution.ts"
        (Path(tmpdir) / sol).write_text(code)
        if context_files:
            for filename, content in context_files.items():
                (Path(tmpdir) / filename).write_text(content)
        (Path(tmpdir) / "solution.test.ts").write_text(test_code)
        # Reuse the installed deps without copying them.
        os.symlink(runner / "node_modules", Path(tmpdir) / "node_modules")
        (Path(tmpdir) / "package.json").write_text(json.dumps({"type": "module"}))

        result = subprocess.run(
            [str(runner / "node_modules" / ".bin" / "vitest"),
             "run", "--no-color", "--reporter=basic"],
            capture_output=True, text=True, cwd=tmpdir, timeout=timeout,
            env={**os.environ, "CI": "true"},
        )
        output = result.stdout + result.stderr
        passed, failed, errors, total = _parse_vitest_output(output, result.returncode)
        return ExecutionResult(passed, failed, errors, total, output, result.returncode)
    except subprocess.TimeoutExpired:
        return ExecutionResult(0, 0, 1, 1, "TIMEOUT: test execution exceeded time limit", 1)
    except Exception as e:
        return ExecutionResult(0, 0, 1, 1, f"ERROR: {e}", 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
