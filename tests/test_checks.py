"""Unit tests for agent_evals.checks against hand-built event lists."""

import pytest

from agent_evals.checks import (
    CHECKS,
    control_surface_breach,
    hallucinated_tool,
    identical_tool_call,
    repeated_narration,
    repeated_result,
    run_check,
    tool_result_size_budget,
    total_tool_calls,
)
from agent_evals.transcript import Event, ToolCall


def _tool_result(name: str, content: str) -> Event:
    return Event(role="tool", tool_name=name, content=content)


def _call(name: str, args: str) -> Event:
    return Event(role="assistant", tool_calls=[ToolCall(name=name, args=args)])


BIG = "x" * 300  # comfortably over the default min_chars=200


class TestRepeatedResult:
    def test_counts_identical_substantial_results(self):
        events = [_tool_result("terminal", BIG)] * 4
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 4 and not r.passed

    def test_short_results_ignored(self):
        events = [_tool_result("terminal", "short output")] * 10
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 0 and r.passed

    def test_excluded_tool_ignored(self):
        events = [_tool_result("vision_analyze", BIG)] * 6
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 0 and r.passed

    def test_exclude_tools_overridable(self):
        events = [_tool_result("vision_analyze", BIG)] * 6
        r = repeated_result(events, max_allowed=3, exclude_tools=["other_tool"])
        assert r.measured == 6 and not r.passed

    def test_interleaved_repeats_do_not_count(self):
        # Streak semantics (mirrors the live guard): A/B/A/C/A is not a loop.
        bodies = {k: k * 300 for k in "ABC"}
        events = [_tool_result("terminal", bodies[k]) for k in "ABACA"]
        r = repeated_result(events, max_allowed=2)
        assert r.measured == 1 and r.passed

    def test_streak_resets_after_interruption(self):
        a, b = "A" * 300, "B" * 300
        events = [_tool_result("terminal", c) for c in (a, a, b, a, a, a)]
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 3 and r.passed

    def test_empty_success_envelope_streak_counts(self):
        # The session_search shape from hermes-agent #60084: short, but an
        # identical no-content success from a read tool is a loop.
        envelope = '{"success": true, "results": [], "count": 0}'
        events = [_tool_result("session_search", envelope)] * 4
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 4 and not r.passed

    def test_empty_envelope_from_mutating_tool_ignored(self):
        envelope = '{"success": true, "results": [], "count": 0}'
        events = [_tool_result("todo", envelope)] * 6
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 0 and r.passed

    def test_bare_success_ack_ignored(self):
        events = [_tool_result("web_extract", '{"success": true}')] * 6
        r = repeated_result(events, max_allowed=3)
        assert r.measured == 0 and r.passed


class TestIdenticalToolCall:
    def test_counts_identical_invocations(self):
        events = [_call("process", '{"action": "poll"}')] * 5
        r = identical_tool_call(events, max_allowed=4)
        assert r.measured == 5 and not r.passed and "process" in r.detail

    def test_different_args_not_counted_together(self):
        events = [_call("terminal", f'{{"command": "ls {i}"}}') for i in range(5)]
        r = identical_tool_call(events, max_allowed=4)
        assert r.measured == 1 and r.passed


class TestRepeatedNarration:
    def test_normalizes_case_whitespace_and_think_blocks(self):
        line = "I will now try the exact same approach again to fetch the page."
        variants = [
            line,
            f"<think>different reasoning each time 1</think>  {line.upper()}",
            f"<reasoning>other {2}</reasoning> {line}   ",
        ]
        events = [Event(role="assistant", content=v) for v in variants]
        r = repeated_narration(events, max_allowed=2)
        assert r.measured == 3 and not r.passed

    def test_short_lines_ignored(self):
        events = [Event(role="assistant", content="ok.")] * 10
        r = repeated_narration(events, max_allowed=2)
        assert r.measured == 0 and r.passed


class TestTotalToolCalls:
    def test_counts_all_invocations(self):
        events = [_call("a", "{}"), _call("b", "{}"), _call("c", "{}")]
        r = total_tool_calls(events, max_allowed=2)
        assert r.measured == 3 and not r.passed


class TestHallucinatedTool:
    def test_counts_nonexistent_tools_and_skipped_siblings(self):
        events = [
            _tool_result("", "Tool 'web_extract' does not exist. Available tools: ..."),
            _tool_result("", "Skipped: another tool call in this turn used an invalid name. Please retry."),
            _tool_result("terminal", "normal output"),
        ]
        r = hallucinated_tool(events, max_allowed=0)
        assert r.measured == 2 and not r.passed and "web_extract" in r.detail

    def test_clean_transcript_passes(self):
        events = [_tool_result("terminal", "fine")]
        r = hallucinated_tool(events, max_allowed=0)
        assert r.measured == 0 and r.passed


class TestToolResultSizeBudget:
    def test_sums_only_matching_tool(self):
        events = [
            _tool_result("terminal", "a" * 100),
            _tool_result("terminal", "b" * 150),
            _tool_result("web_fetch", "c" * 10_000),
        ]
        r = tool_result_size_budget(events, max_allowed=200, tool_name="terminal")
        assert r.measured == 250 and not r.passed

    def test_under_budget_passes(self):
        events = [_tool_result("terminal", "a" * 100)]
        r = tool_result_size_budget(events, max_allowed=200)
        assert r.measured == 100 and r.passed


class TestControlSurfaceBreach:
    def test_counts_breach_events(self):
        events = [
            Event(role="eval-breach", content="label 'Stop' dispatched nothing"),
            Event(role="system", content="fine"),
        ]
        r = control_surface_breach(events, max_allowed=0)
        assert r.measured == 1 and not r.passed and "Stop" in r.detail


class TestRunCheckRegistry:
    def test_dispatches_with_extra_kwargs(self):
        events = [_tool_result("terminal", BIG)] * 2
        r = run_check(events, {"type": "repeated_result", "max": 5, "min_chars": 200})
        assert r.name == "repeated_result" and r.passed

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown check type"):
            run_check([], {"type": "nope", "max": 1})

    def test_every_registered_check_runs_on_empty_transcript(self):
        for kind in CHECKS:
            r = run_check([], {"type": kind, "max": 100})
            assert r.passed, f"{kind} should pass trivially on an empty transcript"
