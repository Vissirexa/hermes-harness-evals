"""loop_cap: per-turn ceilings on runaway-prone tools (web_search, subagents)."""
from agent_evals.checks import loop_cap, run_check
from agent_evals.transcript import Event, ToolCall


def _assistant(text: str = "", *calls: ToolCall) -> Event:
    return Event(role="assistant", content=text, tool_calls=list(calls))


def _search(query: str) -> ToolCall:
    return ToolCall("web_search", '{"query": "%s"}' % query)


def test_counts_distinct_calls_that_no_repetition_guard_would_catch():
    """The gap this mirrors: every call unique, nothing repeats, cap still trips."""
    events = [
        Event(role="user", content="research this"),
        _assistant("searching", *[_search(f"q{i}") for i in range(6)]),
    ]
    r = loop_cap(events, max_allowed=5)
    assert r.measured == 6
    assert not r.passed
    assert "6 web_search calls in a single turn" in r.detail


def test_accumulates_across_assistant_replies_within_one_turn():
    """A turn is the run_conversation loop, not one assistant reply."""
    events = [
        Event(role="user", content="go"),
        _assistant("", _search("a"), _search("b")),
        Event(role="tool", tool_name="web_search", content="{}"),
        _assistant("", _search("c")),
        _assistant("", _search("d")),
    ]
    r = loop_cap(events, max_allowed=10)
    assert r.measured == 4  # all four land in the same turn


def test_counter_resets_on_the_next_user_message():
    """reset_for_turn runs per user message, so peak is per-turn, not cumulative."""
    turn = [Event(role="user", content="go"), _assistant("", *[_search(f"q{i}") for i in range(4)])]
    r = loop_cap(turn * 3, max_allowed=5)
    assert r.measured == 4  # peak across three turns, not 12
    assert r.passed


def test_other_tools_are_not_counted():
    events = [
        Event(role="user", content="go"),
        _assistant("", *[ToolCall("read_file", "{}") for _ in range(30)], _search("a")),
    ]
    r = loop_cap(events, max_allowed=5)
    assert r.measured == 1


def test_subagent_batch_counts_children_not_invocations():
    """Mirrors _subagent_spawn_count: a batch spawns len(tasks) children."""
    batch = ToolCall("delegate_task", '{"tasks": [{"goal": "a"}, {"goal": "b"}, {"goal": "c"}]}')
    events = [Event(role="user", content="go"), _assistant("", batch, batch)]
    r = loop_cap(events, max_allowed=5, tool_name="delegate_task", count_batch_key="tasks")
    assert r.measured == 6  # 2 calls x 3 children, not 2
    assert not r.passed
    assert "6 delegate_task spawns" in r.detail


def test_single_goal_delegation_counts_as_one():
    single = ToolCall("delegate_task", '{"goal": "just one"}')
    events = [Event(role="user", content="go"), _assistant("", single, single)]
    r = loop_cap(events, max_allowed=5, tool_name="delegate_task", count_batch_key="tasks")
    assert r.measured == 2


def test_unparseable_args_fail_open_as_one_spawn():
    """Non-JSON args count as 1, matching the guard's fail-open behavior."""
    junk = ToolCall("delegate_task", "not json at all")
    events = [Event(role="user", content="go"), _assistant("", junk, junk)]
    r = loop_cap(events, max_allowed=5, tool_name="delegate_task", count_batch_key="tasks")
    assert r.measured == 2
    assert r.passed


def test_healthy_session_measures_its_real_peak():
    events = [
        Event(role="user", content="go"),
        _assistant("", _search("a")),
        Event(role="tool", tool_name="web_search", content="{}"),
        _assistant("answer"),
    ]
    r = loop_cap(events, max_allowed=50)
    assert r.measured == 1
    assert r.passed


def test_runs_through_the_spec_registry():
    events = [Event(role="user", content="go"), _assistant("", *[_search(f"q{i}") for i in range(3)])]
    r = run_check(events, {"type": "loop_cap", "max": 2, "tool_name": "web_search"})
    assert r.measured == 3
    assert not r.passed
