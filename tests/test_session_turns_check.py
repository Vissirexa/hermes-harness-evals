"""session_turns: assistant turn count as the O(turns) longevity signal."""
from agent_evals.checks import run_check, session_turns
from agent_evals.transcript import Event, ToolCall


def _assistant(text: str = "", *calls: ToolCall) -> Event:
    return Event(role="assistant", content=text, tool_calls=list(calls))


def _tool_result(name: str, content: str = "{}") -> Event:
    return Event(role="tool", tool_name=name, content=content)


def test_counts_every_assistant_turn_including_toolless_and_dense():
    """A turn is one model reply regardless of how many tool calls it carried."""
    events = [
        Event(role="user", content="go"),
        _assistant("thinking", ToolCall("read_file", "{}"), ToolCall("read_file", "{}")),
        _tool_result("read_file"),
        _tool_result("read_file"),
        _assistant("", ToolCall("terminal", "{}")),  # no narration, still a turn
        _tool_result("terminal"),
        _assistant("done"),                            # final answer, no calls
    ]
    r = session_turns(events, max_allowed=10)
    assert r.measured == 3
    assert r.passed


def test_breaches_when_turns_exceed_threshold():
    """The regime witness: turn count over the ceiling is a breach."""
    events = [_assistant(f"turn {i}") for i in range(250)]
    r = session_turns(events, max_allowed=200)
    assert r.measured == 250
    assert not r.passed


def test_distinct_from_tool_call_volume():
    """Many tool calls in few turns stays under a turn ceiling."""
    dense = _assistant("", *[ToolCall("read_file", "{}") for _ in range(50)])
    events = [dense, dense]
    r = session_turns(events, max_allowed=10)
    assert r.measured == 2  # 2 turns despite 100 tool calls
    assert r.passed


def test_registered_in_check_registry():
    r = run_check(
        [_assistant("a"), _assistant("b")],
        {"type": "session_turns", "max": 1},
    )
    assert r.name == "session_turns"
    assert r.measured == 2
    assert not r.passed
