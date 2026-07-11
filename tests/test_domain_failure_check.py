"""domain_failure: per-host failing-fetch streaks, classified from result payloads."""
import json

from agent_evals.checks import domain_failure, run_check
from agent_evals.transcript import Event


def _fetch_result(url: str, *, status: int | None = None, blocked: bool = False,
                  error: str | None = None, tool: str = "fetch_resilient") -> Event:
    payload: dict = {"url": url, "ok": not blocked and not error}
    if status is not None:
        payload["status"] = status
    if blocked:
        payload["blocked"] = True
    if error:
        payload["error"] = error
    return Event(role="tool", tool_name=tool, content=json.dumps(payload))


def test_slug_mutation_streak_counts_per_host():
    """Unique slugs on one host accumulate; ok:true + status:404 is a failure."""
    events = [
        _fetch_result(f"https://www.example.com/guide-{i}", status=404)
        for i in range(7)
    ]
    r = domain_failure(events, max_allowed=6)
    assert not r.passed
    assert r.measured == 7
    assert "example.com" in r.detail


def test_success_resets_host_streak():
    events = (
        [_fetch_result("https://example.com/a", status=404)] * 5
        + [_fetch_result("https://example.com/real-page", status=200)]
        + [_fetch_result("https://example.com/b", status=404)] * 5
    )
    r = domain_failure(events, max_allowed=6)
    assert r.passed
    assert r.measured == 5


def test_hosts_tracked_independently():
    events = [
        _fetch_result(f"https://site-{i % 4}.com/page", status=404)
        for i in range(12)
    ]
    r = domain_failure(events, max_allowed=6)
    assert r.passed
    assert r.measured == 3


def test_blocked_and_error_shapes_count_as_failures():
    events = (
        [_fetch_result("https://example.com/a", blocked=True)] * 3
        + [_fetch_result("https://example.com/b", error="All fetch tiers failed")] * 3
        + [_fetch_result("https://example.com/c", status=500)]
    )
    r = domain_failure(events, max_allowed=6)
    assert not r.passed
    assert r.measured == 7


def test_www_prefix_folds_into_bare_host():
    events = (
        [_fetch_result("https://www.example.com/a", status=404)] * 3
        + [_fetch_result("https://example.com/b", status=404)] * 3
    )
    r = domain_failure(events, max_allowed=5)
    assert not r.passed
    assert r.measured == 6


def test_non_web_tools_and_unparseable_results_ignored():
    events = [
        Event(role="tool", tool_name="terminal", content='{"url": "https://x.com", "status": 404}'),
        Event(role="tool", tool_name="fetch_resilient", content="not json at all"),
        Event(role="tool", tool_name="fetch_resilient", content='"a bare string"'),
        _fetch_result("https://ok.com/a", status=404),
    ]
    r = domain_failure(events, max_allowed=6)
    assert r.passed
    assert r.measured == 1


def test_registered_in_check_registry():
    r = run_check(
        [_fetch_result("https://example.com/a", status=404)],
        {"type": "domain_failure", "max": 0},
    )
    assert r.name == "domain_failure"
    assert not r.passed
