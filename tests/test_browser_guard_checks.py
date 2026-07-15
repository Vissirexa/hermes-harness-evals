"""canned_halt + idempotent_no_progress: the browser-guard false-positive family.

The recorded failure these mirror: interactive Google-Flights sessions where
the harness's idempotent no-progress guard counted snapshot repeats across
interleaved clicks/types, hard-stopped browser_snapshot, and replaced the
final answer with the fabricated "I stopped retrying..." string.
"""
import json

from agent_evals.checks import canned_halt, idempotent_no_progress, run_check
from agent_evals.transcript import Event


CANNED = (
    "I stopped retrying browser_snapshot because it hit the tool-call "
    "guardrail (idempotent_no_progress_block) after 3 repeated "
    "non-progressing attempts. The last tool result explains the blocker; "
    "the next step is to change strategy instead of repeating the same call."
)


def _assistant(text: str) -> Event:
    return Event(role="assistant", content=text)


def _snapshot(tree: str = "same accessibility tree, long enough to matter") -> Event:
    return Event(role="tool", tool_name="browser_snapshot", content=tree)


def _mutation(tool: str = "browser_click", *, ok: bool = True, error: str | None = None) -> Event:
    payload: dict = {"success": ok}
    if error:
        payload["error"] = error
    return Event(role="tool", tool_name=tool, content=json.dumps(payload))


# ---------------------------------------------------------------------------
# canned_halt
# ---------------------------------------------------------------------------

def test_canned_halt_detects_recorded_template():
    """The verbatim line from the 20260712 sessions counts as a hit."""
    events = [_assistant("Searching flights now."), _assistant(CANNED)]
    r = canned_halt(events, max_allowed=0)
    assert not r.passed
    assert r.measured == 1
    assert "I stopped retrying" in r.detail


def test_canned_halt_ignores_real_summaries():
    """A model-written wrap-up that mentions the guardrail is not canned."""
    events = [_assistant(
        "The search stalled behind the tool-loop guardrail, but before that I "
        "found two 1-stop fares: $743 (BA via LHR) and $802 (AA via LHR)."
    )]
    r = canned_halt(events, max_allowed=0)
    assert r.passed
    assert r.measured == 0


def test_canned_halt_counts_every_occurrence_and_custom_pattern():
    events = [_assistant(CANNED), _assistant("ok"), _assistant(CANNED)]
    assert canned_halt(events, max_allowed=0).measured == 2
    r = canned_halt(events, max_allowed=0, pattern=r"gave up on \S+ after")
    assert r.passed  # custom pattern replaces (not extends) the template


def test_canned_halt_via_run_check_dispatch():
    r = run_check([_assistant(CANNED)], {"type": "canned_halt", "max": 0})
    assert r.name == "canned_halt"
    assert not r.passed


# ---------------------------------------------------------------------------
# idempotent_no_progress
# ---------------------------------------------------------------------------

def test_successful_mutations_reset_the_run():
    """click → snapshot → type → snapshot on a laggy SPA measures 1."""
    events = []
    for _ in range(5):
        events += [_snapshot(), _mutation("browser_click")]
    r = idempotent_no_progress(events, max_allowed=3)
    assert r.passed
    assert r.measured == 1


def test_failed_mutations_do_not_reset():
    """A click that errored changed nothing — identical reads keep counting."""
    events = []
    for _ in range(4):
        events += [_snapshot(), _mutation("browser_click", ok=False, error="no such ref")]
    r = idempotent_no_progress(events, max_allowed=3)
    assert not r.passed
    assert r.measured == 4
    assert "browser_snapshot" in r.detail


def test_pure_read_thrash_counts():
    events = [_snapshot() for _ in range(4)]
    r = idempotent_no_progress(events, max_allowed=3)
    assert not r.passed
    assert r.measured == 4


def test_changed_content_is_progress():
    events = [_snapshot("loading spinner")] * 2 + [_snapshot("results: $743, $802")]
    r = idempotent_no_progress(events, max_allowed=3)
    assert r.passed
    assert r.measured == 2


def test_browser_wait_success_resets():
    """Waiting is the sanctioned poll: wait → identical snapshot is legitimate."""
    events = []
    for _ in range(4):
        events += [_snapshot("loading spinner"), _mutation("browser_wait")]
    r = idempotent_no_progress(events, max_allowed=3)
    assert r.passed
    assert r.measured == 1


def test_non_json_mutating_result_counts_as_success():
    """Matches the guard's failed-flag default for unparseable payloads."""
    events = [
        _snapshot(), Event(role="tool", tool_name="browser_click", content="clicked."),
        _snapshot(), Event(role="tool", tool_name="browser_click", content="clicked."),
        _snapshot(),
    ]
    r = idempotent_no_progress(events, max_allowed=3)
    assert r.passed
    assert r.measured == 1


def test_custom_tool_and_mutators_via_run_check():
    events = [
        Event(role="tool", tool_name="read_file", content="same body"),
        Event(role="tool", tool_name="read_file", content="same body"),
        Event(role="tool", tool_name="patch", content=json.dumps({"success": True})),
        Event(role="tool", tool_name="read_file", content="same body"),
    ]
    r = run_check(events, {
        "type": "idempotent_no_progress", "max": 2,
        "tool_name": "read_file", "mutating_tools": ["patch"],
    })
    assert r.passed
    assert r.measured == 2  # the two pre-patch reads; the post-patch read restarts at 1
