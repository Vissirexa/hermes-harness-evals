"""Unit tests for agent_evals.transcript against a real (temporary) sqlite db."""

import json
import sqlite3

import pytest

from agent_evals.transcript import (
    _normalize_args,
    _parse_tool_calls,
    load_transcript,
    tool_invocations,
    tool_results,
)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "state.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE messages ("
        "id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
        "tool_calls TEXT, tool_name TEXT, timestamp REAL, active INTEGER DEFAULT 1)"
    )
    rows = [
        ("s1", "user", "run ls", None, None, 1.0, 1),
        ("s1", "assistant", "", json.dumps([
            {"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
        ]), None, 2.0, 1),
        ("s1", "tool", "file_a\nfile_b", None, "terminal", 3.0, 1),
        ("s1", "assistant", "Two files found.", None, None, 4.0, 1),
        # inactive row must be filtered by default
        ("s1", "assistant", "compacted-away narration", None, None, 2.5, 0),
        # another session must never leak in
        ("s2", "user", "other session", None, None, 1.0, 1),
    ]
    con.executemany(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_name, timestamp, active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows,
    )
    con.commit()
    con.close()
    return path


def test_loads_session_in_timestamp_order(db):
    events = load_transcript("s1", db_path=db)
    assert [e.role for e in events] == ["user", "assistant", "tool", "assistant"]
    assert tool_invocations(events) == [("terminal", '{"command": "ls"}')]
    assert tool_results(events) == [("terminal", "file_a\nfile_b")]


def test_active_only_false_includes_inactive_rows(db):
    events = load_transcript("s1", db_path=db, active_only=False)
    assert any(e.content == "compacted-away narration" for e in events)


def test_other_sessions_do_not_leak(db):
    events = load_transcript("s1", db_path=db)
    assert all("other session" not in e.content for e in events)


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transcript("s1", db_path=tmp_path / "absent.db")


def test_db_path_is_required():
    with pytest.raises(TypeError):
        load_transcript("s1")  # no default database location


def test_normalize_args_sorts_keys():
    assert _normalize_args('{"b": 1, "a": 2}') == '{"a": 2, "b": 1}'


def test_normalize_args_passes_through_non_json():
    assert _normalize_args("not json") == "not json"


def test_parse_tool_calls_handles_garbage():
    assert _parse_tool_calls(None) == []
    assert _parse_tool_calls("") == []
    assert _parse_tool_calls("{broken") == []
