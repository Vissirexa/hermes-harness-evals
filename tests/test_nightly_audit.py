"""Tests for tools/nightly_audit.py and transcript.sessions_since.

Fixtures build a synthetic Hermes ``state.db`` (the real ``sessions`` +
``messages`` schema, minimal columns) under ``tmp_path`` and populate it with
hand-built sessions — a clean one and a deliberately loopy one. Nothing here
reads the user's real ~/.hermes db.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from agent_evals.transcript import sessions_since
from tools.nightly_audit import (
    DEFAULT_CHECKS,
    audit_sessions,
    main,
    parse_since,
    read_state_since,
)

# Minimal slice of the real Hermes schema — enough for the enumerator and the
# transcript loader (they SELECT explicit columns, so extra columns are absent
# rather than defaulted).
_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
"""


def _tool_call_json(name: str, args: dict) -> str:
    return json.dumps([{"function": {"name": name, "arguments": json.dumps(args)}}])


class DbBuilder:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(path)
        self.con.executescript(_SCHEMA)
        self._ts = 1_000_000.0

    def add_session(self, sid: str, *, source: str, started_at: float, title: str = "") -> None:
        self.con.execute(
            "INSERT INTO sessions (id, source, started_at, title) VALUES (?, ?, ?, ?)",
            (sid, source, started_at, title),
        )

    def add_assistant_call(self, sid: str, name: str, args: dict) -> None:
        self._ts += 1
        self.con.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp, active) "
            "VALUES (?, 'assistant', '', ?, ?, 1)",
            (sid, _tool_call_json(name, args), self._ts),
        )

    def add_tool_result(self, sid: str, tool_name: str, content: str) -> None:
        self._ts += 1
        self.con.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp, active) "
            "VALUES (?, 'tool', ?, ?, ?, 1)",
            (sid, content, tool_name, self._ts),
        )

    def close(self) -> None:
        self.con.commit()
        self.con.close()


@pytest.fixture
def now() -> float:
    return time.time()


@pytest.fixture
def db(tmp_path: Path, now: float) -> Path:
    """A db with: one clean recent session, one loopy recent session, and one
    clean OLD session (outside a 24h window)."""
    path = tmp_path / "state.db"
    b = DbBuilder(path)

    # Clean, recent (cli): three distinct calls, three distinct results.
    b.add_session("clean_recent", source="cli", started_at=now - 3600, title="clean run")
    for i in range(3):
        b.add_assistant_call("clean_recent", "terminal", {"cmd": f"echo {i}"})
        b.add_tool_result("clean_recent", "terminal", f"distinct result {i} " + "x" * 300)

    # Loopy, recent (telegram): same fetch result byte-identical 8 times.
    b.add_session("loopy_recent", source="telegram", started_at=now - 1800, title="stuck fetch")
    identical = "the same 300-char body " + "y" * 300
    for _ in range(8):
        b.add_assistant_call("loopy_recent", "fetch", {"url": "http://x"})
        b.add_tool_result("loopy_recent", "fetch", identical)

    # Clean but OLD (cron), ~3 days ago — should fall outside a 24h window.
    b.add_session("clean_old", source="cron", started_at=now - 3 * 86400, title="old cron")
    b.add_assistant_call("clean_old", "terminal", {"cmd": "date"})
    b.add_tool_result("clean_old", "terminal", "Mon " + "z" * 300)

    b.close()
    return path


# ---------------------------------------------------------------------------
# sessions_since enumerator
# ---------------------------------------------------------------------------


def test_sessions_since_window_excludes_old(db, now):
    recent = sessions_since(db, since=now - 86400)
    ids = {s.session_id for s in recent}
    assert ids == {"clean_recent", "loopy_recent"}
    # chronological order (oldest first): clean_recent (-3600) before loopy (-1800)
    assert [s.session_id for s in recent] == ["clean_recent", "loopy_recent"]


def test_sessions_since_open_lower_bound_includes_all(db):
    everything = sessions_since(db)
    assert {s.session_id for s in everything} == {"clean_recent", "loopy_recent", "clean_old"}


def test_sessions_since_source_filter(db, now):
    tg = sessions_since(db, since=now - 86400, sources=["telegram"])
    assert [s.session_id for s in tg] == ["loopy_recent"]


def test_sessions_since_limit_keeps_most_recent_chronologically(db):
    two = sessions_since(db, limit=2)
    # most-recent two are loopy_recent (-1800) and clean_recent (-3600),
    # returned oldest-first.
    assert [s.session_id for s in two] == ["clean_recent", "loopy_recent"]


def test_sessions_since_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sessions_since(tmp_path / "nope.db", since=0)


def test_sessions_since_metadata_populated(db, now):
    loopy = next(s for s in sessions_since(db) if s.session_id == "loopy_recent")
    assert loopy.source == "telegram"
    assert loopy.title == "stuck fetch"
    assert loopy.started_at == pytest.approx(now - 1800)


# ---------------------------------------------------------------------------
# audit_sessions + the loop-guard defaults
# ---------------------------------------------------------------------------


def test_audit_flags_only_the_loopy_session(db, now):
    infos = sessions_since(db, since=now - 86400)
    audits = audit_sessions(db, infos, DEFAULT_CHECKS)
    by_id = {a.info.session_id: a for a in audits}

    assert by_id["clean_recent"].breached is False
    assert by_id["loopy_recent"].breached is True

    breached_checks = {r.name for r in by_id["loopy_recent"].breaches}
    # 8 identical calls (>10? no) — the identical *result* (8) breaches max 6,
    # and 8 identical calls does not breach max 10, so repeated_result fires.
    assert "repeated_result" in breached_checks


def test_audit_load_failure_becomes_error_not_crash(db, now, monkeypatch):
    infos = sessions_since(db, since=now - 86400)

    import tools.nightly_audit as na

    def boom(*a, **k):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(na, "load_transcript", boom)
    audits = na.audit_sessions(db, infos, DEFAULT_CHECKS)
    assert all(a.error is not None and a.breached for a in audits)


# ---------------------------------------------------------------------------
# --since parsing
# ---------------------------------------------------------------------------


def test_parse_since_relative_units():
    base = 1_000_000.0
    assert parse_since("24h", now=base) == base - 86400
    assert parse_since("7d", now=base) == base - 7 * 86400
    assert parse_since("90m", now=base) == base - 90 * 60


def test_parse_since_epoch_and_iso():
    assert parse_since("1783487810", now=0) == 1783487810.0
    # ISO date round-trips through the local tz the tool formats in
    from datetime import datetime

    assert parse_since("2026-07-01", now=0) == datetime.fromisoformat("2026-07-01").timestamp()


def test_parse_since_garbage_raises():
    with pytest.raises(ValueError):
        parse_since("last tuesday", now=0)


# ---------------------------------------------------------------------------
# CLI: exit codes, JSON, incremental state file
# ---------------------------------------------------------------------------


def test_cli_exit_1_and_json_lists_breach(db, now, capsys):
    code = main(["--db", str(db), "--since", "24h", "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["sessions_breached"] == 1
    assert payload["breaches"][0]["session_id"] == "loopy_recent"


def test_cli_exit_0_when_only_clean_in_window(db, now, capsys):
    # Narrow the window + source to the clean session only.
    code = main(["--db", str(db), "--since", "24h", "--source", "cli"])
    assert code == 0
    assert "Clean" in capsys.readouterr().out


def test_cli_state_file_advances_watermark(db, now, tmp_path, capsys):
    state = tmp_path / "audit-state.json"
    # First run: no watermark → default 24h window, sees both recent sessions,
    # records the newest started_at (loopy_recent at now-1800).
    main(["--db", str(db), "--state-file", str(state)])
    watermark = read_state_since(state)
    assert watermark == pytest.approx(now - 1800)

    # Second run with the same state file: watermark now excludes both recent
    # sessions (both started at/below it), so nothing is audited → exit 0.
    capsys.readouterr()
    code = main(["--db", str(db), "--state-file", str(state)])
    assert code == 0
    assert "Sessions audited: 0" in capsys.readouterr().out


def test_cli_missing_db_exit_2(tmp_path, capsys):
    code = main(["--db", str(tmp_path / "nope.db"), "--since", "24h"])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_cli_checks_file_override(db, now, tmp_path, capsys):
    # A checks file that only looks for hallucinated tools — the loopy session
    # has none, so the whole sweep comes back clean.
    checks_file = tmp_path / "checks.yaml"
    checks_file.write_text("- {type: hallucinated_tool, max: 0}\n")
    code = main(["--db", str(db), "--since", "24h", "--checks-file", str(checks_file)])
    assert code == 0
    assert "Clean" in capsys.readouterr().out
