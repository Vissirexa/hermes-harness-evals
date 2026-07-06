"""Deterministic tests for agent_evals.control_surface and agent_evals.live.

None of these touch a real Hermes install or invoke an LLM: control-surface
tests stop at the input-validation layer (before any subprocess would run),
and the live-drive tests use tiny stub scripts standing in for `hermes`.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
import time

import pytest

from agent_evals.control_surface import simulate_control_surface
import agent_evals.live as live_mod
from agent_evals.live import _newest_session_after, drive_live


# -- simulate_control_surface: input validation ------------------------------ #


def test_missing_install_path_raises_key_error():
    with pytest.raises(KeyError):
        simulate_control_surface({"triggers": [{"kind": "config", "expect": {}}]})


def test_missing_config_path_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        simulate_control_surface({
            "install_path": str(tmp_path),
            "triggers": [{"kind": "config", "expect": {}}],
        })


def test_nonexistent_paths_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        simulate_control_surface({
            "install_path": str(tmp_path / "no-such-install"),
            "config_path": str(tmp_path / "no-such-config.yaml"),
            "triggers": [{"kind": "config", "expect": {}}],
        })


def _make_fake_install(tmp_path):
    install = tmp_path / "install"
    venv_bin = install / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.write_text("")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)

    config = tmp_path / "config.yaml"
    config.write_text("telegram: {}\n")
    return install, config


def test_valid_paths_but_no_triggers_raises_key_error(tmp_path):
    install, config = _make_fake_install(tmp_path)
    with pytest.raises(KeyError):
        simulate_control_surface({
            "install_path": str(install),
            "config_path": str(config),
        })


# -- _newest_session_after ---------------------------------------------------- #


@pytest.fixture()
def sessions_db(tmp_path):
    path = tmp_path / "state.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sessions (id TEXT, started_at REAL)")
    con.commit()
    con.close()
    return path


def test_newest_session_after_finds_newer_row(sessions_db):
    t = time.time()
    con = sqlite3.connect(sessions_db)
    con.executemany(
        "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
        [("old", t - 100), ("new", t + 100)],
    )
    con.commit()
    con.close()

    assert _newest_session_after(sessions_db, t) == "new"
    assert _newest_session_after(sessions_db, t + 200) is None


# -- drive_live ---------------------------------------------------------------- #


def _write_stub(tmp_path, name, body):
    script = tmp_path / name
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_drive_live_raises_runtime_error_when_no_new_session(tmp_path, monkeypatch, sessions_db):
    stub = _write_stub(tmp_path, "hermes_stub.sh", "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(live_mod, "HERMES_BIN", str(stub))

    with pytest.raises(RuntimeError, match="no new session"):
        drive_live("do something", db_path=sessions_db, timeout=10)


def test_drive_live_returns_session_id_on_success(tmp_path, monkeypatch, sessions_db):
    script_body = (
        f"#!{sys.executable}\n"
        "import sqlite3, sys, time\n"
        f"con = sqlite3.connect({str(sessions_db)!r})\n"
        "con.execute(\"INSERT INTO sessions (id, started_at) VALUES (?, ?)\", "
        "(\"live_test_sid\", time.time()))\n"
        "con.commit()\n"
        "con.close()\n"
        "print('ok')\n"
    )
    stub = _write_stub(tmp_path, "hermes_stub.py", script_body)
    monkeypatch.setattr(live_mod, "HERMES_BIN", str(stub))

    session_id, _ = drive_live("do something", db_path=sessions_db, timeout=10)
    assert session_id == "live_test_sid"
