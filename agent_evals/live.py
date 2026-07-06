"""Live agent-drive: run a scenario through the real Hermes agent and return the
session id it produced, so the checks can run against a fresh transcript.

Hermes `-z/--oneshot` runs a single prompt headlessly: tools/memory/rules load as
normal, approvals are auto-bypassed, and it prints only the final response. It
does NOT print the session id, so we bracket the run by time: note the latest
session start before, run, then take the newest session created after.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

HERMES_BIN = os.environ.get("HERMES_BIN", "hermes")


def _newest_session_after(db_path: Path, after_epoch: float) -> str | None:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT id FROM sessions WHERE started_at > ? ORDER BY started_at DESC LIMIT 1",
            (after_epoch, ),
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def drive_live(
    scenario_prompt: str,
    db_path: str | Path,
    model: str | None = None,
    timeout: int = 600,
    extra_args: list[str] | None = None,
) -> tuple[str, str]:
    """Run one scenario through Hermes; return (session_id, final_stdout).

    Raises RuntimeError if the run fails or no new session can be located.
    `db_path` must point at the state.db your install records sessions into —
    there is deliberately no default, same policy as ``load_transcript``.
    """
    db_path = Path(os.path.expanduser(str(db_path)))
    t0 = time.time()

    cmd = [HERMES_BIN, "-z", scenario_prompt, "--yolo", "--accept-hooks"]
    if model:
        cmd += ["-m", model]
    if extra_args:
        cmd += extra_args

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A timeout is itself a finding (the run never converged); still try to
        # locate the session so the checks can report what happened.
        sid = _newest_session_after(db_path, t0)
        if sid:
            return sid, f"TIMEOUT after {timeout}s (session {sid} captured anyway)"
        raise RuntimeError(f"hermes -z timed out after {timeout}s and no session was recorded")

    sid = _newest_session_after(db_path, t0)
    if not sid:
        raise RuntimeError(
            "no new session recorded after the run; "
            f"hermes exited {proc.returncode}. stderr tail:\n{proc.stderr[-500:]}"
        )
    return sid, proc.stdout
