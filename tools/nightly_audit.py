#!/usr/bin/env python3
"""nightly_audit — sweep recent Hermes sessions through the loop-guard checks.

The recorded-session specs in ``agent_evals/specs/`` pin *known* sessions as
regression fixtures. This tool is the other direction: it takes every session
your install recorded in a window (last night, since the last run, a date
range) and runs the same checks over all of them, so a loop/thrash/hallucinated
-tool regression that lands in real traffic surfaces the next morning instead
of waiting to be noticed by hand.

It reuses the exact building blocks the spec runner uses — ``sessions_since``
to enumerate, ``load_transcript`` to normalize, and ``checks.run_check`` to
measure — so the audit and the pinned specs can never drift apart in what
"a breach" means.

Window selection (``--since`` / ``--state-file``):
  - ``--since 24h`` / ``7d`` / ``90m``  — relative to now
  - ``--since 2026-07-01``              — ISO date (or full ISO datetime)
  - ``--since 1783487810``             — raw epoch seconds
  - ``--state-file PATH``               — incremental: lower-bound at the last
        run's newest session, then record the new high-water mark for next time.
  Default when neither is given: the last 24 hours.

Read-only against the session db. The ONLY paths it writes are ``--output``
(the report) and ``--state-file`` (the high-water mark) — never the db.

Usage:
    python -m tools.nightly_audit --db ~/.hermes/profiles/code/state.db
    python -m tools.nightly_audit --db … --since 7d --source cli --source telegram
    python -m tools.nightly_audit --db … --state-file ~/.hermes/audit-state.json
    python -m tools.nightly_audit --db … --checks-file my_thresholds.yaml --json

Exit codes: 0 clean, 1 one or more sessions breached a check, 2 tool error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ``tools/`` is a standalone-script dir (see tests/conftest.py); when run as
# ``python -m tools.nightly_audit`` from the repo root the package imports resolve.
from agent_evals.checks import run_check
from agent_evals.transcript import SessionInfo, load_transcript, sessions_since

# Loop-signature checks that should stay flat on a healthy session regardless of
# how long it legitimately ran. total_tool_calls is deliberately NOT a default —
# a blanket sweep would flag every long-but-fine session on length alone; add it
# via --checks-file if you want a ceiling.
DEFAULT_CHECKS: list[dict] = [
    {"type": "hallucinated_tool", "max": 0},
    {"type": "identical_tool_call", "max": 10},
    {"type": "repeated_result", "max": 6, "min_chars": 200},
    {"type": "repeated_narration", "max": 4, "min_chars": 40},
]

_RELATIVE_UNITS = {"m": 60, "h": 3600, "d": 86400}


def parse_since(value: str, now: float | None = None) -> float:
    """Turn a --since string into an epoch lower bound.

    Accepts relative (``24h``/``7d``/``90m``), ISO date/datetime, or raw epoch.
    """
    now = time.time() if now is None else now
    v = value.strip()

    # Relative: <number><unit>
    if len(v) >= 2 and v[-1] in _RELATIVE_UNITS and v[:-1].replace(".", "", 1).isdigit():
        return now - float(v[:-1]) * _RELATIVE_UNITS[v[-1]]

    # Raw epoch seconds
    try:
        return float(v)
    except ValueError:
        pass

    # ISO date or datetime
    try:
        iso = v.replace("Z", "+00:00")
        return datetime.fromisoformat(iso).timestamp()
    except ValueError as exc:
        raise ValueError(
            f"could not parse --since {value!r}: use e.g. '24h', '7d', "
            f"'2026-07-01', or an epoch like '1783487810'"
        ) from exc


def read_state_since(state_file: Path) -> float | None:
    """The high-water mark recorded by a previous run, or None if absent/empty."""
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text())
    except (ValueError, OSError):
        return None
    watermark = data.get("last_started_at")
    return float(watermark) if isinstance(watermark, (int, float)) else None


def write_state_since(state_file: Path, last_started_at: float) -> None:
    state_file.write_text(
        json.dumps(
            {"last_started_at": last_started_at, "updated_at": time.time()}, indent=2
        )
        + "\n"
    )


class SessionAudit:
    """A single session and every check breach found in it."""

    def __init__(self, info: SessionInfo):
        self.info = info
        self.breaches: list = []  # CheckResult objects that did not pass
        self.error: str | None = None  # transcript/check failure, not a breach

    @property
    def breached(self) -> bool:
        return bool(self.breaches) or self.error is not None

    def to_dict(self) -> dict:
        return {
            "session_id": self.info.session_id,
            "source": self.info.source,
            "started_at": self.info.started_at,
            "title": self.info.title,
            "error": self.error,
            "breaches": [
                {
                    "check": r.name,
                    "measured": r.measured,
                    "threshold": r.threshold,
                    "detail": r.detail,
                }
                for r in self.breaches
            ],
        }


def audit_sessions(
    db_path: str | Path,
    infos: list[SessionInfo],
    checks: list[dict],
    active_only: bool = True,
) -> list[SessionAudit]:
    """Run ``checks`` over each session; return one SessionAudit per session."""
    audits: list[SessionAudit] = []
    for info in infos:
        audit = SessionAudit(info)
        try:
            events = load_transcript(
                session_id=info.session_id, db_path=db_path, active_only=active_only
            )
        except Exception as exc:  # a session we can't read is worth surfacing
            audit.error = f"could not load transcript: {exc}"
            audits.append(audit)
            continue
        for check_spec in checks:
            result = run_check(events, check_spec)
            if not result.passed:
                audit.breaches.append(result)
        audits.append(audit)
    return audits


def _fmt_ts(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, OverflowError):
        return str(epoch)


def format_text(audits: list[SessionAudit], window_lo: float | None, window_hi: float) -> str:
    breached = [a for a in audits if a.breached]
    lines: list[str] = []
    lines.append("nightly_audit report")
    lines.append("=" * 40)
    lo = _fmt_ts(window_lo) if window_lo is not None else "(open)"
    lines.append(f"Window: {lo}  →  {_fmt_ts(window_hi)}")
    lines.append(f"Sessions audited: {len(audits)}")
    lines.append(f"Sessions with breaches: {len(breached)}")
    lines.append("")
    if not breached:
        lines.append("No breaches. Clean.")
        return "\n".join(lines) + "\n"

    for a in breached:
        i = a.info
        lines.append(
            f"[BREACH] {i.session_id}  ({i.source}, {_fmt_ts(i.started_at)})"
            + (f"  — {i.title}" if i.title else "")
        )
        if a.error:
            lines.append(f"    ERROR: {a.error}")
        for r in a.breaches:
            detail = f"   {r.detail}" if r.detail else ""
            lines.append(f"    ✗ {r.name:<22} measured {r.measured} / max {r.threshold}{detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_json(audits: list[SessionAudit], window_lo: float | None, window_hi: float) -> str:
    breached = [a for a in audits if a.breached]
    payload = {
        "window": {"since": window_lo, "until": window_hi},
        "sessions_audited": len(audits),
        "sessions_breached": len(breached),
        "breaches": [a.to_dict() for a in breached],
    }
    return json.dumps(payload, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nightly_audit",
        description=(
            "Sweep recent Hermes sessions through the loop-guard checks and "
            "report every session that breached one."
        ),
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to the Hermes state.db whose sessions to audit (required; no default).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Lower bound: '24h'/'7d'/'90m', an ISO date/datetime, or epoch seconds. "
        "Default: last 24h (unless --state-file supplies a newer bound).",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Optional upper bound (same formats as --since). Default: now.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Incremental mode: audit only sessions newer than the last run's "
        "high-water mark, then record the new one here.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        dest="sources",
        help="Restrict to this session source (repeatable): cli, cron, telegram, ….",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit at most the N most-recent sessions in the window.",
    )
    parser.add_argument(
        "--checks-file",
        type=Path,
        default=None,
        help="YAML file with a list of check specs to run instead of the defaults.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the report here instead of stdout (report file; never the db).",
    )
    return parser


def _load_checks(checks_file: Path | None) -> list[dict]:
    if checks_file is None:
        return DEFAULT_CHECKS
    import yaml

    data = yaml.safe_load(checks_file.read_text())
    if not isinstance(data, list) or not all(isinstance(c, dict) for c in data):
        raise ValueError(f"{checks_file}: expected a YAML list of check-spec mappings")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    now = time.time()

    try:
        checks = _load_checks(args.checks_file)

        # Resolve the lower bound: explicit --since, else the state-file
        # watermark, else default 24h. When both --since and --state-file are
        # given, take the tighter (newer) of the two so an incremental run never
        # re-audits below its watermark.
        bounds: list[float] = []
        if args.since is not None:
            bounds.append(parse_since(args.since, now=now))
        state_watermark = None
        if args.state_file is not None:
            state_watermark = read_state_since(args.state_file)
            if state_watermark is not None:
                bounds.append(state_watermark)
        if not bounds:
            bounds.append(now - 86400.0)  # default: last 24h
        window_lo = max(bounds)
        window_hi = parse_since(args.until, now=now) if args.until else now

        infos = sessions_since(
            db_path=args.db,
            since=window_lo,
            until=window_hi,
            sources=args.sources,
            limit=args.limit,
        )
        audits = audit_sessions(args.db, infos, checks)
    except FileNotFoundError as exc:
        print(f"nightly_audit: error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # tool-level failure, not a finding
        print(f"nightly_audit: error: {exc}", file=sys.stderr)
        return 2

    text = format_json(audits, window_lo, window_hi) if args.json else format_text(
        audits, window_lo, window_hi
    )
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")

    # Advance the watermark only after a successful sweep, to the newest session
    # actually seen (so a session that arrives mid-run isn't skipped next time).
    if args.state_file is not None and infos:
        newest = max(i.started_at for i in infos)
        write_state_since(args.state_file, newest)

    return 1 if any(a.breached for a in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
