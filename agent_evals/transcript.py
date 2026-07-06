"""Load a Hermes Agent session into a normalized transcript the checks can read.

Schema (Hermes state.db, table ``messages``):
  role, content, tool_calls (JSON on assistant rows), tool_name, tool_call_id,
  timestamp, active.
Assistant rows carry text in `content` and zero or more calls in `tool_calls`
(OpenAI shape: [{"function": {"name": ..., "arguments": "<json str>"}}, ...]).
Tool-result rows have role='tool', a tool_name, and the result in `content`.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    name: str
    args: str  # normalized JSON string (sorted keys) for stable comparison


@dataclass
class Event:
    role: str                 # 'user' | 'assistant' | 'tool' | 'system'
    content: str = ""
    tool_name: str = ""       # set on assistant calls and tool results
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: float = 0.0


def _normalize_args(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return (raw or "").strip()


def _parse_tool_calls(raw: str | None) -> list[ToolCall]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    calls = []
    for c in data if isinstance(data, list) else []:
        fn = c.get("function", c) if isinstance(c, dict) else {}
        name = fn.get("name") or c.get("name", "") if isinstance(c, dict) else ""
        args = fn.get("arguments", "") if isinstance(fn, dict) else ""
        calls.append(ToolCall(name=name, args=_normalize_args(args)))
    return calls


def load_transcript(
    session_id: str,
    db_path: str | Path,
    active_only: bool = True,
) -> list[Event]:
    """Read one session's messages (read-only) into ordered Events.

    ``db_path`` points at a Hermes state.db; there is deliberately no default —
    which database holds your sessions depends on your install and profile, so
    every spec states it explicitly.
    """
    db_path = Path(os.path.expanduser(str(db_path)))
    if not db_path.exists():
        raise FileNotFoundError(f"state.db not found: {db_path}")

    where = "session_id = ?"
    params: list = [session_id]
    if active_only:
        where += " AND active = 1"

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"SELECT role, content, tool_calls, tool_name, timestamp "
            f"FROM messages WHERE {where} ORDER BY timestamp, id",
            params,
        ).fetchall()
    finally:
        con.close()

    events = []
    for role, content, tool_calls, tool_name, ts in rows:
        events.append(Event(
            role=role or "",
            content=content or "",
            tool_name=tool_name or "",
            tool_calls=_parse_tool_calls(tool_calls),
            timestamp=ts or 0.0,
        ))
    return events


# -- projections the checks consume ----------------------------------------- #

def tool_invocations(events: list[Event]) -> list[tuple[str, str]]:
    """Every (tool_name, normalized_args) the agent issued, in order."""
    out = []
    for e in events:
        for c in e.tool_calls:
            out.append((c.name, c.args))
    return out


def tool_results(events: list[Event]) -> list[tuple[str, str]]:
    """Every (tool_name, result_content) returned to the agent, in order."""
    return [(e.tool_name, e.content) for e in events if e.role == "tool"]


def assistant_texts(events: list[Event]) -> list[str]:
    """Non-empty assistant narration, in order."""
    return [e.content for e in events if e.role == "assistant" and e.content.strip()]
