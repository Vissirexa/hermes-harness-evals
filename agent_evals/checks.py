"""Checks over a normalized transcript.

Each check returns a CheckResult with the measured value and whether it stayed
within the spec's threshold. The repetition checks mirror the loop guards in
the Hermes harness, so an agent-eval that passes is evidence the guards (or
the model) avoided a known failure mode, and a regression in those guards
shows up here as a threshold breach.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .transcript import (
    Event,
    assistant_texts,
    tool_invocations,
    tool_results,
)


@dataclass
class CheckResult:
    name: str
    measured: float
    threshold: float
    passed: bool
    detail: str = ""


def _max_repeat(items: list) -> tuple[int, object]:
    if not items:
        return 0, None
    counts = Counter(items)
    val, n = counts.most_common(1)[0]
    return n, val


def _normalize_text(s: str) -> str:
    s = re.sub(r"<(reasoning|think)[^>]*>.*?</\1>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip().lower()


def repeated_result(
    events: list[Event],
    max_allowed: int,
    min_chars: int = 200,
    exclude_tools: list[str] | None = None,
) -> CheckResult:
    """Largest count of an identical substantial tool result.

    The classic stuck-agent signature: the same fetch/command result coming
    back byte-identical while the agent keeps retrying.

    ``exclude_tools`` exists for tools whose *persisted* result is a constant
    placeholder while the real payload is multimodal — a vision tool that
    stores "Image loaded into your context…" for every distinct screenshot
    would count as repetition even while the agent makes legitimate progress
    through different images (one recorded session shows 6 such identical
    placeholders for 6 different screenshots); the live guard sees the media
    digests and handles the real same-image loop.
    """
    excluded = set(exclude_tools or ("vision_analyze",))
    results = [
        c for t, c in tool_results(events)
        if len(c) >= min_chars and t not in excluded
    ]
    n, sample = _max_repeat(results)
    return CheckResult(
        "repeated_result", n, max_allowed, n <= max_allowed,
        detail="" if n <= max_allowed else f"a {len(str(sample))}-char result repeated {n}x",
    )


def identical_tool_call(events: list[Event], max_allowed: int) -> CheckResult:
    """Largest count of an identical (tool, args) invocation."""
    calls = tool_invocations(events)
    n, sample = _max_repeat(calls)
    return CheckResult(
        "identical_tool_call", n, max_allowed, n <= max_allowed,
        detail="" if n <= max_allowed else f"call {sample[0] if sample else '?'} repeated {n}x",
    )


def repeated_narration(events: list[Event], max_allowed: int, min_chars: int = 40) -> CheckResult:
    """Largest count of an identical assistant sentence (the 'same wall' loop).

    Reasoning/think blocks are stripped and whitespace/case normalized first,
    so models that re-derive the same conclusion in slightly different
    formatting still register as repeating themselves.
    """
    texts = [_normalize_text(t) for t in assistant_texts(events)]
    texts = [t for t in texts if len(t) >= min_chars]
    n, sample = _max_repeat(texts)
    return CheckResult(
        "repeated_narration", n, max_allowed, n <= max_allowed,
        detail="" if n <= max_allowed else f'narration repeated {n}x: "{str(sample)[:60]}..."',
    )


def total_tool_calls(events: list[Event], max_allowed: int) -> CheckResult:
    """Guards against unbounded runs even when no single call/result repeats."""
    n = len(tool_invocations(events))
    return CheckResult("total_tool_calls", n, max_allowed, n <= max_allowed)


_HALLUCINATED_RE = re.compile(r"^Tool '([^']+)' does not exist")


def hallucinated_tool(events: list[Event], max_allowed: int) -> CheckResult:
    """Count of calls to tools that don't exist (usually max 0).

    Catches two distinct regressions at once: the model inventing tool names,
    and a toolset silently collapsing so real tools vanish — in one recorded
    session the web tools were never registered and this check measures 3
    (one invented web-tool call plus two sibling calls the harness skipped in
    the same turn), followed by a long raw-curl thrash. Skipped-sibling
    results count alongside the direct "does not exist" errors.
    """
    names: list[str] = []
    skipped = 0
    for _, content in tool_results(events):
        m = _HALLUCINATED_RE.match(content)
        if m:
            names.append(m.group(1))
        elif content.startswith("Skipped: another tool call in this turn used an invalid name"):
            skipped += 1
    n = len(names) + skipped
    return CheckResult(
        "hallucinated_tool", n, max_allowed, n <= max_allowed,
        detail="" if n <= max_allowed else
        f"nonexistent tools called: {sorted(set(names))} (+{skipped} skipped siblings)",
    )


_WEB_FETCH_TOOLS = ("fetch_resilient", "web_extract", "browser_navigate")


def _registrable_host(url: str) -> str:
    """Lowercased netloc with a leading www. stripped — matches the live guard."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def domain_failure(
    events: list[Event],
    max_allowed: int,
    tools: list[str] | None = None,
) -> CheckResult:
    """Largest per-host streak of failing web fetches (the slug-roulette loop).

    The identical-call guards miss a model that mutates the URL slug on every
    retry while staying on the same host — each call is unique, each result is
    a fresh 404 body, and nothing repeats verbatim. In one recorded session
    this pattern accumulated 34 fetch_resilient 404s, and those results
    carried ``ok: true`` with ``status: 404``, so a check keyed on tool-level
    errors alone would also have stayed quiet. Classify from the result
    payload instead: blocked, an error, or HTTP status >= 400 is a failure; a
    2xx/3xx success resets that host's streak. Fails open on payloads it
    can't parse as JSON, same as the live guard. Mirrors the live
    ``hard_stop_after.domain_failure`` guard in tool_guardrails.py.
    """
    import json as _json

    watched = set(tools or _WEB_FETCH_TOOLS)
    streaks: Counter = Counter()
    peak, peak_host = 0, ""
    for tool, content in tool_results(events):
        if tool not in watched:
            continue
        try:
            payload = _json.loads(content)
        except (ValueError, TypeError):
            continue  # fail open, like the guard
        if not isinstance(payload, dict):
            continue
        host = _registrable_host(
            str(payload.get("final_url") or payload.get("url") or "")
        )
        if not host:
            continue
        status = payload.get("status")
        failed = bool(payload.get("blocked")) or bool(payload.get("error")) or (
            isinstance(status, int) and status >= 400
        )
        if failed:
            streaks[host] += 1
            if streaks[host] > peak:
                peak, peak_host = streaks[host], host
        else:
            streaks[host] = 0
    return CheckResult(
        "domain_failure", peak, max_allowed, peak <= max_allowed,
        detail="" if peak <= max_allowed else
        f"{peak} consecutive failing fetches on {peak_host} without a success",
    )


def tool_result_size_budget(
    events: list[Event],
    max_allowed: int,
    tool_name: str = "terminal",
) -> CheckResult:
    """Total bytes of every ``tool_name`` result in the transcript stays under budget.

    Exists to catch an output-compression hook (e.g. an rtk-style
    ``pre_tool_call`` rewrite) silently regressing back to raw, uncompressed
    tool output. Sums across *every* matching result in the transcript, not
    just one — a scenario that deliberately repeats the same call several
    times accumulates bytes across all of them, so the budget must be sized
    for the whole run, not a single call.
    """
    total = sum(len(c) for t, c in tool_results(events) if t == tool_name)
    return CheckResult(
        "tool_result_size_budget", total, max_allowed, total <= max_allowed,
        detail="" if total <= max_allowed else
        f"{tool_name} results totaled {total} bytes (budget {max_allowed})",
    )


def control_surface_breach(events: list[Event], max_allowed: int) -> CheckResult:
    """Count of breaches emitted by a control-surface simulation source.

    A sim source turns every failed expectation — feature disabled in the
    live config, a button/label not dispatching, wrong action kind, missing
    anchor, sim crash — into an ``eval-breach`` event. Usually ``max: 0``.
    """
    breaches = [e.content for e in events if e.role == "eval-breach"]
    n = len(breaches)
    return CheckResult(
        "control_surface_breach", n, max_allowed, n <= max_allowed,
        detail="" if n <= max_allowed else "; ".join(breaches[:3]),
    )


# Registry: spec `type` -> (function, extra-kwarg names it accepts)
CHECKS = {
    "repeated_result": (repeated_result, ("min_chars", "exclude_tools")),
    "identical_tool_call": (identical_tool_call, ()),
    "repeated_narration": (repeated_narration, ("min_chars",)),
    "total_tool_calls": (total_tool_calls, ()),
    "hallucinated_tool": (hallucinated_tool, ()),
    "domain_failure": (domain_failure, ("tools",)),
    "control_surface_breach": (control_surface_breach, ()),
    "tool_result_size_budget": (tool_result_size_budget, ("tool_name",)),
}


def run_check(events: list[Event], spec: dict) -> CheckResult:
    kind = spec["type"]
    if kind not in CHECKS:
        raise ValueError(f"unknown check type: {kind!r} (have {sorted(CHECKS)})")
    fn, extra = CHECKS[kind]
    kwargs = {k: spec[k] for k in extra if k in spec}
    return fn(events, spec["max"], **kwargs)
