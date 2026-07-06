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
    and a toolset silently collapsing so real tools vanish — one recorded
    session shows 7 web-family hallucinations in a row because the web tools
    were never registered, followed by a long raw-curl thrash. Also counts
    sibling calls the harness skipped because of an invalid name in the same
    turn.
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
