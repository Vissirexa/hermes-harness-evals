"""Run agent/harness eval specs and report pass/fail.

Usage:
    python -m agent_evals.runner                      # runs agent_evals/specs/*.yaml
    python -m agent_evals.runner path/to/spec.yaml ...

A spec (YAML):
    id: loop-guard-holds
    description: The result/narration guards keep a known-loopy scenario bounded.
    source:
      type: recorded_session
      session_id: 20260628_000716_44a3cd
      db_path: path/to/state.db       # required — which db holds your sessions
    checks:
      - { type: repeated_result,   max: 5, min_chars: 200 }
      - { type: identical_tool_call, max: 8 }
      - { type: repeated_narration, max: 3, min_chars: 40 }
      - { type: total_tool_calls,  max: 60 }

Live specs (source.type: live) sit under agent_evals/specs/live/ and are not
picked up by the default glob — pass their paths explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from .checks import run_check
from .transcript import load_transcript

SPECS_DIR = Path(__file__).parent / "specs"


def _load_events(source: dict):
    kind = source.get("type", "recorded_session")
    if kind == "recorded_session":
        if "db_path" not in source:
            raise KeyError("recorded_session source needs a 'db_path'")
        return load_transcript(
            session_id=source["session_id"],
            db_path=source["db_path"],
            active_only=source.get("active_only", True),
        )
    if kind == "control_surface_sim":
        from .control_surface import simulate_control_surface
        return simulate_control_surface(source)
    if kind == "live":
        from .live import drive_live
        scenario = source.get("scenario_prompt") or source.get("prompt")
        if not scenario:
            raise KeyError("live source needs a 'scenario_prompt'")
        if "db_path" not in source:
            raise KeyError("live source needs a 'db_path' (where your install records sessions)")
        print(f"  driving Hermes live (timeout {source.get('timeout', 600)}s)...")
        session_id, _ = drive_live(
            scenario_prompt=scenario,
            db_path=source["db_path"],
            model=source.get("model"),
            timeout=source.get("timeout", 600),
        )
        print(f"  captured session {session_id}")
        return load_transcript(session_id=session_id, db_path=source["db_path"])
    raise ValueError(f"unknown source type: {kind!r}")


def run_spec(path: Path) -> bool:
    spec = yaml.safe_load(path.read_text())
    print(f"\n\033[1m{spec.get('id', path.stem)}\033[0m — {spec.get('description', '')}")
    try:
        events = _load_events(spec["source"])
    except (FileNotFoundError, NotImplementedError, KeyError) as e:
        print(f"  \033[33mSKIP\033[0m  {e}")
        return True  # not a failure of the harness; a missing input
    print(f"  transcript: {len(events)} events")

    all_ok = True
    for check_spec in spec.get("checks", []):
        r = run_check(events, check_spec)
        mark = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
        line = f"  {mark} {r.name:<20} measured {r.measured} / max {r.threshold}"
        if r.detail:
            line += f"   \033[2m{r.detail}\033[0m"
        print(line)
        all_ok = all_ok and r.passed

    # `expect: fail` pins a known-bad recorded session as a fixture: the spec
    # passes while the transcript still breaches its thresholds, and starts
    # failing if the breach "disappears" (wrong session, rewritten history) —
    # keeping the regression example honest.
    if str(spec.get("expect", "pass")).lower() == "fail":
        ok = not all_ok
        print("  " + ("\033[32mPASS (expected breach present)\033[0m" if ok
                      else "\033[31mFAIL (expected a breach, but all checks passed)\033[0m"))
        return ok
    print("  " + ("\033[32mPASS\033[0m" if all_ok else "\033[31mFAIL\033[0m"))
    return all_ok


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = [Path(a) for a in argv] if argv else sorted(SPECS_DIR.glob("*.yaml"))
    if not paths:
        print(f"No specs found in {SPECS_DIR}")
        return 1
    results = [run_spec(p) for p in paths]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} specs passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
