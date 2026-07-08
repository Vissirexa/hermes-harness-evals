#!/usr/bin/env python3
"""config_drift_check — precedence-aware Hermes config drift detector.

Born from a real incident (2026-07-04): the gateway launchd unit
(`ai.hermes.gateway-code.plist`) sets `HERMES_HOME=~/.hermes/profiles/code`, so
the gateway reads ONLY `~/.hermes/profiles/code/config.yaml`. The base
`~/.hermes/config.yaml` is not read by that unit at all — but nothing says so
anywhere, so `telegram.reaction_commands` was configured in the base file
while the gateway read the profile file, and the feature was silently dead.
A stale duplicate block also lingered in the base file afterward.

lintlang v0.2.2 cannot catch either failure mode: it scans one file at a time
(no cross-file awareness of which file a running unit actually reads), and it
calls `yaml.safe_load()`, which silently collapses duplicate top-level keys
within a single file (last one wins) before any detector runs.

This tool does two things neither lintlang nor a human skim reliably catches:

1. **Cross-file drift** — maps each Hermes-invoking launchd unit to the
   config.yaml its `HERMES_HOME` resolves to, then flags feature keys that
   live only in a file no discovered unit reads (the reaction_commands shape).
2. **In-file duplicate keys** — parses the raw YAML text with the node tree
   (`yaml.compose`), which does NOT collapse duplicates the way
   `yaml.safe_load()` does, and reports every duplicated key path with both
   line numbers.

Read-only: this tool never writes anywhere except stdout, or a file the user
names with --output. It only reads ~/.hermes/**/config.yaml and
~/Library/LaunchAgents/*.plist (both paths are overridable for testing).

Usage:
    python -m tools.config_drift_check
    python -m tools.config_drift_check --json
    python -m tools.config_drift_check --hermes-root /tmp/fixture/.hermes \\
        --launchagents-dir /tmp/fixture/LaunchAgents

Exit codes: 0 clean, 1 findings, 2 tool error (bad input, unreadable paths).
"""
from __future__ import annotations

import argparse
import json
import plistlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HERMES_ROOT = Path.home() / ".hermes"
DEFAULT_LAUNCHAGENTS_DIR = Path.home() / "Library" / "LaunchAgents"

# Sections inspected one level deep for feature-key drift. `telegram` is where
# the reaction_commands incident lived; the others are the other places
# Hermes keeps on/off feature toggles rather than tuning knobs.
WATCHED_SECTIONS: tuple[str, ...] = ("telegram", "plugins", "browser", "memory")

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"

_SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    """A launchd unit that invokes hermes, and the config file it reads."""

    label: str
    plist_path: Path
    hermes_home: Path
    config_path: Path
    had_environment_variables_block: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "plist_path": str(self.plist_path),
            "hermes_home": str(self.hermes_home),
            "config_path": str(self.config_path),
            "had_environment_variables_block": self.had_environment_variables_block,
        }


@dataclass
class Finding:
    severity: str
    kind: str
    path: str
    detail: str
    unit: str | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "path": self.path,
            "unit": self.unit,
            "detail": self.detail,
        }


@dataclass
class Report:
    units: list[Unit] = field(default_factory=list)
    config_files: list[Path] = field(default_factory=list)
    read_files: list[Path] = field(default_factory=list)
    unread_files: list[Path] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "units": [u.to_dict() for u in self.units],
            "config_files": [str(p) for p in self.config_files],
            "read_files": [str(p) for p in self.read_files],
            "unread_files": [str(p) for p in self.unread_files],
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# 1. Launchd unit discovery (read-only; plistlib, no `plutil` subprocess)
# ---------------------------------------------------------------------------


def _invokes_hermes(program_arguments: list) -> bool:
    return any("hermes" in str(arg).lower() for arg in program_arguments)


def discover_units(launchagents_dir: Path, default_hermes_home: Path) -> list[Unit]:
    """Scan ``*.plist`` for units whose ProgramArguments invoke hermes.

    HERMES_HOME comes from EnvironmentVariables; a unit with no
    EnvironmentVariables block at all (or none set) falls back to
    ``default_hermes_home`` — the same default hermes itself uses.
    """
    units: list[Unit] = []
    if not launchagents_dir.is_dir():
        return units

    for plist_path in sorted(launchagents_dir.glob("*.plist")):
        try:
            with plist_path.open("rb") as fh:
                data = plistlib.load(fh)
        except Exception:
            # Not a plist we can parse (corrupt / unrelated file) — skip quietly.
            continue

        program_args = data.get("ProgramArguments") or []
        if not _invokes_hermes(program_args):
            continue

        had_env_block = "EnvironmentVariables" in data
        env = data.get("EnvironmentVariables") or {}
        hermes_home_str = env.get("HERMES_HOME")
        hermes_home = (
            Path(hermes_home_str).expanduser() if hermes_home_str else default_hermes_home
        )
        label = data.get("Label", plist_path.stem)
        units.append(
            Unit(
                label=label,
                plist_path=plist_path,
                hermes_home=hermes_home,
                config_path=hermes_home / "config.yaml",
                had_environment_variables_block=had_env_block,
            )
        )
    return units


# ---------------------------------------------------------------------------
# 2. Config file discovery
# ---------------------------------------------------------------------------


def discover_config_files(hermes_root: Path) -> list[Path]:
    """The base config plus every profile config under ``hermes_root``.

    Deliberately does not recurse into e.g. ``state-snapshots/`` — those are
    point-in-time backups, not live participants in the precedence chain.
    """
    files: list[Path] = []
    base = hermes_root / "config.yaml"
    if base.is_file():
        files.append(base)
    profiles_dir = hermes_root / "profiles"
    if profiles_dir.is_dir():
        for p in sorted(profiles_dir.glob("*/config.yaml")):
            if p.is_file():
                files.append(p)
    return files


def load_config(path: Path) -> dict:
    """The effective (last-key-wins) view of a config file, as Hermes itself sees it."""
    text = path.read_text()
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML is not a mapping (got {type(data).__name__})")
    return data


# ---------------------------------------------------------------------------
# 3. In-file duplicate-key detection
# ---------------------------------------------------------------------------


def find_duplicate_keys(path: Path) -> list[tuple[str, str, int, int]]:
    """Return ``(dotted_path, key, first_line, dup_line)`` for every duplicated
    mapping key in the RAW yaml text (1-indexed lines).

    Uses ``yaml.compose()``, which builds the raw node tree WITHOUT
    constructing Python objects — duplicate keys survive here, unlike in
    ``yaml.safe_load()`` (which silently keeps only the last occurrence).
    That collapse is exactly what a per-file linter that calls safe_load
    cannot see.
    """
    text = path.read_text()
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    dups: list[tuple[str, str, int, int]] = []

    def walk(node, prefix: str) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[str, int] = {}
            for key_node, value_node in node.value:
                key_repr = key_node.value if isinstance(key_node.value, str) else repr(key_node.value)
                line = key_node.start_mark.line + 1
                dotted = f"{prefix}.{key_repr}" if prefix else key_repr
                if key_repr in seen:
                    dups.append((dotted, key_repr, seen[key_repr], line))
                seen[key_repr] = line
                walk(value_node, dotted)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                walk(item, f"{prefix}[{i}]")
        # ScalarNode: leaf, nothing to walk further.

    if root is not None:
        walk(root, "")
    return dups


def find_duplicate_findings(path: Path) -> list[Finding]:
    findings = []
    for dotted, key, line1, line2 in find_duplicate_keys(path):
        findings.append(
            Finding(
                severity=SEVERITY_CRITICAL,
                kind="duplicate_key",
                path=str(path),
                detail=(
                    f"key '{key}' (path '{dotted}') is duplicated at lines {line1} and "
                    f"{line2} — PyYAML's safe_load silently keeps only the line-{line2} "
                    f"value; anything configured under the line-{line1} block is dead"
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# 4. Cross-file drift
# ---------------------------------------------------------------------------


def _section_dict(cfg: dict, section: str) -> dict:
    value = cfg.get(section)
    return value if isinstance(value, dict) else {}


def find_drift(
    read_files: list[Path],
    unread_files: list[Path],
    configs: dict[Path, dict],
    watched_sections: tuple[str, ...] = WATCHED_SECTIONS,
) -> list[Finding]:
    """Flag feature keys living only in a file no discovered unit reads.

    A key present in some ``unread_files`` entry but absent from the same
    top-level section in *every* ``read_files`` entry is the
    reaction_commands incident shape: CRITICAL. A key present in both but
    with a different value is lower-severity drift (INFO) — the value in the
    unread file is inert, but it's evidence of a stale edit worth a look.
    """
    findings: list[Finding] = []
    if not read_files:
        return findings

    read_top_keys: set[str] = set()
    read_section_keys: dict[str, set[str]] = {s: set() for s in watched_sections}
    read_section_values: dict[str, dict[str, list[tuple[Path, Any]]]] = {
        s: {} for s in watched_sections
    }
    for rf in read_files:
        cfg = configs[rf]
        read_top_keys |= set(cfg.keys())
        for s in watched_sections:
            sect = _section_dict(cfg, s)
            read_section_keys[s] |= set(sect.keys())
            for k, v in sect.items():
                read_section_values[s].setdefault(k, []).append((rf, v))

    read_labels = ", ".join(str(p) for p in read_files)

    for uf in unread_files:
        cfg = configs[uf]

        for key in cfg.keys():
            if key not in read_top_keys:
                findings.append(
                    Finding(
                        severity=SEVERITY_WARNING,
                        kind="unread_top_level_key",
                        path=str(uf),
                        detail=(
                            f"top-level key '{key}' is set in {uf}, but no config file "
                            f"actually read by a discovered launchd unit ({read_labels}) "
                            f"has that key at all"
                        ),
                    )
                )

        for s in watched_sections:
            sect = _section_dict(cfg, s)
            if not sect:
                continue
            for k, v in sect.items():
                if k not in read_section_keys[s]:
                    findings.append(
                        Finding(
                            severity=SEVERITY_CRITICAL,
                            kind="configured_in_unread_file",
                            path=str(uf),
                            detail=(
                                f"'{s}.{k}' is set in {uf} but is absent from '{s}:' in "
                                f"every config file a discovered launchd unit reads "
                                f"({read_labels}) — this is the reaction_commands incident "
                                f"shape: the feature is silently dead"
                            ),
                        )
                    )
                else:
                    for rf, rv in read_section_values[s][k]:
                        if rv != v:
                            findings.append(
                                Finding(
                                    severity=SEVERITY_INFO,
                                    kind="differing_value",
                                    path=str(uf),
                                    detail=(
                                        f"'{s}.{k}' = {v!r} in {uf} but = {rv!r} in {rf} "
                                        f"(the file actually read) — likely-stale override, "
                                        f"has no runtime effect"
                                    ),
                                )
                            )
    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_check(
    hermes_root: Path,
    launchagents_dir: Path,
    watched_sections: tuple[str, ...] = WATCHED_SECTIONS,
) -> Report:
    hermes_root = Path(hermes_root).expanduser()
    launchagents_dir = Path(launchagents_dir).expanduser()

    report = Report()
    report.units = discover_units(launchagents_dir, default_hermes_home=hermes_root)

    if not report.units:
        report.notes.append(
            f"no launchd units invoking hermes found under {launchagents_dir}; "
            "cross-file drift check skipped (duplicate-key check still runs on "
            "every discovered config file)"
        )

    config_files = discover_config_files(hermes_root)

    # A unit's HERMES_HOME might point somewhere discover_config_files doesn't
    # walk (e.g. outside hermes_root entirely) — still include it if it exists,
    # so the read/unread classification is accurate.
    for u in report.units:
        if u.config_path.is_file() and u.config_path not in config_files:
            config_files.append(u.config_path)

    configs: dict[Path, dict] = {}
    for f in config_files:
        try:
            configs[f] = load_config(f)
        except Exception as exc:
            report.findings.append(
                Finding(
                    severity=SEVERITY_CRITICAL,
                    kind="parse_error",
                    path=str(f),
                    detail=f"failed to parse as a YAML mapping: {exc}",
                )
            )

    parsed_files = [f for f in config_files if f in configs]
    read_paths = {u.config_path.resolve() for u in report.units}
    read_files = [f for f in parsed_files if f.resolve() in read_paths]
    unread_files = [f for f in parsed_files if f.resolve() not in read_paths]

    report.config_files = parsed_files
    report.read_files = read_files
    report.unread_files = unread_files

    report.findings.extend(
        find_drift(read_files, unread_files, configs, watched_sections=watched_sections)
    )

    for f in parsed_files:
        report.findings.extend(find_duplicate_findings(f))

    for u in report.units:
        if not u.config_path.is_file():
            report.findings.append(
                Finding(
                    severity=SEVERITY_WARNING,
                    kind="missing_config",
                    path=str(u.config_path),
                    unit=u.label,
                    detail=(
                        f"unit '{u.label}' (HERMES_HOME={u.hermes_home}) reads "
                        f"{u.config_path}, but that file does not exist"
                    ),
                )
            )

    report.findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.path, f.kind))
    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_text(report: Report) -> str:
    lines: list[str] = []
    lines.append("config_drift_check report")
    lines.append("=" * 40)

    if report.units:
        lines.append("")
        lines.append("Discovered launchd units:")
        for u in report.units:
            env_note = "" if u.had_environment_variables_block else " (no EnvironmentVariables block; defaulted)"
            lines.append(f"  - {u.label}: HERMES_HOME={u.hermes_home}{env_note}")
            lines.append(f"      plist:  {u.plist_path}")
            lines.append(f"      reads:  {u.config_path}")
    else:
        lines.append("")
        lines.append("Discovered launchd units: none")

    lines.append("")
    lines.append(f"Config files scanned ({len(report.config_files)}):")
    for f in report.read_files:
        lines.append(f"  [READ]   {f}")
    for f in report.unread_files:
        lines.append(f"  [unread] {f}")

    if report.notes:
        lines.append("")
        lines.append("Notes:")
        for n in report.notes:
            lines.append(f"  - {n}")

    lines.append("")
    if not report.findings:
        lines.append("Findings: none. Clean.")
    else:
        lines.append(f"Findings ({len(report.findings)}):")
        for f in report.findings:
            unit_note = f" [unit: {f.unit}]" if f.unit else ""
            lines.append(f"  [{f.severity}] ({f.kind}) {f.path}{unit_note}")
            lines.append(f"      {f.detail}")

    return "\n".join(lines) + "\n"


def format_json(report: Report) -> str:
    return json.dumps(report.to_dict(), indent=2) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config_drift_check",
        description=(
            "Flag Hermes config keys that live only in a file no launchd unit "
            "reads, and in-file duplicate keys PyYAML would silently collapse."
        ),
    )
    parser.add_argument(
        "--hermes-root",
        type=Path,
        default=DEFAULT_HERMES_ROOT,
        help=f"Hermes home to scan for config.yaml files (default: {DEFAULT_HERMES_ROOT})",
    )
    parser.add_argument(
        "--launchagents-dir",
        type=Path,
        default=DEFAULT_LAUNCHAGENTS_DIR,
        help=f"Directory of *.plist launchd units (default: {DEFAULT_LAUNCHAGENTS_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the report to this file instead of stdout (the only file this tool ever writes).",
    )
    parser.add_argument(
        "--watch-section",
        action="append",
        default=None,
        help=(
            "A second-level section to check for feature-key drift (repeatable). "
            f"Default: {', '.join(WATCHED_SECTIONS)}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    watched_sections = tuple(args.watch_section) if args.watch_section else WATCHED_SECTIONS

    try:
        report = run_check(
            hermes_root=args.hermes_root,
            launchagents_dir=args.launchagents_dir,
            watched_sections=watched_sections,
        )
    except Exception as exc:  # tool-level failure, not a "finding"
        print(f"config_drift_check: error: {exc}", file=sys.stderr)
        return 2

    text = format_json(report) if args.json else format_text(report)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")

    return 1 if report.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
