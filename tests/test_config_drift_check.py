"""Tests for tools/config_drift_check.py.

Everything runs against synthetic fixtures under ``tmp_path`` — a fake Hermes
root and a fake LaunchAgents dir built with plistlib. Nothing here reads the
real ~/.hermes or ~/Library/LaunchAgents; the tool's --hermes-root /
--launchagents-dir flags (and the equivalent run_check() kwargs) exist
specifically so tests never touch the user's live config.
"""
from __future__ import annotations

import json
import plistlib
import textwrap
from pathlib import Path

import pytest

from tools.config_drift_check import (
    SEVERITY_CRITICAL,
    discover_units,
    find_duplicate_keys,
    main,
    run_check,
)

GATEWAY_PROGRAM_ARGUMENTS = [
    "/usr/bin/python3",
    "-m",
    "hermes_cli.main",
    "--profile",
    "code",
    "gateway",
    "run",
    "--replace",
]


def _write_plist(path: Path, *, program_arguments: list, environment_variables: dict | None) -> None:
    data = {
        "Label": path.stem,
        "RunAtLoad": True,
        "ProgramArguments": program_arguments,
    }
    if environment_variables is not None:
        data["EnvironmentVariables"] = environment_variables
    with path.open("wb") as fh:
        plistlib.dump(data, fh)


def _base_telegram_yaml(reaction_commands: bool) -> str:
    extra = (
        "  reaction_commands:\n"
        "    thumbsup: \"/approve\"\n"
        "    thumbsdown: \"/reject\"\n"
        if reaction_commands
        else ""
    )
    return (
        "model:\n"
        "  default: test-model\n"
        "telegram:\n"
        "  reactions: false\n"
        f"{extra}"
        "  allowed_chats: ''\n"
    )


@pytest.fixture
def fixture_dirs(tmp_path: Path):
    hermes_root = tmp_path / ".hermes"
    launchagents_dir = tmp_path / "LaunchAgents"
    (hermes_root / "profiles" / "code").mkdir(parents=True)
    launchagents_dir.mkdir(parents=True)
    return hermes_root, launchagents_dir


# ---------------------------------------------------------------------------
# 1. The reaction_commands incident, reproduced
# ---------------------------------------------------------------------------


def test_reaction_commands_incident_shape(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs

    # Base file has the feature; nothing reads the base file.
    (hermes_root / "config.yaml").write_text(_base_telegram_yaml(reaction_commands=True))
    # The profile the gateway actually reads does NOT have it.
    (hermes_root / "profiles" / "code" / "config.yaml").write_text(
        _base_telegram_yaml(reaction_commands=False)
    )
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    report = run_check(hermes_root, launchagents_dir)

    drift = [f for f in report.findings if f.kind == "configured_in_unread_file"]
    assert len(drift) == 1
    finding = drift[0]
    assert finding.severity == SEVERITY_CRITICAL
    assert "reaction_commands" in finding.detail
    assert str(hermes_root / "config.yaml") == finding.path

    # The read/unread classification is the whole point of the tool.
    assert report.read_files == [hermes_root / "profiles" / "code" / "config.yaml"]
    assert report.unread_files == [hermes_root / "config.yaml"]


def test_reaction_commands_incident_cli_exit_code(fixture_dirs, capsys):
    hermes_root, launchagents_dir = fixture_dirs
    (hermes_root / "config.yaml").write_text(_base_telegram_yaml(reaction_commands=True))
    (hermes_root / "profiles" / "code" / "config.yaml").write_text(
        _base_telegram_yaml(reaction_commands=False)
    )
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    exit_code = main(
        ["--hermes-root", str(hermes_root), "--launchagents-dir", str(launchagents_dir), "--json"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert any(f["kind"] == "configured_in_unread_file" for f in payload["findings"])


# ---------------------------------------------------------------------------
# 2. Duplicate-key collapse within a single file
# ---------------------------------------------------------------------------


def test_duplicate_key_collapse_detected(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs

    profile_config = hermes_root / "profiles" / "code" / "config.yaml"
    profile_config.write_text(
        textwrap.dedent(
            """\
            model:
              default: test-model
            telegram:
              reactions: false
              reaction_commands:
                thumbsup: "/approve"
            telegram:
              reactions: false
            """
        )
    )
    (hermes_root / "config.yaml").write_text("model:\n  default: test-model\n")
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    # Unit-level: the raw duplicate-key finder sees both `telegram:` blocks
    # and reports both line numbers.
    dups = find_duplicate_keys(profile_config)
    assert ("telegram", "telegram", 3, 7) in dups

    report = run_check(hermes_root, launchagents_dir)
    dup_findings = [f for f in report.findings if f.kind == "duplicate_key"]
    assert len(dup_findings) == 1
    assert dup_findings[0].severity == SEVERITY_CRITICAL
    assert "lines 3 and 7" in dup_findings[0].detail
    assert dup_findings[0].path == str(profile_config)

    # And because safe_load only sees the *second* `telegram:` block, the
    # reaction_commands set under the first block is invisible at runtime —
    # exactly the silent-collapse failure mode lintlang cannot see.
    from tools.config_drift_check import load_config

    effective = load_config(profile_config)
    assert "reaction_commands" not in effective["telegram"]


# ---------------------------------------------------------------------------
# 3. Clean pass
# ---------------------------------------------------------------------------


def test_clean_pass_no_findings(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs

    identical_yaml = _base_telegram_yaml(reaction_commands=False)
    (hermes_root / "config.yaml").write_text(identical_yaml)
    (hermes_root / "profiles" / "code" / "config.yaml").write_text(identical_yaml)
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    report = run_check(hermes_root, launchagents_dir)
    assert report.findings == []


def test_clean_pass_cli_exit_code_zero(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs
    identical_yaml = _base_telegram_yaml(reaction_commands=False)
    (hermes_root / "config.yaml").write_text(identical_yaml)
    (hermes_root / "profiles" / "code" / "config.yaml").write_text(identical_yaml)
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    exit_code = main(["--hermes-root", str(hermes_root), "--launchagents-dir", str(launchagents_dir)])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# 4. A unit with no EnvironmentVariables block at all
# ---------------------------------------------------------------------------


def test_unit_with_no_environment_variables_defaults_hermes_home(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs

    # No profiles/code config needed for this one — the unit has no
    # EnvironmentVariables block, so HERMES_HOME should default to hermes_root
    # itself (mirroring what hermes does when the launchd unit doesn't set it).
    (hermes_root / "config.yaml").write_text("model:\n  default: test-model\n")
    _write_plist(
        launchagents_dir / "ai.hermes.some-unit.plist",
        program_arguments=["/usr/bin/python3", "-m", "hermes_cli.main", "gateway", "run"],
        environment_variables=None,
    )

    units = discover_units(launchagents_dir, default_hermes_home=hermes_root)
    assert len(units) == 1
    unit = units[0]
    assert unit.had_environment_variables_block is False
    assert unit.hermes_home == hermes_root
    assert unit.config_path == hermes_root / "config.yaml"

    # And the whole pipeline doesn't blow up over it.
    report = run_check(hermes_root, launchagents_dir)
    assert report.read_files == [hermes_root / "config.yaml"]
    assert report.findings == []


def test_non_hermes_plist_is_ignored(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs
    (hermes_root / "config.yaml").write_text("model:\n  default: test-model\n")
    _write_plist(
        launchagents_dir / "com.example.unrelated.plist",
        program_arguments=["/usr/bin/some-other-daemon"],
        environment_variables={"HERMES_HOME": str(hermes_root)},
    )

    units = discover_units(launchagents_dir, default_hermes_home=hermes_root)
    assert units == []


def test_no_units_found_notes_but_still_checks_duplicates(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs
    (hermes_root / "config.yaml").write_text(
        textwrap.dedent(
            """\
            model:
              default: test-model
            model:
              default: other-model
            """
        )
    )
    # No plist files at all in launchagents_dir.

    report = run_check(hermes_root, launchagents_dir)
    assert report.units == []
    assert any("no launchd units" in n for n in report.notes)
    dup_findings = [f for f in report.findings if f.kind == "duplicate_key"]
    assert len(dup_findings) == 1


# ---------------------------------------------------------------------------
# Misc: severity differing_value, missing config file for a discovered unit
# ---------------------------------------------------------------------------


def test_differing_value_reported_as_info_not_critical(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs
    (hermes_root / "config.yaml").write_text(
        "model:\n  default: test-model\nmemory:\n  memory_char_limit: 2200\n"
    )
    (hermes_root / "profiles" / "code" / "config.yaml").write_text(
        "model:\n  default: test-model\nmemory:\n  memory_char_limit: 3000\n"
    )
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    report = run_check(hermes_root, launchagents_dir)
    diffs = [f for f in report.findings if f.kind == "differing_value"]
    assert len(diffs) == 1
    assert diffs[0].severity == "INFO"


def test_missing_config_for_discovered_unit(fixture_dirs):
    hermes_root, launchagents_dir = fixture_dirs
    # profiles/code exists as a dir but has no config.yaml in it.
    _write_plist(
        launchagents_dir / "ai.hermes.gateway-code.plist",
        program_arguments=GATEWAY_PROGRAM_ARGUMENTS,
        environment_variables={"HERMES_HOME": str(hermes_root / "profiles" / "code")},
    )

    report = run_check(hermes_root, launchagents_dir)
    missing = [f for f in report.findings if f.kind == "missing_config"]
    assert len(missing) == 1
    assert missing[0].unit == "ai.hermes.gateway-code"
