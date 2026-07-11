"""Validate every shipped spec YAML against the schema the runner expects."""

from pathlib import Path

import pytest
import yaml

from agent_evals.checks import CHECKS
from agent_evals.runner import _load_events

REPO_ROOT = Path(__file__).parent.parent
SPEC_PATHS = sorted((REPO_ROOT / "agent_evals" / "specs").glob("**/*.yaml"))


def _extra_kwargs_for(kind: str) -> set[str]:
    _, extra = CHECKS[kind]
    return set(extra) | {"type", "max"}


@pytest.fixture(scope="module")
def specs():
    loaded = []
    for path in SPEC_PATHS:
        data = yaml.safe_load(path.read_text())
        loaded.append((path, data))
    return loaded


def test_at_least_one_spec_found():
    assert SPEC_PATHS, "expected shipped specs under agent_evals/specs/"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_spec_parses_to_dict(path):
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_spec_has_id_and_source(path):
    data = yaml.safe_load(path.read_text())
    assert isinstance(data.get("id"), str) and data["id"]
    source = data.get("source")
    assert isinstance(source, dict)
    assert "type" in source


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_spec_checks_are_well_formed(path):
    data = yaml.safe_load(path.read_text())
    checks = data.get("checks")
    assert isinstance(checks, list) and checks, f"{path} needs a non-empty checks list"
    for check in checks:
        assert isinstance(check, dict)
        kind = check.get("type")
        assert kind in CHECKS, f"{path}: unknown check type {kind!r}"
        assert isinstance(check.get("max"), int), f"{path}: check {kind!r} needs an integer max"
        allowed = _extra_kwargs_for(kind)
        extra_keys = set(check) - allowed
        assert not extra_keys, f"{path}: check {kind!r} has unexpected keys {extra_keys}"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_spec_expect_is_pass_or_fail(path):
    data = yaml.safe_load(path.read_text())
    if "expect" in data:
        assert str(data["expect"]).lower() in ("pass", "fail")


@pytest.mark.parametrize("path", SPEC_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_live_specs_only_live_under_specs_live(path):
    data = yaml.safe_load(path.read_text())
    source = data.get("source", {})
    is_under_live_dir = "specs/live" in str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    if source.get("type") == "live":
        assert is_under_live_dir, f"{path}: type: live specs must live under specs/live/"


@pytest.mark.parametrize(
    "path", [p for p in SPEC_PATHS if "specs/live" in str(p.relative_to(REPO_ROOT)).replace("\\", "/")],
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_live_specs_have_required_fields(path):
    data = yaml.safe_load(path.read_text())
    source = data.get("source", {})
    if source.get("type") != "live":
        return
    assert source.get("db_path"), f"{path}: live source needs a 'db_path'"
    assert source.get("scenario_prompt"), f"{path}: live source needs a 'scenario_prompt'"


def test_spec_ids_are_unique(specs):
    ids = [data.get("id") for _, data in specs]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate spec ids: {dupes}"


# -- runner dispatch errors --------------------------------------------------- #


def test_unknown_source_type_raises_value_error():
    with pytest.raises(ValueError):
        _load_events({"type": "nope"})


def test_live_source_missing_db_path_raises_key_error():
    with pytest.raises(KeyError):
        _load_events({"type": "live", "scenario_prompt": "x"})


def test_live_source_missing_scenario_prompt_raises_key_error():
    with pytest.raises(KeyError):
        _load_events({"type": "live", "db_path": "x.db"})


def test_live_source_nonexistent_db_skips_before_driving():
    """A placeholder db_path must FileNotFoundError (-> SKIP) *before* the
    live drive launches a yolo-approved agent run against the real install."""
    with pytest.raises(FileNotFoundError):
        _load_events({
            "type": "live",
            "db_path": "~/.hermes/profiles/<profile>/state.db",
            "scenario_prompt": "anything",
        })
