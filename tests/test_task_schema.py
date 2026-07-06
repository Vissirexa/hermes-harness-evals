"""Validate that every task YAML has required fields."""

from pathlib import Path

import yaml

import pytest

TASKS_DIR = Path(__file__).parent.parent / "tasks"
REQUIRED_FIELDS = {"id", "category", "difficulty", "chunk_size", "language", "title", "prompt", "test_code"}


def _collect_task_files():
    return sorted(TASKS_DIR.rglob("*.yaml"))


def _parametrize_ids(yaml_file):
    return yaml_file.name


@pytest.mark.parametrize("yaml_file", _collect_task_files(), ids=_parametrize_ids)
def test_task_has_required_fields(yaml_file):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    missing = REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"{yaml_file.name} missing fields: {missing}"


@pytest.mark.parametrize("yaml_file", _collect_task_files(), ids=_parametrize_ids)
def test_task_id_is_nonempty(yaml_file):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    assert data["id"], f"{yaml_file.name} has empty id"


def test_task_ids_unique_across_suite():
    ids = {}
    for yaml_file in _collect_task_files():
        with open(yaml_file) as f:
            task_id = yaml.safe_load(f)["id"]
        assert task_id not in ids, (
            f"duplicate task id {task_id!r} in {yaml_file.name} and {ids[task_id]}"
        )
        ids[task_id] = yaml_file.name


@pytest.mark.parametrize("yaml_file", _collect_task_files(), ids=_parametrize_ids)
def test_test_code_is_nonempty(yaml_file):
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    assert data["test_code"].strip(), f"{yaml_file.name} has empty test_code"
