from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .client import LLMClient, ModelResponse
from .extractor import extract_primary_code
from .sandbox import ExecutionResult, run_tests

DEFAULT_SYSTEM_PROMPT = """\
You are a senior software engineer executing a coding ticket.
You receive a well-specified ticket with description, context, and requirements.
Respond with ONLY the implementation code in a single markdown code block.
Do not include tests, examples, or explanations outside the code block.
The code must be complete and runnable — include all necessary imports."""

# Map a task's tier to a default solution filename when one isn't specified.
_DEFAULT_SOLUTION = {"python": "solution.py", "typescript": "solution.ts"}

VALID_TIERS = ("easy", "medium", "hard")


@dataclass
class Task:
    id: str
    category: str
    difficulty: str
    chunk_size: str
    language: str
    title: str
    prompt: str
    test_code: str
    tier: str = "medium"
    solution_file: str = "solution.py"
    context_files: dict[str, str] = field(default_factory=dict)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class TaskResult:
    task: Task
    response: ModelResponse
    extracted_code: str
    execution: ExecutionResult

    @property
    def passed(self) -> bool:
        return self.execution.all_passed


def _tier_from(data: dict, yaml_file: Path, tasks_dir: Path) -> str:
    """Resolve the tier from an explicit field, else the parent dir name."""
    if data.get("tier") in VALID_TIERS:
        return data["tier"]
    # Infer from path: tasks/<language>/<tier>/file.yaml
    for part in yaml_file.relative_to(tasks_dir).parts:
        if part in VALID_TIERS:
            return part
    # Back-compat with the old junior/senior difficulty field.
    return {"junior": "easy", "senior": "hard"}.get(data.get("difficulty", ""), "medium")


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Load every *.yaml task under tasks_dir, recursing into subdirectories.

    Layout is tasks/<language>/<tier>/<name>.yaml, but a flat directory of
    yaml files still works (tier inferred from the `tier`/`difficulty` field).
    """
    tasks = []
    for yaml_file in sorted(tasks_dir.rglob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        language = data.get("language", "python")
        tasks.append(Task(
            id=data["id"],
            category=data["category"],
            difficulty=data["difficulty"],
            chunk_size=data["chunk_size"],
            language=language,
            title=data["title"],
            prompt=data["prompt"],
            test_code=data["test_code"],
            tier=_tier_from(data, yaml_file, tasks_dir),
            solution_file=data.get("solution_file", _DEFAULT_SOLUTION.get(language, "solution.py")),
            context_files=data.get("context_files") or {},
            system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        ))
    return tasks


def run_task(
    client: LLMClient,
    task: Task,
    model: str,
    timeout: int = 120,
    max_tokens: int = 4096,
    extra_body: dict | None = None,
) -> TaskResult:
    response = client.complete(
        prompt=task.prompt,
        system=task.system_prompt,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    code = extract_primary_code(response.content, task.language)

    if not code:
        execution = ExecutionResult(
            passed=0, failed=0, errors=1, total=1,
            output="No code extracted from model response",
            return_code=1,
        )
        return TaskResult(task=task, response=response, extracted_code="", execution=execution)

    execution = run_tests(
        code=code,
        test_code=task.test_code,
        solution_file=task.solution_file,
        context_files=task.context_files,
        language=task.language,
    )

    return TaskResult(task=task, response=response, extracted_code=code, execution=execution)
