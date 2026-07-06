import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .runner import TaskResult

console = Console()


def _model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(" ", "_")


def print_results(results: list[TaskResult], model: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = passed / total if total > 0 else 0.0
    color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"

    console.print()
    console.print(Panel(
        f"[bold]{model}[/bold]\n"
        f"Passed: [bold {color}]{passed} / {total} ({rate:.0%})[/bold {color}]",
        title="Results Summary",
        expand=False,
    ))

    table = Table(title="Task Results", box=box.SIMPLE_HEAVY)
    table.add_column("Task", style="bold", min_width=35)
    table.add_column("Category", style="dim")
    table.add_column("Chunk", style="dim")
    table.add_column("Tests", justify="right")
    table.add_column("Result", justify="center")
    table.add_column("Time", justify="right", style="dim")

    for r in results:
        ex = r.execution
        tests_str = f"{ex.passed}/{ex.total}" if ex.total > 0 else "—"
        result_cell = Text("✓ PASS", style="green") if r.passed else Text("✗ FAIL", style="red")
        table.add_row(
            r.task.title,
            r.task.category,
            r.task.chunk_size,
            tests_str,
            result_cell,
            f"{r.response.elapsed_seconds:.1f}s",
        )

    console.print(table)
    _print_breakdown(results, "category", "Category Breakdown")
    _print_breakdown(results, "chunk_size", "Chunk Size Breakdown")

    times = [r.response.elapsed_seconds for r in results]
    if times:
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        console.print(
            f"\n[dim]Timing — avg: {statistics.mean(times):.1f}s  "
            f"median: {statistics.median(times):.1f}s  "
            f"p95: {p95:.1f}s[/dim]"
        )


def _print_breakdown(results: list[TaskResult], attr: str, title: str) -> None:
    buckets: dict[str, dict[str, int]] = {}
    for r in results:
        key = getattr(r.task, attr)
        if key not in buckets:
            buckets[key] = {"passed": 0, "total": 0}
        buckets[key]["total"] += 1
        if r.passed:
            buckets[key]["passed"] += 1

    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column(attr.replace("_", " ").title())
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Rate", justify="right")

    for key, counts in sorted(buckets.items()):
        p, t = counts["passed"], counts["total"]
        rate = p / t if t > 0 else 0.0
        color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
        table.add_row(key, str(p), str(t), f"[{color}]{rate:.0%}[/{color}]")

    console.print(table)


def save_results(results: list[TaskResult], model: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{_model_slug(model)}_{timestamp}.json"

    total = len(results)
    passed = sum(1 for r in results if r.passed)

    by_category: dict[str, Any] = {}
    by_chunk: dict[str, Any] = {}

    for r in results:
        for bucket, key in [(by_category, r.task.category), (by_chunk, r.task.chunk_size)]:
            if key not in bucket:
                bucket[key] = {"passed": 0, "total": 0}
            bucket[key]["total"] += 1
            if r.passed:
                bucket[key]["passed"] += 1

    for bucket in [by_category, by_chunk]:
        for v in bucket.values():
            v["rate"] = v["passed"] / v["total"] if v["total"] > 0 else 0.0

    data = {
        "model": model,
        "timestamp": timestamp,
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0.0,
        },
        "by_category": by_category,
        "by_chunk_size": by_chunk,
        "results": [
            {
                "task_id": r.task.id,
                "title": r.task.title,
                "category": r.task.category,
                "chunk_size": r.task.chunk_size,
                "difficulty": r.task.difficulty,
                "passed": r.passed,
                "tests_passed": r.execution.passed,
                "tests_total": r.execution.total,
                "elapsed_seconds": r.response.elapsed_seconds,
                "prompt_tokens": r.response.prompt_tokens,
                "completion_tokens": r.response.completion_tokens,
                "extracted_code": r.extracted_code,
                "test_output": r.execution.output,
            }
            for r in results
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    return path


def print_comparison(result_files: list[str]) -> None:
    runs = [json.loads(Path(f).read_text()) for f in result_files]

    # Collect all task IDs in first-seen order
    all_task_ids: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for r in run["results"]:
            if r["task_id"] not in seen:
                all_task_ids.append(r["task_id"])
                seen.add(r["task_id"])

    lookups: list[dict[str, dict]] = [
        {r["task_id"]: r for r in run["results"]} for run in runs
    ]
    model_names = [run["model"] for run in runs]

    table = Table(title="Quant Comparison", box=box.SIMPLE_HEAVY)
    table.add_column("Task", min_width=35)
    for name in model_names:
        table.add_column(name, justify="center")

    has_cliff = False
    for task_id in all_task_ids:
        row_results = [lookups[i].get(task_id) for i in range(len(runs))]
        title = next((r["title"] for r in row_results if r), task_id)
        pass_flags = [bool(r and r["passed"]) for r in row_results]
        is_cliff = any(pass_flags) and not all(pass_flags)
        if is_cliff:
            has_cliff = True

        cells = []
        for i, r in enumerate(row_results):
            if r is None:
                cells.append("[dim]—[/dim]")
            elif r["passed"]:
                cells.append(f"[green]✓  {r['tests_passed']}/{r['tests_total']}[/green]")
            else:
                cliff = " [yellow]⚠[/yellow]" if is_cliff and i > 0 and pass_flags[0] else ""
                cells.append(f"[red]✗  {r['tests_passed']}/{r['tests_total']}[/red]{cliff}")

        table.add_row(title, *cells)

    # Summary row
    summary_cells = []
    for run in runs:
        rate = run["summary"]["pass_rate"]
        color = "green" if rate >= 0.8 else "yellow" if rate >= 0.5 else "red"
        summary_cells.append(f"[bold {color}]{rate:.0%}[/bold {color}]")
    table.add_row("[bold]Total Pass Rate[/bold]", *summary_cells)

    console.print(table)
    if has_cliff:
        console.print("[yellow]⚠[/yellow] = cliff: this task passes at a higher quant but fails here")
