import argparse
import sys
from pathlib import Path

from . import __version__
from .client import LLMClient
from .report import console, print_comparison, print_results, save_results
from .runner import load_tasks, run_task


def _find_tasks_dir() -> Path:
    pkg_tasks = Path(__file__).parent.parent / "tasks"
    if pkg_tasks.exists():
        return pkg_tasks
    cwd_tasks = Path.cwd() / "tasks"
    if cwd_tasks.exists():
        return cwd_tasks
    raise FileNotFoundError(
        "Could not find a tasks/ directory. Use --tasks-dir to specify one."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="local-bench",
        description="Opinionated benchmark for local LLMs on chunked coding tasks",
    )
    parser.add_argument("--version", action="version", version=f"local-bench {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser("compare", help="Compare results from multiple runs")
    compare_parser.add_argument("files", nargs="+", help="JSON result files to compare")

    parser.add_argument("--model", "-m", help="Model name as the server knows it")
    parser.add_argument(
        "--base-url", "-u",
        default="http://localhost:11434/v1",
        help="API base URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument("--api-key", default="local", help="API key (default: local)")
    parser.add_argument("--tasks-dir", "-t", type=Path, help="Tasks directory")
    parser.add_argument(
        "--category", "-c",
        choices=["implement", "fix", "refactor", "extend", "test_gen"],
        help="Filter by category",
    )
    parser.add_argument(
        "--chunk-size",
        choices=["small", "medium", "large"],
        help="Filter by chunk size",
    )
    parser.add_argument(
        "--language", "-l",
        choices=["python", "typescript"],
        help="Filter by task language",
    )
    parser.add_argument(
        "--tier",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty tier",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for JSON results (default: ./results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show model responses and test output on failure",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for model response (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max completion tokens per response (default: 4096). "
             "Raise for reasoning/thinking models that emit chain-of-thought.",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable model 'thinking' mode via "
             "chat_template_kwargs={'enable_thinking': false} (Qwen3-style "
             "reasoning models). Much faster; verify quality holds.",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="Extra JSON merged into the request body, e.g. "
             "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'.",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        help="Run the suite N times for variance measurement (default: 1)",
    )

    args = parser.parse_args()

    if args.command == "compare":
        print_comparison(args.files)
        return

    if not args.model:
        parser.error("--model / -m is required")

    tasks_dir = args.tasks_dir
    if tasks_dir is None:
        try:
            tasks_dir = _find_tasks_dir()
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    tasks = load_tasks(tasks_dir)

    if args.category:
        tasks = [t for t in tasks if t.category == args.category]
    if args.chunk_size:
        tasks = [t for t in tasks if t.chunk_size == args.chunk_size]
    if args.language:
        tasks = [t for t in tasks if t.language == args.language]
    if args.tier:
        tasks = [t for t in tasks if t.tier == args.tier]

    if not tasks:
        filters = []
        if args.category:
            filters.append(f"category={args.category}")
        if args.chunk_size:
            filters.append(f"chunk_size={args.chunk_size}")
        if args.language:
            filters.append(f"language={args.language}")
        if args.tier:
            filters.append(f"tier={args.tier}")
        console.print(f"[yellow]No tasks match filters: {', '.join(filters)}[/yellow]")
        sys.exit(1)

    import json as _json
    extra_body: dict = {}
    if args.extra_body:
        try:
            extra_body.update(_json.loads(args.extra_body))
        except _json.JSONDecodeError as e:
            parser.error(f"--extra-body is not valid JSON: {e}")
    if args.no_think:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    client = LLMClient(base_url=args.base_url, api_key=args.api_key)

    for run_num in range(args.runs):
        if args.runs > 1:
            console.print(f"\n[dim]── Run {run_num + 1} of {args.runs} ──[/dim]")
        else:
            console.print()

        console.print(
            f"[bold]local-bench[/bold] — {len(tasks)} tasks against [cyan]{args.model}[/cyan]"
        )
        console.print()

        all_results = []
        for i, task in enumerate(tasks, 1):
            console.print(f"  [{i}/{len(tasks)}] {task.title:<45}", end=" ")
            try:
                result = run_task(
                    client, task, args.model,
                    timeout=args.timeout, max_tokens=args.max_tokens,
                    extra_body=extra_body or None,
                )
            except (ConnectionError, TimeoutError, RuntimeError) as e:
                console.print("[red]ERROR[/red]")
                console.print(f"  [red]{e}[/red]")
                sys.exit(1)

            all_results.append(result)
            symbol = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            console.print(f"{symbol} ({result.response.elapsed_seconds:.1f}s)")

            if args.verbose and not result.passed:
                console.print("\n[dim]--- model response ---[/dim]")
                console.print(result.response.content[:3000])
                console.print("\n[dim]--- test output ---[/dim]")
                console.print(result.execution.output[:3000])
                console.print()

        print_results(all_results, args.model)
        path = save_results(all_results, args.model, args.output_dir)
        console.print(f"Results saved to [dim]{path}[/dim]\n")


if __name__ == "__main__":
    main()
