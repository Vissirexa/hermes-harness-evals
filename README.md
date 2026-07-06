# hermes-harness-evals

[![CI](https://github.com/Vissirexa/hermes-harness-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Vissirexa/hermes-harness-evals/actions/workflows/ci.yml)

Eval suite for the [Hermes Agent](https://github.com/NousResearch/hermes-agent)
harness running fully local models (currently Qwen3.6-35B-A3B on Apple
Silicon). Single-shot model benchmarks can tell you whether a model writes
working code; they can't tell you where agents actually break — tool loops,
silent config drift, toolset collapse all live in the harness, not the model,
and they regress silently.

## The case study

A platform toolset config regression once silently collapsed the
web/browser toolsets. Asked to fetch and summarize a JS-heavy,
client-rendered page, the agent hallucinated web-tool names, thrashed through
roughly 30 raw curl attempts and a dozen vision calls — 66 tool calls, and
the task still failed. After the harness fixes, the same scenario converged
in **2 tool calls**: a resilient fetch, a render retry, and an honest
"blocked" report with concrete next steps.

- 66 tool calls, task failed → 2 tool calls, honest and correct
- Three harness-side fixes: config validation, a tool-aware fetch steer, a
  multimodal-aware repetition guard
- The harness-tier evals below exist so that regression can never come back
  silently

Full story, numbers, and the specs that pin it:
[`docs/case-study-toolset-collapse.md`](docs/case-study-toolset-collapse.md).

## Two tiers

Independent of each other:

- **`local_bench/`** — model tier: does the local model write working code at
  all? 36 sandboxed codegen tasks (Python + TypeScript × easy/medium/hard),
  verified by pytest/vitest, runnable against any OpenAI-compatible endpoint.
- **`agent_evals/`** — harness tier: regression evals over normalized agent
  transcripts. A check registry (repeated results, repeated narration,
  identical tool calls, hallucinated tools, total tool calls, ...) runs
  against transcripts loaded from recorded sessions, live Hermes drives, or
  control-surface simulation, via spec YAMLs under `agent_evals/specs/`.

## Quickstart: model tier

```bash
pip install -e ".[dev]"
npm ci --prefix ts_runner        # only needed for the TypeScript tasks
python -m local_bench.cli -m Qwen3.6-35B-A3B-MLX-8bit -u http://127.0.0.1:8000/v1
```

Point `-u/--base-url` at any OpenAI-compatible endpoint (LM Studio, oMLX,
vLLM, llama.cpp server) — it defaults to `http://localhost:11434/v1`. Useful
flags:

- `--language {python,typescript}`, `--tier {easy,medium,hard}`,
  `--category {implement,fix,refactor,extend,test_gen}` — filter which tasks
  run
- `--runs RUNS` — repeat the suite N times for variance measurement
- `--no-think` — disable thinking mode (see
  [`docs/thinking-sweep.md`](docs/thinking-sweep.md) for why this is the
  default recommendation)
- `--extra-body EXTRA_BODY` — merge extra JSON into the request body
- `--timeout` — seconds to wait for a model response
- `--output-dir` — where JSON results land (default `./results`)

Results land as JSON under `./results/` plus a console table. Use the
`compare` subcommand to compare results from multiple runs.

## Quickstart: harness tier

Checks run over a normalized transcript loaded from a Hermes session db. A
minimal spec looks like:

```yaml
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
```

```bash
python -m agent_evals.runner
```

runs every spec under `agent_evals/specs/*.yaml`. Live specs under
`agent_evals/specs/live/` run explicitly (they invoke Hermes and can take
minutes) and need a Hermes Agent install.

The shipped incident specs pin real recorded incidents, but their session
dbs are not shipped — they SKIP until pointed at your own recordings.
`expect: fail` pins known-bad sessions so the regression evidence stays
honest: the spec passes only while the breach is still visible.

Full design notes: [`agent_evals/DESIGN.md`](agent_evals/DESIGN.md).

## Docs

- [`docs/case-study-toolset-collapse.md`](docs/case-study-toolset-collapse.md)
  — the 66-to-2 incident, in full
- [`docs/thinking-sweep.md`](docs/thinking-sweep.md) — the config and
  thinking-mode sweep data behind the current defaults

## Layout

```
local_bench/    model-tier bench harness (cli, client, sandbox, runner, report)
tasks/          36 codegen task definitions (python + typescript × difficulty)
ts_runner/      pinned vitest environment for the TypeScript tasks
agent_evals/    harness-tier eval framework + specs (recorded / live / control-surface)
docs/           case study and config-sweep write-ups
tests/          tests for the framework itself — parsers, task schema, checks, spec schema
```

## Requirements

Python 3.11+. Model tier: any OpenAI-compatible endpoint; TypeScript tasks need
Node 20+ (`npm ci` in `ts_runner/`). Harness tier: replay specs need only a
recorded session; live specs need a Hermes Agent install.

MIT licensed.
