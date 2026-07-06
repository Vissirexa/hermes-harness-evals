# hermes-harness-evals

[![CI](https://github.com/Vissirexa/hermes-harness-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/Vissirexa/hermes-harness-evals/actions/workflows/ci.yml)

Eval suite for the [Hermes Agent](https://github.com/NousResearch/hermes-agent)
harness running fully local models (currently Qwen3.6-35B-A3B on Apple Silicon).

Two tiers, independent of each other:

- **`local_bench/`** — model tier: does the local model write working code at all?
  36 sandboxed codegen tasks (Python + TypeScript × easy/medium/hard), verified by
  pytest/vitest, runnable against any OpenAI-compatible endpoint.
- **`agent_evals/`** — harness tier (landing next): regression evals over recorded
  Hermes agent sessions. The interesting failures (tool loops, silent config drift,
  toolset collapse) live in the harness, not the model, and they regress silently —
  single-shot model benchmarks can't catch them.

## Layout

```
local_bench/    model-tier bench harness (cli, client, sandbox, runner, report)
tasks/          36 codegen task definitions (python + typescript × difficulty)
ts_runner/      pinned vitest environment for the TypeScript tasks
agent_evals/    harness-tier eval framework + specs (in progress)
tests/          tests for the framework itself — parsers, task schema
```

## Running the model tier

```bash
pip install -e ".[dev]"
python -m local_bench.cli --help
```

Point it at any OpenAI-compatible endpoint (LM Studio, oMLX, vLLM, llama.cpp
server). A proper quickstart with example output lands with the docs.

## Requirements

Python 3.11+. Model tier: any OpenAI-compatible endpoint; TypeScript tasks need
Node 20+ (`npm ci` in `ts_runner/`). Harness tier: replay specs need only a
recorded session; live specs need a Hermes Agent install.

MIT licensed.
