# hermes-harness-evals

Eval suite for the [Hermes Agent](https://github.com/NousResearch/hermes-agent)
harness running fully local models (currently Qwen3.6-35B-A3B on Apple Silicon).

Two tiers, independent of each other:

- **`local_bench/`** — model tier: does the local model write working code at all?
  Sandboxed codegen tasks, pytest/vitest verified, works against any
  OpenAI-compatible endpoint.
- **`agent_evals/`** — harness tier: regression evals over recorded Hermes agent
  sessions. The interesting failures (tool loops, silent config drift, toolset
  collapse) live in the harness, not the model, and they regress silently —
  single-shot model benchmarks can't catch them.

Early days — structure is being laid down, code landing incrementally.

## Layout (planned)

```
local_bench/    model-tier bench harness
tasks/          codegen task definitions (python + typescript)
agent_evals/    harness-tier eval framework + specs
tests/          tests for the framework itself
```

## Requirements

Python 3.11+. Model tier: any OpenAI-compatible endpoint. Harness tier: replay
specs need only a recorded session; live specs need a Hermes Agent install.

MIT licensed.
