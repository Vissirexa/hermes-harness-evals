# Config and thinking-mode sweep

Iterative tuning of the Hermes `code` profile and an oMLX inference server for
Qwen3.6-35B-A3B coding workloads, measured with this repo's model tier
(`local_bench`).

## TL;DR

We systematically tuned two layers — Hermes agent config and the oMLX
inference server — using the model-tier suite (`local_bench`) to measure real
impact. Key changes:

1. **oMLX sampling:** `temperature 0.2 → 0.7`, `top_p 0.95 → 0.80`, `top_k 0 →
   20` (per HuggingFace Qwen3.6 recommendations). Result: 10/10 pass rate,
   33% faster, 60–94% fewer output tokens — same correctness, much more
   concise code.
2. **Hermes max_turns:** `40 → 50` — more headroom for complex multi-turn
   tasks (`extend-middleware` is a known 300s+ runaway that benefits from
   extra turns).
3. **Thinking disabled:** `enable_thinking: false` at the oMLX server level —
   ~7x faster than thinking mode with identical quality on this suite.
4. **Environment probe off:** `environment_probe: false` — free latency win,
   no quality loss.

Bottom line: the biggest wins came from *not* making the model talk (disable
thinking, tighten sampling), not from exotic decode tricks.

## Settings audit & rationale

Below is a systematic walkthrough of every active setting, why it's set this
way, and what evidence supports it.

### 1. oMLX server — global sampling (`~/.omlx/settings.json`)

| Setting | Value | Why |
|---|---|---|
| `temperature` | **0.7** | HuggingFace recommended for Qwen3.6 instruct. At 0.2 the model was too conservative — verbose, repetitive, slow. At 0.7 it writes concise, direct code. Model-tier evidence: 10/10 pass, 33% faster, 60–94% fewer tokens per task. |
| `top_p` | **0.80** | Tighter nucleus sampling. Combined with `top_k=20`, keeps the model focused on likely tokens — good for code generation where wandering is costly. |
| `top_k` | **20** | Restricts sampling to top-20 tokens. Was 0 (disabled). This was a blind spot — the model was choosing from thousands of tokens at 0.95 top_p, leading to verbose rambling. |
| `repetition_penalty` | 1.0 | No penalty. Correct for code — you don't want to penalize legitimate repetition (variable names, method calls). |
| `max_tokens` | 32,768 | Matches HuggingFace standard recommendation. Sufficient for any single task. |
| `max_context_window` | 32,768 (sampling default) | Not the model's context limit — the Hermes agent/harness runs the model's full native window (262,144 tokens for Qwen3.6-35B-A3B). Very long prefills are where memory pressure bites (prefill-guard rejections observed at ~118GB peaks); the memory-guard tier below handles that. |
| `memory_guard_tier` | `balanced` | Good default. `aggressive` would be safer for memory but might evict the model more often. |
| `preserve_mid_system_cache` | `true` | Critical for prefix-cache stability in the agentic loop. Keeps the system prompt byte-stable across turns. |
| `max_concurrent_requests` | 8 | Default is fine for single-user coding. Lowering to 1 didn't change throughput (oMLX already dedicates full compute to one request). |
| `chunked_prefill` | `false` | Left as-is. Could help with very long prefills at the 262K window, but unvalidated so far. |

### 2. oMLX server — per-model settings (`~/.omlx/model_settings.json`)

| Setting | Value | Why |
|---|---|---|
| `enable_thinking` | **`false`** | The single biggest win. Disabling thinking makes Qwen3.6-35B-A3B faster and far more token-efficient with identical quality on our suite — see the 3-run sweep below for the exact numbers. The model stops writing pages of reasoning before the code. |
| `force_sampling` | `false` | Default. Correct — we want the sampling params above to apply. |
| `turboquant_kv_enabled` | `false` | Not needed. The model fits comfortably in memory at 8-bit. |
| `dflash_in_memory_cache` | `true` | Helps with KV cache reuse across requests. 8GB limit is reasonable. |
| `mtp_enabled` | `false` | Deliberately off. MTP (multi-token prediction) on the unsloth 8-bit checkpoint produces deterministically worse code. Speedup (~1.5x) came at a quality cost. |
| `specprefill_enabled` | `false` | Not applicable for this model type. |

### 3. Hermes agent config (`~/.hermes/profiles/code/config.yaml`)

#### Core model settings

| Setting | Value | Why |
|---|---|---|
| `model.default` | `Qwen3.6-35B-A3B-MLX-8bit` | Our chosen model. MoE (~3B active params), fast decode (~83 tok/s), 8-bit quant. |
| `model.provider` | `lmstudio` | Note: the actual inference server is oMLX, not LM Studio. Hermes uses the `lmstudio` provider type because it doesn't have a built-in `omlx` provider. The `base_url` points to oMLX at `http://127.0.0.1:8000/v1`. |
| `model.base_url` | `http://127.0.0.1:8000/v1` | oMLX inference server. |
| `model.supports_vision` | `true` | Qwen3.6-35B-A3B is genuinely a VLM (has a `vision_tower`). Required for correct image routing. |

#### Agent loop settings

| Setting | Value | Why |
|---|---|---|
| `agent.max_turns` | **`50`** | Increased from 40. `extend-middleware` is a known runaway that can loop for 300s+ with self-correction. 40 was enough for most tasks but 50 gives more headroom for genuinely complex multi-turn work without being the runaway-inducing 90 we had before. |
| `agent.task_completion_guidance` | `true` | Enables a self-verify-and-fix pass. Turning it off cost 3 tasks in benchmarking — it's quality, not waste. |
| `agent.parallel_tool_call_guidance` | `true` | Per-turn guidance text. Candidate for `false` to trim prompt bloat, but not yet validated. |
| `agent.environment_probe` | `false` | Critical optimization. Skips the session-start probe round-trip. Free latency win; kept 14/15 quality. The single best lever in the agent config. |
| `agent.tool_use_enforcement` | `auto` | oMLX serves native OpenAI tools, so `auto` isn't doing expensive fallback. |

#### Context compression

| Setting | Value | Why |
|---|---|---|
| `compression.enabled` | `true` | Necessary for long sessions. |
| `compression.threshold` | 0.5 | Compress when context is 50% full. |
| `compression.target_ratio` | 0.2 | Compress down to 20% of original. Tight for the 35B-A3B model (small active params). |
| `compression.protect_last_n` | 20 | Protect last 20 messages from compression. May be too high for a 35B model's context window. |

#### Tool loop guardrails

| Setting | Value | Why |
|---|---|---|
| `tool_loop_guardrails.hard_stop_after.exact_failure` | 3 | Stop after 3 exact failures. Prevents infinite loops. |
| `tool_loop_guardrails.hard_stop_after.same_tool_failure` | 5 | Stop after 5 failures from the same tool. |
| `tool_loop_guardrails.hard_stop_after.idempotent_no_progress` | 3 | Stop if 3 turns produce no progress. |
| `tool_loop_guardrails.hard_stop_after.repeated_result` | 5 | Stop after 5 repeated tool results. |
| `tool_loop_guardrails.hard_stop_after.assistant_repeat` | 3 | Stop after 3 repeated assistant messages. |

These thresholds are exactly what the harness-tier specs mirror
(`agent_evals/specs/loop_guard_holds.yaml`).

#### Memory

| Setting | Value | Why |
|---|---|---|
| `memory.memory_char_limit` | 3000 | Tight but functional for the code profile. |
| `memory.user_char_limit` | 1375 | User profile size. |

#### Approvals

| Setting | Value | Why |
|---|---|---|
| `approvals.mode` | `smart` | Uses an auxiliary LLM to auto-approve low-risk commands, prompts on high-risk. Better than manual for coding workflow. |

### 4. What we tried that didn't work

| Attempt | Result | Why |
|---|---|---|
| `temperature: 0.2` | Slow, verbose, repetitive | Too conservative — model over-explains, writes hand-holding prose |
| `top_p: 0.95, top_k: 0` | Verbose output | Sampling from too many tokens leads to rambling |
| `thinking: ON` | Much slower, many more tokens | Model writes pages of reasoning before code |
| `max_turns: 90` | 13/15 pass, 31.3s avg | Too many turns — agent rabbit-holes on hard tasks |
| `max_turns: 20` | 10/15 pass, 14.6s avg | Too few turns — legitimate tasks get truncated |
| `max_turns: 40` | 14/15 pass, 40.3s avg | Good balance, but `extend-middleware` still runs 300s+ |
| `max_turns: 50` | Adopted setting | More headroom; validation tracked in next steps below |
| MTP speculative decoding | 1.5x faster but worse code | Deterministically produces different (worse) output on the 8-bit checkpoint |
| Auxiliary model (gemma-4-e4b) | Slower overall | No true parallelism on Apple Silicon; load/evict churn drags the main model |
| `chunked_prefill: true` | Not tested | Could help with large contexts but unvalidated |
| KV hot cache (`16GB`) | No measurable improvement | At these context sizes, the warm path is forward-pass overhead, not I/O |
| `max_concurrent_requests: 1` | No change | oMLX already dedicates full compute to one request |

### 5. Benchmark evidence

**Current run (June 28, 15:22 UTC) — `temp=0.7, top_p=0.80, top_k=20,
thinking=off`:**

| Metric | Value |
|---|---|
| Pass rate | 10/10 (100%) |
| Avg time/task | 30.7 s |
| Median time/task | 20.7 s |
| Total time | 307 s |
| p95 time | 87.7 s |

**vs. previous best (June 23, 09:20 UTC) — `temp=0.2, top_p=0.95, top_k=0,
thinking=off`:**

| Metric | Previous | Current | Change |
|---|---|---|---|
| Pass rate | 10/10 (100%) | 10/10 (100%) | same |
| Avg time/task | 45.9 s | 30.7 s | 33% faster |
| Median time/task | 34.7 s | 20.7 s | 40% faster |
| Total time | 459 s | 307 s | 33% faster |

**Token reduction (representative tasks):**

| Task | Old tokens | New tokens | Reduction |
|---|---|---|---|
| impl-from-tests | 10,008 | 897 | 91% less |
| impl-flatten-dict | 2,809 | 168 | 94% less |
| extend-pagination | 5,615 | 1,005 | 82% less |
| fix-state-machine | 3,127 | 448 | 86% less |
| refactor-to-dataclass | 2,517 | 401 | 84% less |

The sampling numbers above (10/10) come from the 10-task Python tier that the
sweep used at the time. The `max_turns` and environment-probe evidence
(13/15, 14/15, etc.) come from the 15-task suite that was current during this
round of tuning.

**Suite-evolution honesty note:** these sweeps were run on the original
15-task suite, which passed 15/15 under the final config; the task suite has
since expanded to 36 tasks (33/36 under the same config at last run).

## Thinking ON vs OFF: 3-run sweep

To remove single-run noise, the headline thinking comparison was re-run 3x
per cell (ON forced via `--extra-body
'{"chat_template_kwargs":{"enable_thinking":true}}'`, OFF via `--no-think`).
Per-task averages over all 3 runs:

| Tier | Mode | Pass (each of 3 runs) | Avg latency/task | Tokens/run |
|---|---|---|---|---|
| Easy (10) | ON | 10/10, 10/10, 10/10 | 48.9 s | 38,756 |
| Easy (10) | OFF | 10/10, 10/10, 10/10 | **8.2 s** | **6,625** |
| Senior (5) | ON | 5/5, 5/5, 5/5 | 65.2 s | 26,975 |
| Senior (5) | OFF | 5/5, 5/5, 5/5 | **7.2 s** | **2,699** |
| **All 15** | ON | **15/15 x3** | 54.3 s | 65,731 |
| **All 15** | OFF | **15/15 x3** | **7.9 s** | **9,324** |

Net: approximately 6.9x faster, approximately 7.0x fewer tokens, zero quality
loss — OFF scored 15/15 on all three runs. One earlier single run had scored
14/15 (a recursive-descent-parser task at the non-thinking capability
boundary); it did not recur across the 3-run sweep. The safety net for harder
out-of-suite tasks is escalation: retry a failed task with thinking on.

## Next steps / open questions

1. **Validate `max_turns: 50`** — run the suite again with the new turn limit
   to confirm no regressions and better handling of the `extend-middleware`
   runaway.
2. **Test `presence_penalty`** — oMLX doesn't expose this in config, but it's
   part of the OpenAI-compatible API. Could be passed via a Hermes provider
   `extra_body`.
3. **Consider `chunked_prefill: true`** — for very long prefills at the
   262K window, chunked prefill could help avoid prefill-guard errors.
4. **Re-validate after Hermes config regen** — Hermes auto-regenerates
   `config.yaml` on version updates. Always back up first.

This is a living log — update it as configs change, and re-run the
model-tier suite to track progress.
