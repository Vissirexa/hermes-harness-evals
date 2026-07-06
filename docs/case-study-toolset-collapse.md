# Case study: toolset collapse — 66 tool calls to 2

## The incident (2026-07-01)

A platform toolsets config value that should have been a YAML list arrived as
a JSON string. Toolset resolution didn't error — it silently returned no
web/browser toolsets, so the agent's real fetch tools never registered.

Asked to fetch and summarize a JS-heavy, client-rendered page, the model:

- Hallucinated web-tool names. The shipped `hallucinated_tool` check
  measures 3 on the recorded transcript: one invented tool call plus two
  sibling calls the harness skipped in the same turn.
- Fell back to roughly 30 raw curl attempts that could only ever return the
  page shell.
- Burned a dozen vision calls on manually-supplied screenshots.

66 tool calls in total, and the task still failed. Nothing caught it live:
the config loaded fine, every individual call "succeeded", and no single
guard axis (repeated result, repeated narration, identical call) tripped.

## The fixes (three, harness-side)

1. **Config validation.** Platform toolset config is now validated/coerced to
   a list, so the collapse is impossible to hit silently.
2. **Tool-aware web-fetch steer.** The model is steered to route JS-heavy
   pages through the resilient fetch tool instead of inventing tools or
   shelling out.
3. **Multimodal-aware repetition guard.** A vision loop over near-identical
   screenshots now counts as repetition.

## The re-run (same scenario, after the fixes)

2 tool calls. The resilient fetch returned HTTP 200 but only the site shell
(the page is client-rendered), so the agent retried once with `render=true`.
No connected browser was available, so it reported the blocker honestly with
concrete options — paste the content, or connect a browser — and stopped. No
hallucinated tools, no curl thrash, no vision loop.

## Check-level before/after

Measured by the shipped checks:

| Check | Before (2026-07-01) | After re-run | Spec threshold |
|---|---|---|---|
| hallucinated_tool | **3** | 0 | max 0 |
| total_tool_calls | **66** | 2 | 40 (recorded) / 25 (live) |
| repeated_result | — | 1 | max 5 |
| identical_tool_call | — | 1 | max 8 |
| repeated_narration | — | 1 | max 3 |

## How the regression is pinned

Two specs pin this incident so it can never regress silently:

- [`agent_evals/specs/toolset_collapse_20260701.yaml`](../agent_evals/specs/toolset_collapse_20260701.yaml)
  — recorded, `expect: fail`. It passes only while the breach is still
  visible in the recorded transcript.
- [`agent_evals/specs/live/spa_page_fetch.yaml`](../agent_evals/specs/live/spa_page_fetch.yaml)
  — drives the live scenario shape against a public client-rendered page.

Honest note: the original session db contains unrelated personal content and
is not shipped, so the recorded spec SKIPs unless pointed at a db containing
that session. The live spec is the reproducible half.

## The general lesson

Harness failures compose: a config regression caused tool loss, tool loss
caused hallucination, hallucination caused fallback thrash. Each layer looks
fine in isolation — the config loads, each call "succeeds" — and it's only
the shape of the whole transcript that reveals the failure. Transcript-level
checks over recorded sessions are how you turn an incident like this into a
permanent regression test instead of an anecdote.
