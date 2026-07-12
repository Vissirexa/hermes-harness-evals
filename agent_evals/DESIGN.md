# Agent / harness eval tier — design

The model tier (`local_bench`) answers "is the *model* good at single-shot
code-gen under these sampling settings?" It cannot answer "did my *harness*
change break the loop guards / convergence / fetch behaviour?", because none of
that fires in one shot. This tier answers the second question.

## The unit: a normalized transcript

Every check reads a `list[Event]` (see `transcript.py`): assistant narration,
the tool calls the agent issued (name + normalized args), and the tool results
it got back. That representation is deliberately source-agnostic, so the same
checks can run over any input mode that produces events.

## Input mode: recorded session

`source.type: recorded_session` loads an existing run straight from a Hermes
state.db by `session_id` (the spec names the db explicitly — no assumed
install layout). This is the regression workflow:

1. Change something in the harness (a guard, a steer, a config value).
2. Run your scenario through Hermes as normal.
3. Grab the new `session_id` and point a spec at it.
4. `python -m agent_evals.runner` asserts the guards still held.

Tune the check thresholds to mirror your live guard config (e.g. if your
harness hard-stops after 5 repeated results, spec `repeated_result: max: 5`) —
then a passing spec is positive evidence the guard did its job, and a
regression that lets a loop through surfaces here as a threshold breach.

The repetition checks all key on something being identical across calls, so
they're blind to a model that mutates the URL slug on every retry while
staying on one host — each call is unique, each result is a fresh failure
body, and nothing repeats verbatim for them to catch. `domain_failure` closes
that gap by classifying success/failure from the *result payload* (blocked,
an error, or an HTTP status >= 400 counts as a failure; a 2xx/3xx resets that
host's streak) and tracking the largest streak per host rather than per call.
Set its threshold to mirror your live per-domain budget guard — e.g. a
hard-stop of 6 — the same way you would for any other check here.

`session_turns` measures a different dimension: not a loop, but plain
longevity — the number of assistant turns (model invocations) in a session.
A turn is one model reply whether it carried ten tool calls, one, or none, so
this is distinct from `total_tool_calls`, and it is the O(turns) multiplier
that per-turn and per-task harness state scales against — the regime a
long-running process spends most of its memory in. Like `total_tool_calls`
it measures length, so a breach is not itself a defect: a legitimately long
session runs long. It is therefore opt-in (never a nightly_audit default,
which would flag every long-but-fine session) and used either as a generous
ceiling in a `--checks-file` or, via `expect: fail`, to pin a real long-run
session as a standing witness of that regime.

### Pinned known-bad fixtures (`expect: fail`)

A spec may set `expect: fail` to pin a *known-bad* session as a permanent
fixture: the spec passes while the breach is still visible in the transcript,
and starts failing if the breach "disappears" (wrong session id, rewritten
history). This keeps regression examples honest — the eval suite itself
notices if its evidence goes stale. Example of a real breach this catches:
one recorded session has a narration line repeated 4× against a max of 3.

## Input mode: live agent-drive (`agent_evals/live.py`)

`source.type: live` drives Hermes for real: `drive_live()` runs the scenario
through `hermes -z/--oneshot` (headless: tools/memory load as normal, approvals
auto-bypassed, prints only the final text), then reads the resulting transcript
back through the *same* `load_transcript` path and runs the checks.

`hermes -z` deliberately does not print a session id, so the driver brackets
the run by time: it records the latest session start before, runs, then takes
the newest session created after. A timeout is treated as a finding (the run
never converged) and the partial session is still captured if one exists.

Because live specs actually invoke Hermes (and can take minutes), they live in
`agent_evals/specs/live/` and are NOT picked up by the default
`python -m agent_evals.runner`. Run them explicitly:

```
python -m agent_evals.runner agent_evals/specs/live/coding_smoke.yaml
```

## Input mode: control-surface simulation (`agent_evals/control_surface.py`)

`source.type: control_surface_sim` drives the Telegram adapter's deterministic
dispatch layer (quick-keyboard labels, `qa:` quick-action callbacks, reaction
commands) in a subprocess under the target install's own venv python, against
the profile config the spec names — no Telegram, no LLM, runs in seconds. Each
trigger carries an `expect:` block (action kind, exact text / substring,
anchoring); every miss becomes an `eval-breach` event counted by the
`control_surface_breach` check. This is the config-drift detector: a feature
enabled in the wrong file, buttons dropped by validation, or an install that
predates the feature all fail loud with an actionable message.

The specs shipped under `agent_evals/specs/` pin real incidents from a
development install. Their session dbs are not shipped, so they SKIP until
pointed at your own recordings — each spec's description says what it pins.

## The research-is-read-only contract (`state_mutation` / `deliverable_missing`)

Asked to *research the provider profiles and write the findings to MD files*,
one session instead started editing the live config and creating a new
profile. "Research X" plus available write tools gets misread as "set up X" —
the harness answer is a system-prompt steer (`research_read_only_guidance`)
plus approval patterns on `hermes config set` / `hermes profile <mutating
verb>`; this tier's answer is the `specs/live/research_readonly_*.yaml`
family asserting the contract end to end.

`state_mutation` counts mutation *attempts* deliberately — a command the
approval layer denied still fails the spec, because the steer under test is
supposed to stop the model from trying, not lean on the approval net. The
mutation definition is mechanical (no LLM judge): config-shaped file writes
outside the spec's `allowed_paths`, terminal commands matching the mutating
patterns (the hermes config/profile regexes mirror the harness's approval.py
guard — keep them in sync), and mutating HTTP verbs aimed at localhost (the
oMLX admin API case, which no file or terminal guard covers).
`deliverable_missing` closes the other half: a run that mutates nothing but
also never writes the requested report is not a pass.

The control spec (`research_readonly_control.yaml`) reuses `expect: fail` to
invert the check: an explicit "create a profile" ask must still *attempt* the
mutation, so over-steering — refusing legitimate change requests, or writing
a plan document instead of acting — fails loud. Run each spec several times
per configuration (local-model variance is high), once with the steer on for
the regression bar and once with `agent.research_read_only_guidance: false`
to record the unsteered baseline and make the steer's effect size visible.

## Adding a check

Add a function in `checks.py` that takes `(events, max_allowed, **kw)` and
returns a `CheckResult`, then register it in the `CHECKS` dict. Keep checks
pure over the transcript so they work for every input mode.
