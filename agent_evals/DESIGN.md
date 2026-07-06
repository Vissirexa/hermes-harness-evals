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

### Pinned known-bad fixtures (`expect: fail`)

A spec may set `expect: fail` to pin a *known-bad* session as a permanent
fixture: the spec passes while the breach is still visible in the transcript,
and starts failing if the breach "disappears" (wrong session id, rewritten
history). This keeps regression examples honest — the eval suite itself
notices if its evidence goes stale. Example of a real breach this catches:
one recorded session has a narration line repeated 4× against a max of 3.

Further source types (live agent-driving, deterministic control-surface
simulation) build on the same Event contract and land with their harnesses.

## Adding a check

Add a function in `checks.py` that takes `(events, max_allowed, **kw)` and
returns a `CheckResult`, then register it in the `CHECKS` dict. Keep checks
pure over the transcript so they work for every input mode.
