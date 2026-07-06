"""Agent/harness eval tier.

Where the model tier (local_bench) measures a raw single-shot model on coding
tasks, this tier evaluates the agent loop itself: runaway tool loops,
repeated-result spirals, repeated narration, hallucinated tools, unbounded
runs.

A check operates on a normalized transcript (a list of ``Event``s) loaded from
a recorded Hermes session db — see DESIGN.md for the model and the workflow.
"""
