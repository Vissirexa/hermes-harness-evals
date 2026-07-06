"""Simulate Telegram control-surface triggers against the live install.

Drive the Telegram adapter's control-surface dispatch layer (quick-keyboard
label taps, ``qa:<key>`` quick-action callbacks, emoji reaction commands)
in-process — no Telegram, no LLM — and assert the synthetic events the
adapter would hand the gateway.

What it actually guards:
  * config drift — the spec's triggers must resolve against the *live profile
    config* (the config file the spec points at). Enabling a feature in the
    wrong file, or a malformed button silently dropped at validation, shows up
    as a breach here instead of as a dead button in the chat.
  * dispatch regressions — a label/key/emoji that stops mapping to its
    command/prompt, loses its anchor, or flips action kind.

What it cannot guard (still manual): real Telegram
delivery — reply-markup attach, callback round-trips, reaction updates.

The simulation runs as a subprocess under the live install's own venv python
(the same interpreter the gateway runs), with HERMES_HOME pointed at the
profile, so import surface and config resolution match production exactly.
Auth is pinned to a synthetic allowlisted user — live allowlist wiring is out
of scope here.

Spec source shape:

    source:
      type: control_surface_sim
      install_path: ~/.hermes/hermes-agent                  # your Hermes install
      config_path: ~/.hermes/profiles/<profile>/config.yaml # the profile the gateway runs
      # python: <install_path>/venv/bin/python              (default)
      triggers:
        - { kind: config, expect: { quick_keyboard: true, quick_actions: true,
                                    reaction_commands: true } }
        - { kind: keyboard_label, label: "⏹ Stop",
            expect: { action: command, text: "/stop" } }
        - { kind: quick_action, key: retry, message_id: 424242,
            expect: { action: prompt, anchored: true, contains: "try again" } }
        - { kind: reaction, emoji: "👎",
            expect: { action: command, text: "/stop" } }
        # negative assertion: surface disabled in config -> must dispatch nothing
        - { kind: reaction, emoji: "👎", expect: { dispatch: none } }

Every failed expectation (or a trigger that dispatches nothing, unless the
trigger says ``expect: { dispatch: none }``) becomes an ``eval-breach`` event;
pair with ``{ type: control_surface_breach, max: 0 }``. ``dispatch: none``
inverts the usual rule for a single trigger: dispatching nothing passes, and
any dispatch at all is the breach — for guarding that a disabled surface stays
a no-op.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .transcript import Event

_JSON_MARKER = "===CONTROL_SURFACE_JSON==="

# Runs under the live install's python. Reads the payload from stdin, builds a
# bare adapter over the live profile's telegram config, fires the triggers,
# and prints the resulting events as JSON after the marker line.
_SIM_SCRIPT = r'''
import asyncio, json, os, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

payload = json.loads(sys.stdin.read())
sys.path.insert(0, payload["install"])

import yaml

raw = yaml.safe_load(open(payload["config"], encoding="utf-8")) or {}
tg = raw.get("telegram") or {}
extra = {
    k: tg[k]
    for k in ("reaction_commands", "quick_keyboard", "quick_actions")
    if k in tg
}

# Deterministic auth: a synthetic allowlisted user. Clear the feature env
# overrides so the *config file* decides enablement (that's what we guard).
SIM_USER = str(payload.get("sim_user", "929292"))
os.environ["TELEGRAM_ALLOWED_USERS"] = SIM_USER
for var in ("TELEGRAM_REACTION_COMMANDS", "TELEGRAM_QUICK_KEYBOARD",
            "TELEGRAM_QUICK_ACTIONS", "GATEWAY_ALLOW_ALL_USERS"):
    os.environ.pop(var, None)

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter

config = PlatformConfig(enabled=True, token="sim-token")
config.extra.update(extra)
adapter = object.__new__(TelegramAdapter)
adapter.config = config
adapter._config = config
adapter.platform = Platform.TELEGRAM
adapter._connected = True
adapter._message_handler = None
adapter._bot = SimpleNamespace()
adapter.handle_message = AsyncMock()

out = []

def emit(role, content, tool_name=""):
    out.append({"role": role, "content": content, "tool_name": tool_name})

def breach(msg):
    emit("eval-breach", msg)

def dispatched():
    if not adapter.handle_message.await_args:
        return None
    ev = adapter.handle_message.await_args.args[0]
    adapter.handle_message.reset_mock()
    return ev

def _user():
    return SimpleNamespace(id=int(SIM_USER), is_bot=False, full_name="Sim",
                           username="sim", first_name="Sim")

def _chat():
    return SimpleNamespace(id=616161, type="private", title=None, full_name="Sim")

_CONFIG_FNS = {
    "reaction_commands": "_reaction_commands_config",
    "quick_keyboard": "_quick_keyboard_config",
    "quick_actions": "_quick_actions_config",
}

async def run_trigger(t):
    kind = t["kind"]
    expect = t.get("expect") or {}
    desc = t.get("label") or t.get("key") or t.get("emoji") or kind

    if kind == "config":
        for feature, want in expect.items():
            fn = getattr(adapter, _CONFIG_FNS.get(feature, ""), None)
            if fn is None:
                breach(f"live adapter has no {feature} support "
                       f"(install outdated? pull + restart)")
                continue
            cfg = fn()
            if bool(cfg.get("enabled")) != bool(want):
                breach(f"{feature}.enabled is {cfg.get('enabled')} in the live "
                       f"profile config, expected {want} — check "
                       f"{payload['config']}")
            entries = cfg.get("map") or cfg.get("labels") or cfg.get("keys") or {}
            if want and not entries:
                breach(f"{feature} is enabled but validation left no usable "
                       f"entries — malformed buttons/map in config?")
            emit("system", f"config {feature}: enabled={cfg.get('enabled')} "
                           f"entries={sorted(entries)}")
        return

    if kind == "keyboard_label":
        if not hasattr(adapter, "_maybe_dispatch_quick_keyboard_label"):
            breach("live adapter lacks quick-keyboard support"); return
        msg = SimpleNamespace(
            text=t["label"], chat=_chat(), from_user=_user(), message_id=101,
            date=None, is_topic_message=False, message_thread_id=None,
        )
        if not await adapter._maybe_dispatch_quick_keyboard_label(msg):
            breach(f"keyboard label {t['label']!r} was not intercepted "
                   f"(disabled, or label not in live config)")
            return
    elif kind == "quick_action":
        if not hasattr(adapter, "_handle_quick_action_callback"):
            breach("live adapter lacks quick-actions support"); return
        mid = str(t.get("message_id", 424242))
        q = SimpleNamespace(
            data=f"qa:{t['key']}:{mid}",
            from_user=_user(),
            message=SimpleNamespace(chat=_chat(), chat_id=616161,
                                    message_thread_id=None, date=None),
            answer=AsyncMock(),
        )
        await adapter._handle_quick_action_callback(
            q, q.data, query_chat_id=616161, query_chat_type="private",
            query_thread_id=None, query_user_name="Sim",
        )
        if not adapter.handle_message.await_args and expect.get("dispatch") != "none":
            toast = ""
            if q.answer.await_args:
                toast = (q.answer.await_args.kwargs or {}).get("text", "")
            breach(f"quick action {t['key']!r} dispatched nothing "
                   f"(toast: {toast!r})")
            return
    elif kind == "reaction":
        mr = SimpleNamespace(
            chat=_chat(), user=_user(),
            message_id=int(t.get("message_id", 424242)), date=None,
            old_reaction=(),
            new_reaction=(SimpleNamespace(emoji=t["emoji"]),),
        )
        await adapter._handle_message_reaction(
            SimpleNamespace(message_reaction=mr, update_id=1), None,
        )
    else:
        breach(f"unknown trigger kind {kind!r}"); return

    ev = dispatched()
    if expect.get("dispatch") == "none":
        if ev is not None:
            action = "command" if ev.message_type == MessageType.COMMAND else "prompt"
            breach(f"{kind} {desc!r} dispatched a {action}, expected no dispatch "
                   f"(surface should be disabled)")
        else:
            emit("system", f"{kind} {desc!r} dispatched nothing (as expected)")
        return
    if ev is None:
        breach(f"{kind} {desc!r} dispatched nothing")
        return
    action = "command" if ev.message_type == MessageType.COMMAND else "prompt"
    emit("user", ev.text or "", tool_name=action)
    if "action" in expect and expect["action"] != action:
        breach(f"{kind} {desc!r} dispatched a {action}, expected {expect['action']}")
    if "text" in expect and (ev.text or "") != expect["text"]:
        breach(f"{kind} {desc!r} text {ev.text!r} != expected {expect['text']!r}")
    if "contains" in expect and expect["contains"] not in (ev.text or ""):
        breach(f"{kind} {desc!r} text missing {expect['contains']!r}")
    if expect.get("anchored") and not getattr(ev, "reply_to_message_id", None):
        breach(f"{kind} {desc!r} turn is not anchored to a message")

async def main():
    for t in payload["triggers"]:
        try:
            await run_trigger(t)
        except Exception as e:
            breach(f"{t.get('kind')} trigger crashed: {type(e).__name__}: {e}")

asyncio.run(main())
print("\n" + "===CONTROL_SURFACE_JSON===")
print(json.dumps({"events": out}, ensure_ascii=False))
'''


def simulate_control_surface(source: dict) -> list[Event]:
    """Run the trigger script under the target install's python; return Events."""
    for key in ("install_path", "config_path"):
        if key not in source:
            raise KeyError(f"control_surface_sim source needs {key!r}")
    install = Path(os.path.expanduser(source["install_path"]))
    config_path = Path(os.path.expanduser(source["config_path"]))
    python = Path(os.path.expanduser(
        source.get("python", str(install / "venv" / "bin" / "python"))))
    for p, what in ((install, "hermes install"),
                    (config_path, "profile config"),
                    (python, "install python")):
        if not p.exists():
            raise FileNotFoundError(f"{what} not found: {p}")
    if not source.get("triggers"):
        raise KeyError("control_surface_sim source needs a 'triggers' list")

    payload = {
        "install": str(install),
        "config": str(config_path),
        "triggers": source["triggers"],
        "sim_user": str(source.get("sim_user", "929292")),
    }
    env = dict(os.environ, HERMES_HOME=str(config_path.parent))
    proc = subprocess.run(
        [str(python), "-c", _SIM_SCRIPT],
        input=json.dumps(payload),
        capture_output=True, text=True, env=env, cwd=str(install), timeout=180,
    )

    # Import-time noise (profile-fallback warnings etc.) can precede the JSON,
    # so take only what follows the marker. A crashed sim is a breach, not a
    # SKIP — the whole point is failing loud on a broken live install.
    marker_split = proc.stdout.rsplit(_JSON_MARKER, 1)
    if proc.returncode != 0 or len(marker_split) != 2:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
        return [Event(role="eval-breach",
                      content="simulation failed to run: " + " | ".join(tail))]
    data = json.loads(marker_split[1])
    return [
        Event(role=e.get("role", ""), content=e.get("content", ""),
              tool_name=e.get("tool_name", ""))
        for e in data.get("events", [])
    ]
