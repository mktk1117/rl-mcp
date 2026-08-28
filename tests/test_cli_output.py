"""Two readers, one CLI: the agent's bytes and the person's screen.

The contract worth a test is the first one. An agent shells out and parses
stdout, so JSON mode must stay byte-identical no matter what the text renderer
grows -- that is what lets the formatted side be changed freely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytest

from rlmcp import cli, cli_output
from rlmcp.session import Response, Session


# Mode resolution: a pipe is an agent, a terminal is a person.


class _Tty:
  """A stdout that claims to be a terminal."""

  def __init__(self, tty: bool):
    self._tty = tty

  def isatty(self) -> bool:
    return self._tty

  def write(self, text: str) -> int:  # pragma: no cover - not exercised
    return len(text)


@pytest.mark.parametrize("tty, expected", [(False, "json"), (True, "text")])
def test_mode_follows_whether_stdout_is_a_terminal(monkeypatch, tty, expected):
  monkeypatch.delenv("RLMCP_OUTPUT", raising=False)
  monkeypatch.setattr(sys, "stdout", _Tty(tty))
  assert cli_output.resolve_mode() == expected


def test_explicit_flag_and_env_override_the_guess(monkeypatch):
  monkeypatch.setattr(sys, "stdout", _Tty(True))
  assert cli_output.resolve_mode("json") == "json"
  monkeypatch.setenv("RLMCP_OUTPUT", "json")
  assert cli_output.resolve_mode() == "json"
  # An explicit flag still wins over the environment.
  assert cli_output.resolve_mode("text") == "text"


def test_a_closed_stdout_falls_back_to_json(monkeypatch):
  monkeypatch.delenv("RLMCP_OUTPUT", raising=False)
  monkeypatch.delattr(sys, "stdout")
  assert cli_output.resolve_mode() == "json"


# The agent's contract: piped bytes do not move.


PAYLOADS = [
    {"ok": True, "result": {"image_path": "/tmp/x.png", "metrics": ["a", "b"]}},
    {"ok": False, "error": "no such parameter", "hint": "run rlmcp params"},
    {"count": 2, "records": [{"id": "001", "verdict": "supported"}]},
    [{"session": "/logs/a", "iteration": 3}, {"session": "/logs/b", "iteration": 9}],
    {"ok": True, "result": None, "error": None},
    {"nested": {"deep": {"deeper": [1, 2.5, None, True]}}},
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_json_mode_emits_exactly_what_it_always_has(payload, capsys, monkeypatch):
  monkeypatch.setattr(cli, "_MODE", "json")
  cli._emit(payload, command="screenshot")
  assert capsys.readouterr().out == json.dumps(payload, indent=2, default=str) + "\n"


def test_the_command_name_never_reaches_json_output(capsys, monkeypatch):
  """Bytes must not depend on how the call was spelled."""
  monkeypatch.setattr(cli, "_MODE", "json")
  payload = {"ok": True, "result": {"a": 1}}
  cli._emit(payload, command="get_metrics")
  named = capsys.readouterr().out
  cli._emit(payload)
  assert capsys.readouterr().out == named


# Text mode: presentation changes, content does not.


def test_long_prose_wraps_rather_than_truncating():
  hypothesis = " ".join(f"word{i}" for i in range(80))
  text = cli_output.render_generic({"hypothesis": hypothesis}, width=60)
  assert "..." not in text
  # Every word survives the wrap.
  assert all(f"word{i}" in text for i in range(80))
  assert max(len(line) for line in text.splitlines()) <= 60


def test_a_path_is_not_broken_across_lines():
  path = "/logs/rsl_rl/2026-08-24_09-12-51_walk003/rlmcp/status.json"
  text = cli_output.render_generic({"session": path}, width=40)
  assert path in text.replace("\n", "").replace(" ", "") or path in text


def test_the_envelope_is_unwrapped():
  assert cli_output.render_generic({"ok": True, "result": None, "error": None}) == "ok"
  failed = cli_output.render_generic({"ok": False, "error": "boom", "hint": "try x"})
  assert failed.startswith("error: boom")
  assert "try x" in failed


def test_scalar_lists_render_as_aligned_rows():
  text = cli_output.render_generic({"series": [[324, -4.4532], [325, -4.79775]]})
  rows = [r for r in text.splitlines() if "324" in r or "325" in r]
  assert len(rows) == 2
  assert len({len(r) for r in rows}) == 1  # Columns line up.


def test_timestamps_are_not_printed_in_scientific_notation():
  text = cli_output.render_generic({"started_at": 1787588427.474})
  assert "e+" not in text and "2026" in text


def test_an_unknown_command_falls_through_to_the_generic_renderer():
  """Extension verbs the core has never heard of must still be readable."""
  payload = {"ok": True, "result": {"terrains": ["flat", "rough"], "max_level": 4}}
  text = cli_output.render(payload, command="set_terrain_invented_by_a_plugin")
  assert "flat, rough" in text
  assert "max_level" in text


# Artifacts: what gets shown, and what must not be.


def test_only_media_suffixes_are_treated_as_artifacts(tmp_path):
  made = {}
  for name in ("shot.png", "clip.mp4", "tree.html", "model.pt", "report.md"):
    path = tmp_path / name
    path.write_bytes(b"x")
    made[name] = str(path)
  found = cli_output.find_artifacts({
      "image_path": made["shot.png"],
      "video_path": made["clip.mp4"],
      "result": {"path": made["tree.html"]},
      "checkpoint": made["model.pt"],
      "report": made["report.md"],
  })
  assert [p.name for p in found] == ["shot.png", "clip.mp4", "tree.html"]


def test_a_path_that_does_not_exist_is_not_offered(tmp_path):
  assert cli_output.find_artifacts({"image_path": str(tmp_path / "gone.png")}) == []


def test_never_policy_shows_nothing(tmp_path, monkeypatch):
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.setattr(cli_output, "show", lambda p: pytest.fail("must not open"))
  assert cli_output.show_artifacts({"image_path": str(path)}, "never") == ([], [])


def test_json_mode_opens_nothing(tmp_path, capsys, monkeypatch):
  """An agent gets the path; a window must not appear on anyone's screen."""
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.setattr(cli, "_MODE", "json")
  monkeypatch.setattr(cli, "_OPEN", "always")
  monkeypatch.setattr(cli_output, "show", lambda p: pytest.fail("must not open"))
  cli._emit({"ok": True, "result": {"image_path": str(path)}}, command="screenshot")
  assert json.loads(capsys.readouterr().out)["result"]["image_path"] == str(path)


def test_text_mode_notes_what_it_opened(tmp_path, capsys, monkeypatch):
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.setattr(cli, "_MODE", "text")
  monkeypatch.setattr(cli, "_OPEN", "auto")
  monkeypatch.setattr(cli_output, "show", lambda p: "xdg-open")
  cli._emit({"ok": True, "result": {"image_path": str(path)}}, command="screenshot")
  out = capsys.readouterr().out
  assert str(path) in out          # The path is still printed, always.
  assert "opened" in out


def test_an_inline_draw_is_not_announced(tmp_path, capsys, monkeypatch):
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.setattr(cli, "_MODE", "text")
  monkeypatch.setattr(cli_output, "show", lambda p: "inline")
  cli._emit({"ok": True, "result": {"image_path": str(path)}}, command="screenshot")
  assert "opened" not in capsys.readouterr().out


def test_show_declines_when_there_is_no_display(tmp_path, monkeypatch):
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.delenv("DISPLAY", raising=False)
  monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
  monkeypatch.setattr(cli_output, "_kitty_capable", lambda: False)
  monkeypatch.setattr(cli_output.shutil, "which", lambda tool: None)
  assert cli_output.show(path) is None


# The parser seam.


def test_output_and_open_flags_are_mutually_exclusive():
  parser = cli.build_parser()
  assert parser.parse_args(["--json", "status"]).output == "json"
  assert parser.parse_args(["--text", "status"]).output == "text"
  assert parser.parse_args(["--no-open", "shot"]).open_policy == "never"
  assert parser.parse_args(["status"]).output is None
  with pytest.raises(SystemExit):
    parser.parse_args(["--json", "--text", "status"])


def test_a_shelf_of_artifacts_is_listed_rather_than_opened(tmp_path, monkeypatch):
  """Showing a record's history must not put a window per file on the screen."""
  paths = []
  for i in range(5):
    path = tmp_path / f"shot{i}.png"
    path.write_bytes(b"x")
    paths.append(str(path))
  monkeypatch.setattr(cli_output, "show", lambda p: pytest.fail("must not open"))
  shown, held = cli_output.show_artifacts({"assets": paths}, "auto")
  assert shown == []
  assert len(held) == 5


def test_open_always_raises_the_cap(tmp_path, monkeypatch):
  paths = []
  for i in range(5):
    path = tmp_path / f"shot{i}.png"
    path.write_bytes(b"x")
    paths.append(str(path))
  monkeypatch.setattr(cli_output, "show", lambda p: "xdg-open")
  shown, held = cli_output.show_artifacts({"assets": paths}, "always")
  assert len(shown) == 5 and held == []


def test_a_held_back_artifact_is_announced(tmp_path, capsys, monkeypatch):
  path = tmp_path / "shot.png"
  path.write_bytes(b"x")
  monkeypatch.setattr(cli, "_MODE", "text")
  monkeypatch.setattr(cli, "_OPEN", "auto")
  monkeypatch.setattr(cli_output, "show", lambda p: None)  # No viewer available.
  cli._emit({"ok": True, "result": {"image_path": str(path)}}, command="screenshot")
  out = capsys.readouterr().out
  assert str(path) in out and "not opened" in out


# The envelope split: which commands wrap their payload, which emit it bare.
#
# Each command's JSON shape is a contract, but it is one contract *per
# command*, not one shape across the CLI. Commands answered from session files
# emit their payload directly (`sessions` and `events` as a bare list);
# commands that reach the trainer -- or stand in for it offline, like `play`
# and `analyze` -- wrap it in the `{ok, result, error}` envelope; `record`
# subcommands put `ok` beside their own keys instead. README.md states the
# rule next to the parsing-contract paragraph; the table here is what keeps it
# true. A new subcommand fails the walk below until it declares its family.

BARE = "bare"          # top-level object without "ok": the payload itself
LIST = "list"          # top-level JSON list: the payload itself
ENVELOPE = "envelope"  # {"ok": ..., "result": ...} / {"ok": false, "error": ...}

SHAPES = {
    "sessions": LIST,
    "tasks": BARE,         # answered from the registry; there may be no session
    "events": LIST,
    "status": BARE,
    "info": BARE,
    "params": BARE,        # --live flips it to the envelope; pinned below
    "extensions": BARE,    # --available flips it; pinned below
    "record": BARE,        # via `record list`; the record family has its own test
    "help": ENVELOPE,
    "get": ENVELOPE,
    "set": ENVELOPE,
    "reset": ENVELOPE,
    "reset-envs": ENVELOPE,
    "metrics": ENVELOPE,
    "plot": ENVELOPE,
    "shot": ENVELOPE,
    "video": ENVELOPE,
    "view": ENVELOPE,
    "play": ENVELOPE,
    "trace": ENVELOPE,
    "diagnose": ENVELOPE,
    "analyze": ENVELOPE,
    "plot-trace": ENVELOPE,
    "curriculum": ENVELOPE,
    "commands": ENVELOPE,
    "pause": ENVELOPE,
    "resume": ENVELOPE,
    "step-once": ENVELOPE,
    "checkpoint": ENVELOPE,
    "checkpoints": ENVELOPE,
    "load": ENVELOPE,
    "note": ENVELOPE,
    "feedback": ENVELOPE,
    "stop": ENVELOPE,
    "run": ENVELOPE,
    "raw": ENVELOPE,
}

# Arguments that make a command well-formed, as in test_cli_dispatch. The two
# paths that do not exist are deliberate: `play` and `analyze` build their own
# result rather than round-tripping, so their *refusal* is what must carry the
# envelope -- a live success would need a checkpoint and a trace.
_SHAPE_ARGS = {
    "get": ["reward.x.weight"],
    "set": ["reward.x.weight", "-0.2"],
    "run": ["get_status"],
    "raw": ["get_status"],
    "note": ["a note"],
    "feedback": ["they said something"],
    "load": ["/nonexistent/checkpoint.pt"],
    "analyze": ["/nonexistent/trace.npz"],
    "play": ["/nonexistent/checkpoint.pt"],
    "record": ["list"],
}
_REFUSALS = {"play", "analyze"}

# `train` and `serve` hand the line to another program; there is no payload.
_NO_OUTPUT = {"train", "serve"}


@pytest.fixture
def live_session(tmp_path):
  """A session whose trainer looks genuinely alive.

  Liveness is not stubbed: the session carries this test's own pid and a fresh
  heartbeat, the same evidence the CLI weighs for a real run. Only the
  transport is faked, in the test body, so dispatch, `_call` and `_emit` all
  run for real.
  """
  session = tmp_path / "logs" / "run1" / "rlmcp"
  session.mkdir(parents=True)
  now = time.time()
  (session / "session.json").write_text(json.dumps({
      "schema_version": 1, "pid": os.getpid(), "started_at": now,
      "task": "Fake-Task-v0", "num_envs": 4,
  }))
  (session / "status.json").write_text(json.dumps({
      "updated_at": now, "iteration": 7, "num_envs": 4,
  }))
  (session / "metrics.jsonl").write_text(
      json.dumps({"iteration": 7, "t": now, "Train/mean_reward": 1.0}) + "\n")
  (session / "events.jsonl").write_text(
      json.dumps({"t": now, "kind": "note", "text": "hello"}) + "\n")
  (session / "params.json").write_text(json.dumps({
      "reward.x.weight": {"current_value": -0.1, "category": "reward"},
  }))
  return session


def _shape_subcommands():
  sub = next(
      a for a in cli.build_parser()._actions
      if isinstance(a, argparse._SubParsersAction)
  )
  return set(sub.choices) - _NO_OUTPUT


def _run_json(argv, tmp_path, capsys, monkeypatch):
  """Run the CLI as an agent would and parse what landed on stdout."""
  monkeypatch.setenv("RLMCP_OUTPUT", "json")
  monkeypatch.setenv("RLMCP_OPEN", "never")
  monkeypatch.setenv("RLMCP_RECORDS", str(tmp_path / "records"))
  monkeypatch.setattr(
      Session, "call",
      lambda self, cmd, timeout=120.0, **args: Response(
          req_id="pin", ok=True, result={}, error=None),
  )
  code = cli.main(argv)
  return code, json.loads(capsys.readouterr().out)


def test_every_command_declares_its_envelope_family():
  """A command added without a row in SHAPES is a contract nobody wrote down."""
  assert set(SHAPES) == _shape_subcommands()


@pytest.mark.parametrize("command", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_declared_family_is_the_emitted_family(
    command, live_session, tmp_path, capsys, monkeypatch):
  argv = [
      "--session", str(live_session),
      "--root", str(tmp_path),
      "--timeout", "1",
      command,
      *_SHAPE_ARGS.get(command, []),
  ]
  code, payload = _run_json(argv, tmp_path, capsys, monkeypatch)

  family = SHAPES[command]
  if family == LIST:
    assert isinstance(payload, list)
  elif family == BARE:
    assert isinstance(payload, dict) and "ok" not in payload
  elif command in _REFUSALS:
    # The failure half of the envelope: ok is false and the story is in error.
    assert code == 1 and payload["ok"] is False and payload["error"]
  else:
    # A real round-trip emits exactly the three keys, ok true, payload under
    # result. Anything else here means the command did not reach the (faked)
    # trainer -- a dead-session refusal would also carry "ok", so the strict
    # form is what proves the mechanism ran.
    assert code == 0
    assert set(payload) == {"ok", "result", "error"}
    assert payload["ok"] is True


@pytest.mark.parametrize("argv_tail, family", [
    (["params", "--live"], ENVELOPE),
    (["metrics", "--list"], BARE),
    (["metrics", "--offline"], ENVELOPE),
    (["extensions", "--available"], ENVELOPE),
], ids=["params-live", "metrics-list", "metrics-offline", "extensions-available"])
def test_flags_that_move_the_answer_move_the_family(
    argv_tail, family, live_session, tmp_path, capsys, monkeypatch):
  """The flip follows where the answer comes from, so the flip is the pin.

  `--offline` answers from files yet keeps the envelope on purpose: it stands
  in for a trainer round-trip, and its callers parse it as one.
  """
  argv = ["--session", str(live_session), "--timeout", "1", *argv_tail]
  code, payload = _run_json(argv, tmp_path, capsys, monkeypatch)
  assert code == 0
  if family == BARE:
    assert "ok" not in payload
  else:
    assert payload["ok"] is True and "result" in payload


def test_record_subcommands_tag_their_payload_rather_than_wrapping_it(
    tmp_path, capsys, monkeypatch):
  """Listings answer bare; everything else puts `ok` beside its own keys."""
  root = ["record", "--records-root", str(tmp_path / "records")]

  code, listing = _run_json([*root, "list"], tmp_path, capsys, monkeypatch)
  assert code == 0 and "ok" not in listing
  assert set(listing) == {"count", "records"}

  code, made = _run_json(
      [*root, "new", "a-slug", "--falsifier", "it never walks"],
      tmp_path, capsys, monkeypatch)
  assert code == 0 and made["ok"] is True
  assert "result" not in made and "record" in made

  code, checked = _run_json([*root, "check"], tmp_path, capsys, monkeypatch)
  assert code == 0 and "ok" in checked and "result" not in checked
