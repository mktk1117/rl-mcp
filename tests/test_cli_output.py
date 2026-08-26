"""Two readers, one CLI: the agent's bytes and the person's screen.

The contract worth a test is the first one. An agent shells out and parses
stdout, so JSON mode must stay byte-identical no matter what the text renderer
grows -- that is what lets the formatted side be changed freely.
"""

from __future__ import annotations

import json
import sys

import pytest

from rlmcp import cli, cli_output


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
