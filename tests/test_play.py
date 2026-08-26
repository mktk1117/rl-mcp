"""Playing a finished run: everything that happens before the simulator.

``rlmcp play`` is the one command that builds an environment of its own, so
most of it needs a GPU and a task package to exercise. The half in front of
that does not, and it is the half that decides *what* gets played: which
checkpoint, which session, which task, which packages to import. Getting any of
those wrong produces a clip of the wrong thing, which is worse than no clip.

Nothing here imports a simulator, and every fixture is built in ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from rlmcp.play import (
    PlayConfig,
    PlayError,
    add_arguments,
    checkpoint_iteration,
    config_from_args,
    find_checkpoint,
    packages_to_import,
    run_play,
    session_for,
    task_for,
)


def _run(tmp_path, name: str = "run", task: str = "Example-Task-v0",
         checkpoints=("model_100.pt", "model_final_900.pt")):
  """A finished run: checkpoints in the run directory, a session beside them."""
  run = tmp_path / name
  session = run / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps({"task": task, "pid": 1,
                                                    "started_at": 0.0,
                                                    "schema_version": 1}))
  for stem in checkpoints:
    (run / stem).write_bytes(b"weights")
  return run


# Which checkpoint.


def test_a_run_directory_resolves_to_the_checkpoint_it_ended_on(tmp_path):
  run = _run(tmp_path)

  assert find_checkpoint(run).name == "model_final_900.pt"


def test_the_latest_is_by_iteration_not_by_modification_time(tmp_path):
  """A run whose final checkpoint had an earlier one copied back in beside it
  for comparison should still play the one it ended on."""
  run = _run(tmp_path)
  (run / "model_100.pt").write_bytes(b"copied back in, so newer on disk")

  assert find_checkpoint(run).name == "model_final_900.pt"


def test_pointing_at_the_session_finds_the_run_above_it(tmp_path):
  """The path in hand is usually the session directory -- it is what the
  trainer printed and what $RLMCP_SESSION holds -- not the run."""
  run = _run(tmp_path)

  assert find_checkpoint(run / "rlmcp").name == "model_final_900.pt"


def test_a_checkpoint_saved_through_the_session_is_found_in_its_own_directory(tmp_path):
  run = _run(tmp_path, checkpoints=())
  (run / "rlmcp" / "checkpoints").mkdir()
  (run / "rlmcp" / "checkpoints" / "it000400.pt").write_bytes(b"weights")

  assert find_checkpoint(run / "rlmcp").name == "it000400.pt"


def test_a_named_file_is_taken_as_given(tmp_path):
  run = _run(tmp_path)

  assert find_checkpoint(run / "model_100.pt").name == "model_100.pt"


def test_a_directory_with_no_checkpoints_says_where_it_looked(tmp_path):
  empty = tmp_path / "nothing"
  empty.mkdir()

  with pytest.raises(PlayError) as caught:
    find_checkpoint(empty)

  assert "--checkpoint" in str(caught.value)


def test_a_path_that_is_not_there_is_refused_before_anything_is_built(tmp_path):
  with pytest.raises(PlayError):
    find_checkpoint(tmp_path / "no-such-run")


@pytest.mark.parametrize("stem,expected", [
    ("model_4300", 4300), ("model_final_4375", 4375), ("it000400", -1),
    ("policy", -1), ("model-900", 900),
])
def test_the_iteration_is_read_out_of_the_checkpoint_name(tmp_path, stem, expected):
  assert checkpoint_iteration(tmp_path / f"{stem}.pt") == expected


# Which session, and which task.


def test_a_checkpoint_finds_the_session_beside_it(tmp_path):
  run = _run(tmp_path)

  assert session_for(run / "model_final_900.pt") == run / "rlmcp"


def test_a_checkpoint_saved_inside_the_session_finds_it_too(tmp_path):
  run = _run(tmp_path, checkpoints=())
  (run / "rlmcp" / "checkpoints").mkdir()
  saved = run / "rlmcp" / "checkpoints" / "it000400.pt"
  saved.write_bytes(b"weights")

  assert session_for(saved) == run / "rlmcp"


def test_a_checkpoint_with_no_session_anywhere_near_it_has_none(tmp_path):
  loose = tmp_path / "model_900.pt"
  loose.write_bytes(b"weights")

  assert session_for(loose) is None
  assert task_for(None) == ""


def test_an_unreadable_session_file_reads_as_no_task(tmp_path):
  """Half a session.json is not a reason to raise; it is a reason to ask for
  --task."""
  session = tmp_path / "rlmcp"
  session.mkdir()
  (session / "session.json").write_text("{torn")

  assert task_for(session) == ""


def test_a_checkpoint_whose_task_cannot_be_told_asks_for_it(tmp_path):
  """This refusal happens before any simulator import, which is why it is
  testable here at all -- and why it is fast."""
  loose = tmp_path / "model_900.pt"
  loose.write_bytes(b"weights")

  with pytest.raises(PlayError) as caught:
    run_play(PlayConfig(checkpoint=str(loose)))

  assert "--task" in str(caught.value)


def test_an_unknown_mode_is_refused_by_name(tmp_path):
  with pytest.raises(PlayError) as caught:
    run_play(PlayConfig(checkpoint=str(tmp_path), mode="hologram"))

  assert "video" in str(caught.value)


# Which packages to import.


def test_task_packages_come_from_the_flag(monkeypatch):
  monkeypatch.delenv("RLMCP_TASK_PACKAGES", raising=False)

  assert packages_to_import(PlayConfig(task_package=["a.tasks"])) == ["a.tasks"]


def test_the_environment_can_say_it_once_for_a_whole_shell(monkeypatch):
  """A project whose tasks always live in the same package should not have to
  pass --task-package to every replay."""
  monkeypatch.setenv("RLMCP_TASK_PACKAGES", " a.tasks , b.tasks ")

  assert packages_to_import(PlayConfig()) == ["a.tasks", "b.tasks"]


def test_a_package_named_twice_is_imported_once(monkeypatch):
  monkeypatch.setenv("RLMCP_TASK_PACKAGES", "a.tasks")

  assert packages_to_import(PlayConfig(task_package=["a.tasks"])) == ["a.tasks"]


def test_nothing_named_anywhere_means_nothing_to_import(monkeypatch):
  monkeypatch.setenv("RLMCP_TASK_PACKAGES", "")

  assert packages_to_import(PlayConfig()) == []


# The command line.


def test_the_arguments_round_trip_into_the_config():
  import argparse

  parser = add_arguments(argparse.ArgumentParser())
  args = parser.parse_args([
      "/runs/one/model_900.pt", "--mode", "native", "--seconds", "3",
      "--device", "cpu", "--stage", "1_harder", "--no-replay",
      "--task-package", "a.tasks", "--task-package", "b.tasks",
      "--allow-partial", "--num-envs", "4",
  ])

  cfg = config_from_args(args, {"reward.a.weight": -0.2})

  assert cfg.checkpoint == "/runs/one/model_900.pt"
  assert (cfg.mode, cfg.seconds, cfg.device) == ("native", 3.0, "cpu")
  assert cfg.stage == "1_harder"
  assert cfg.replay is False
  assert cfg.task_package == ["a.tasks", "b.tasks"]
  assert cfg.allow_partial is True
  assert cfg.num_envs == 4
  assert cfg.overrides == {"reward.a.weight": -0.2}


def test_the_defaults_are_the_common_case():
  """Point at a run, take the last checkpoint, replay what it trained under,
  write a clip. Everything else is a deviation from that."""
  import argparse

  cfg = config_from_args(add_arguments(argparse.ArgumentParser()).parse_args([]), {})

  assert (cfg.mode, cfg.replay, cfg.allow_partial) == ("video", True, False)
  assert cfg.num_envs == 1 and cfg.extra_envs == 0


def test_a_gl_backend_is_chosen_per_mode_and_never_overrides_one_already_set(
    monkeypatch):
  """MuJoCo reads MUJOCO_GL once, at import. Offscreen wants EGL, which needs
  no display; the native viewer wants a window, and EGL would make it fail
  with an error about the frame buffer instead of about the display."""
  import os

  from rlmcp.play import _choose_gl_backend

  monkeypatch.delenv("MUJOCO_GL", raising=False)
  _choose_gl_backend(PlayConfig(mode="video", device="cpu"))
  assert os.environ["MUJOCO_GL"] == "egl"

  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  _choose_gl_backend(PlayConfig(mode="native", device="cpu"))
  assert os.environ["MUJOCO_GL"] == "osmesa"


def test_the_clip_lands_beside_the_runs_other_evidence(tmp_path):
  from rlmcp.play import _default_out

  run = _run(tmp_path)
  out = _default_out(run / "model_final_900.pt", run / "rlmcp",
                     PlayConfig(stage="1_harder"))

  assert out == run / "rlmcp" / "artifacts" / "play_model_final_900_1_harder.mp4"


def test_a_checkpoint_with_no_session_writes_its_clip_next_to_itself(tmp_path):
  from rlmcp.play import _default_out

  loose = tmp_path / "model_900.pt"

  assert _default_out(loose, None, PlayConfig()) == tmp_path / "play_model_900.mp4"


@pytest.mark.parametrize("kind,advice", [
    ("missing_command", "--task-package"),
    ("changed_command", "--allow-partial"),
    ("parameter", "no longer has"),
])
def test_a_failed_restore_advises_the_fix_for_that_particular_failure(kind, advice):
  """Three different problems with three different fixes. A message that
  guesses at the wrong one sends the reader to check something unbroken."""
  from rlmcp.play import _cannot_restore

  message = _cannot_restore({"errors": ["set_mode(): boom"], "error_kinds": [kind]})

  assert advice in message


# The CLI verb.


def test_play_reports_a_missing_checkpoint_as_a_result_not_a_traceback(
    tmp_path, capsys, monkeypatch):
  """An agent parses stdout. A refusal has to arrive as the same envelope
  everything else uses."""
  from rlmcp.cli import main

  monkeypatch.setenv("RLMCP_OUTPUT", "json")
  monkeypatch.setenv("RLMCP_OPEN", "never")
  empty = tmp_path / "nothing"
  empty.mkdir()

  assert main(["play", str(empty)]) == 1

  payload = json.loads(capsys.readouterr().out)
  assert payload["ok"] is False
  assert "checkpoint" in payload["error"]


def test_play_is_registered_with_the_options_that_make_it_useful():
  import argparse

  from rlmcp.cli import build_parser

  sub = next(a for a in build_parser()._actions
             if isinstance(a, argparse._SubParsersAction))
  flags = {s for action in sub.choices["play"]._actions
           for s in action.option_strings}

  assert {"--mode", "--device", "--stage", "--set", "--task-package",
          "--allow-partial", "--no-replay"} <= flags


# Play sessions do not shadow the run they came from.


def test_discovery_skips_play_sessions_but_sessions_lists_them(tmp_path):
  """A play session is newer than the run it replays, so left in the ordering
  it becomes the answer to every bare rlmcp command."""
  from rlmcp.session import PLAY_SESSION_KIND, Session, iter_sessions

  run = _run(tmp_path)
  play = run / "rlmcp" / "play" / "2026-01-01_00-00-00_video"
  play.mkdir(parents=True)
  (play / "session.json").write_text(json.dumps(
      {"kind": PLAY_SESSION_KIND, "task": "Example-Task-v0", "pid": 1,
       "started_at": 100.0, "schema_version": 1}
  ))

  assert Session.find_latest(tmp_path).dir == run / "rlmcp"
  assert len(list(iter_sessions(tmp_path, include_play=True))) == 2
