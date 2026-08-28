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
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from rlmcp.play import (
    POLICIES,
    PlayConfig,
    PlayError,
    UntrainedPolicy,
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


# Live controls: which stage a checkpoint belongs to.


def _session(tmp_path, name="run", events=(), task="Example-Task-v0"):
  """A run directory with a session and an event log to fold."""
  run = tmp_path / name
  session = run / "rlmcp"
  session.mkdir(parents=True)
  (session / "session.json").write_text(json.dumps(
      {"task": task, "pid": 1, "started_at": 0.0, "schema_version": 1}))
  if events:
    (session / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n")
  return run


def _stage_event(name, iteration, parameters=None):
  return {"t": 0.0, "kind": "curriculum_stage", "to": name, "iteration": iteration,
          "applied": {"parameters": parameters or {}, "calls": []}}


def test_the_stage_a_checkpoint_was_saved_under_is_the_last_one_entered_before_it(
    tmp_path):
  run = _session(tmp_path, events=[
      _stage_event("0_flat", 0), _stage_event("1_rough", 500),
      _stage_event("2_stairs", 2000),
  ])

  from rlmcp.play import stage_at_iteration

  assert stage_at_iteration(run / "rlmcp", 900) == "1_rough"
  assert stage_at_iteration(run / "rlmcp", 500) == "1_rough"
  assert stage_at_iteration(run / "rlmcp", 499) == "0_flat"


def test_a_checkpoint_whose_name_carries_no_iteration_reads_as_the_end_of_the_run(
    tmp_path):
  """`policy.pt` says nothing about when it was saved, and an unnamed
  checkpoint is nearly always the last one."""
  run = _session(tmp_path, events=[_stage_event("0_flat", 0),
                                   _stage_event("1_rough", 500)])

  from rlmcp.play import stage_at_iteration

  assert stage_at_iteration(run / "rlmcp", -1) == "1_rough"


def test_a_run_with_no_curriculum_has_no_stage_to_report(tmp_path):
  from rlmcp.play import stage_at_iteration

  run = _session(tmp_path, events=[{"t": 0.0, "kind": "note", "text": "hello"}])

  assert stage_at_iteration(run / "rlmcp", 100) == ""
  assert stage_at_iteration(None, 100) == ""


# Live controls: swapping the acting policy.


class _Policy:
  """A stand-in for an inference policy: says which checkpoint it came from."""

  def __init__(self, name: str, fits: bool = True):
    self.name = name
    self.fits = fits

  def __call__(self, obs):
    if not self.fits:
      raise RuntimeError("expected 48 observations, got 36")
    return f"{self.name}:{obs}"


@pytest.fixture
def play_lab(tmp_path, fake_sim, fake_terrain):
  """A play session's controller: the real one, over a fake simulator."""
  from rlmcp.core.controller import RlMcp
  from rlmcp.session import PLAY_SESSION_KIND

  controller = RlMcp(
      sim_adapter=fake_sim,
      session_dir=tmp_path / "play-session",
      extensions=[fake_terrain],
      session_info={"kind": PLAY_SESSION_KIND, "task": "Example-Task-v0"},
  )
  yield controller
  controller.close()


def _swap(play_lab, holder, *, stage="", session_dir=None, loads=None, probes=True):
  """Register the play-side swap extension against a controller."""
  from rlmcp.core.replay import apply_conditions
  from rlmcp.play import PolicySwap

  def load(path):
    return (loads or {}).get(path.name) or _Policy(path.stem)

  def probe(policy):
    policy("obs")

  extension = PolicySwap(
      holder=holder,
      load=load,
      session_dir=session_dir,
      stage=stage,
      restore=lambda conditions: apply_conditions(play_lab, conditions),
      probe=probe if probes else None,
  )
  assert play_lab.add_extension(extension)
  return extension


def _holder(tmp_path, name="model_100.pt"):
  from rlmcp.play import SwappablePolicy

  path = tmp_path / name
  path.write_bytes(b"weights")
  return SwappablePolicy(_Policy(path.stem), path)


def test_the_rollout_calls_one_object_that_starts_forwarding_somewhere_else(tmp_path):
  """The viewer is handed the policy once and never told about the swap."""
  holder = _holder(tmp_path)
  assert holder("obs") == "model_100:obs"

  holder.swap(_Policy("model_900"), tmp_path / "model_900.pt")

  assert holder("obs") == "model_900:obs"
  assert holder.checkpoint.name == "model_900.pt"


def test_swapping_the_policy_is_a_command_the_play_session_answers(play_lab, tmp_path):
  """Contributed through the extension registry, so it is reachable from
  `rlmcp run`, from MCP and from a curriculum stage at once."""
  _swap(play_lab, _holder(tmp_path))

  assert "load_policy" in play_lab.cmd_help()["commands"]
  assert "list_policies" in play_lab.cmd_help()["commands"]
  assert play_lab.cmd_status()["extensions"]["play_policy"]["iteration"] == 100


def test_a_swap_loads_the_named_checkpoint_and_says_where_it_came_from(
    play_lab, tmp_path):
  holder = _holder(tmp_path)
  _swap(play_lab, holder)
  later = tmp_path / "model_900.pt"
  later.write_bytes(b"weights")

  result = play_lab.run_command("load_policy", checkpoint=str(later))

  assert result["loaded"] is True
  assert result["checkpoint"] == str(later)
  assert result["iteration"] == 900
  assert result["previous_checkpoint"].endswith("model_100.pt")
  assert holder("obs") == "model_900:obs"


def test_a_swap_is_written_to_the_event_log_like_every_other_steering_action(
    play_lab, tmp_path):
  from rlmcp.session import Session

  _swap(play_lab, _holder(tmp_path))
  (tmp_path / "model_900.pt").write_bytes(b"weights")

  play_lab.run_command("load_policy", checkpoint=str(tmp_path / "model_900.pt"))

  logged = [e for e in Session.open(play_lab.session.dir).events()
            if e["kind"] == "load_policy"]
  assert logged[-1]["checkpoint"].endswith("model_900.pt")


def test_a_checkpoint_that_is_not_there_leaves_the_running_policy_alone(
    play_lab, tmp_path):
  holder = _holder(tmp_path)
  _swap(play_lab, holder)

  with pytest.raises(PlayError):
    play_lab.run_command("load_policy", checkpoint=str(tmp_path / "gone" / "x.pt"))

  assert holder("obs") == "model_100:obs"
  assert holder.swaps == []


def test_weights_that_do_not_fit_this_environment_are_refused_before_the_swap(
    play_lab, tmp_path):
  """A checkpoint whose observation width has moved on loads clean and only
  fails when something asks it to act. Asking here, while the old policy is
  still the one driving, is what keeps a swap from half-applying."""
  holder = _holder(tmp_path)
  wrong = tmp_path / "model_900.pt"
  wrong.write_bytes(b"weights")
  _swap(play_lab, holder, loads={"model_900.pt": _Policy("model_900", fits=False)})

  with pytest.raises(PlayError) as caught:
    play_lab.run_command("load_policy", checkpoint=str(wrong))

  assert "unchanged" in str(caught.value)
  assert "48 observations" in str(caught.value)
  assert holder("obs") == "model_100:obs"


def test_a_swap_within_the_same_rung_reports_a_match_and_says_nothing_more(
    play_lab, tmp_path):
  run = _session(tmp_path, events=[_stage_event("0_flat", 0),
                                   _stage_event("1_rough", 500)])
  (run / "model_900.pt").write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="1_rough", session_dir=run / "rlmcp")

  result = play_lab.run_command("load_policy", checkpoint=str(run / "model_900.pt"))

  assert result["conditions"]["match"] is True
  assert result["conditions"]["same_run"] is True
  assert result["conditions"]["warning"] is None


def test_a_checkpoint_from_another_rung_is_loaded_but_the_conditions_are_not(
    play_lab, tmp_path, fake_sim):
  """The bug core/replay.py exists to prevent, stated in the response instead
  of committed: the weights move, the environment holds still, and the payload
  names both stages so nothing reading it can miss the difference."""
  run = _session(tmp_path, events=[
      _stage_event("0_flat", 0),
      _stage_event("2_stairs", 800, {"reward.action_rate_l2.weight": -0.9}),
  ])
  (run / "model_900.pt").write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="0_flat", session_dir=run / "rlmcp")
  before = fake_sim.get_parameter("reward.action_rate_l2.weight")

  result = play_lab.run_command("load_policy", checkpoint=str(run / "model_900.pt"))

  conditions = result["conditions"]
  assert result["loaded"] is True
  assert conditions["match"] is False
  assert conditions["checkpoint_stage"] == "2_stairs"
  assert conditions["applied_stage"] == "0_flat"
  assert "2_stairs" in conditions["warning"] and "replay=true" in conditions["warning"]
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") == before


def test_replay_puts_the_environment_back_where_that_checkpoint_trained(
    play_lab, tmp_path, fake_sim):
  run = _session(tmp_path, events=[
      _stage_event("0_flat", 0),
      _stage_event("2_stairs", 800, {"reward.action_rate_l2.weight": -0.9}),
  ])
  (run / "model_900.pt").write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="0_flat", session_dir=run / "rlmcp")

  result = play_lab.run_command(
      "load_policy", checkpoint=str(run / "model_900.pt"), replay=True)

  conditions = result["conditions"]
  assert conditions["replayed"] is True
  assert conditions["applied_stage"] == "2_stairs"
  assert conditions["restored_parameters"] == 1
  assert conditions["restore_errors"] == []
  assert fake_sim.get_parameter("reward.action_rate_l2.weight") == -0.9


def test_a_second_swap_compares_against_the_conditions_the_first_one_restored(
    play_lab, tmp_path):
  """Replaying moves where the environment is, so the next swap must be judged
  against the new position rather than the one the session started at."""
  run = _session(tmp_path, events=[_stage_event("0_flat", 0),
                                   _stage_event("2_stairs", 800)])
  (run / "model_900.pt").write_bytes(b"weights")
  (run / "model_400.pt").write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="0_flat", session_dir=run / "rlmcp")

  play_lab.run_command("load_policy", checkpoint=str(run / "model_900.pt"),
                       replay=True)
  second = play_lab.run_command("load_policy", checkpoint=str(run / "model_400.pt"))

  assert second["conditions"]["applied_stage"] == "2_stairs"
  assert second["conditions"]["checkpoint_stage"] == "0_flat"
  assert second["conditions"]["match"] is False


def test_a_checkpoint_with_no_session_beside_it_says_it_cannot_tell(
    play_lab, tmp_path):
  """Unknown is not the same answer as mismatched, and an agent deciding what
  to do next needs the difference."""
  loose = tmp_path / "loose" / "model_900.pt"
  loose.parent.mkdir()
  loose.write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="1_rough")

  result = play_lab.run_command("load_policy", checkpoint=str(loose))

  assert result["loaded"] is True
  assert result["conditions"]["match"] is None
  assert result["conditions"]["checkpoint_stage"] is None
  assert "no way to tell" in result["conditions"]["warning"]


def test_replay_needs_a_run_to_replay_and_refuses_without_one(play_lab, tmp_path):
  holder = _holder(tmp_path)
  loose = tmp_path / "loose" / "model_900.pt"
  loose.parent.mkdir()
  loose.write_bytes(b"weights")
  _swap(play_lab, holder)

  with pytest.raises(PlayError) as caught:
    play_lab.run_command("load_policy", checkpoint=str(loose), replay=True)

  assert "no rlmcp session beside it" in str(caught.value)
  assert holder("obs") == "model_100:obs"


def test_naming_a_stage_without_replay_is_refused_rather_than_ignored(
    play_lab, tmp_path):
  run = _session(tmp_path, events=[_stage_event("0_flat", 0)])
  (run / "model_900.pt").write_bytes(b"weights")
  _swap(play_lab, _holder(tmp_path), stage="0_flat", session_dir=run / "rlmcp")

  with pytest.raises(PlayError):
    play_lab.run_command("load_policy", checkpoint=str(run / "model_900.pt"),
                         stage="0_flat")


def test_a_run_directory_swaps_to_the_checkpoint_that_run_ended_on(
    play_lab, tmp_path):
  """The same resolution `rlmcp play` itself uses, so 'show me that run'
  needs no path to a file."""
  run = _run(tmp_path, name="other")
  _swap(play_lab, _holder(tmp_path))

  result = play_lab.run_command("load_policy", checkpoint=str(run))

  assert result["checkpoint"].endswith("model_final_900.pt")


def test_the_session_lists_what_else_could_be_loaded(play_lab, tmp_path):
  run = _run(tmp_path, name="listed")
  from rlmcp.play import SwappablePolicy

  holder = SwappablePolicy(_Policy("model_100"), run / "model_100.pt")
  _swap(play_lab, holder)

  listed = play_lab.run_command("list_policies")

  assert listed["current"].endswith("model_100.pt")
  names = [(row["path"].split("/")[-1], row["iteration"]) for row in
           listed["checkpoints"]]
  assert ("model_final_900.pt", 900) in names
  assert [row["current"] for row in listed["checkpoints"]].count(True) == 1


# Live controls: stopping cleanly.


def test_a_requested_stop_is_reported_as_a_result_not_as_an_exception(play_lab):
  """`rlmcp stop` against a play session has to end like closing the window:
  the loop unwinds, and what comes back describes the stop."""
  from rlmcp.play import _stop_state
  from rlmcp.core.controller import SessionStopped

  assert _stop_state(play_lab) == ""

  play_lab.run_command("stop_training", reason="seen enough")

  assert play_lab.should_stop()
  assert _stop_state(play_lab) == "seen enough"
  assert _stop_state(play_lab, SessionStopped("unwound out of the viewer")) == (
      "unwound out of the viewer")


def test_a_viewer_that_swallows_the_exception_still_reports_the_stop(play_lab):
  """The reason a session ended must not depend on how thoroughly somebody
  else's loop catches things."""
  from rlmcp.play import _stop_state

  play_lab.run_command("stop_training", reason="")

  assert _stop_state(play_lab) == "stop requested"


def test_closing_a_stopped_play_session_records_its_end(tmp_path, fake_sim):
  from rlmcp.core.controller import RlMcp
  from rlmcp.session import PLAY_SESSION_KIND, Session

  controller = RlMcp(sim_adapter=fake_sim, session_dir=tmp_path / "ending",
                     session_info={"kind": PLAY_SESSION_KIND})
  controller.run_command("stop_training", reason="closed from another shell")
  controller.close()

  ended = [e for e in Session.open(controller.session.dir).events()
           if e["kind"] == "session_end"]
  assert ended and ended[-1]["stop_reason"] == "closed from another shell"


# A task with no policy yet.
#
# `play` was built around a checkpoint, and a checkpoint is what tells it the
# session, the task and the conditions. A task being *written* has none of
# those, and looking at it is the cheapest way to find out that it terminates
# on the first step. These cover the half of that which needs no simulator.


def test_zero_actions_are_zero_and_the_right_shape():
  policy = UntrainedPolicy((4, 12), "cpu", "zero")

  actions = policy()

  assert tuple(actions.shape) == (4, 12)
  assert float(actions.abs().sum()) == 0.0


def test_random_actions_stay_in_the_normalised_span():
  """mjlab action terms scale and offset a normalised action, so [-1, 1] is the
  span a policy would emit -- not a guess at joint limits."""
  policy = UntrainedPolicy((8, 6), "cpu", "random")

  actions = policy()

  assert tuple(actions.shape) == (8, 6)
  assert bool((actions.abs() <= 1.0).all())


def test_an_untrained_policy_needs_a_task():
  """With no checkpoint there is no session to read the task from, so asking
  for one is the only honest thing to do."""
  with pytest.raises(PlayError, match="needs --task"):
    run_play(PlayConfig(policy="zero"))


def test_an_unknown_policy_is_refused_by_name():
  with pytest.raises(PlayError, match="Unknown policy"):
    run_play(PlayConfig(policy="borrowed", task="Some-Task"))


def test_an_untrained_play_never_looks_for_a_checkpoint(tmp_path, monkeypatch):
  """The regression this whole change is about: `find_checkpoint` ran first and
  raised, so a task with no checkpoint could not be opened at all -- whatever
  else was asked for."""
  looked = []
  monkeypatch.setattr("rlmcp.play.find_checkpoint",
                      lambda target: looked.append(target) or (_ for _ in ()).throw(
                          AssertionError("looked for a checkpoint")))
  monkeypatch.setattr("rlmcp.play._choose_gl_backend", lambda cfg: None)
  monkeypatch.setattr("rlmcp.play._build_env",
                      lambda cfg, task, session_dir: (_ for _ in ()).throw(
                          PlayError("stop here: past the checkpoint lookup")))

  with pytest.raises(PlayError, match="stop here"):
    run_play(PlayConfig(policy="zero", task="Some-Task"))

  assert looked == []


def test_the_policy_choice_round_trips_through_the_command_line():
  import argparse

  parser = add_arguments(argparse.ArgumentParser())
  args = parser.parse_args(["--task", "Some-Task", "--policy", "random"])

  assert config_from_args(args, {}).policy == "random"
  assert "checkpoint" in POLICIES


# ── one server, and mjlab's panel on it ───────────────────────────────────
class _RecordingViewer:
  """A stand-in for mjlab's viewer that remembers how it was built."""

  seen: Dict[str, Any] = {}

  def __init__(self, vec_env, policy, **kwargs):
    _RecordingViewer.seen = dict(kwargs)

  def run(self) -> None:
    pass


def _stub_viewers(monkeypatch) -> None:
  """Stand in for both mjlab viewers.

  Patched on `mjlab.viewer`, not on `rlmcp.play`: `_view` imports them inside
  the function, so the name this test has to replace is the one it reads.
  """
  import mjlab.viewer as mjlab_viewer

  monkeypatch.setattr(mjlab_viewer, "NativeMujocoViewer", _RecordingViewer)
  monkeypatch.setattr(mjlab_viewer, "ViserPlayViewer", _RecordingViewer)
  _RecordingViewer.seen = {}


def _viewer_cfg(mode: str):
  from rlmcp.play import PlayConfig
  return PlayConfig(checkpoint=None, mode=mode, task="Example-Task-v0")


def test_the_viewer_is_given_the_session_s_own_server(play_lab, monkeypatch):
  """The bug: a play session served the same environment twice -- mjlab's
  viewer on its own port with the full GUI, and the session's live view on the
  port `status` publishes with a much smaller one. Whoever read `status` went
  to the poorer panel. One server now, and mjlab draws on it."""
  import rlmcp.play as play_module

  sentinel = object()
  monkeypatch.setattr(play_lab.live_view, "host_for_viewer", lambda: sentinel)
  _stub_viewers(monkeypatch)

  env = SimpleNamespace(rlmcp=play_lab)
  play_module._view(_viewer_cfg("viser"), env, object(), object())

  assert _RecordingViewer.seen.get("viser_server") is sentinel


def test_the_native_viewer_is_not_handed_a_viser_server(play_lab, monkeypatch):
  """`--mode native` opens a window, not a port, and would reject the argument."""
  import rlmcp.play as play_module

  asked = []
  monkeypatch.setattr(play_lab.live_view, "host_for_viewer",
                      lambda: asked.append(1))
  _stub_viewers(monkeypatch)

  env = SimpleNamespace(rlmcp=play_lab)
  play_module._view(_viewer_cfg("native"), env, object(), object())

  assert not asked, "a native viewer needs no server bound on its behalf"
  assert "viser_server" not in _RecordingViewer.seen


def test_an_install_without_viser_still_plays(play_lab, monkeypatch):
  """`host_for_viewer` answers None when no server could be opened. That is the
  old behaviour exactly: let the viewer open whatever it can."""
  import rlmcp.play as play_module

  monkeypatch.setattr(play_lab.live_view, "host_for_viewer", lambda: None)
  _stub_viewers(monkeypatch)

  env = SimpleNamespace(rlmcp=play_lab)
  play_module._view(_viewer_cfg("viser"), env, object(), object())

  assert "viser_server" not in _RecordingViewer.seen
