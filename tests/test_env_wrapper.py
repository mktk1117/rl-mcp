"""``rlmcp.wrap`` itself -- the call every user makes first.

Nothing exercised this before. The suite covers the layers underneath and the
example's import lines, but no test ever constructed the wrapper, because doing
it *looked* like it needed mjlab and a GPU. It does not: the module imports only
torch, and the adapter duck-types the environment. So a rename left
``lab.start(...)`` referring to a local that had become ``records``, and
``wrap()`` raised ``NameError`` on every call while 421 tests passed.

These build the smallest environment the wrapper accepts and call it. They
assert little about behaviour on purpose -- the point is that every line of the
constructor runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from test_access import FakeEnv

from rlmcp.core.curriculum import CurriculumStage, StageSchedule


class _Robot:
  is_articulated = True
  joint_names = ["a_joint", "b_joint"]


class _Scene(dict):
  @property
  def entities(self) -> Dict[str, Any]:
    return self


def wrappable_env() -> FakeEnv:
  """``FakeEnv`` plus the handful of attributes the sim adapter reads."""
  env = FakeEnv()
  env.scene = _Scene(robot=_Robot())
  env.num_envs = 4
  env.step_dt = 0.02
  env.device = "cpu"
  env.render_mode = None
  return env


@pytest.fixture
def wrapper_cls():
  """Import late so a torch-less environment skips rather than errors."""
  torch = pytest.importorskip("torch", reason="the wrapper module imports torch")
  del torch
  from rlmcp.adapters.mjlab.env_wrapper import RlMcpEnvWrapper

  return RlMcpEnvWrapper


def test_wrap_constructs(wrapper_cls, tmp_path):
  """The regression: every line of ``__init__`` runs, or this raises."""
  wrapper = wrapper_cls(wrappable_env(), session_dir=tmp_path / "session")

  assert wrapper.rlmcp is not None
  assert (tmp_path / "session").exists()


def test_wrap_helper_matches_the_class(tmp_path):
  """``rlmcp.wrap(env)`` is the documented entry point, not the class."""
  pytest.importorskip("torch", reason="the wrapper module imports torch")
  from rlmcp.adapters.mjlab.env_wrapper import RlMcpEnvWrapper, wrap

  wrapper = wrap(wrappable_env(), session_dir=tmp_path / "session")
  assert isinstance(wrapper, RlMcpEnvWrapper)


def test_the_session_records_what_the_env_reported(wrapper_cls, tmp_path):
  wrapper = wrapper_cls(
      wrappable_env(), session_dir=tmp_path / "session", task_id="Fake-Task-v0"
  )

  info = wrapper.rlmcp.session.info()
  assert info["task"] == "Fake-Task-v0"
  assert info["num_envs"] == 4


def test_parameters_are_discovered_through_the_wrapper(wrapper_cls, tmp_path):
  """The walk runs at construction, so the run is steerable immediately."""
  wrapper = wrapper_cls(wrappable_env(), session_dir=tmp_path / "session")

  keys = {spec.key for spec in wrapper.rlmcp.parameters.get_all_specs()}
  assert "reward.foot_slip.weight" in keys


def test_an_explicit_schedule_is_kept(wrapper_cls, tmp_path):
  schedule = StageSchedule(
      [CurriculumStage(name="only", parameters={"reward.foot_slip.weight": -0.2})]
  )
  wrapper = wrapper_cls(
      wrappable_env(), session_dir=tmp_path / "session", curriculum=schedule
  )

  assert wrapper.rlmcp.curriculum is schedule


def test_curriculum_terrain_declines_without_a_terrain_grid(wrapper_cls, tmp_path, capsys):
  """Asking for the terrain ladder on a scene that has none says so, not raises."""
  wrapper = wrapper_cls(
      wrappable_env(), session_dir=tmp_path / "session", curriculum="terrain"
  )

  assert wrapper.rlmcp.curriculum is None
  assert "terrain" in capsys.readouterr().out.lower()


def test_unknown_attributes_reach_the_wrapped_env(wrapper_cls, tmp_path):
  """The wrapper is transparent; a training loop must not notice it."""
  env = wrappable_env()
  env.some_task_specific_thing = 7
  wrapper = wrapper_cls(env, session_dir=tmp_path / "session")

  assert wrapper.some_task_specific_thing == 7
  assert wrapper.unwrapped is env


def test_service_every_steps_is_at_least_one(wrapper_cls, tmp_path):
  """A zero would divide the service clock by nothing."""
  wrapper = wrapper_cls(
      wrappable_env(), session_dir=tmp_path / "session", service_every_steps=0
  )

  assert wrapper.service_every_steps >= 1


def test_a_record_run_binds_the_session_to_the_record(wrapper_cls, tmp_path):
  """``records.start(...)`` is the line the rename broke; this reaches it."""
  from rlmcp.records import open_store

  store = open_store(tmp_path / "records")
  record = store.new_record("wrapper-smoke", hypothesis="the link binds")

  wrapper = wrapper_cls(
      wrappable_env(),
      session_dir=tmp_path / "session",
      record_run=record.id,
      records_root=str(tmp_path / "records"),
  )

  assert wrapper.rlmcp.records is not None
  reloaded = open_store(tmp_path / "records").get_record(record.id)
  assert reloaded is not None


def test_a_stop_unwinds_as_the_core_signal_under_the_backend_name(wrapper_cls,
                                                                  tmp_path):
  """`TrainingStopped` is what the training entrypoints catch, and a play
  session's viewer loop is not a training entrypoint. Both catch the same
  object: the backend name is the core signal, so neither side has to learn
  about the other."""
  from rlmcp.adapters.mjlab.env_wrapper import TrainingStopped
  from rlmcp.core.controller import SessionStopped

  assert issubclass(TrainingStopped, SessionStopped)

  wrapper = wrapper_cls(wrappable_env(), session_dir=tmp_path / "session")
  wrapper.rlmcp.run_command("stop_training", reason="seen enough")

  with pytest.raises(SessionStopped) as caught:
    wrapper._service(iteration=1)

  assert "seen enough" in str(caught.value)


# Where "once per learning iteration" lives, per runner.


class _MjlabStyleRunner:
  """mjlab: the runner owns a logger object with a `log(it=...)`."""

  def __init__(self):
    self.logger = SimpleNamespace(log=self._log)
    self.seen = []

  def _log(self, *args, **kwargs):
    self.seen.append(kwargs.get("it", args[0] if args else None))


class _RslRlStyleRunner:
  """plain rsl_rl, which IsaacLab uses: `log(locs)` on the runner itself."""

  def __init__(self):
    self.seen = []

  def log(self, locs, width=80, pad=35):
    self.seen.append(locs.get("it"))


@pytest.mark.parametrize("runner_type", [_MjlabStyleRunner, _RslRlStyleRunner],
                         ids=["mjlab-logger", "rsl_rl-runner"])
def test_the_iteration_hook_is_found_wherever_the_runner_keeps_it(
    runner_type, wrapper_cls, tmp_path, monkeypatch):
  """A backend that logs from the runner rather than from a logger object still
  gets serviced on iteration boundaries -- not every N steps, which is the
  fallback and answers commands at the wrong moment."""
  wrapper = wrapper_cls(wrappable_env(), session_dir=tmp_path / "session",
                        video_every=0)
  serviced = []
  monkeypatch.setattr(type(wrapper), "_service",
                      lambda self, iteration=None: serviced.append(iteration))
  runner = runner_type()
  try:
    wrapper.attach_runner(runner)
    if isinstance(runner, _MjlabStyleRunner):
      runner.logger.log(it=7)
    else:
      runner.log({"it": 7})
  finally:
    wrapper.rlmcp.close()

  assert wrapper._runner_hooked is True
  assert serviced == [7]
  assert runner.seen == [7]     # and the runner's own logging still happened


# The shared base is not a backend.


def test_the_shared_base_cannot_be_wrapped_round_an_environment(tmp_path):
  """`manager_based` holds the half that is not any simulator's, so it has no
  simulator to talk to. Constructing it directly has to fail where a backend
  would have answered -- and say which file to copy -- rather than half-build a
  wrapper whose every sim call is broken."""
  pytest.importorskip("torch", reason="the wrapper module imports torch")
  from rlmcp.adapters import manager_based
  from rlmcp.adapters.manager_based import env_wrapper as shared

  # No module-level `wrap()` here: the entry point belongs to the backend that
  # can name a SimAdapter, next to the subclass it builds.
  assert not hasattr(shared, "wrap")
  assert not hasattr(manager_based, "wrap")

  with pytest.raises(NotImplementedError) as caught:
    shared.RlMcpEnvWrapper(wrappable_env(), session_dir=tmp_path / "session")
  assert "build_sim_adapter" in str(caught.value)
  assert "mjlab" in str(caught.value)


# The live view is on unless somebody says otherwise.
#
# It is a default rather than a flag because of what it costs: with no browser
# open, and while paused, the tick returns before it reads a clock. What an
# unwatched view uses is a port. What a missing one costs is the hour of a run
# somebody now wants to look at and cannot.


def test_a_training_run_serves_a_live_view_without_being_asked():
  from rlmcp.adapters.manager_based.env_wrapper import serve_a_live_view

  assert serve_a_live_view(None, "") is True


def test_a_play_session_does_not():
  # It is opening a viewer of its own; a second one on a second port is only a
  # way to end up watching the wrong scene.
  from rlmcp.adapters.manager_based.env_wrapper import serve_a_live_view

  assert serve_a_live_view(None, "play") is False


def test_saying_so_wins_either_way():
  from rlmcp.adapters.manager_based.env_wrapper import serve_a_live_view

  assert serve_a_live_view(False, "") is False
  assert serve_a_live_view(True, "play") is True


def test_rlmcp_train_defaults_the_view_on_and_no_viser_turns_it_off():
  from rlmcp.train import _parse_args

  assert _parse_args(["Some-Task"]).viser is True
  assert _parse_args(["Some-Task", "--no-viser"]).viser is False
  assert _parse_args(["Some-Task", "--viser"]).viser is True
