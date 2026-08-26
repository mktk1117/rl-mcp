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
