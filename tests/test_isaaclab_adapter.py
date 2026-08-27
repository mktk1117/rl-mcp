"""The IsaacLab adapter, against a fake env shaped like IsaacLab's.

No Isaac Sim, no GPU, no display. What these pin is the part that is genuinely
IsaacLab's -- how the robot is found in a scene that keeps articulations in
their own mapping, and how a frame is asked for through the Kit app -- plus the
claim the whole split rests on: the shared manager-based machinery works
against IsaacLab's shapes without knowing which simulator it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from rlmcp.adapters.base import NotSupported
from rlmcp.adapters.isaaclab import IsaacLabSimAdapter, wrap
from rlmcp.core.parameters.spec import ParameterCategory


@dataclass
class TermCfg:
  func: Any = None
  weight: float = 0.0
  params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardsCfg:
  track_lin_vel: TermCfg = field(
      default_factory=lambda: TermCfg(weight=1.0, params={"std": 0.5}))
  action_rate_l2: TermCfg = field(default_factory=lambda: TermCfg(weight=-0.01))


@dataclass
class TerminationsCfg:
  base_contact: TermCfg = field(
      default_factory=lambda: TermCfg(params={"threshold": 1.0}))


@dataclass
class ViewerCfg:
  """IsaacLab's viewer config: which env the camera is looking at."""

  env_index: int = 0
  resolution: tuple = (128, 96)


@dataclass
class EnvCfg:
  rewards: RewardsCfg = field(default_factory=RewardsCfg)
  terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
  viewer: ViewerCfg = field(default_factory=ViewerCfg)


class FakeArticulation:
  def __init__(self, names: List[str]):
    self.joint_names = list(names)
    self.data = SimpleNamespace(
        joint_pos=torch.zeros(4, len(names)),
        joint_vel=torch.zeros(4, len(names)),
        root_lin_vel_b=torch.zeros(4, 3),
        root_ang_vel_b=torch.zeros(4, 3),
    )


class FakeScene:
  """IsaacLab keeps articulations in their own mapping, unlike mjlab."""

  def __init__(self, articulations: Dict[str, FakeArticulation]):
    self.articulations = articulations
    self.sensors: Dict[str, Any] = {}

  def __getitem__(self, key: str) -> Any:
    return self.articulations[key]


class FakeManager:
  """IsaacLab's manager API, as much of it as discovery uses."""

  def __init__(self, terms: Dict[str, TermCfg]):
    self._terms = dict(terms)

  @property
  def active_terms(self) -> List[str]:
    return list(self._terms)

  def get_term_cfg(self, name: str) -> TermCfg:
    return self._terms[name]


class FakeIsaacLabEnv:
  """Shaped like ``ManagerBasedRLEnv`` as far as the adapter can tell."""

  def __init__(self, articulations=None, render_mode=None, frame=None):
    self.cfg = EnvCfg()
    # The managers hold the same cfg objects the env config does, which is what
    # makes an edit visible to the running environment in IsaacLab too.
    self.reward_manager = FakeManager({
        "track_lin_vel": self.cfg.rewards.track_lin_vel,
        "action_rate_l2": self.cfg.rewards.action_rate_l2,
    })
    self.termination_manager = FakeManager(
        {"base_contact": self.cfg.terminations.base_contact})
    self.scene = FakeScene(articulations or {"robot": FakeArticulation(["hip", "knee"])})
    self.num_envs = 4
    self.step_dt = 0.02
    self.max_episode_length = 500
    self.device = "cpu"
    self.common_step_counter = 7
    self.episode_length_buf = torch.arange(4)
    self.render_mode = render_mode
    self.render_calls: List[int] = []
    self.reset_calls: List[List[int]] = []
    self._frame = frame

  def render(self, recompute: bool = False):
    self.render_calls.append(self.cfg.viewer.env_index)
    if self.render_mode != "rgb_array":
      return None
    return self._frame

  def _reset_idx(self, env_ids):
    self.reset_calls.append([int(i) for i in env_ids])


# The robot, in a scene that names its articulations.


def test_the_robot_is_the_one_called_robot():
  assert IsaacLabSimAdapter(FakeIsaacLabEnv()).robot_name == "robot"


def test_a_scene_with_one_articulation_needs_no_convention():
  env = FakeIsaacLabEnv({"anymal": FakeArticulation(["hip"])})
  assert IsaacLabSimAdapter(env).robot_name == "anymal"


def test_an_ambiguous_scene_asks_instead_of_guessing():
  """Two articulations and no convention is a question, not a default: picking
  one silently is how a trace ends up describing the wrong body."""
  env = FakeIsaacLabEnv({"arm": FakeArticulation(["j1"]),
                         "hand": FakeArticulation(["f1"])})

  with pytest.raises(ValueError, match="2 articulations"):
    IsaacLabSimAdapter(env)

  assert IsaacLabSimAdapter(env, robot_name="hand").joint_names() == ["f1"]


# The shared machinery, against IsaacLab's shapes.


def test_parameters_are_discovered_by_walking_isaaclabs_config():
  """The claim the split rests on: nothing in the discovery layer knows which
  simulator this is, and IsaacLab's term configs are the same shape."""
  adapter = IsaacLabSimAdapter(FakeIsaacLabEnv())

  found = {spec.key: spec for spec in adapter.discover_parameters()}

  assert "reward.track_lin_vel.weight" in found
  assert found["reward.track_lin_vel.weight"].category == ParameterCategory.REWARD
  # Not just weights: a term's function parameters are found by walking too.
  assert found["reward.track_lin_vel.params.std"].current_value == 0.5
  assert "termination.base_contact.params.threshold" in found


def test_an_edit_lands_on_the_config_the_env_re_reads():
  env = FakeIsaacLabEnv()
  adapter = IsaacLabSimAdapter(env)

  assert adapter.set_parameter("reward.action_rate_l2.weight", -0.25) is True

  assert env.cfg.rewards.action_rate_l2.weight == -0.25
  assert adapter.get_parameter("reward.action_rate_l2.weight") == -0.25


def test_the_basics_come_off_the_environment():
  adapter = IsaacLabSimAdapter(FakeIsaacLabEnv())

  assert adapter.num_envs() == 4
  assert adapter.step_dt() == 0.02
  assert adapter.max_episode_length() == 500.0
  assert adapter.joint_names() == ["hip", "knee"]
  assert adapter.get_env_state() == {"common_step_counter": 7}


def test_resetting_goes_through_the_path_a_termination_takes():
  env = FakeIsaacLabEnv()
  adapter = IsaacLabSimAdapter(env)

  answer = adapter.reset_envs([1, 3])

  assert answer == {"num_reset": 2, "method": "_reset_idx"}
  assert env.reset_calls == [[1, 3]]
  assert env.episode_length_buf[1] == 0 and env.episode_length_buf[2] == 2


def test_an_environment_that_does_not_exist_is_refused_by_number():
  adapter = IsaacLabSimAdapter(FakeIsaacLabEnv())

  with pytest.raises(ValueError, match=r"\[9\]"):
    adapter.reset_envs([9])


# Frames, which IsaacLab can only give under conditions set before rlmcp exists.


def test_no_frames_without_cameras_says_what_to_do_about_it():
  adapter = IsaacLabSimAdapter(FakeIsaacLabEnv(render_mode=None))

  assert adapter.renderer_ready() is False
  with pytest.raises(NotSupported, match="enable_cameras"):
    adapter.render(0)


def test_a_frame_comes_back_as_rgb_and_the_camera_is_put_back():
  """The viewer config is shared with whatever else is watching, so pointing it
  at env 2 for one frame must not leave it there."""
  rgba = np.full((6, 8, 4), 200, dtype=np.uint8)
  env = FakeIsaacLabEnv(render_mode="rgb_array", frame=rgba)
  adapter = IsaacLabSimAdapter(env)

  frame = adapter.render(2)

  assert frame.shape == (6, 8, 3)                  # RGBA in, RGB out
  assert env.render_calls == [2]                   # it looked at env 2
  assert env.cfg.viewer.env_index == 0             # and looked away again


def test_a_frame_request_beyond_the_last_env_is_clamped_not_crashed():
  env = FakeIsaacLabEnv(render_mode="rgb_array",
                        frame=np.zeros((4, 4, 3), dtype=np.uint8))

  IsaacLabSimAdapter(env).render(99)

  assert env.render_calls == [3]                   # the last environment there is


# The wrapper.


def test_wrapping_gives_a_run_the_agent_can_reach(tmp_path):
  env = FakeIsaacLabEnv()

  wrapped = wrap(env, session_dir=tmp_path / "session", task_id="Isaac-Fake-v0",
                 video_every=0)
  try:
    status = wrapped.rlmcp.run_command("status")
    assert status["num_envs"] == 4
    assert wrapped.rlmcp.session.info()["task"] == "Isaac-Fake-v0"
    # And the transparent-forwarding promise: the env is still the env.
    assert wrapped.num_envs == 4
  finally:
    wrapped.rlmcp.close()


def test_a_run_with_no_frames_says_so_at_wrap_time(tmp_path, capsys):
  """Four hours in, when you finally want to look at the robot, is the
  expensive moment to discover the app was launched without cameras."""
  wrapped = wrap(FakeIsaacLabEnv(), session_dir=tmp_path / "s", video_every=0)
  try:
    assert "enable_cameras" in capsys.readouterr().out
  finally:
    wrapped.rlmcp.close()
