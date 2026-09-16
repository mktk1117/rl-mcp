"""The single-file family, against a fake env of that shape.

No simulator, no GPU. The fake is the shape ``docs/single-file.md`` describes:
one declared dataclass on ``env.cfg``, the conventional state buffers, a
``reset(env_ids)``, and a physics object on ``env.sim`` that the environment
reads into those buffers. What these pin is the promise the family makes --
that every declared value is found, that a write lands where the environment
reads it, that a ``Static`` field is refused with a reason rather than
accepted and ignored, and that a term appended at runtime scores on the next
step because the loop reads the table rather than a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pytest
import torch

from rlmcp import declare
from rlmcp.adapters.base import NotSupported
from rlmcp.adapters.reward_terms import RewardInstallError
from rlmcp.adapters.single_file import (
  AlgorithmAdapter,
  NotASingleFileEnv,
  SingleFileEnv,
  SingleFileSimAdapter,
  SingleFileSpec,
  TrainingStopped,
  wrap,
)
from rlmcp.adapters.single_file.spec import detect
from rlmcp.core.parameters.spec import Liveness, ParameterCategory
from rlmcp.declare import Static, Term, term

# The fake environment, written the way an env.py would be.


@dataclass
class Rewards:
  tracking_lin_vel: Term = term(4.0, sigma=0.5)
  upright: Term = term(1.0, sigma=0.4472)
  action_rate: Term = term(-0.1)


@dataclass
class Commands:
  lin_vel_x: tuple[float, float] = (0.5, 1.5)
  lin_vel_y: tuple[float, float] = (0.0, 0.0)
  ang_vel_yaw: tuple[float, float] = (-0.3, 0.3)
  resample_time_s: float = 7.0


@dataclass
class Termination:
  max_tilt_rad: float = 0.8
  contact_threshold: float = 50.0


@dataclass
class Noise:
  level: float = 1.0
  dof_pos: float = 0.01


@dataclass
class EnvConfig:
  num_envs: Static[int] = 4
  dt: Static[float] = 0.005
  decimation: Static[int] = 4
  episode_length_s: Static[float] = 20.0
  backend: Static[str] = "fake"
  action_scale: float = 0.25
  reward: Rewards = field(default_factory=Rewards)
  command: Commands = field(default_factory=Commands)
  termination: Termination = field(default_factory=Termination)
  noise: Noise = field(default_factory=Noise)


class FakeBackend:
  name = "fake"
  joint_names: ClassVar[list[str]] = ["FL_hip", "FL_thigh"]
  mj_model = None

  def render(self, env_id: int):
    import numpy as np
    return np.zeros((6, 8, 3), dtype=np.uint8)


class FakeSingleFileEnv(SingleFileEnv):
  """An env.py-shaped environment: config on cfg, state in buffers, sim behind."""

  def __init__(self, cfg: EnvConfig | None = None, with_sim: bool = True):
    self.cfg = cfg or EnvConfig()
    self.num_envs = self.cfg.num_envs
    self.device = torch.device("cpu")
    self.control_dt = self.cfg.dt * self.cfg.decimation
    self.max_episode_steps = int(self.cfg.episode_length_s / self.control_dt)
    if with_sim:
      self.sim = FakeBackend()
    n, j = self.num_envs, 2
    self.dof_pos = torch.zeros(n, j)
    self.dof_vel = torch.zeros(n, j)
    self.actions = torch.zeros(n, j)
    self.last_actions = torch.ones(n, j)
    self.base_lin_vel = torch.zeros(n, 3)
    self.base_ang_vel = torch.zeros(n, 3)
    self.base_pos = torch.zeros(n, 3)
    self.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]] * n)
    self.commands = torch.zeros(n, 3)
    self.resets: list = []
    self.steps = 0

  def reset(self, env_ids=None):
    self.resets.append(None if env_ids is None else env_ids.tolist())
    ids = torch.arange(self.num_envs) if env_ids is None else env_ids
    c = self.cfg.command
    self.commands[ids, 0] = torch.empty(len(ids)).uniform_(*c.lin_vel_x)
    self.commands[ids, 1] = torch.empty(len(ids)).uniform_(*c.lin_vel_y)
    self.commands[ids, 2] = torch.empty(len(ids)).uniform_(*c.ang_vel_yaw)
    return torch.zeros(self.num_envs, 8)

  def compute_reward_terms(self) -> dict[str, torch.Tensor]:
    """What an env.py computes inline, keyed by term name."""
    ones = torch.ones(self.num_envs)
    return {
        "tracking_lin_vel": ones * self.cfg.reward.tracking_lin_vel.sigma,
        "upright": ones,
        "action_rate": ones * 2.0,
    }

  def step(self, actions):
    self.steps += 1
    self.actions[:] = actions * self.cfg.action_scale
    total, terms = self.compute_reward(scale=self.control_dt)
    info = self.step_info(
        terms, episode_rewards=torch.tensor([1.0, 2.0]),
        episode_lengths=torch.tensor([10.0, 20.0]),
        time_outs=torch.zeros(self.num_envs, dtype=torch.bool))
    return torch.zeros(self.num_envs, 8), total, torch.zeros(self.num_envs, dtype=torch.bool), info


class ShapeOnlyEnv:
  """The same shape without the base class: what a file that cannot inherit
  looks like to the wrapper."""

  def __init__(self):
    inner = FakeSingleFileEnv()
    self.__dict__.update(inner.__dict__)
    self._inner = inner

  def reset(self, env_ids=None):
    return self._inner.reset(env_ids)

  def step(self, actions):
    return self._inner.step(actions)


@dataclass
class FakePPOConfig:
  learning_rate: float = 1e-3
  entropy_coef: float = 0.01
  clip_param: float = 0.2
  gamma: float = 0.99
  num_learning_epochs: int = 5
  hidden: Static[int] = 64
  schedule: str = "adaptive"


class FakePPO:
  """A hand-rolled PPO, as much of it as the adapter touches: a declared cfg,
  mirrored attributes, the change hook, metrics, save and load."""

  def __init__(self):
    self.cfg = FakePPOConfig()
    self.weights = torch.nn.Parameter(torch.zeros(3))
    self.optimizer = torch.optim.Adam([self.weights], lr=1e-3)
    self.learning_rate = 1e-3
    self.entropy_coef = 0.01
    self.schedule = "adaptive"
    self.hook_calls: list = []
    self.actor = torch.nn.Linear(2, 2)
    self.actor.output_std = torch.tensor([0.5, 0.7])

  def on_hyperparameter_change(self, name, value):
    self.hook_calls.append((name, value))
    if name == "learning_rate":
      for group in self.optimizer.param_groups:
        group["lr"] = value
      self.schedule = "fixed"

  def metrics(self):
    return {"Policy/mean_std": float(self.actor.output_std.mean())}

  def save(self) -> dict:
    return {"weights": self.weights.detach().clone(), "entropy_coef": self.entropy_coef}

  def load(self, state: dict) -> None:
    with torch.no_grad():
      self.weights.copy_(state["weights"])
    self.entropy_coef = state["entropy_coef"]


@pytest.fixture
def env() -> FakeSingleFileEnv:
  return FakeSingleFileEnv()


@pytest.fixture
def sim(env) -> SingleFileSimAdapter:
  return SingleFileSimAdapter(env)


def keys(sim: SingleFileSimAdapter) -> dict:
  return {spec.key: spec for spec in sim.discover_parameters()}


# The markers.


def test_each_config_gets_its_own_terms():
  """A shared Term default would leak one run's edits into the next config."""
  a, b = EnvConfig(), EnvConfig()
  a.reward.upright.weight = 9.0
  assert b.reward.upright.weight == 1.0


def test_term_params_read_as_attributes():
  t = Term(4.0, sigma=0.5)
  assert t.sigma == 0.5
  with pytest.raises(AttributeError) as excinfo:
    _ = t.missing
  assert "sigma" in str(excinfo.value)


def test_static_is_read_off_the_annotation():
  assert declare.static_fields(EnvConfig) == {
      "num_envs", "dt", "decimation", "episode_length_s", "backend"}
  assert not declare.is_static(EnvConfig, "action_scale")


def test_terms_includes_what_was_added_at_runtime():
  cfg = EnvConfig()
  cfg.reward.later = Term(0.5, func=lambda env: None)
  assert list(declare.terms(cfg.reward)) == [
      "tracking_lin_vel", "upright", "action_rate", "later"]


# Detection.


def test_a_conventional_env_needs_no_spec(env):
  assert detect(env) == SingleFileSpec()


def test_an_env_that_cannot_inherit_is_accepted_by_shape():
  duck = ShapeOnlyEnv()
  assert not isinstance(duck, SingleFileEnv)
  assert detect(duck) == SingleFileSpec()
  sim = SingleFileSimAdapter(duck)
  assert "reward.upright.weight" in {s.key for s in sim.discover_parameters()}


def test_the_base_class_scores_a_term_it_did_not_compute(env):
  env.cfg.reward.bonus = Term(0.5, func=lambda e, k=2.0: torch.full((e.num_envs,), k), k=3.0)
  total, scored = env.compute_reward()
  assert torch.equal(scored["bonus"], torch.full((4,), 3.0))
  assert float(total[0]) == pytest.approx(4.0 * 0.5 + 1.0 + (-0.1) * 2.0 + 0.5 * 3.0)


def test_the_base_class_names_a_term_nobody_scores(env):
  env.cfg.reward.orphan = Term(1.0)
  with pytest.raises(KeyError) as excinfo:
    env.compute_reward()
  assert "orphan" in str(excinfo.value)


def test_an_env_of_the_wrong_shape_is_refused_by_name():
  class NotOne:
    def __init__(self):
      self.observations = {}

  with pytest.raises(NotASingleFileEnv) as excinfo:
    detect(NotOne())
  message = str(excinfo.value)
  assert "env.cfg" in message and "num_envs" in message and "reset" in message
  assert "SingleFileSpec" in message
  assert "observations" in message


def test_a_config_that_is_not_a_dataclass_is_refused(env):
  env.cfg = {"reward": {}}
  with pytest.raises(NotASingleFileEnv) as excinfo:
    detect(env)
  assert "not a dataclass" in str(excinfo.value)


# Discovery.


def test_every_declared_value_is_found_under_the_shared_vocabulary(sim):
  found = keys(sim)
  assert {"reward.tracking_lin_vel.weight",
          "reward.tracking_lin_vel.params.sigma",
          "reward.action_rate.weight",
          "command.lin_vel_x",
          "command.resample_time_s",
          "termination.max_tilt_rad",
          "noise.dof_pos",
          "env.action_scale",
          "env.num_envs"} <= set(found)
  assert found["command.lin_vel_x"].data_type == "range"
  assert found["command.lin_vel_x"].current_value == [0.5, 1.5]


def test_categories_come_from_the_group_names(sim):
  found = keys(sim)
  assert found["reward.upright.weight"].category is ParameterCategory.REWARD
  assert found["command.lin_vel_x"].category is ParameterCategory.CURRICULUM
  assert found["termination.max_tilt_rad"].category is ParameterCategory.TERMINATION
  assert found["noise.dof_pos"].category is ParameterCategory.DOMAIN_RANDOMIZATION
  assert found["env.action_scale"].category is ParameterCategory.OTHER


def test_static_fields_are_listed_as_at_startup_and_refused(sim):
  found = keys(sim)
  assert found["env.num_envs"].liveness is Liveness.AT_STARTUP
  assert found["env.action_scale"].liveness is Liveness.LIVE
  with pytest.raises(ValueError) as excinfo:
    sim.set_parameter("env.num_envs", 8)
  assert "at_startup" in str(excinfo.value)
  assert sim.env.cfg.num_envs == 4


def test_strings_are_not_offered_as_parameters(sim):
  assert "env.backend" not in keys(sim)


# Writes land where the environment reads.


def test_a_weight_write_changes_the_next_step(sim, env):
  _, before, _, _ = env.step(torch.zeros(4, 2))
  sim.set_parameter("reward.upright.weight", 3.0)
  assert env.cfg.reward.upright.weight == 3.0
  _, after, _, _ = env.step(torch.zeros(4, 2))
  assert float(after[0]) == pytest.approx(float(before[0]) + 2.0 * env.control_dt)


def test_a_term_param_write_lands_in_params(sim, env):
  sim.set_parameter("reward.tracking_lin_vel.params.sigma", 0.25)
  assert env.cfg.reward.tracking_lin_vel.sigma == 0.25
  assert sim.get_parameter("reward.tracking_lin_vel.params.sigma") == 0.25


def test_a_command_range_keeps_the_tuple_the_config_declared(sim, env):
  sim.set_parameter("command.lin_vel_x", [1.0, 2.0])
  assert env.cfg.command.lin_vel_x == (1.0, 2.0)
  assert isinstance(env.cfg.command.lin_vel_x, tuple)
  env.reset(torch.tensor([0, 1]))
  assert (env.commands[:2, 0] >= 1.0).all()


def test_a_bad_range_is_refused_before_anything_changes(sim, env):
  with pytest.raises(ValueError):
    sim.set_parameter("command.lin_vel_x", 1.0)
  assert env.cfg.command.lin_vel_x == (0.5, 1.5)


def test_an_unknown_reward_term_says_how_to_add_one(sim):
  with pytest.raises(KeyError) as excinfo:
    sim.set_parameter("reward.foot_slip.weight", 1.0)
  assert "add-reward" in str(excinfo.value)


# Adding a term.


def test_an_added_term_scores_from_the_next_step(sim, env):
  def foot_slip(env, gain=2.0):
    return torch.ones(env.num_envs) * gain

  _, before, _, _ = env.step(torch.zeros(4, 2))
  installed = sim.add_reward_term("foot_slip", foot_slip, weight=-0.5, params={"gain": 2.0})
  assert installed["name"] == "foot_slip" and installed["index"] == 3
  _, after, _, info = env.step(torch.zeros(4, 2))
  assert "foot_slip" in info["reward_terms"]
  assert float(after[0]) == pytest.approx(float(before[0]) - 0.5 * 2.0 * env.control_dt)
  assert "reward.foot_slip.weight" in keys(sim)
  assert "reward.foot_slip.params.gain" in keys(sim)
  sim.set_parameter("reward.foot_slip.weight", -1.0)
  assert env.cfg.reward.foot_slip.weight == -1.0


def test_a_failing_term_leaves_the_table_untouched(sim, env):
  def broken(env):
    raise RuntimeError("no such buffer")

  with pytest.raises(RewardInstallError):
    sim.add_reward_term("broken", broken, weight=1.0)
  assert "broken" not in declare.terms(env.cfg.reward)


def test_a_duplicate_term_is_refused(sim):
  with pytest.raises(RewardInstallError):
    sim.add_reward_term("upright", lambda env: torch.ones(4), weight=1.0)


# State.


def test_the_basics_come_off_the_environment(sim, env):
  assert sim.num_envs() == 4
  assert sim.step_dt() == pytest.approx(0.02)
  assert sim.joint_names() == ["FL_hip", "FL_thigh"]
  assert sim.max_episode_length() == pytest.approx(1000)


def test_traces_publish_the_command_as_a_velocity(sim, env):
  env.reset()
  sample = sim.sample_state(0)
  assert "joint_pos" in sample and "command" in sample
  assert sim.trace_labels()["joint_pos"] == ["FL_hip", "FL_thigh"]


def test_a_command_that_is_not_a_velocity_is_not_published_as_one():
  @dataclass
  class Goals:
    goal_x: tuple[float, float] = (0.0, 1.0)
    goal_y: tuple[float, float] = (0.0, 1.0)
    goal_yaw: tuple[float, float] = (0.0, 1.0)

  @dataclass
  class Cfg(EnvConfig):
    command: Goals = field(default_factory=Goals)

  env = FakeSingleFileEnv(Cfg())
  sample = SingleFileSimAdapter(env).sample_state(0)
  assert "command" not in sample and "command_raw" in sample


def test_summary_metrics_are_prefixed(sim):
  assert all(k.startswith("rlmcp/") for k in sim.summary_metrics())


def test_resetting_goes_through_the_environment(sim, env):
  assert sim.reset_envs([3, 1]) == {"num_reset": 2, "env_ids": [1, 3]}
  assert env.resets[-1] == [1, 3]
  assert sim.reset_envs(None) == {"num_reset": 4, "env_ids": None}
  assert env.resets[-1] is None
  with pytest.raises(ValueError):
    sim.reset_envs([9])


def test_frames_come_from_the_backend(sim):
  assert sim.renderer_ready()
  assert sim.render(0).shape == (6, 8, 3)


def test_a_backend_without_render_says_so(env):
  env.sim = object()
  sim = SingleFileSimAdapter(env)
  assert not sim.renderer_ready()
  with pytest.raises(NotSupported):
    sim.render(0)


# The wrapper: the loop's two lines.


def test_the_wrapper_parks_the_reward_and_logs_the_terms(env, tmp_path):
  wrapped = wrap(env, session_dir=tmp_path / "s", viser=False)
  out = wrapped.step(torch.zeros(4, 2))
  assert torch.equal(env.rew_buf, out[1])
  logged = wrapped._flush_log()
  assert "rewards/upright" in logged
  assert logged["episode/reward"] == pytest.approx(1.5)


def test_service_is_the_iteration_boundary(env, tmp_path):
  wrapped = wrap(env, session_dir=tmp_path / "s", viser=False)
  adapter = wrapped.attach_algorithm(FakePPO())
  wrapped.step(torch.zeros(4, 2))
  wrapped.service(7, metrics={"losses/value": 0.25, "not_a_number": "x"})
  assert adapter.current_iteration() == 7
  assert wrapped.rlmcp.iteration == 7
  wrapped.rlmcp.run_command("stop_training", reason="seen enough")
  with pytest.raises(TrainingStopped) as caught:
    wrapped.service(8)
  assert "seen enough" in str(caught.value)


def test_hyperparameters_reach_the_algorithm(tmp_path, env):
  ppo = FakePPO()
  wrapped = wrap(env, session_dir=tmp_path / "s", viser=False)
  wrapped.attach_algorithm(ppo)
  specs = {s.key: s for s in wrapped.rlmcp.runner.discover_hyperparameters()}
  assert {"rl.learning_rate", "rl.entropy_coef", "rl.clip_param", "rl.gamma",
          "rl.num_learning_epochs", "rl.hidden"} == set(specs), "every numeric leaf of cfg"
  assert specs["rl.hidden"].liveness is Liveness.AT_STARTUP
  assert specs["rl.num_learning_epochs"].data_type == "int"
  runner = wrapped.rlmcp.runner
  runner.set_hyperparameter("rl.learning_rate", 5e-4)
  assert ppo.optimizer.param_groups[0]["lr"] == 5e-4, "the hook did its job"
  assert ppo.schedule == "fixed", "adaptive would overwrite the edit next update"
  assert ppo.cfg.learning_rate == 5e-4 and ppo.learning_rate == 5e-4
  runner.set_hyperparameter("rl.entropy_coef", 0.02)
  assert ppo.entropy_coef == 0.02 and ppo.cfg.entropy_coef == 0.02
  assert ppo.hook_calls == [("learning_rate", 5e-4), ("entropy_coef", 0.02)]
  runner.set_hyperparameter("rl.num_learning_epochs", 7)
  assert ppo.cfg.num_learning_epochs == 7 and isinstance(ppo.cfg.num_learning_epochs, int)
  with pytest.raises(ValueError) as excinfo:
    runner.set_hyperparameter("rl.hidden", 128)
  assert "at_startup" in str(excinfo.value) and ppo.cfg.hidden == 64
  with pytest.raises(KeyError) as excinfo:
    runner.set_hyperparameter("rl.schedule", "fixed")
  assert "Available" in str(excinfo.value)
  assert runner.runner_metrics()["Policy/mean_std"] == pytest.approx(0.6)


def test_an_algorithm_without_a_cfg_declares_nothing():
  adapter = AlgorithmAdapter(object())
  assert adapter.discover_hyperparameters() == []
  with pytest.raises(KeyError) as excinfo:
    adapter.set_hyperparameter("rl.learning_rate", 1e-3)
  assert "cfg" in str(excinfo.value)


def test_checkpoints_round_trip_through_save_and_load(tmp_path):
  ppo = FakePPO()
  adapter = AlgorithmAdapter(ppo)
  adapter.iteration = 12
  path = tmp_path / "ckpt.pt"
  assert adapter.save_checkpoint(str(path), {"parameters": {"a": 1}}) == str(path)
  with torch.no_grad():
    ppo.weights.fill_(3.0)
  ppo.entropy_coef = 0.5
  infos = adapter.load_checkpoint(str(path))
  assert infos == {"parameters": {"a": 1}}
  assert torch.equal(ppo.weights.detach(), torch.zeros(3))
  assert ppo.entropy_coef == 0.01


def test_an_algorithm_without_save_is_refused_not_faked():
  adapter = AlgorithmAdapter(object())
  with pytest.raises(NotSupported):
    adapter.save_checkpoint("/nowhere/x.pt")
