"""Shared fakes: a simulator and runner that behave like the real ones.

These are also the minimal reference implementation of the adapter contract in
:mod:`rlmcp.adapters.base` -- read them alongside it when wiring up a new
backend. They follow the contract's error convention (unknown keys raise,
setters return True, absent capabilities raise NotSupported via the base
defaults) and emit trace channels under the shared ``CHANNEL_*`` vocabulary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from rlmcp.adapters.base import (
    CHANNEL_ACTION,
    CHANNEL_BASE_ANG_VEL,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_COMMAND,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_VEL,
    RunnerAdapter,
    SimAdapter,
)
from rlmcp.core.extensions import Extension
from rlmcp.core.parameters.spec import ParameterCategory, ParameterSpec

TERRAINS = ["flat", "random_rough", "pyramid_stairs"]
NUM_LEVELS = 6


class FakeSimAdapter(SimAdapter):
  """In-memory stand-in for a vectorised env with a terrain grid."""

  def __init__(self, num_envs: int = 12):
    self._num_envs = num_envs
    self._params: dict[str, Any] = {
        "reward.track_linear_velocity.weight": 2.0,
        "reward.action_rate_l2.weight": -0.1,
        "command.twist.ranges.lin_vel_x": [-1.0, 1.0],
    }
    self.terrain_types = np.zeros(num_envs, dtype=int)
    self.terrain_levels = np.zeros(num_envs, dtype=int)
    self.level_ceiling = NUM_LEVELS
    self.enabled = ["flat"]
    self.step_count = 0
    self.command_range_calls: list[dict[str, Any]] = []
    self.resets: list[list[int]] = []

  def discover_parameters(self) -> list[ParameterSpec]:
    specs = []
    for key, value in self._params.items():
      is_range = isinstance(value, list)
      specs.append(
          ParameterSpec(
              key=key,
              data_type="range" if is_range else "float",
              current_value=value,
              default_value=value,
              # This fake's own bounds, so registry enforcement has something
              # to enforce; the real providers declare none.
              min_value=None if is_range else -10.0,
              max_value=None if is_range else 10.0,
              description=f"fake parameter {key}",
              category=ParameterCategory.REWARD,
          )
      )
    return specs

  def get_parameter(self, key: str) -> Any:
    if key not in self._params:
      raise KeyError(key)
    return self._params[key]

  def set_parameter(self, key: str, value: Any) -> bool:
    # Contract: failures raise, success returns True explicitly -- the
    # registry reads a falsy return as "not applied".
    if key not in self._params:
      raise KeyError(key)
    self._params[key] = value
    return True

  def num_envs(self) -> int:
    return self._num_envs

  def step_dt(self) -> float:
    return 0.02

  def max_episode_length(self) -> float:
    return 100.0

  def joint_names(self) -> list[str]:
    return ["left_knee", "right_knee"]

  def sample_state(self, env_id: int = 0) -> dict[str, np.ndarray]:
    # Keys are the trace vocabulary from rlmcp.adapters.base; the command is
    # a plane-velocity twist, which is what CHANNEL_COMMAND means.
    self.step_count += 1
    phase = self.step_count * 0.1
    return {
        CHANNEL_JOINT_POS: np.array([np.sin(phase), np.cos(phase)], dtype=np.float32),
        CHANNEL_JOINT_VEL: np.array([np.cos(phase), -np.sin(phase)], dtype=np.float32),
        CHANNEL_ACTION: np.array([0.1, -0.1], dtype=np.float32),
        CHANNEL_COMMAND: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        CHANNEL_BASE_LIN_VEL: np.array([0.9, 0.0, 0.0], dtype=np.float32),
        CHANNEL_BASE_ANG_VEL: np.array([0.0, 0.0, 0.05], dtype=np.float32),
    }

  def trace_labels(self) -> dict[str, list[str]]:
    # One label per component of every channel sample_state emits.
    return {
        CHANNEL_JOINT_POS: self.joint_names(),
        CHANNEL_JOINT_VEL: self.joint_names(),
        CHANNEL_ACTION: [f"act:{j}" for j in self.joint_names()],
        CHANNEL_COMMAND: ["cmd_vx", "cmd_vy", "cmd_wz"],
        CHANNEL_BASE_LIN_VEL: ["vx", "vy", "vz"],
        CHANNEL_BASE_ANG_VEL: ["wx", "wy", "wz"],
    }

  def summary_metrics(self) -> dict[str, float]:
    return {
        "rlmcp/terrain_level_mean": float(self.terrain_levels.mean()),
        "rlmcp/terrain_level_frac": float(
            self.terrain_levels.mean() / max(1, self.level_ceiling - 1)
        ),
    }

  def reset_envs(self, env_ids=None) -> dict[str, Any]:
    # Optional capability: a backend without one leaves this to the base class,
    # which raises NotSupported. Records what it was asked to restart so a test
    # can see that a --where query narrowed it.
    chosen = (list(range(self._num_envs)) if env_ids is None
              else [int(i) for i in env_ids])
    self.resets.append(chosen)
    return {"num_reset": len(chosen)}

  def render(self, env_id: int = 0) -> np.ndarray:
    return np.full((8, 8, 3), fill_value=env_id % 255, dtype=np.uint8)

  def renderer_ready(self) -> bool:
    # This fake renders from nothing, so a frame never costs a construction.
    return True

  # Terrain state, driven by FakeTerrainExtension below. Not part of the
  # SimAdapter contract -- that is the point of the extension mechanism.

  def set_terrain(self, terrains=None, weights=None, max_level=None) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    if max_level is not None:
      self.level_ceiling = int(max_level)
      self.terrain_levels = np.clip(self.terrain_levels, 0, self.level_ceiling - 1)
      changed["level_ceiling"] = self.level_ceiling
    if terrains is not None:
      unknown = [t for t in terrains if t not in TERRAINS]
      if unknown:
        raise ValueError(f"Unknown terrain(s) {unknown}")
      self.enabled = list(terrains)
      columns = [TERRAINS.index(t) for t in self.enabled]
      self.terrain_types = np.array(
          [columns[i % len(columns)] for i in range(self._num_envs)]
      )
      changed["terrains"] = list(self.enabled)
    return changed

  def env_ids_on(self, terrain=None, level=None) -> list[int]:
    mask = np.ones(self._num_envs, dtype=bool)
    if terrain is not None:
      if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain '{terrain}'")
      mask &= self.terrain_types == TERRAINS.index(terrain)
    if level is not None:
      mask &= self.terrain_levels == int(level)
    return [int(i) for i in np.nonzero(mask)[0]]

  def get_env_state(self) -> dict[str, Any]:
    return {"step": self.step_count}

  def set_env_state(self, state: dict[str, Any]) -> None:
    self.step_count = int(state.get("step", 0))


class FakeTerrainExtension(Extension):
  """A stand-in capability, shaped like the real terrain extension.

  It exists to test the mechanism rather than terrain itself: commands, metrics,
  env selection and checkpoint state all arrive through this interface.
  """

  name = "terrain"

  def __init__(self, sim: FakeSimAdapter):
    super().__init__(env=None)
    self.sim = sim

  def available(self) -> bool:
    return True

  def commands(self):
    return {
        "terrain_status": self.cmd_terrain_status,
        "set_terrain": self.cmd_set_terrain,
    }

  def metrics(self) -> dict[str, float]:
    ceiling = max(1, self.sim.level_ceiling - 1)
    return {
        "rlmcp/terrain_level_mean": float(self.sim.terrain_levels.mean()),
        "rlmcp/terrain_level_frac": float(self.sim.terrain_levels.mean() / ceiling),
    }

  def select_envs(self, **criteria):
    terrain = criteria.pop("terrain", None)
    level = criteria.pop("level", None)
    if criteria or (terrain is None and level is None):
      return None
    return self.sim.env_ids_on(terrain=terrain, level=level)

  def selectors(self) -> dict[str, dict[str, Any]]:
    return {
        "terrain": {"label": "terrain", "values": list(self.sim.enabled)},
        "level": {"label": "difficulty level",
                  "values": list(range(self.sim.level_ceiling))},
    }

  def describe(self) -> dict[str, Any]:
    return {"active_terrains": list(self.sim.enabled),
            "level_ceiling": self.sim.level_ceiling}

  def snapshot(self) -> dict[str, Any]:
    return {"enabled": list(self.sim.enabled), "ceiling": self.sim.level_ceiling}

  def restore(self, state: dict[str, Any]) -> None:
    self.sim.set_terrain(terrains=state["enabled"], max_level=state["ceiling"])

  def cmd_terrain_status(self) -> dict[str, Any]:
    """Per-terrain environment counts."""
    return {"active_terrains": list(self.sim.enabled),
            "level_ceiling": self.sim.level_ceiling}

  def cmd_set_terrain(self, terrains=None, weights=None, max_level=None,
                      rationale: str = "") -> dict[str, Any]:
    """Choose which terrains spawn environments and cap how hard they get."""
    return {"changed": self.sim.set_terrain(terrains, weights, max_level)}


class FakeRunnerAdapter(RunnerAdapter):
  """Runner stand-in that records what was asked of it."""

  def __init__(self):
    self.hyper = {"learning_rate": 1e-3, "entropy_coef": 0.01}
    self.iteration = 0
    self.saved: list[str] = []
    self.loaded: list[str] = []
    self.stop_called = False
    self.checkpoint_infos: dict[str, Any] = {}

  def discover_hyperparameters(self) -> list[ParameterSpec]:
    return [
        ParameterSpec(
            key=f"rl.{name}",
            data_type="float",
            current_value=value,
            default_value=value,
            min_value=0.0,
            max_value=1.0,
            description=f"fake {name}",
            category=ParameterCategory.RL_HYPERPARAMETER,
        )
        for name, value in self.hyper.items()
    ]

  def get_hyperparameter(self, key: str) -> Any:
    return self.hyper[key.split(".", 1)[-1]]

  def set_hyperparameter(self, key: str, value: Any) -> bool:
    # Same setter convention as the sim adapter: unknown keys raise, success
    # returns True.
    name = key.split(".", 1)[-1]
    if name not in self.hyper:
      raise KeyError(name)
    self.hyper[name] = float(value)
    return True

  def current_iteration(self) -> int:
    return self.iteration

  def runner_metrics(self) -> dict[str, float]:
    return {"Train/mean_reward": 1.0, "Train/mean_episode_length": 80.0}

  def save_checkpoint(self, path: str, infos: dict[str, Any] | None = None) -> str:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("fake-checkpoint")
    self.saved.append(path)
    # Mirror mjlab's Runner.save exactly: the runner replaces any caller-provided
    # "env_state" wholesale with its own payload, so nothing the caller packs
    # under that key survives a real checkpoint.
    self.checkpoint_infos = {
        **(infos or {}),
        "env_state": {"common_step_counter": self.iteration},
    }
    return path

  def load_checkpoint(self, path: str) -> dict[str, Any]:
    # Like the real runner's load: hand back the stored infos verbatim.
    self.loaded.append(path)
    return self.checkpoint_infos

  def request_stop(self) -> bool:
    # Advisory, like the mjlab adapter: record that the controller asked, and
    # return False because no runner-native stop mechanism was armed -- stop
    # is delivered by the loop polling RlMcp.should_stop().
    self.stop_called = True
    return False


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path_factory):
  """Every test gets a private, empty registry.

  Session.create registers the sessions it makes; without this, the suite
  would write the developer's real ~/.local/state/rlmcp and, worse, tests
  would see each other's (and the developer's actual) runs.
  """
  monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("xdg-state")))


@pytest.fixture
def fake_sim() -> FakeSimAdapter:
  return FakeSimAdapter()


@pytest.fixture
def fake_terrain(fake_sim) -> FakeTerrainExtension:
  return FakeTerrainExtension(fake_sim)


@pytest.fixture
def fake_runner() -> FakeRunnerAdapter:
  return FakeRunnerAdapter()


# A legged-gym-shaped environment, for the second adapter family.


class FakeFlatEnv:
  """A Go2Env-shaped environment, down to the traps."""

  def __init__(self, num_envs: int = 4):
    self.num_envs = num_envs
    self.dt = 0.02

    self.env_cfg = {
        "num_actions": 12,
        "action_scale": 0.25,
        "clip_actions": 100.0,
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "termination_if_pitch_greater_than": 10.0,
        "termination_if_roll_greater_than": 10.0,
        "kp": 20.0,
        "kd": 0.5,
        "joint_names": ["FL_hip", "FL_thigh"],
        "base_init_pos": [0.0, 0.0, 0.42],
    }
    self.reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "lin_vel_z": -1.0,
            "action_rate": -0.005,
        },
    }
    self.command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 1.5],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
    self.obs_cfg = {"obs_scales": {"lin_vel": 2.0}}

    # Exactly what Go2Env.__init__ does, in the order it does it.
    self.reward_scales = self.reward_cfg["reward_scales"]
    self.commands_limits = tuple(
        torch.tensor(values, dtype=torch.float32)
        for values in zip(
            self.command_cfg["lin_vel_x_range"],
            self.command_cfg["lin_vel_y_range"],
            self.command_cfg["ang_vel_range"],
            strict=True,
        )
    )
    self.reward_functions = {}
    for name in self.reward_scales:
      self.reward_scales[name] *= self.dt
      self.reward_functions[name] = lambda: torch.ones(self.num_envs)

    # The per-step buffers step() writes, which is where traces read from.
    n, j = num_envs, len(self.env_cfg["joint_names"])
    self.dof_pos = torch.zeros(n, j)
    self.dof_vel = torch.zeros(n, j)
    self.actions = torch.zeros(n, j)
    self.last_actions = torch.zeros(n, j)
    self.base_lin_vel = torch.zeros(n, 3)
    self.base_ang_vel = torch.zeros(n, 3)
    self.base_pos = torch.zeros(n, 3)
    self.projected_gravity = torch.tensor([[0.0, 0.0, -1.0]] * n)
    self.rew_buf = torch.zeros(n)
    self.commands = torch.zeros(n, 3)
    self.episode_length_buf = torch.zeros(n, dtype=torch.long)
    self.max_episode_length = 1000
    self.extras = {}

  def total_reward(self) -> float:
    """One step's reward, computed the way the env computes it."""
    return float(
        sum(fn() * self.reward_scales[name]
            for name, fn in self.reward_functions.items())[0]
    )

  def sample_command(self, channel: int) -> tuple:
    """The bounds the resampler would draw channel ``channel`` from."""
    lower, upper = self.commands_limits
    return float(lower[channel]), float(upper[channel])


@pytest.fixture
def flat_env() -> FakeFlatEnv:
  """A Go2Env-shaped environment: dict configs, flat buffers, and the traps
  that come with both. See tests/test_flat_env_access.py for what they are."""
  return FakeFlatEnv()


# A Genesis environment: the flat fake, plus a scene with cameras.


class FakeCamera:
  """A Genesis camera, as much of one as rendering uses."""

  def __init__(self, debug: bool = False, res=(8, 6), batched: bool = False):
    self.debug = debug
    self.res = res
    self.pos = (2.0, 0.0, 2.5)
    self.lookat = (0.0, 0.0, 0.5)
    self.batched = batched
    self.poses = []

  def set_pose(self, transform=None, pos=None, lookat=None, up=None,
               envs_idx=None):
    if pos is not None:
      self.pos = tuple(pos)
    if lookat is not None:
      self.lookat = tuple(lookat)
    self.poses.append((self.pos, self.lookat))

  def render(self, rgb=True, depth=False, segmentation=False,
             colorize_seg=False, normal=False, antialiasing=False,
             force_render=False):
    width, height = self.res
    # The frame encodes where the camera was, so a test can tell one env's
    # picture from another's.
    frame = np.full((height, width, 3), int(abs(self.lookat[0])) % 256,
                    dtype=np.uint8)
    if self.batched:
      frame = np.stack([frame, frame])
    return frame, None, None, None


class FakeVisualizer:
  def __init__(self, cameras):
    self.cameras = list(cameras)


class FakeScene:
  """Genesis keeps the drawn-env list in two places that can disagree.

  ``vis_options`` is what the script asked for; the visualizer's ``context`` is
  what the renderer reads. ``options_say`` lets a test set them apart, which is
  how a real run turned `shot --env-id 2` away from an env it was drawing.
  """

  def __init__(self, cameras=(), rendered_envs_idx=None, options_say=None):
    self.visualizer = FakeVisualizer(cameras)
    self.visualizer.context = type(
        "Context", (), {"rendered_envs_idx": rendered_envs_idx})()
    self.vis_options = type(
        "VisOptions", (),
        {"rendered_envs_idx": options_say if options_say is not None
         else rendered_envs_idx})()


@pytest.fixture
def genesis_env(flat_env):
  """The flat fake, plus the scene and reset a Genesis env would have."""
  flat_env.scene = FakeScene(cameras=[FakeCamera(debug=True)],
                             rendered_envs_idx=list(range(flat_env.num_envs)))
  flat_env.device = None
  flat_env.reset_calls = []

  def _reset_idx(mask=None):
    flat_env.reset_calls.append(mask)

  flat_env._reset_idx = _reset_idx
  # Envs sit at spaced world origins, which is what makes per-env framing a
  # pose change rather than an anchor negotiation.
  for i in range(flat_env.num_envs):
    flat_env.base_pos[i, 0] = 10.0 * i
  return flat_env
