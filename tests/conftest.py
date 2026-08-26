"""Shared fakes: a simulator and runner that behave like the real ones.

These are also the minimal reference implementation of the adapter contract in
:mod:`rlmcp.adapters.base` -- read them alongside it when wiring up a new
backend. They follow the contract's error convention (unknown keys raise,
setters return True, absent capabilities raise NotSupported via the base
defaults) and emit trace channels under the shared ``CHANNEL_*`` vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

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
    self._params: Dict[str, Any] = {
        "reward.track_linear_velocity.weight": 2.0,
        "reward.action_rate_l2.weight": -0.1,
        "command.twist.ranges.lin_vel_x": [-1.0, 1.0],
    }
    self.terrain_types = np.zeros(num_envs, dtype=int)
    self.terrain_levels = np.zeros(num_envs, dtype=int)
    self.level_ceiling = NUM_LEVELS
    self.enabled = ["flat"]
    self.step_count = 0
    self.command_range_calls: List[Dict[str, Any]] = []

  def discover_parameters(self) -> List[ParameterSpec]:
    specs = []
    for key, value in self._params.items():
      is_range = isinstance(value, list)
      specs.append(
          ParameterSpec(
              key=key,
              data_type="range" if is_range else "float",
              current_value=value,
              default_value=value,
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

  def joint_names(self) -> List[str]:
    return ["left_knee", "right_knee"]

  def sample_state(self, env_id: int = 0) -> Dict[str, np.ndarray]:
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

  def trace_labels(self) -> Dict[str, List[str]]:
    # One label per component of every channel sample_state emits.
    return {
        CHANNEL_JOINT_POS: self.joint_names(),
        CHANNEL_JOINT_VEL: self.joint_names(),
        CHANNEL_ACTION: [f"act:{j}" for j in self.joint_names()],
        CHANNEL_COMMAND: ["cmd_vx", "cmd_vy", "cmd_wz"],
        CHANNEL_BASE_LIN_VEL: ["vx", "vy", "vz"],
        CHANNEL_BASE_ANG_VEL: ["wx", "wy", "wz"],
    }

  def summary_metrics(self) -> Dict[str, float]:
    return {
        "rlmcp/terrain_level_mean": float(self.terrain_levels.mean()),
        "rlmcp/terrain_level_frac": float(
            self.terrain_levels.mean() / max(1, self.level_ceiling - 1)
        ),
    }

  def render(self, env_id: int = 0) -> np.ndarray:
    return np.full((8, 8, 3), fill_value=env_id % 255, dtype=np.uint8)

  def renderer_ready(self) -> bool:
    # This fake renders from nothing, so a frame never costs a construction.
    return True

  # Terrain state, driven by FakeTerrainExtension below. Not part of the
  # SimAdapter contract -- that is the point of the extension mechanism.

  def set_terrain(self, terrains=None, weights=None, max_level=None) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}
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

  def env_ids_on(self, terrain=None, level=None) -> List[int]:
    mask = np.ones(self._num_envs, dtype=bool)
    if terrain is not None:
      if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain '{terrain}'")
      mask &= self.terrain_types == TERRAINS.index(terrain)
    if level is not None:
      mask &= self.terrain_levels == int(level)
    return [int(i) for i in np.nonzero(mask)[0]]

  def get_env_state(self) -> Dict[str, Any]:
    return {"step": self.step_count}

  def set_env_state(self, state: Dict[str, Any]) -> None:
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

  def metrics(self) -> Dict[str, float]:
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

  def describe(self) -> Dict[str, Any]:
    return {"active_terrains": list(self.sim.enabled),
            "level_ceiling": self.sim.level_ceiling}

  def snapshot(self) -> Dict[str, Any]:
    return {"enabled": list(self.sim.enabled), "ceiling": self.sim.level_ceiling}

  def restore(self, state: Dict[str, Any]) -> None:
    self.sim.set_terrain(terrains=state["enabled"], max_level=state["ceiling"])

  def cmd_terrain_status(self) -> Dict[str, Any]:
    """Per-terrain environment counts."""
    return {"active_terrains": list(self.sim.enabled),
            "level_ceiling": self.sim.level_ceiling}

  def cmd_set_terrain(self, terrains=None, weights=None, max_level=None,
                      rationale: str = "") -> Dict[str, Any]:
    """Choose which terrains spawn environments and cap how hard they get."""
    return {"changed": self.sim.set_terrain(terrains, weights, max_level)}


class FakeRunnerAdapter(RunnerAdapter):
  """Runner stand-in that records what was asked of it."""

  def __init__(self):
    self.hyper = {"learning_rate": 1e-3, "entropy_coef": 0.01}
    self.iteration = 0
    self.saved: List[str] = []
    self.loaded: List[str] = []
    self.stop_called = False
    self.checkpoint_infos: Dict[str, Any] = {}

  def discover_hyperparameters(self) -> List[ParameterSpec]:
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

  def runner_metrics(self) -> Dict[str, float]:
    return {"Train/mean_reward": 1.0, "Train/mean_episode_length": 80.0}

  def save_checkpoint(self, path: str, infos: Optional[Dict[str, Any]] = None) -> str:
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

  def load_checkpoint(self, path: str) -> Dict[str, Any]:
    # Like the real runner's load: hand back the stored infos verbatim.
    self.loaded.append(path)
    return self.checkpoint_infos

  def request_stop(self) -> bool:
    # Advisory, like the mjlab adapter: record that the controller asked, and
    # return False because no runner-native stop mechanism was armed -- stop
    # is delivered by the loop polling RlMcp.should_stop().
    self.stop_called = True
    return False


@pytest.fixture
def fake_sim() -> FakeSimAdapter:
  return FakeSimAdapter()


@pytest.fixture
def fake_terrain(fake_sim) -> FakeTerrainExtension:
  return FakeTerrainExtension(fake_sim)


@pytest.fixture
def fake_runner() -> FakeRunnerAdapter:
  return FakeRunnerAdapter()
