"""SimAdapter for IsaacLab's ``ManagerBasedRLEnv``.

Thin on purpose, and thinner than it looks: parameter discovery, trace sampling
and summary metrics come from :mod:`rlmcp.adapters.manager_based`, which is
written against the shape both IsaacLab and mjlab share -- manager term configs
that are live dataclasses, a scene of named entities, per-env buffers on the
environment. What is genuinely IsaacLab's own is small: how the robot is found
in the scene, and how a frame is rendered.

Everything an agent changes lands on a config object the environment re-reads,
so nothing here needs a restart or a checkpoint round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rlmcp.adapters.base import NotSupported, SimAdapter
from rlmcp.adapters.isaaclab import rendering
from rlmcp.adapters.manager_based import metrics as state_metrics
from rlmcp.adapters.manager_based import terms as state_terms
from rlmcp.adapters.manager_based.access import ParameterAccess
from rlmcp.adapters.manager_based.sampling import StateSampler
from rlmcp.core.parameters.spec import ParameterSpec


class IsaacLabSimAdapter(SimAdapter):
  """Live control surface over one IsaacLab ``ManagerBasedRLEnv``."""

  def __init__(self, env: Any, robot_name: str | None = None):
    self.env = env
    self.robot_name = robot_name or self._resolve_robot_name(env)
    self.parameters = ParameterAccess(env)
    self.sampler = StateSampler(env, self.robot_name)
    self._last_set_notes: dict[str, Any] = {}

  @staticmethod
  def _resolve_robot_name(env: Any) -> str:
    """The articulation an agent means when it says "the robot".

    IsaacLab's scene keeps articulations in their own mapping, which makes this
    easier than guessing: "robot" by convention, otherwise the only
    articulation there is. Two unnamed articulations is a scene where the
    question is genuinely ambiguous, so it is asked rather than answered.
    """
    scene = getattr(env, "scene", None)
    articulations = dict(getattr(scene, "articulations", {}) or {})
    if "robot" in articulations:
      return "robot"
    if len(articulations) == 1:
      return next(iter(articulations))
    if articulations:
      raise ValueError(
          f"This scene has {len(articulations)} articulations "
          f"({', '.join(sorted(articulations))}); say which one is the robot "
          "with wrap(robot_name=...).")
    raise ValueError("Scene has no articulations; cannot identify the robot.")

  @property
  def robot(self) -> Any:
    return self.env.scene[self.robot_name]

  # Basics.

  def num_envs(self) -> int:
    return int(self.env.num_envs)

  def step_dt(self) -> float:
    return float(self.env.step_dt)

  def joint_names(self) -> list[str]:
    return list(self.robot.joint_names)

  def max_episode_length(self) -> float | None:
    value = getattr(self.env, "max_episode_length", None)
    return float(value) if value else None

  # Parameters.

  def discover_parameters(self) -> list[ParameterSpec]:
    return self.parameters.discover()

  def get_parameter(self, key: str) -> Any:
    return self.parameters.get(key)

  def set_parameter(self, key: str, value: Any) -> bool:
    """Accurate by exception, per the contract in :mod:`rlmcp.adapters.base`."""
    self._last_set_notes = self.parameters.set(key, value)
    return True

  def last_set_notes(self) -> dict[str, Any]:
    return dict(self._last_set_notes)

  def parameter_domains(self) -> list[str]:
    return self.parameters.domains()

  # State sampling and metrics.

  def sample_state(self, env_id: int = 0) -> dict[str, np.ndarray]:
    return self.sampler.sample(env_id)

  def trace_labels(self) -> dict[str, list[str]]:
    return self.sampler.labels()

  def summary_metrics(self) -> dict[str, float]:
    return state_metrics.summary_metrics(self.env, self.robot_name)

  # Manager terms. The same two managers, read by the same shared module: an
  # IsaacLab task's reward is a sum of named terms exactly as mjlab's is.

  def reward_terms(self) -> dict[str, float]:
    found = state_terms.reward_terms(self.env)
    if not found:
      raise NotSupported("reward_terms")
    return found

  def termination_terms(self) -> dict[str, float]:
    return state_terms.termination_terms(self.env)

  # Rendering.

  def render(self, env_id: int = 0) -> np.ndarray:
    return rendering.render(self.env, env_id, self.robot_name)

  def renderer_ready(self) -> bool:
    return rendering.renderer_ready(self.env)

  # Episodes.

  def reset_envs(self, env_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Start fresh episodes through the path a termination takes.

    ``_reset_idx`` is what IsaacLab's own ``step`` calls for whichever
    environments ended this tick, so reusing it is what makes a requested reset
    indistinguishable from a natural one -- the event and command terms that
    run on reset run here too.
    """
    total = self.num_envs()
    if env_ids is None:
      chosen = list(range(total))
    else:
      chosen = [int(i) for i in env_ids]
      out_of_range = [i for i in chosen if i < 0 or i >= total]
      if out_of_range:
        raise ValueError(
            f"Environment id(s) {out_of_range} do not exist; this run has "
            f"{total} environments (0..{total - 1}).")

    reset_idx = getattr(self.env, "_reset_idx", None)
    if callable(reset_idx):
      import torch

      reset_idx(torch.as_tensor(chosen, dtype=torch.long,
                                device=getattr(self.env, "device", None)))
      clock = getattr(self.env, "episode_length_buf", None)
      if clock is not None:
        try:
          clock[chosen] = 0
        except Exception:
          pass
      return {"num_reset": len(chosen), "method": "_reset_idx"}

    if env_ids is None and callable(getattr(self.env, "reset", None)):
      self.env.reset()
      return {"num_reset": total, "method": "reset"}

    raise NotSupported(
        "reset_envs: this environment exposes no way to restart a subset of "
        "its environments, so only a full reset is possible -- ask for one by "
        "leaving env_ids and where unset.")

  # Env state, for checkpointing the curriculum alongside the policy.

  def get_env_state(self) -> dict[str, Any]:
    return {"common_step_counter": int(getattr(self.env, "common_step_counter", 0))}

  def set_env_state(self, state: dict[str, Any]) -> None:
    if "common_step_counter" in state:
      self.env.common_step_counter = int(state["common_step_counter"])
