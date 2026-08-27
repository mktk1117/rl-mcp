"""SimAdapter for mjlab's ``ManagerBasedRlEnv``.

This file is deliberately thin. It implements the :class:`SimAdapter` contract
by delegating to the pieces that do the work:

* :mod:`rlmcp.adapters.manager_based.access` -- parameter discovery, reads and writes,
  found by reflection over the environment's manager term configs rather than
  by a hand-written list. Add a family by adding a provider there.
* :mod:`rlmcp.adapters.mjlab.state` -- reading live state: per-step sampling,
  batch metrics, rendering.

Anything specific to one kind of task -- terrain for locomotion, say -- lives
in :mod:`rlmcp.extensions` instead, so wrapping a manipulation environment
yields fewer commands rather than broken ones.

Every change lands on a live config object that mjlab reads on its next step,
so nothing here needs a restart or a checkpoint round-trip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rlmcp.adapters.base import NotSupported, SimAdapter
from rlmcp.adapters.manager_based.access import ParameterAccess
from rlmcp.adapters.manager_based import metrics as state_metrics
from rlmcp.adapters.manager_based import terms as state_terms
from rlmcp.adapters.mjlab.state import rendering
from rlmcp.adapters.manager_based.sampling import StateSampler
from rlmcp.core.parameters.spec import ParameterSpec


class MjlabSimAdapter(SimAdapter):
  """Live control surface over one ``ManagerBasedRlEnv``."""

  def __init__(self, env: Any, robot_name: Optional[str] = None):
    self.env = env
    self.robot_name = robot_name or self._resolve_robot_name(env)
    self.parameters = ParameterAccess(env)
    self.sampler = StateSampler(env, self.robot_name)
    self._last_set_notes: Dict[str, Any] = {}

  @staticmethod
  def _resolve_robot_name(env: Any) -> str:
    entities = getattr(env.scene, "entities", {}) or {}
    if "robot" in entities:
      return "robot"
    for name, entity in entities.items():
      if getattr(entity, "is_articulated", False):
        return name
    if entities:
      return next(iter(entities))
    raise ValueError("Scene has no entities; cannot identify the robot.")

  @property
  def robot(self) -> Any:
    return self.env.scene[self.robot_name]

  # Basics.

  def num_envs(self) -> int:
    return int(self.env.num_envs)

  def step_dt(self) -> float:
    return float(self.env.step_dt)

  def joint_names(self) -> List[str]:
    return list(self.robot.joint_names)

  def max_episode_length(self) -> Optional[float]:
    value = getattr(self.env, "max_episode_length", None)
    return float(value) if value else None

  # Parameters.

  def discover_parameters(self) -> List[ParameterSpec]:
    return self.parameters.discover()

  def get_parameter(self, key: str) -> Any:
    return self.parameters.get(key)

  def set_parameter(self, key: str, value: Any) -> bool:
    """Write through the access layer: raises on any failure, True on success.

    Accurate by exception, per the contract in :mod:`rlmcp.adapters.base`: an
    unknown key, a refused shape, or a dead (non-live) leaf raises with the
    reason, and the ``True`` is returned explicitly because the registry reads
    a falsy return as "not applied".
    """
    # Providers can report side effects (e.g. a curriculum term that had to be
    # overridden); keep them for the caller that asks next.
    self._last_set_notes = self.parameters.set(key, value)
    return True

  def last_set_notes(self) -> Dict[str, Any]:
    return dict(self._last_set_notes)

  def parameter_domains(self) -> List[str]:
    return self.parameters.domains()

  # State sampling and metrics.

  def sample_state(self, env_id: int = 0) -> Dict[str, np.ndarray]:
    """One step of trace channels (vocabulary in :mod:`rlmcp.adapters.base`)."""
    return self.sampler.sample(env_id)

  def trace_labels(self) -> Dict[str, List[str]]:
    """Component names derived from the live env: joints, action terms, command."""
    return self.sampler.labels()

  def summary_metrics(self) -> Dict[str, float]:
    return state_metrics.summary_metrics(self.env, self.robot_name)

  # Manager terms.

  def reward_terms(self) -> Dict[str, float]:
    found = state_terms.reward_terms(self.env)
    if not found:
      raise NotSupported("reward_terms")
    return found

  def termination_terms(self) -> Dict[str, float]:
    return state_terms.termination_terms(self.env)

  # Rendering.

  def render(self, env_id: int = 0) -> np.ndarray:
    return rendering.render(self.env, env_id)

  def renderer_ready(self) -> bool:
    """Whether a frame would reuse an existing renderer (else render() builds one)."""
    return rendering.renderer_ready(self.env)

  # Episodes.

  def reset_envs(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Start fresh episodes, through the same path a termination takes.

    A manager-based environment restarts a subset of its environments with
    ``_reset_idx(env_ids)`` -- the method its own ``step`` calls for whichever
    environments terminated this tick. Reusing it is what makes a requested
    reset indistinguishable from a natural one: the event, command and
    randomisation terms that run on reset all run here too, which is the whole
    point of asking the environment to reset rather than writing state back
    into it by hand.

    ``reset()`` is the fallback for a backend that exposes no per-env path, and
    only when every environment was asked for -- resetting all twelve because
    the caller asked for two would be worse than refusing.
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
            f"{total} environments (0..{total - 1})."
        )

    # Public spelling first, then the manager-based convention. Named rather
    # than duck-typed so a backend that happens to have some other _reset_idx
    # is not called by accident.
    for name in ("reset_idx", "_reset_idx"):
      reset_idx = getattr(self.env, name, None)
      if callable(reset_idx):
        import torch

        reset_idx(torch.as_tensor(chosen, dtype=torch.long,
                                  device=getattr(self.env, "device", None)))
        # Manager-based envs zero the episode clock inside _reset_idx; doing it
        # again is harmless, and a backend that does not would otherwise hand
        # the restarted episode a clock already at the time limit.
        clock = getattr(self.env, "episode_length_buf", None)
        if clock is not None:
          try:
            clock[chosen] = 0
          except Exception:
            pass
        return {"num_reset": len(chosen), "method": name}

    if env_ids is None and callable(getattr(self.env, "reset", None)):
      self.env.reset()
      return {"num_reset": total, "method": "reset"}

    raise NotSupported(
        "reset_envs: this environment exposes neither reset_idx nor a way to "
        "restart a subset of its environments, so only a full reset is "
        "possible here -- ask for one by leaving env_ids and where unset."
    )

  # Env state, for checkpointing the curriculum alongside the policy.

  def get_env_state(self) -> Dict[str, Any]:
    """Generic env state only; extensions add their own through the registry."""
    return {"common_step_counter": int(getattr(self.env, "common_step_counter", 0))}

  def set_env_state(self, state: Dict[str, Any]) -> None:
    if "common_step_counter" in state:
      self.env.common_step_counter = int(state["common_step_counter"])
