"""SimAdapter for a single-file environment.

Thin. Trace sampling and summary metrics read the environment's variables
(:mod:`rlmcp.adapters.single_file.state`); parameters come from the declared
config tree and the observation blocks
(:mod:`rlmcp.adapters.single_file.access`). What is genuinely this adapter's
own is small: how episodes are restarted, how a frame is asked for, and how
a reward term is appended to a table that is a dataclass rather than a dict.

The physics backend is not this adapter's business. ``env.sim`` may be MuJoCo
Warp, mjbatch or Genesis; the environment reads it into the same buffers
either way, and the only two things asked of it here are ``render(env_id)``
and, when it has one, ``mj_model`` for the live view.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rlmcp import declare
from rlmcp.adapters.base import NotSupported, SimAdapter
from rlmcp.adapters.reward_terms import RewardInstallError, trial_call
from rlmcp.adapters.single_file.access import ParameterAccess
from rlmcp.adapters.single_file.spec import SingleFileSpec, detect
from rlmcp.adapters.single_file.state import StateSampler, summary_metrics
from rlmcp.core.parameters.spec import ParameterSpec


class SingleFileSimAdapter(SimAdapter):
  """Live control surface over one single-file environment."""

  def __init__(self, env: Any, spec: SingleFileSpec | None = None):
    self.env = env
    self.spec = detect(env, spec)
    self.parameters = ParameterAccess(env, self.spec)
    self.sampler = StateSampler(env, self.spec, command_names=self.parameters.command_channels())
    self._last_set_notes: dict[str, Any] = {}

  @property
  def cfg(self) -> Any:
    return self.spec.resolve(self.env, "cfg")

  @property
  def sim(self) -> Any:
    return self.spec.resolve(self.env, "sim", None)

  # Required.

  def discover_parameters(self) -> list[ParameterSpec]:
    return self.parameters.discover()

  def get_parameter(self, key: str) -> Any:
    return self.parameters.get(key)

  def set_parameter(self, key: str, value: Any) -> bool:
    self._last_set_notes = self.parameters.set(key, value)
    return True

  def last_set_notes(self) -> dict[str, Any]:
    return dict(self._last_set_notes)

  # Reward terms.

  def add_reward_term(
      self,
      name: str,
      func: Any,
      weight: float,
      params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Append a :class:`~rlmcp.declare.Term` carrying ``func`` to the table.

    The environment's reward loop reads the table with
    :func:`rlmcp.declare.terms` every step, so a term set on the group here
    scores from the next step on, and its weight is served under
    ``reward.<name>.weight`` like the ones the task shipped with. The trial
    call comes first and nothing is written on any refusal.
    """
    params = dict(params or {})
    group = getattr(self.cfg, self.spec.reward_group, None)
    if group is None:
      raise NotSupported(
          f"add_reward_term: the config has no '{self.spec.reward_group}' "
          "section to append to."
      )
    table = declare.terms(group)
    if name in table:
      raise RewardInstallError(
          f"Reward term '{name}' already exists; pick another name or set "
          f"reward.{name}.weight instead."
      )
    if hasattr(group, name):
      raise RewardInstallError(
          f"'{name}' is already a field of the reward section and not a term; "
          "pick another name."
      )

    import torch

    with torch.inference_mode():
      trial = trial_call(
          self.env, name=name, func=func, params=params,
          num_envs=int(self.env.num_envs),
      )

    setattr(group, name, declare.Term(float(weight), func, **params))
    self.parameters._refresh_synthetic()
    return {
        "name": name,
        "index": len(table),
        "weight": float(weight),
        "params": params,
        "trial": trial,
        "scores_from": "next step",
    }

  # Introspection.

  def num_envs(self) -> int:
    return int(self.env.num_envs)

  def step_dt(self) -> float:
    dt = self.spec.resolve(self.env, "control_dt", None)
    if dt is None:
      cfg = self.cfg
      base = getattr(cfg, "dt", None)
      decimation = getattr(cfg, "decimation", 1)
      if base is None:
        raise NotSupported("step_dt")
      dt = float(base) * int(decimation)
    return float(dt)

  def joint_names(self) -> list[str]:
    names = list(getattr(self.sim, "joint_names", None) or [])
    if not names:
      names = self.sampler.joint_names()
    if not names:
      raise NotSupported("joint_names")
    return names

  def max_episode_length(self) -> float | None:
    length = self.spec.resolve(self.env, "max_episode_steps", None)
    return float(length) if length else None

  # State.

  def sample_state(self, env_id: int = 0) -> dict[str, np.ndarray]:
    sample = self.sampler.sample(env_id)
    if not sample:
      raise NotSupported("sample_state")
    return sample

  def trace_labels(self) -> dict[str, list[str]]:
    labels = self.sampler.labels()
    names = list(getattr(self.sim, "joint_names", None) or [])
    if names:
      for channel in ("joint_pos", "joint_vel", "joint_torque"):
        labels.setdefault(channel, names)
    return labels

  def summary_metrics(self) -> dict[str, float]:
    return summary_metrics(self.env, self.sampler)

  def reset_envs(self, env_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Start fresh episodes through the environment's own ``reset(env_ids)``.

    Inside inference mode, for the reason the Genesis adapter gives: the
    buffers were made under it during rollout, and torch refuses an in-place
    write to an inference tensor from outside.
    """
    reset = getattr(self.env, self.spec.reset_fn, None)
    if not callable(reset):
      raise NotSupported(f"reset_envs: this environment has no {self.spec.reset_fn}().")
    total = self.num_envs()

    import torch

    if env_ids is None:
      with torch.inference_mode():
        reset(None)
      return {"num_reset": total, "env_ids": None}

    ids = sorted({int(i) for i in env_ids})
    out_of_range = [i for i in ids if not 0 <= i < total]
    if out_of_range:
      raise ValueError(
          f"No environment {out_of_range[0]}: this run has {total} "
          f"(0 to {total - 1})."
      )
    if not ids:
      return {"num_reset": 0, "env_ids": []}
    device = getattr(self.env, "device", None)
    with torch.inference_mode():
      reset(torch.tensor(ids, dtype=torch.long, device=device))
    return {"num_reset": len(ids), "env_ids": ids}

  # Rendering.

  def render(self, env_id: int = 0) -> np.ndarray:
    sim = self.sim
    render = getattr(sim, "render", None)
    if not callable(render):
      raise NotSupported(
          "This environment's backend has no render(env_id); frames are not "
          "available on it."
      )
    frame = render(int(env_id))
    return np.asarray(frame)

  def renderer_ready(self) -> bool:
    return callable(getattr(self.sim, "render", None))


__all__ = ["SingleFileSimAdapter"]
