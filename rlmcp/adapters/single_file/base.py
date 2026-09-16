"""The base class of a single-file environment: the contract, written down once.

A single-file environment is one ``env.py``: the config is a dataclass at the
top of the file, declared with :mod:`rlmcp.declare`, and ``reset()``,
``step()`` and the reward terms are written out below it. rlmcp needs to
know four things about such a file, and this class is where they are said:

* **the config** -- ``self.cfg``, the declared dataclass. Every numeric leaf
  of it is a parameter an agent can list and set; ``Static[...]`` marks the
  ones read once at construction; ``term(...)`` declares a reward term.
* **the state** -- tensors on the environment with a leading ``num_envs``
  axis, under the names the trace reads. ``dof_pos``, ``dof_vel`` and
  ``actions`` are required. ``base_pos``, ``base_quat``, ``base_lin_vel``,
  ``base_ang_vel``, ``projected_gravity`` and ``commands`` are for a floating
  base and a commanded task; a fixed-base arm or hand simply does not define
  them, and the channels they feed are dropped rather than faked.
* **the boundaries** -- ``reset(env_ids)`` restarts episodes and ``step()``
  returns ``(obs, reward, done, info)``, with ``info`` carrying the keys
  :meth:`step_info` builds.
* **the reward table** -- :meth:`compute_reward_terms` scores the terms the
  file computes inline; :meth:`compute_reward` weights them by the table on
  ``cfg.reward`` and scores any term that was appended at runtime through its
  own function. That loop lives here so ``rlmcp add-reward`` works on every
  subclass without each file carrying the loop.

The physics is not part of the contract. ``self.sim`` is whatever the file
built -- a MuJoCo Warp batch, a Genesis scene, anything -- and rlmcp asks it
for two optional things only: ``render(env_id)`` for frames, and ``mj_model``
for the live view when it has one.

Inheriting is the documented path. The wrapper also accepts any object of
the same shape (:mod:`rlmcp.adapters.single_file.spec` checks it at wrap
time), so a file that cannot inherit still works.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from rlmcp import declare


class SingleFileEnv(ABC):
  """One environment in one file. See the module docstring for the contract."""

  cfg: Any
  """The declared config dataclass."""
  num_envs: int
  device: torch.device
  control_dt: float
  """Seconds per policy step."""
  max_episode_steps: int
  sim: Any = None
  """The physics, if the file keeps a handle. Optional ``render(env_id)`` and
  ``mj_model`` are all rlmcp asks of it."""

  reward_group: str = "reward"
  """The field of ``cfg`` holding the reward table."""

  # The boundaries.

  @abstractmethod
  def reset(self, env_ids: Tensor | None = None) -> Any:
    """Start fresh episodes for ``env_ids``; ``None`` restarts every one.
    Returns the observation."""

  @abstractmethod
  def step(self, actions: Tensor) -> tuple[Any, Tensor, Tensor, dict[str, Any]]:
    """One control step. Returns ``(obs, reward, done, info)``; build ``info``
    with :meth:`step_info` so the wrapper finds what it logs."""

  # The reward table.

  @abstractmethod
  def compute_reward_terms(self) -> dict[str, Tensor]:
    """Every term the file computes inline, ``{name: (num_envs,) tensor}``,
    keyed by the field name in the reward table. Unweighted."""

  def reward_table(self) -> dict[str, declare.Term]:
    """The terms on ``cfg.<reward_group>``, including any added at runtime."""
    return declare.terms(getattr(self.cfg, self.reward_group, None))

  def compute_reward(self, scale: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    """``sum(weight * value) * scale`` over the table, plus the per-term values.

    A term the table has and :meth:`compute_reward_terms` did not score is
    one that was appended at runtime; it carries its function and is scored
    by ``func(env, **params)``. ``scale`` is for a task that multiplies every
    term by the control timestep, as mjlab's reward manager does.
    """
    computed = self.compute_reward_terms()
    total = torch.zeros(self.num_envs, device=self.device)
    scored: dict[str, Tensor] = {}
    for name, term in self.reward_table().items():
      if name in computed:
        value = computed[name]
      elif term.func is not None:
        value = term.func(self, **term.params)
      else:
        raise KeyError(
            f"Reward term '{name}' is in the table but compute_reward_terms() "
            "did not score it and it carries no function."
        )
      scored[name] = value
      total += term.weight * value * scale
    return total, scored

  # What step() reports.

  @staticmethod
  def step_info(
      reward_terms: dict[str, Tensor],
      episode_rewards: Tensor,
      episode_lengths: Tensor,
      time_outs: Tensor,
  ) -> dict[str, Any]:
    """The ``info`` dict the wrapper reads into per-iteration telemetry.

    ``reward_terms`` is the per-term value (means are taken here);
    ``episode_rewards`` and ``episode_lengths`` are the totals of the episodes
    that ended this step; ``time_outs`` marks the ``done`` envs that ended on
    the clock rather than by failing, which a PPO update bootstraps.
    """
    return {
        "reward_terms": {name: float(v.mean()) for name, v in reward_terms.items()},
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "time_outs": time_outs,
    }


__all__ = ["SingleFileEnv"]
