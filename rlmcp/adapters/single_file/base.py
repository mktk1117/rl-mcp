"""The base class of a single-file environment: the contract, written down once.

A single-file environment is one ``env.py``, built from the blocks in
:mod:`rlmcp.adapters.single_file.blocks`: the config is a dataclass at the top of the file,
declared with :mod:`rlmcp.declare`; the variables ``step()`` writes are a
:class:`~rlmcp.adapters.single_file.blocks.Variables` container; the observations are
:class:`~rlmcp.adapters.single_file.blocks.Obs` groups; and each reward term is a method named in
the config's reward table. rlmcp needs to know four things about such a
file, and this class is where they are said:

* **the config** -- ``self.cfg``, the declared dataclass. Every numeric leaf
  of it is a parameter an agent can list and set; ``Static[...]`` marks the
  ones read once at construction; ``term(...)`` declares a reward term.
* **the variables** -- ``self.state``, a
  :class:`~rlmcp.adapters.single_file.blocks.Variables` with one tensor per
  name and a leading ``num_envs`` axis. Every variable is sampled into a
  trace under its own name. The library attaches no meaning to a name: a
  variable named after a trace channel of :mod:`rlmcp.adapters.base`
  (``joint_pos``, ``joint_vel``, ``action``, ``base_lin_vel``,
  ``base_ang_vel``, ``base_pos``, ``projected_gravity``, ``foot_contact``,
  ``command``, ``reward``) also feeds the diagnostics that read it. A
  fixed-base arm simply does not declare the base ones, and the channels
  they feed are dropped rather than faked.
* **the boundaries** -- ``reset(env_ids)`` restarts episodes and ``step()``
  returns ``(obs, reward, done, info)``, with ``info`` carrying the keys
  :meth:`step_info` builds.
* **the reward table** -- ``cfg.reward`` holds a ``term(weight, **params)``
  per term, and each names a method of the environment with the same
  signature. :meth:`compute_reward` sums ``weight * method(**params)`` over
  the table; a term appended at runtime through ``rlmcp add-reward``
  brings its own function and is scored the same way, so the file never
  has to know a term was added.

The physics is not part of the contract. ``self.sim`` is whatever the file
built -- a MuJoCo Warp batch, a Genesis scene, anything -- and rlmcp asks it
for two optional things only: ``render(env_id)`` for frames, and ``mj_model``
for the live view when it has one.

The observation groups and pipes are found by looking: any
:class:`~rlmcp.adapters.single_file.blocks.Obs` or
:class:`~rlmcp.adapters.single_file.blocks.Pipe` assigned to an
attribute of the environment is served under that attribute's name
(``actor_obs.joint_vel.uniform_noise.half_width``), and :meth:`reset_blocks` clears
their history on an episode reset.

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
from rlmcp.adapters.single_file import blocks


class SingleFileEnv(ABC):
  """One environment in one file. See the module docstring for the contract."""

  cfg: Any
  """The declared config dataclass."""
  state: Any = None
  """The :class:`~rlmcp.adapters.single_file.blocks.Variables` holding every
  variable ``step()`` writes."""
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

  # The blocks.

  def blocks(self) -> dict[str, blocks.Block]:
    """The :class:`~rlmcp.adapters.single_file.blocks.Obs` and
    :class:`~rlmcp.adapters.single_file.blocks.Pipe`
    attributes of this environment, by attribute name."""
    return blocks.blocks(self)

  def reset_blocks(self, env_ids: Tensor | None = None) -> None:
    """Tell every pipe stage with history (a :class:`~rlmcp.adapters.single_file.blocks.Delay`)
    that ``env_ids`` start over. Call it from ``reset()``."""
    for block in self.blocks().values():
      block.reset(env_ids)

  # The reward table.

  def reward_table(self) -> dict[str, declare.Term]:
    """The terms on ``cfg.<reward_group>``, including any added at runtime."""
    return declare.terms(getattr(self.cfg, self.reward_group, None))

  def reward_function(self, name: str, term: declare.Term) -> Any:
    """The callable behind a term: its own ``func(env, **params)`` when it
    carries one, else the method of this environment named ``name``,
    called as ``method(**params)``."""
    if term.func is not None:
      return lambda **params: term.func(self, **params)
    method = getattr(self, name, None)
    if not callable(method):
      raise KeyError(
          f"Reward term '{name}' is in the table but this environment has no "
          f"method '{name}' to score it, and the term carries no function."
      )
    return method

  def compute_reward(self, scale: float = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
    """``sum(weight * term(**params)) * scale`` over the table, plus the
    per-term values.

    ``scale`` is for a task that multiplies every term by the control
    timestep, as mjlab's reward manager does.
    """
    total = torch.zeros(self.num_envs, device=self.device)
    scored: dict[str, Tensor] = {}
    for name, term in self.reward_table().items():
      value = self.reward_function(name, term)(**term.params)
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
