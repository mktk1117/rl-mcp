"""Reading a single-file environment's variables: the trace sampler and the
per-iteration summary metrics.

The environment keeps its variables in a :class:`~rlmcp.blocks.Vars`
container on ``env.state`` -- every tensor ``step()`` writes, declared once
with its shape and labels. The sampler here serves *all* of them: the ones
with conventional names go out under the trace channels the diagnostics know
(``joint_pos``, ``base_lin_vel``, ``command``, ...), and everything else under
its own name, so an agent tracing the run sees the foot air time and the
push timer next to the joint positions without the file listing anything.

An environment that keeps its variables as plain attributes instead (the
legged_gym layout) is served the same way through the conventional names,
which is what keeps the old shape working.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlmcp.adapters.base import CHANNEL_COMMAND
from rlmcp.adapters.legged_gym_style import metrics as flat_metrics
from rlmcp.adapters.legged_gym_style.sampling import (
    CHANNEL_BUFFERS,
    VELOCITY_COMMAND_PREFIXES,
)
from rlmcp.adapters.single_file.spec import SingleFileSpec
from rlmcp.blocks import Vars

ALIASES = {
    "dof_pos": "joint_pos",
    "dof_vel": "joint_vel",
    "dof_torque": "joint_torque",
    "torques": "joint_torque",
    "actions": "action",
    "commands": "command",
    "rew_buf": "reward",
    "episode_length_buf": "episode_length",
}
"""Variable name -> trace channel, for the names legged_gym spelled
differently. A variable already named after its channel needs no entry."""


def channel_of(name: str) -> str:
  return ALIASES.get(name, name)


class StateSampler:
  """Reads one environment's variables, one device sync per step."""

  def __init__(self, env: Any, spec: SingleFileSpec | None = None,
               command_names: list[str] | None = None):
    self.env = env
    self.spec = spec or SingleFileSpec()
    self._command_names = list(command_names or [])

  @property
  def state(self) -> Vars | None:
    state = self.spec.resolve(self.env, "state", None)
    return state if isinstance(state, Vars) else None

  # Lookup by either spelling.

  def buffer(self, name: str) -> torch.Tensor | None:
    """The tensor behind ``name`` -- a variable or an attribute, under its
    own name or its legged_gym alias."""
    channel = channel_of(name)
    candidates = [name, channel] + [k for k, v in ALIASES.items() if v == channel]
    state = self.state
    for candidate in candidates:
      if state is not None and candidate in state:
        return state[candidate]
      value = getattr(self.env, candidate, None)
      if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return value
    return None

  def command_names(self) -> list[str]:
    """The command columns: the variable's labels, else the config's ranges."""
    state = self.state
    if state is not None:
      for name in ("commands", "command"):
        if name in state and state.labels(name):
          return list(state.labels(name) or [])
    return self._command_names

  def commands_are_velocities(self, width: int) -> bool:
    names = self.command_names()
    if width not in (2, 3) or not names:
      return False
    return all(n.startswith(VELOCITY_COMMAND_PREFIXES) for n in names[:width])

  # Channels.

  def channels(self) -> dict[str, torch.Tensor]:
    """``{channel: tensor}`` for everything this environment has right now."""
    out: dict[str, torch.Tensor] = {}
    state = self.state
    if state is not None:
      for name, tensor in state.vars().items():
        out[channel_of(name)] = tensor
    else:
      for channel, attr in CHANNEL_BUFFERS.items():
        value = getattr(self.env, attr, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
          out[channel] = value
      commands = getattr(self.env, "commands", None)
      if isinstance(commands, torch.Tensor) and commands.ndim == 2:
        out[CHANNEL_COMMAND] = commands
    commands = out.pop(CHANNEL_COMMAND, None)
    if commands is not None and commands.ndim == 2:
      width = int(commands.shape[-1])
      out[CHANNEL_COMMAND if self.commands_are_velocities(width) else "command_raw"] = commands
    return out

  def labels(self) -> dict[str, list[str]]:
    """Component names per channel, from the variables' labelled axes."""
    out: dict[str, list[str]] = {}
    state = self.state
    if state is None:
      return out
    for name in state.names():
      labels = state.labels(name)
      if labels:
        channel = channel_of(name)
        if channel == CHANNEL_COMMAND and not self.commands_are_velocities(len(labels)):
          channel = "command_raw"
        out[channel] = list(labels)
    return out

  def joint_names(self) -> list[str]:
    state = self.state
    if state is not None:
      for name in ("joint_pos", "dof_pos"):
        if name in state and state.labels(name):
          return list(state.labels(name) or [])
    return []

  def sample(self, env_id: int = 0) -> dict[str, np.ndarray]:
    """One step of every channel for ``env_id``, in one copy off the device."""
    channels = self.channels()
    if not channels:
      return {}
    rows, widths = [], []
    for name in sorted(channels):
      row = channels[name][env_id]
      row = row.reshape(1) if row.ndim == 0 else row.reshape(-1)
      rows.append(row.to(torch.float32))
      widths.append((name, int(row.numel())))
    flat = torch.cat(rows).detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    start = 0
    for name, width in widths:
      out[name] = flat[start:start + width]
      start += width
    return out


class _Buffers:
  """What the shared summary metrics read, resolved through the sampler, so
  they work over a ``Vars`` container and over plain attributes alike."""

  def __init__(self, env: Any, sampler: StateSampler):
    self._env = env
    self._sampler = sampler

  @property
  def max_episode_length(self) -> float:
    return float(self._sampler.spec.resolve(self._env, "max_episode_steps", 0) or 0)

  def __getattr__(self, name: str) -> Any:
    value = self._sampler.buffer(name)
    if value is None:
      raise AttributeError(name)
    return value


def summary_metrics(env: Any, sampler: StateSampler) -> dict[str, float]:
  """The legged_gym-style batch statistics, over whichever layout the
  environment uses."""
  return flat_metrics.summary_metrics(_Buffers(env, sampler), sampler)


__all__ = ["ALIASES", "StateSampler", "channel_of", "summary_metrics"]
