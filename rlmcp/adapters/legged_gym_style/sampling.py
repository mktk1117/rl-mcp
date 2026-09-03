"""Per-step state sampling for a legged-gym-shaped environment.

Same job and same cost model as the manager-based sampler: a trace pulls a
dozen signals off the GPU every step, and doing that one field at a time would
pay a dozen synchronisations. Everything is concatenated into one tensor,
copied once, and split back on the host.

What differs is only the lookup. A manager-based env is asked
``scene[robot].data.joint_pos``; here the buffers are attributes on the
environment itself, written by ``step()`` and named by convention. So the map
below *is* the adapter -- and a channel whose buffer is absent is omitted
rather than faked, which is what keeps the diagnosis honest: Genesis's own
``Go2Env`` keeps no contact or torque buffer, so the gait and effort sections
of ``diagnose`` are skipped for it rather than computed from something else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlmcp.adapters.base import (
    CHANNEL_ACTION,
    CHANNEL_BASE_ANG_VEL,
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_BASE_POS,
    CHANNEL_COMMAND,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_VEL,
    CHANNEL_PROJECTED_GRAVITY,
    CHANNEL_REWARD,
)

CHANNEL_BUFFERS = {
    CHANNEL_JOINT_POS: "dof_pos",
    CHANNEL_JOINT_VEL: "dof_vel",
    CHANNEL_ACTION: "actions",
    CHANNEL_BASE_LIN_VEL: "base_lin_vel",
    CHANNEL_BASE_ANG_VEL: "base_ang_vel",
    CHANNEL_BASE_POS: "base_pos",
    CHANNEL_PROJECTED_GRAVITY: "projected_gravity",
    CHANNEL_REWARD: "rew_buf",
}
"""Trace channel -> the attribute this family keeps it in. Conventional names,
checked at sample time rather than assumed: a missing one drops its channel."""

VELOCITY_COMMAND_PREFIXES = ("lin_vel", "ang_vel")
"""What makes a command channel a *plane velocity*, which is what
``CHANNEL_COMMAND`` means. A goal pose or a motion target under this key would
be silently subtracted from the base velocity by the tracking diagnostics, so
the command buffer is only published under that name when the channel names say
it is one."""


class StateSampler:
  """Reads one environment's signals, one device sync per step."""

  def __init__(self, env: Any, command_names: list[str] | None = None):
    self.env = env
    self.command_names = list(command_names or [])

  # Which channels this environment can actually serve.

  def _buffer(self, attr: str) -> torch.Tensor | None:
    value = getattr(self.env, attr, None)
    return value if isinstance(value, torch.Tensor) and value.ndim >= 1 else None

  def commands_are_velocities(self, width: int) -> bool:
    if width not in (2, 3):
      return False
    if not self.command_names:
      return False
    return all(
        name.startswith(VELOCITY_COMMAND_PREFIXES)
        for name in self.command_names[:width]
    )

  def channels(self) -> dict[str, torch.Tensor]:
    """``{channel: buffer}`` for everything this environment has right now."""
    out: dict[str, torch.Tensor] = {}
    for channel, attr in CHANNEL_BUFFERS.items():
      buffer = self._buffer(attr)
      if buffer is not None:
        out[channel] = buffer
    commands = self._buffer("commands")
    if commands is not None and commands.ndim == 2:
      width = int(commands.shape[-1])
      name = (
          CHANNEL_COMMAND if self.commands_are_velocities(width)
          else "command_raw"
      )
      out[name] = commands
    return out

  # Sampling.

  def sample(self, env_id: int = 0) -> dict[str, np.ndarray]:
    """One step of signals for ``env_id``, in one copy off the device."""
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

  # Labels, derived from the environment rather than assumed from the task.

  def labels(self) -> dict[str, list[str]]:
    names = self.joint_names()
    out: dict[str, list[str]] = {}
    if names:
      out[CHANNEL_JOINT_POS] = list(names)
      out[CHANNEL_JOINT_VEL] = list(names)
      actions = self._buffer("actions")
      if actions is not None and int(actions.shape[-1]) == len(names):
        out[CHANNEL_ACTION] = list(names)
    if self.command_names:
      commands = self._buffer("commands")
      width = int(commands.shape[-1]) if commands is not None else 0
      if width and len(self.command_names) >= width:
        key = (
            CHANNEL_COMMAND if self.commands_are_velocities(width)
            else "command_raw"
        )
        out[key] = [f"cmd_{n}" for n in self.command_names[:width]]
    return out

  def joint_names(self) -> list[str]:
    """The names the task gave its joints, when it gave them any.

    ``env_cfg["joint_names"]`` is the convention. Without it the components are
    numbered by the recorder, which is honest; a wrong label is worse than no
    label, because it misdirects whoever reads the diagnosis.
    """
    cfg = getattr(self.env, "env_cfg", None)
    names = cfg.get("joint_names") if isinstance(cfg, dict) else None
    if not isinstance(names, (list, tuple)):
      return []
    return [str(n) for n in names]
