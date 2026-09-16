"""Reading a single-file environment's variables: the trace sampler and the
per-iteration summary metrics.

The environment keeps its variables in a
:class:`~rlmcp.adapters.single_file.blocks.Variables` container on
``env.state``: every tensor ``step()`` writes, declared once with its shape
and labels. The sampler here serves *all* of them, each under its own name
with its own labels. The library attaches no meaning to a name. The trace
vocabulary in :mod:`rlmcp.adapters.base` (``joint_pos``, ``base_lin_vel``,
``command``, ``foot_contact``, ``reward``, ...) is the one convention: a
variable named after a channel feeds the diagnostics that read that
channel, and ``command`` in particular is, by that vocabulary's definition,
a plane velocity ``[vx, vy(, wz)]`` -- a goal or a motion target goes under
another name. Everything else is recorded and plotted by name.

An environment that keeps its variables as plain attributes instead (the
legged_gym layout: ``dof_pos``, ``base_lin_vel``, ``commands``) is read by
that family's sampler and metrics, unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlmcp.adapters.base import (
    CHANNEL_BASE_LIN_VEL,
    CHANNEL_COMMAND,
    CHANNEL_JOINT_POS,
    CHANNEL_JOINT_VEL,
    CHANNEL_PROJECTED_GRAVITY,
)
from rlmcp.adapters.legged_gym_style import metrics as legged_metrics
from rlmcp.adapters.legged_gym_style.sampling import StateSampler as LeggedSampler
from rlmcp.adapters.single_file.blocks import Variables
from rlmcp.adapters.single_file.spec import SingleFileSpec


class StateSampler:
  """Reads one environment's variables, one device sync per step."""

  def __init__(self, env: Any, spec: SingleFileSpec | None = None,
               command_names: list[str] | None = None):
    self.env = env
    self.spec = spec or SingleFileSpec()
    self._legged = LeggedSampler(env, command_names=command_names)

  @property
  def state(self) -> Variables | None:
    state = self.spec.resolve(self.env, "state", None)
    return state if isinstance(state, Variables) else None

  def channels(self) -> dict[str, torch.Tensor]:
    """``{name: tensor}`` for every variable, or the legged_gym buffers."""
    state = self.state
    if state is None:
      return self._legged.channels()
    return dict(state.vars())

  def labels(self) -> dict[str, list[str]]:
    """Component names per variable, from the labelled axes."""
    state = self.state
    if state is None:
      return self._legged.labels()
    return {name: list(state.labels(name) or []) for name in state.names() if state.labels(name)}

  def joint_names(self) -> list[str]:
    """The labels of the ``joint_pos`` variable, when it has any."""
    state = self.state
    if state is None:
      return self._legged.joint_names()
    if CHANNEL_JOINT_POS in state:
      return list(state.labels(CHANNEL_JOINT_POS) or [])
    return []

  def sample(self, env_id: int = 0) -> dict[str, np.ndarray]:
    """One step of every variable for ``env_id``, in one copy off the device."""
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


def summary_metrics(env: Any, sampler: StateSampler) -> dict[str, float]:
  """Cheap batch statistics, prefixed ``rlmcp/``, from the channels the
  environment declares. Each is computed on its own and skipped when its
  channel is absent; a metric whose meaning cannot be verified is omitted
  rather than computed from the nearest thing to hand."""
  state = sampler.state
  if state is None:
    return legged_metrics.summary_metrics(env, sampler._legged)
  out: dict[str, float] = {}
  channels = state.vars()

  joint_vel = channels.get(CHANNEL_JOINT_VEL)
  if joint_vel is not None:
    out["rlmcp/joint_vel_rms"] = float(torch.sqrt(torch.mean(joint_vel.float() ** 2)).item())

  gravity = channels.get(CHANNEL_PROJECTED_GRAVITY)
  if gravity is not None and gravity.ndim == 2 and gravity.shape[1] == 3:
    degrees = torch.rad2deg(torch.arccos(torch.clamp(-gravity[:, 2].float(), -1.0, 1.0)))
    out["rlmcp/tilt_deg_mean"] = float(torch.mean(degrees).item())

  command = channels.get(CHANNEL_COMMAND)
  measured = channels.get(CHANNEL_BASE_LIN_VEL)
  if (command is not None and measured is not None and command.ndim == 2
      and command.shape[1] in (2, 3) and measured.ndim == 2 and measured.shape[1] >= 2):
    planar = measured[:, :2].float()
    error = torch.linalg.norm(planar - command[:, :2].float(), dim=1)
    out["rlmcp/lin_vel_error_mean"] = float(torch.mean(error).item())
    out["rlmcp/commanded_speed_mean"] = float(
        torch.mean(torch.linalg.norm(command[:, :2].float(), dim=1)).item())
    # The do-nothing detector: a policy that learned to stand still is the one
    # bad policy whose reward and episode length both rise; its ground speed
    # is the number that goes to zero.
    out["rlmcp/achieved_speed_mean"] = float(torch.mean(torch.linalg.norm(planar, dim=1)).item())
  return out


__all__ = ["StateSampler", "summary_metrics"]
