"""Always-on batch statistics.

These are computed across every environment once per iteration, so they have to
stay cheap: reductions on tensors already resident on the GPU, one ``.item()``
each. Anything that would need a per-step copy belongs in a trace instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from rlmcp.adapters.manager_based.sampling import first_command, is_velocity_command


def _try(out: dict[str, float], fn: Callable[[], None]) -> None:
  """Run one metric, skipping it if this environment cannot provide it."""
  try:
    fn()
  except Exception:
    pass


def summary_metrics(env: Any, robot_name: str) -> dict[str, float]:
  """Cheap scalars describing the current batch, prefixed ``rlmcp/``."""
  out: dict[str, float] = {}
  data = env.scene[robot_name].data

  def joint_motion() -> None:
    out["rlmcp/joint_vel_rms"] = float(torch.sqrt(torch.mean(data.joint_vel**2)).item())
    out["rlmcp/joint_acc_rms"] = float(torch.sqrt(torch.mean(data.joint_acc**2)).item())

  def action_rate() -> None:
    manager = env.action_manager
    delta = manager.action - manager.prev_action
    out["rlmcp/action_rate_rms"] = float(torch.sqrt(torch.mean(delta**2)).item())

  def tilt() -> None:
    projected = data.projected_gravity_b
    degrees = torch.rad2deg(torch.arccos(torch.clamp(-projected[:, 2], -1.0, 1.0)))
    out["rlmcp/tilt_deg_mean"] = float(torch.mean(degrees).item())

  def tracking() -> None:
    # Only a plane-velocity command can be subtracted from the base velocity;
    # on any other first command term (a motion target, a goal pose) the
    # metric is omitted rather than computed from the wrong signal.
    resolved = first_command(env)
    if resolved is None:
      return
    _, term, command = resolved
    if not is_velocity_command(term, command):
      return
    error = torch.linalg.norm(data.root_link_lin_vel_b[:, :2] - command[:, :2], dim=1)
    out["rlmcp/lin_vel_error_mean"] = float(torch.mean(error).item())
    out["rlmcp/commanded_speed_mean"] = float(
        torch.mean(torch.linalg.norm(command[:, :2], dim=1)).item()
    )

  def episode_progress() -> None:
    max_length = float(env.max_episode_length)
    if max_length > 0:
      out["rlmcp/episode_progress_mean"] = float(
          torch.mean(env.episode_length_buf.float()).item() / max_length
      )

  for metric in (joint_motion, action_rate, tilt, tracking, episode_progress):
    _try(out, metric)
  return out
