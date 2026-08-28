"""Always-on batch statistics for a legged-gym-shaped environment.

Computed across every environment once per iteration, so they stay cheap:
reductions on tensors already on the device, one ``.item()`` each. Anything
needing a per-step copy belongs in a trace instead.

Every metric is optional and each is computed on its own. An environment
missing a buffer loses that number and keeps the rest -- and a metric whose
meaning cannot be verified is omitted rather than computed from the nearest
thing to hand. The tracking error is the one that matters here: subtracting a
command from a base velocity is only meaningful when the command *is* a plane
velocity, so it is skipped unless the sampler says so.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from rlmcp.adapters.legged_gym_style.sampling import StateSampler


def _try(out: Dict[str, float], fn: Callable[[], None]) -> None:
  """Run one metric, skipping it if this environment cannot provide it."""
  try:
    fn()
  except Exception:
    pass


def summary_metrics(env: Any, sampler: Optional[StateSampler] = None) -> Dict[str, float]:
  """Cheap scalars describing the current batch, prefixed ``rlmcp/``."""
  out: Dict[str, float] = {}
  sampler = sampler or StateSampler(env)

  def joint_motion() -> None:
    out["rlmcp/joint_vel_rms"] = float(
        torch.sqrt(torch.mean(env.dof_vel.float() ** 2)).item()
    )

  def action_rate() -> None:
    """Only when the environment still holds two different actions.

    Genesis's Go2Env ends `step()` with `last_actions.copy_(actions)`, so by
    the time rlmcp services a command at the iteration boundary the two buffers
    are identical and the difference is exactly zero -- not "the policy is
    smooth", but "the question cannot be asked here". Publishing 0.0 would be a
    number computed from the wrong signal, and it read as a perfectly smooth
    policy on a robot that was visibly buzzing.

    Exact equality is the detector rather than a threshold: a real converged
    policy gets close to zero but not to the bit pattern. A fork that keeps the
    previous action distinct still gets the metric. Per-step action rate is
    always available from `trace` and `diagnose`, which record the action
    channel every step and measure the rate there.
    """
    if torch.equal(env.actions, env.last_actions):
      return
    delta = env.actions - env.last_actions
    out["rlmcp/action_rate_rms"] = float(
        torch.sqrt(torch.mean(delta.float() ** 2)).item()
    )

  def tilt() -> None:
    projected = env.projected_gravity
    degrees = torch.rad2deg(
        torch.arccos(torch.clamp(-projected[:, 2].float(), -1.0, 1.0))
    )
    out["rlmcp/tilt_deg_mean"] = float(torch.mean(degrees).item())

  def tracking() -> None:
    commands = getattr(env, "commands", None)
    if commands is None or commands.ndim != 2:
      return
    width = int(commands.shape[-1])
    if not sampler.commands_are_velocities(width):
      return
    measured = env.base_lin_vel[:, :2].float()
    error = torch.linalg.norm(measured - commands[:, :2].float(), dim=1)
    out["rlmcp/lin_vel_error_mean"] = float(torch.mean(error).item())
    out["rlmcp/commanded_speed_mean"] = float(
        torch.mean(torch.linalg.norm(commands[:, :2].float(), dim=1)).item()
    )
    # The do-nothing detector. Ask of any locomotion run: if the policy learned
    # to stand still, which logged number would go down? Reward will not --
    # a standing robot collects posture and action-rate rewards and stops
    # falling over, so reward and episode length both *rise*. Tracking error
    # rises too, but it rises for every bad policy and says nothing about which
    # kind. Measured ground speed goes to zero and nothing else does, which is
    # what makes it worth its own line rather than an inference from two others.
    out["rlmcp/achieved_speed_mean"] = float(
        torch.mean(torch.linalg.norm(measured, dim=1)).item()
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


def episode_log(extras: Any) -> Dict[str, float]:
  """The per-term episode means this family reports, as flat telemetry.

  ``_reset_idx`` fills ``extras["episode"]`` with one entry per reward term,
  already averaged over the environments that reset. The controller wants flat
  ``{name: float}``, so that is what this returns -- and nothing else from
  ``extras``, since ``time_outs`` is a per-env tensor the runner consumes, not
  a scalar anybody plots.
  """
  if not isinstance(extras, dict):
    return {}
  episode = extras.get("episode")
  if not isinstance(episode, dict):
    return {}
  out: Dict[str, float] = {}
  for key, value in episode.items():
    try:
      out[str(key)] = float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
      continue
  return out
