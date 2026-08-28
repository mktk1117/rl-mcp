"""The Genesis half of the wrapper: its adapter, and one startup check.

Everything else -- servicing, telemetry, curricula, records, progress clips --
comes from :mod:`rlmcp.adapters.env_wrapper`, unchanged. That is the point of
the split: adding a backend should be a page, not a fork.

Usage::

    env = Go2Env(num_envs=4096, env_cfg=..., obs_cfg=..., reward_cfg=...,
                 command_cfg=...)
    env = rlmcp.adapters.genesis.wrap(env, session_dir=log_dir / "rlmcp")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    env.attach_runner(runner)
"""

from __future__ import annotations

from typing import Any

from rlmcp.adapters.env_wrapper import RlMcpEnvWrapper as _BaseWrapper
from rlmcp.adapters.env_wrapper import TrainingStopped
from rlmcp.adapters.genesis import rendering
from rlmcp.adapters.genesis.sim_adapter import GenesisSimAdapter
from rlmcp.adapters.legged_gym_style.spec import FlatEnvSpec


class RlMcpEnvWrapper(_BaseWrapper):
  """Transparent wrapper over a Genesis environment."""

  def __init__(self, env: Any, spec: FlatEnvSpec | None = None, **kwargs: Any):
    self._spec = spec
    super().__init__(env, **kwargs)

  def build_sim_adapter(self, env: Any, robot_name: str | None) -> Any:
    # robot_name is part of the contract for scenes that hold several named
    # articulations. A Genesis environment holds its robot in one attribute, so
    # the name is carried by the spec instead and this argument is unused.
    return GenesisSimAdapter(env, spec=self._spec)

  def startup_checks(self) -> None:
    """Say at wrap time what this run cannot show you, and what was assumed.

    Both rendering facts were settled before the scene was built and neither
    can be changed now, so the useful moment to hear them is when the run
    starts -- not four hours in, when you finally want to look at the robot.
    The third is the one thing rlmcp cannot check for itself.
    """
    env = self.unwrapped
    sim = self.rlmcp.sim

    if sim.spec.scales_premultiplied_by_dt:
      print(
          "[rlmcp] reward weights are read and written in the units the task "
          "config used. This environment multiplies its scales by dt at "
          "construction, in place, so the configured number cannot be read "
          "back and is reconstructed instead. Pass "
          "wrap(spec=FlatEnvSpec(scales_premultiplied_by_dt=False)) if yours "
          "does not do that.",
          flush=True,
      )

    if rendering.observing_camera(env) is None:
      print(f"[rlmcp] {rendering.NO_CAMERA}", flush=True)
      return
    drawn = rendering.rendered_envs(env)
    if drawn is not None and len(drawn) < sim.num_envs():
      print(
          f"[rlmcp] frames are available for environments {sorted(drawn)} "
          "only; the scene was built to draw those. `shot --env-id` outside "
          "that set is refused rather than answered with the wrong robot.",
          flush=True,
      )


def wrap(env: Any, **kwargs: Any) -> RlMcpEnvWrapper:
  """Wrap a Genesis environment so rlmcp can watch and steer the run.

  See :class:`~rlmcp.adapters.env_wrapper.RlMcpEnvWrapper` for the keyword
  arguments, and ``spec=`` for an environment whose attributes are not the
  conventional ones.
  """
  return RlMcpEnvWrapper(env, **kwargs)


__all__ = ["RlMcpEnvWrapper", "TrainingStopped", "wrap"]
