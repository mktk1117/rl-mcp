"""The IsaacLab half of the wrapper: its adapter, and one startup check.

Everything else -- servicing, telemetry, curricula, records, progress clips --
comes from :mod:`rlmcp.adapters.manager_based.env_wrapper`, unchanged. That is
the point of the split: adding a backend should be a page, not a fork.

Usage, after IsaacLab's app is launched::

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    env = rlmcp.adapters.isaaclab.wrap(env, session_dir=log_dir / "rlmcp")
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg, log_dir, device)
    env.attach_runner(runner)
"""

from __future__ import annotations

from typing import Any

from rlmcp.adapters.isaaclab.sim_adapter import IsaacLabSimAdapter
from rlmcp.adapters.manager_based.env_wrapper import (
    RlMcpEnvWrapper as _ManagerBasedWrapper,
)
from rlmcp.adapters.manager_based.env_wrapper import TrainingStopped


class RlMcpEnvWrapper(_ManagerBasedWrapper):
  """Transparent wrapper over an IsaacLab ``ManagerBasedRLEnv``."""

  def build_sim_adapter(self, env: Any, robot_name: str | None) -> Any:
    return IsaacLabSimAdapter(env, robot_name=robot_name)

  def startup_checks(self) -> None:
    """Say at wrap time that this run cannot be looked at, not at the first ask.

    A frame needs `--enable_cameras` and `render_mode="rgb_array"`, both decided
    before rlmcp exists. Discovering that four hours in, when you finally want
    to see the robot, is the expensive way to find out.
    """
    if getattr(self.unwrapped, "render_mode", None) == "rgb_array":
      return
    print(
        "[rlmcp] this environment has no frames: `shot`, `video` and progress "
        "clips need the app launched with --enable_cameras and the env built "
        "with render_mode='rgb_array'. Everything else works without them.",
        flush=True,
    )


def wrap(env: Any, **kwargs: Any) -> RlMcpEnvWrapper:
  """Wrap an IsaacLab environment so rlmcp can watch and steer the run.

  See :class:`~rlmcp.adapters.manager_based.env_wrapper.RlMcpEnvWrapper` for the
  keyword arguments.
  """
  return RlMcpEnvWrapper(env, **kwargs)


__all__ = ["RlMcpEnvWrapper", "TrainingStopped", "wrap"]
