"""The mjlab half of the wrapper: its adapter, and its startup check.

Everything else -- servicing, telemetry, curricula, records, progress clips --
is in :mod:`rlmcp.adapters.env_wrapper`, because none of it is
mjlab's. What is mjlab's is which SimAdapter to build and one diagnostic that
has caught the same class of bug twice.
"""

from __future__ import annotations

from typing import Any

from rlmcp.adapters.env_wrapper import (
    RlMcpEnvWrapper as _ManagerBasedWrapper,
)
from rlmcp.adapters.env_wrapper import TrainingStopped
from rlmcp.adapters.mjlab.sim_adapter import MjlabSimAdapter
from rlmcp.adapters.mjlab.viz_check import check_marker_colors
from rlmcp.core.palette import format_report


class RlMcpEnvWrapper(_ManagerBasedWrapper):
  """Transparent wrapper over an mjlab ``ManagerBasedRlEnv``."""

  def build_sim_adapter(self, env: Any, robot_name: str | None) -> Any:
    return MjlabSimAdapter(env, robot_name=robot_name)

  def startup_checks(self) -> None:
    self._warn_about_marker_colors()

  def _warn_about_marker_colors(self) -> None:
    """Say so at startup if a debug overlay is the colour of a scene object.

    Twice in this project a visualisation has made the system look like it was
    doing something it was not -- a landing marker the same yellow as a ball,
    and neighbouring envs composited into one frame -- and both times the search
    went after a physics bug that did not exist. The check is cheap, the failure
    is silent, and nobody thinks to look for it, so it runs unasked. It only
    ever warns: a colour clash is a legibility problem, not a reason to refuse
    to train.
    """
    try:
      collisions = check_marker_colors(self.unwrapped)
    except Exception as exc:  # a diagnostic must never take the run down
      self.rlmcp.session.append_event(
          "viz_check_failed", {"error": f"{type(exc).__name__}: {exc}"}
      )
      return
    if not collisions:
      return
    print(f"[rlmcp] {format_report(collisions)}", flush=True)
    self.rlmcp.session.append_event("viz_color_collision", {"collisions": collisions})


def wrap(env: Any, **kwargs: Any) -> RlMcpEnvWrapper:
  """Wrap an mjlab environment so rlmcp can watch and steer the run.

  See :class:`~rlmcp.adapters.env_wrapper.RlMcpEnvWrapper` for the
  keyword arguments.
  """
  return RlMcpEnvWrapper(env, **kwargs)


__all__ = ["RlMcpEnvWrapper", "TrainingStopped", "wrap"]
