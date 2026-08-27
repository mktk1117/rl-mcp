"""Frames from an IsaacLab environment.

IsaacLab renders through the running Kit app rather than through an offscreen
renderer the adapter can build, so there is nothing to construct lazily here --
either the app was launched with cameras enabled and the environment was built
with ``render_mode="rgb_array"``, or no frame exists. Both of those are
decisions made before rlmcp is in the picture, which is why the failure has to
be an explanation rather than an exception nobody can act on.

Which environment a frame shows is ``cfg.viewer.env_index``, the same field the
viewport controller reads, so pointing the camera at env 7 and putting it back
is the whole of per-env rendering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlmcp.adapters.base import NotSupported

_ENABLE_CAMERAS = (
    "IsaacLab renders through the Kit app: launch it with `--enable_cameras` "
    "and build the environment with `render_mode=\"rgb_array\"`. Without both, "
    "no frame exists to return."
)


def renderer_ready(env: Any) -> bool:
  """Whether a frame would come back without an app-level change."""
  return str(getattr(env, "render_mode", "") or "") == "rgb_array"


def render(env: Any, env_id: int = 0) -> np.ndarray:
  """One environment as RGB, with the viewer put back where it was."""
  if not renderer_ready(env):
    raise NotSupported(
        f"This environment's render_mode is "
        f"{getattr(env, 'render_mode', None)!r}. {_ENABLE_CAMERAS}")

  viewer = getattr(getattr(env, "cfg", None), "viewer", None)
  previous = getattr(viewer, "env_index", None) if viewer is not None else None
  if viewer is not None and previous is not None:
    viewer.env_index = int(max(0, min(env_id, int(env.num_envs) - 1)))
    # The controller owns the camera; asking it to follow is what makes the
    # field take effect on this frame rather than the next reset.
    controller = getattr(env, "viewport_camera_controller", None)
    setter = getattr(controller, "set_view_env_index", None)
    if callable(setter):
      try:
        setter(viewer.env_index)
      except Exception:  # noqa: BLE001 -- a camera that will not move is not fatal.
        pass
  try:
    frame = env.render()
  except RuntimeError as exc:  # the simulator's own "cannot render" message
    raise NotSupported(f"{exc} {_ENABLE_CAMERAS}") from exc
  finally:
    if viewer is not None and previous is not None:
      viewer.env_index = previous

  if frame is None:
    raise NotSupported(f"IsaacLab returned no frame. {_ENABLE_CAMERAS}")
  array = np.asarray(frame)
  # The annotator hands back RGBA; everything downstream writes RGB.
  return array[:, :, :3] if array.ndim == 3 and array.shape[2] == 4 else array
