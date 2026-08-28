"""Offscreen rendering of a chosen environment, built only when asked for.

mjlab constructs its offscreen renderer in ``ManagerBasedRlEnv.__init__`` when
``render_mode="rgb_array"``, which means asking for the *option* of a screenshot
costs GPU memory for the whole run whether or not one is ever taken.

On a task that already sits near the limit that is the difference between
fitting and not: the SMP steering task at 2048 envs occupies 21-22.5 GB of a
24 GB card, and the eager renderer pushes it into a Warp CUDA-graph-capture
failure. So rlmcp builds the renderer on the first frame instead, and a run
that never takes a screenshot pays nothing.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from rlmcp.adapters.base import NotSupported

_ATTR = "_offline_renderer"


def renderer_ready(env: Any) -> bool:
  """Whether a renderer already exists, eagerly or from an earlier frame."""
  return getattr(env, _ATTR, None) is not None


def ensure_renderer(env: Any) -> Any:
  """Return the env's offscreen renderer, constructing it if needed.

  Built exactly as mjlab builds it, from the environment's own model, scene and
  viewer config -- so ``env.render()`` works afterwards for anything else that
  calls it, including the task's own video wrapper.
  """
  existing = getattr(env, _ATTR, None)
  if existing is not None:
    return existing

  # The lazy build only works because mjlab's ManagerBasedRlEnv keeps its
  # renderer in `_offline_renderer` (declared None in __init__ as of mjlab
  # 1.5, and read by env.render()). An env without that attribute is a
  # different mjlab -- plugging our renderer into a renamed slot would leave
  # env.render() blind to it, so refuse loudly instead.
  if not hasattr(env, _ATTR):
    raise NotSupported(
        f"This environment has no '{_ATTR}' attribute. rlmcp's lazy "
        "screenshot renderer plugs into mjlab 1.5's "
        f"ManagerBasedRlEnv.{_ATTR}; this mjlab version keeps its renderer "
        "elsewhere, so construct the env with render_mode='rgb_array' to get "
        "frames the eager way."
    )

  try:
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
  except ImportError as exc:  # pragma: no cover - depends on the install.
    raise NotSupported(f"mjlab's offscreen renderer is unavailable: {exc}") from exc

  try:
    renderer = OffscreenRenderer(
        model=env.sim.mj_model,
        cfg=env.cfg.viewer,
        scene=env.scene,
        sim_model=env.sim.model,
        expanded_fields=env.sim.expanded_fields,
    )
    renderer.initialize()
  except Exception as exc:
    raise NotSupported(
        f"Could not create an offscreen renderer: {exc}. On a GPU that is "
        "already near its limit this is usually memory -- a screenshot needs a "
        "render context, and this task may not have room for one."
    ) from exc

  setattr(env, _ATTR, renderer)
  # mjlab's render() checks render_mode before using the renderer.
  if getattr(env, "render_mode", None) != "rgb_array":
    env.render_mode = "rgb_array"
  return renderer


def render(env: Any, env_id: int = 0) -> np.ndarray:
  """Render one environment as RGB.

  The renderer reads its target index from the viewer config every frame, so
  pointing it at another environment is a matter of setting that field around
  the call -- and restoring it, since the same config drives any video the task
  itself is recording.
  """
  ensure_renderer(env)

  viewer_cfg = env.cfg.viewer
  previous = getattr(viewer_cfg, "env_idx", 0)
  viewer_cfg.env_idx = int(max(0, min(env_id, int(env.num_envs) - 1)))
  try:
    frame = env.render()
  finally:
    viewer_cfg.env_idx = previous
  if frame is None:
    raise NotSupported("Renderer returned no frame.")
  return np.asarray(frame)


def release(env: Any) -> bool:
  """Close and drop the renderer, giving its memory back.

  Worth doing on a card that is otherwise full: a screenshot is usually a
  one-off, and holding the context for the rest of a twenty-hour run is not.
  """
  renderer = getattr(env, _ATTR, None)
  if renderer is None:
    return False
  with contextlib.suppress(Exception):
    renderer.close()
  setattr(env, _ATTR, None)
  return True
