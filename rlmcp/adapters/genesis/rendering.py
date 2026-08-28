"""Frames, from a camera the training script made before it built the scene.

``scene.add_camera`` carries ``@gs.assert_unbuilt`` and a Genesis environment
builds its scene inside ``__init__``, so by the time rlmcp is wrapped around it
there is no way to add one. That single fact shapes everything here: this module
*finds* a camera and points it, and when there is none it says so in terms of
what the training script would have to change.

Which camera
------------

A run may have several -- the robot's own sensors among them -- so the choice
is not arbitrary. Genesis marks observing cameras with ``debug=True`` and
documents them as recording the simulation without joining the sensor set, so
those are preferred; a camera that changes what the policy perceives is not an
observation of the run. Failing that, the first camera is used, and it is a
mild imposition rather than a correct one.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from rlmcp.adapters.base import NotSupported

NO_CAMERA = (
    "This run has no camera, so it has no frames. Genesis cameras must be "
    "added before the scene is built -- `scene.add_camera(res=(640, 480), "
    "pos=..., lookat=..., GUI=False, debug=True)` before `scene.build(...)` -- "
    "so rlmcp cannot make one for a scene that is already built. Everything "
    "else about this run works without it."
)


def cameras(env: Any) -> List[Any]:
  """Every camera on this environment's scene, in the order they were added."""
  scene = getattr(env, "scene", None)
  visualizer = getattr(scene, "visualizer", None) or getattr(
      scene, "_visualizer", None)
  found = getattr(visualizer, "cameras", None)
  return list(found) if found else []


def observing_camera(env: Any) -> Optional[Any]:
  """The camera to record through, or None when the scene has none."""
  found = cameras(env)
  if not found:
    return None
  for camera in found:
    if getattr(camera, "debug", False):
      return camera
  return found[0]


def rendered_envs(env: Any) -> Optional[Sequence[int]]:
  """Which environments the renderer draws, when the scene will say.

  Genesis defaults this to ``[0]``, and it is fixed before the build like the
  camera is. A run that means to look at env 7 has to have said so at
  construction, and being told that up front beats being handed a picture of
  env 0 with env 7 written under it. ``None`` means the scene did not say,
  which is not the same as "every env" and is reported as not knowing.
  """
  scene = getattr(env, "scene", None)
  for holder in (scene, getattr(scene, "vis_options", None)):
    options = getattr(holder, "vis_options", holder)
    found = getattr(options, "rendered_envs_idx", None)
    if found is not None:
      return list(found)
  return None


def can_render(env: Any, env_id: int) -> Optional[str]:
  """Why a frame of ``env_id`` cannot be made, or None when it can."""
  if observing_camera(env) is None:
    return NO_CAMERA
  drawn = rendered_envs(env)
  if drawn is not None and env_id not in drawn:
    return (
        f"Environment {env_id} is not rendered: this scene draws "
        f"{sorted(drawn)}. Which environments can be looked at is fixed before "
        "the scene is built, with "
        "`vis_options=gs.options.VisOptions(rendered_envs_idx=[...])`, so this "
        "run can only be shown the ones it was built to draw."
    )
  return None


def render(env: Any, env_id: int = 0, follow: bool = True) -> np.ndarray:
  """An RGB frame of ``env_id``, with shape (H, W, 3).

  The camera is moved onto the robot in the environment asked for and put back
  afterwards, so a screenshot does not quietly re-aim a camera the training
  script set up deliberately. Parallel environments sit at spaced world
  origins, so "point at env 7" is an ordinary pose change here rather than the
  anchor negotiation IsaacLab needs.
  """
  refusal = can_render(env, env_id)
  if refusal:
    raise NotSupported(refusal)
  camera = observing_camera(env)
  restore = _pose_of(camera)
  try:
    if follow:
      _aim_at(camera, env, env_id)
    frame = camera.render(rgb=True)[0]
  finally:
    _restore(camera, restore)
  return _as_rgb(frame)


def _aim_at(camera: Any, env: Any, env_id: int) -> None:
  """Point the camera at one environment's robot, keeping its framing.

  The offset between the camera and what it was looking at is preserved, so a
  script that framed its shot closely keeps that shot and only changes subject.
  """
  base = getattr(env, "base_pos", None)
  if base is None or len(base) <= env_id:
    return
  target = np.asarray([float(v) for v in base[env_id]], dtype=float)
  pos = _vector(getattr(camera, "pos", None))
  lookat = _vector(getattr(camera, "lookat", None))
  offset = (
      pos - lookat if pos is not None and lookat is not None
      else np.array([2.0, 0.0, 1.0])
  )
  camera.set_pose(pos=tuple(target + offset), lookat=tuple(target))


def _vector(value: Any) -> Optional[np.ndarray]:
  try:
    out = np.asarray([float(v) for v in value], dtype=float)
  except (TypeError, ValueError):
    return None
  return out if out.shape == (3,) else None


def _pose_of(camera: Any) -> Optional[tuple]:
  pos, lookat = _vector(getattr(camera, "pos", None)), _vector(
      getattr(camera, "lookat", None))
  return (pos, lookat) if pos is not None and lookat is not None else None


def _restore(camera: Any, pose: Optional[tuple]) -> None:
  if pose is None:
    return
  try:
    camera.set_pose(pos=tuple(pose[0]), lookat=tuple(pose[1]))
  except Exception:
    # A camera that cannot be put back is not a reason to lose the frame that
    # was already taken; the next render re-aims from wherever it now is.
    pass


def _as_rgb(frame: Any) -> np.ndarray:
  """Genesis returns ``(rgb, depth, seg, normal)``; this is the rgb, as uint8.

  A batched renderer hands back one image per rendered environment, so a stack
  is reduced to its first image rather than returned as something no caller
  expects.
  """
  array = np.asarray(frame)
  if array.ndim == 4:
    array = array[0]
  if array.ndim != 3:
    raise NotSupported(
        f"This camera returned an image of shape {array.shape}, which is not a "
        "single RGB frame."
    )
  array = array[..., :3]
  if array.dtype != np.uint8:
    array = np.clip(array * 255.0 if array.max() <= 1.0 else array,
                    0, 255).astype(np.uint8)
  return array
