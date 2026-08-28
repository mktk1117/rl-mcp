"""Frames from an IsaacLab environment.

IsaacLab renders through the running Kit app rather than through an offscreen
renderer the adapter can build, so there is nothing to construct lazily here --
either the app was launched with cameras enabled and the environment was built
with ``render_mode="rgb_array"``, or no frame exists. Both of those are
decisions made before rlmcp is in the picture, which is why the failure has to
be an explanation rather than an exception nobody can act on.

Pointing that camera at *one* environment is the part that takes work, and the
mechanism differs by IsaacLab version:

* A run with a viewport has a ``viewport_camera_controller``, and moving the
  camera means moving its anchor -- ``cfg.viewer.env_index`` alone does nothing
  while ``origin_type`` is ``"world"``, which is the default.
* A headless IsaacLab 6 run has no controller at all. Its frames come from
  ``env.video_recorder``, whose camera is placed once at construction from a
  fixed eye/lookat and never consults ``cfg.viewer`` again.

Either way the camera goes back where it was found, because the viewer belongs
to whoever else is watching. Without any of this, ``shot --env-id 7`` and
``shot --where …`` answer with the same overview of the whole grid every time:
a picture that looks like an answer and is not one.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from rlmcp.adapters.base import NotSupported

_ENABLE_CAMERAS = (
    "IsaacLab renders through the Kit app: launch it with `--enable_cameras` "
    "and build the environment with `render_mode=\"rgb_array\"`. Without both, "
    "no frame exists to return."
)

# IsaacLab's own ViewerCfg defaults, used when a run has no viewer to ask.
_EYE = (7.5, 7.5, 7.5)
_LOOKAT = (0.0, 0.0, 0.0)

Restore = Callable[[], None]


class _Anchor(NamedTuple):
  """What a viewport camera is tied to, in the terms IsaacLab uses."""

  origin_type: str
  env_index: int
  asset_name: Optional[str] = None
  body_name: Optional[str] = None


def renderer_ready(env: Any) -> bool:
  """Whether a frame would come back without an app-level change."""
  return str(getattr(env, "render_mode", "") or "") == "rgb_array"


def _offsets(env: Any) -> Tuple[Sequence[float], Sequence[float]]:
  """Where the camera sits relative to whatever it is following.

  These are ``cfg.viewer.eye`` and ``lookat``, read the way IsaacLab reads them
  for a per-env origin: as an offset from the thing being followed, not as a
  place in the world.
  """
  viewer = getattr(getattr(env, "cfg", None), "viewer", None)
  return (tuple(getattr(viewer, "eye", None) or _EYE),
          tuple(getattr(viewer, "lookat", None) or _LOOKAT))


def _robot_root(env: Any, robot_name: Optional[str],
                env_index: int) -> Optional[np.ndarray]:
  """Where that env's robot is, in world coordinates."""
  if not robot_name:
    return None
  try:
    positions = env.scene[robot_name].data.root_pos_w
    # IsaacLab 6 hands back a Warp array that carries its torch view.
    positions = getattr(positions, "torch", positions)
    row = positions[env_index]
    if hasattr(row, "detach"):
      row = row.detach().cpu().numpy()
    return np.asarray(row, dtype=float).reshape(3)
  except Exception:  # noqa: BLE001 -- a camera that will not move is not fatal.
    return None


# The viewport path: a run that has a camera controller.


def _anchor_of(viewer: Any) -> _Anchor:
  return _Anchor(str(getattr(viewer, "origin_type", "world") or "world"),
                 int(getattr(viewer, "env_index", 0) or 0),
                 getattr(viewer, "asset_name", None),
                 getattr(viewer, "body_name", None))


def _apply(controller: Any, anchor: _Anchor) -> bool:
  """Move the viewport camera onto ``anchor``. False if it would not move.

  The camera is the controller's to move: writing ``cfg.viewer`` alone leaves
  the viewport where it was. Each origin type has its own method, and a missing
  one is a version difference rather than a fault.
  """
  if controller is None:
    return False
  cfg = getattr(controller, "cfg", None)
  if cfg is not None:
    cfg.env_index = anchor.env_index
  calls = {
      "world": ("update_view_to_world", ()),
      "env": ("update_view_to_env", ()),
      "asset_root": ("update_view_to_asset_root", (anchor.asset_name,)),
      "asset_body": ("update_view_to_asset_body",
                     (anchor.asset_name, anchor.body_name)),
  }
  name, argv = calls.get(anchor.origin_type, ("update_view_to_env", ()))
  if any(arg is None for arg in argv):
    return False
  method = getattr(controller, name, None)
  if not callable(method):
    return False
  try:
    method(*argv)
  except Exception:  # noqa: BLE001
    return False
  return True


def _aim_viewport(env: Any, env_index: int,
                  robot_name: Optional[str]) -> Optional[Restore]:
  controller = getattr(env, "viewport_camera_controller", None)
  viewer = getattr(getattr(env, "cfg", None), "viewer", None)
  if controller is None or viewer is None:
    return None
  previous = _anchor_of(viewer)
  viewer.env_index = env_index
  # Follow the robot when we know which one it is: on a locomotion task the env
  # origin is where the robot started, not where it is now. The env origin is
  # still one environment rather than the whole grid.
  if not _apply(controller, _Anchor("asset_root", env_index, robot_name)):
    if not _apply(controller, _Anchor("env", env_index)):
      viewer.env_index = previous.env_index
      return None

  def restore() -> None:
    viewer.env_index = previous.env_index
    if not _apply(controller, previous):
      viewer.origin_type = previous.origin_type

  return restore


# The recorder path: a headless IsaacLab 6 run, where the frame comes from
# env.video_recorder and no controller exists.


def _place_camera(capture: Any, eye: Sequence[float],
                  lookat: Sequence[float]) -> bool:
  """Put the recording camera at ``eye`` looking at ``lookat``.

  A backend that can move its own camera is asked to. The Kit one cannot: it
  writes its pose to the stage once, on the first frame, and then leaves it
  alone -- so moving it afterwards means saying so to Kit directly as well as
  to the config the recorder would use if it were rebuilt.
  """
  cfg = capture.cfg
  update = getattr(capture, "update_camera", None)
  if callable(update):
    try:
      update(tuple(eye), tuple(lookat))
    except Exception:  # noqa: BLE001
      return False
  else:
    try:
      from isaacsim.core.rendering_manager import ViewportManager
    except ImportError:
      return False
    try:
      ViewportManager.set_camera_view(cfg.camera_prim_path, eye=list(eye),
                                      target=list(lookat))
    except Exception:  # noqa: BLE001
      return False
  cfg.eye = tuple(eye)
  cfg.lookat = tuple(lookat)
  return True


def _aim_recorder(env: Any, env_index: int,
                  robot_name: Optional[str]) -> Optional[Restore]:
  capture = getattr(getattr(env, "video_recorder", None), "_capture", None)
  cfg = getattr(capture, "cfg", None)
  if cfg is None or not hasattr(cfg, "eye") or not hasattr(cfg, "lookat"):
    return None
  root = _robot_root(env, robot_name, env_index)
  if root is None:
    return None
  home = (tuple(cfg.eye), tuple(cfg.lookat))
  eye_offset, lookat_offset = _offsets(env)
  if not _place_camera(capture, root + np.asarray(eye_offset, dtype=float),
                       root + np.asarray(lookat_offset, dtype=float)):
    return None

  def restore() -> None:
    _place_camera(capture, *home)

  return restore


def _aim(env: Any, env_index: int,
         robot_name: Optional[str]) -> Optional[Restore]:
  """Point whichever camera makes the frames at one environment.

  The recorder goes first: when a run has both, the recorder is the one whose
  pose the returned frame actually has.
  """
  for aimer in (_aim_recorder, _aim_viewport):
    restore = aimer(env, env_index, robot_name)
    if restore is not None:
      return restore
  return None


def render(env: Any, env_id: int = 0,
           robot_name: Optional[str] = None) -> np.ndarray:
  """One environment as RGB, with the camera put back where it was."""
  if not renderer_ready(env):
    raise NotSupported(
        f"This environment's render_mode is "
        f"{getattr(env, 'render_mode', None)!r}. {_ENABLE_CAMERAS}")

  wanted = int(max(0, min(env_id, int(env.num_envs) - 1)))
  viewer = getattr(getattr(env, "cfg", None), "viewer", None)
  # Kept in step even when nothing reads it, so anything else looking at the
  # config sees which environment was asked for.
  previous_index = getattr(viewer, "env_index", None)
  if previous_index is not None:
    viewer.env_index = wanted
  restore = _aim(env, wanted, robot_name)
  try:
    frame = env.render()
  except RuntimeError as exc:  # the simulator's own "cannot render" message
    raise NotSupported(f"{exc} {_ENABLE_CAMERAS}") from exc
  finally:
    if restore is not None:
      restore()
    if previous_index is not None:
      viewer.env_index = previous_index

  if frame is None:
    raise NotSupported(f"IsaacLab returned no frame. {_ENABLE_CAMERAS}")
  array = np.asarray(frame)
  # The annotator hands back RGBA; everything downstream writes RGB.
  return array[:, :, :3] if array.ndim == 3 and array.shape[2] == 4 else array
